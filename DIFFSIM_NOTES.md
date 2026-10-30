# DiffSim gradient (de Saxcé full-cone central-path) 이식 노트

작성 2026-10-30. 코드: `diffsim_grad.py`, 검증: `verify_diffsim.py`, 학습: `train_cimpc.py --backend diffsim`.
CI-MPC 이식 노트 (`CIMPC_NOTES.md`) 의 후속. 같은 비대칭 구조 (forward hard, backward 만 교체) 이고
`cimpc_grad.make_rollout` 의 `jacobian_fn` 규약 ((f_x, f_u) 공급) 을 그대로 쓴다.

## 1. 참조

| 무엇 | 어디 |
|---|---|
| 논문 | `~/Downloads/DiffSim.pdf` "Differentiable Contact Dynamics for Legged Locomotion" (T-RO 2026 투고). Sec. V (eq. 12-16, Alg. 1), App. A-B, VIII-D |
| C++ 원본 | `~/workspace/DiffSim/aQuadrupedBPTT/lib/contactSolverGradient.cpp` `GetGradientRelaxedV3` (mode_of_relaxation 5). Hound 45 kg, Pinocchio + RaiSim, dt 25 ms |
| 저자 노트 | `~/workspace/DiffSim/PROJECT_NOTES.md` (mode 0/1/5 사다리, delta 의 논리, CI-MPC 와의 두 가지 차이) |
| 2D 재현 | `~/workspace/canonex_relaxedynaflow/contact_diffsim.py` (JAX, custom_vjp 형태) |

## 2. 수학 (한 substep, 접촉별 [T, T, N] 순서)

    sig = A lam + b,     A = J M^-1 J^T,   b = J (v + qacc_smooth dt),  b_n += phi/dt (+ delta)
    lam~ = (mu lam_n, lam_t),   sig~ = (sig_n/mu + |sig_t|, sig_t)          (7) 단위 마찰 스케일링
    F(lam) = lam~ o sig~ - rho e = 0                                          (12) central path
        F_t = mu lam_n sig_t + y0 lam_t,  F_n = mu lam_n y0 + lam_t . sig_t - rho,  y0 = sig_n/mu + |sig_t|
    M_lam = L(sig~) C,  M_sig = L(lam~) D(sig)                                (14) / App. B
    Newton:  (M_lam + M_sig A + beta |F| I) dlam = -F,  lam <- lam + alpha dlam   Alg. 1
    감도:    dlam/dxi = -(M_lam + M_sig A)^-1 M_sig (dA/dxi lam* + db/dxi)       (15)
    f_x, f_u: v_next = v + qacc_smooth dt + M^-1 J^T lam*,  q_next = integrate(q, v_next dt)  (cimpc_grad 와 동일 조립)

CI-MPC 와의 차이 (저자 노트 §7): (a) 원뿔 전체 완화 vs normal 만, (b) 평가점을 Newton 으로 central path 위에
올림 vs 닫힌꼴 재구성, (c) separating 접촉도 시스템에 남음 (인덱스 집합 없음, 항상 12x12).
normal-only 극한에서 (15) 는 CI-MPC eq.(19) `A + rho D` 로 환원된다.

## 3. 구현 대응

| 논문 / C++ | `diffsim_grad.py` |
|---|---|
| `compute_F_perfoot` | `central_path_residual(lam, sig, mu, rho)` |
| `LyC`, `LxD` (M_lam, M_sig) | `arrow_blocks(lam, sig, mu)` -> (fnum,3,3) x 2, `block_diag` |
| Newton 루프 (kMaxNewton, LM beta, 옵션 LS) | `relaxed_point(lam0, A, b, mu, rho, beta, n_newton, n_linesearch, newton_tol)` (fori_loop, 고정 반복수) |
| `DIFFSIM_V3_SPURIOUS_CUT` 조건 + 잔차 | `physical_mask` (lam_n>0, cone<1e-6, sig_n>-0.1, 상대 잔차) |
| `A_full`, `b_eq`, kinematic hessian, phi/dt 보정 | `make_delassus(...)` 의 `A_and_b(theta, v, u)` 를 jacfwd (dA, db 가 dJ/dq, dM/dq, d(phi/dt)/dq 를 전부 포함) |
| `df_dz` 조립 | `_assemble` (cimpc_grad 와 동일, 위치 적분은 Lie-group 정확 미분) |
| substep 연쇄 | `diffsim_jacobian(..., n_sub, forward)` (cimpc_jacobian 과 동일 구조) |

sig_t / |sig_t| 는 |sig_t| <= 1e-12 에서 0 (App. B 규약, C++ kEpsT). Newton 의 clamping 시작점 (sig = 0) 에서 쓰인다.

## 4. 원본과 다르게 한 것 (의도적)

1. **line search 기본 ON** (`n_linesearch=8`; C++ 는 env `DIFFSIM_V3_LS_ENABLE` 기본 OFF). 2D 재현에서 원뿔 경계 근처
   clamping 시작점이 spurious root 로 가는 것이 확인돼 켰다. `--n-linesearch 0` 이면 C++ 와 동일.
   (검증 [A] 에서는 6 regime x 4 rho 모두 LS 유무와 무관하게 같은 물리적 root 로 수렴했다.)
2. **spurious root 는 접촉별 절단** (dlam_k = 0, 2D 재현과 동일). C++ 는 (옵션, 기본 OFF) 스텝의 f_x, f_u 전체를 0 으로 둔다.
   논문 VIII-D 의 "sets the corresponding contact gradient to zero" 를 따랐다. `--no-cut` 으로 끌 수 있다.
3. **수렴 / 절단 판정이 상대 기준** `|F_k| <= cut_tol max(rho, |lam~_k||sig~_k|)`. 학습이 float32 라 C++ 의 절대 1e-10 은 못 쓴다.
4. **lam_eval**: 논문 VI-A 와 C++ 는 df/dx 의 나머지 인자 (M^-1 J^T lam, dJ^T lam / dq) 도 lam* 에서 평가한다 -> 기본 `"relaxed"`.
   2D 재현은 lam_h 를 썼다 (`"hard"`, 옵션). "relaxed" 가 완화 동역학의 정확한 미분이다 (검증 [D]).
5. **forward="mjx" 경로**: DynaFlow 의 MJX soft forward 를 그대로 두고 backward 만 DiffSim. Newton warm start 를
   MJX 접촉력에서 뽑는다. 논문 설정은 아니고 ablation 용이다 (CI-MPC 의 (b) 비교군과 짝).

## 5. rho, delta 의 단위와 스케일

F 는 impulse x velocity. 논문 rho ~ 1 (closed loop) / 1e-4..1e-2 (진단) 은 45 kg, dt 25 ms (lam_n ~ 2.8 N s) 기준.
Go1 (12 kg) substep: lam_n ~ 0.15 N s (dt 0.005), ~0.3 N s (dt 0.01). clamping 접촉의 완화 접촉속도는 sig_n* ~ rho / lam_n 이므로
같은 완화 정도는 **rho ~ 0.05-0.1** (논문의 1 에 해당), 진단은 1e-5..1e-3. 검증 [A]: |lam* - lam_h| / rho ~ 0.5-2 (clamping).
delta 는 속도 [m/s] 라 로봇 무관: 논문 0.2 (고마찰) ~ 0.5 (gait 탐색).

CI-MPC 의 rho 와는 **다른 양** 이다 (CI-MPC: A 에 rho/lam_n^2 를 더함, 1e-4 가 A 의 ~1%). 두 백엔드의 rho 를 같은 값으로 두고 비교하면 안 된다.

## 6. 검증 결과 (`JAX_PLATFORMS=cpu python verify_diffsim.py`, x64, solver 40/50/1e-11)

(아래 채움)
전체 PASS (A/B 117+5 조정 후, C/D/E 37, F 1). 로그: 세션 scratchpad `verify_diffsim_{AB,CDEF,F}.log`.

### [A] Newton (Alg. 1), dt=0.01
* hard 해의 잔차 |F(lam_h)| (rho=0): clamping / separating / flight / landing / mixed 발은 1e-14 이하 (= hard 해가 정확히 de Saxcé root).
  **sliding 발은 0.02-0.03 (|lam||sig| 의 9-16%)**. hard solver (Hwangbo bisection) 의 sliding 해는 이방성 Mapp 에서의 eq.(7)
  최소점이라 lam_t 가 sig_t 에 정확히 반평행이 아니다 (CIMPC_NOTES 9절의 cos = -0.98). de Saxcé 는 등방 MDP 를 가정하므로
  F_t != 0. 원본 (RaiSim forward) 도 같은 상황이고 C++ 는 이걸 `F_norm_warm` 으로 로그한다. 구현 문제가 아니다.
* 6 regime x rho in {1e-6, 1e-4, 1e-2, 1e-1} x {LS, noLS}: 전부 |F| < 1e-15 로 수렴, 모든 접촉이 물리적 root. LS 유무로 결과 차이 없음.
* |lam* - lam_h| / rho = 0.5-2 (clamping 계열, rho 에 선형). sliding 은 rho 와 무관하게 2.3e-2 (위 불일치만큼).
* 작은 rho 에서 반복수 40 은 freeze 임계가 1e-14 상대라 안 멈춘 것이고 |F| 는 2-4 회에 이미 1e-16 이다.

### [B] C++ 직역 대조
`compute_F_perfoot`, `LyC`(M_lam), `LxD`(M_sig) 를 numpy 로 직역해 랜덤 (lam, sig) 에서 0 오차. sig_t = 0 케이스는 C++ 가
|sig_t| 를 1e-12 로 클램프해서 y0 에 넣는 탓에 2e-13 / 8e-13 차이 (의도된 규약 차이). Newton 1 스텝 (beta=0.1, LS 없음) 1e-12
(LU 구현 차이), 수렴점 1e-16.

### [C] 감도 식 (15)
명시적 블록 `M_lam + M_sig A` vs 잔차 R(lam, xi) 의 jacfwd: 1e-10 이하. dlam/dxi (명시적 vs autodiff IFT) 1e-9 이하,
(f_x, f_u) 를 `verify_cimpc._assemble` (독립 J, 독립 조립) 로 만든 것과 1e-9 이하. 6 regime 전부.

### [D] end-to-end (가장 강한 검증)
"hard 해 -> Newton -> lam* 적용" 완화 동역학 한 스텝을 중앙 FD (48 열, x64) 한 것과 diffsim step Jacobian
(lam_eval=relaxed) 이 **sliding 포함 6 regime + delta=0.2 3 케이스 전부 1e-5 이내** 로 일치. 즉 (15) 와 조립은 완화 문제의
정확한 미분이고, delta 는 평가점만 바꾸고 db 에는 안 들어간다는 논문 V-C 의 서술과 일치.

### [E] rho -> 0 극한 (rho=1e-10, delta=0)
clamping / flight / mixed / landing / partial: hard step 의 FD 와 1e-5 이내, cimpc(rho=0) 와도 1e-5 이내 -> 논문 V-D 의
"exact fixed-mode 감도 회복" 확인. sliding (4발, slip 1.1 m/s): FD 와 1.3e-1, cimpc(rho=0) 와 4.4e-1, 정확한 implicit 미분과
1.3e-1. 이것은 [A] 의 불일치의 귀결: rho -> 0 에서 Newton 은 solver 해가 아니라 등방 MDP root 로 가므로 평가점 자체가 다르다.
**sliding 에서는 세 backward (FD = solver 의 진짜 미분, CI-MPC 의 E_s 고정, DiffSim 의 등방 MDP) 가 모두 서로 다르다.**

### [F] 연속성 스캔 (논문 Fig. 3 유사)
서 있는 로봇의 base v_z 를 -0.2 -> +0.3 m/s 로 스캔, FL 발 d lam_n / d v_z:

| | exact (FD) | rho=1e-3 | rho=1e-2 | rho=5e-2 | rho=1e-2, delta=0.1 |
|---|---|---|---|---|---|
| clamping (v_z=-0.2) | -0.609 | -0.604 | -0.566 | -0.469 | -0.539 |
| 전이 직전 (0.22) | -0.416 | -0.235 | -0.247 | -0.260 | -0.147 |
| separating (0.24) | 0 (점프) | -0.179 | -0.224 | -0.249 | -0.132 |
| 인접 샘플 최대 변화 | 0.42 (점프) | 0.056 | 0.026 | 0.011 | 0.026 |

rho > 0 이면 전이에서 연속이고 separating 쪽으로 꼬리가 남는다 (liftoff 방향 정보). delta=0.1 열은 delta=0 열을 정확히
v_z + 0.1 만큼 당긴 것 (b_n 평행이동) 으로 논문 Fig. 4(b) 그대로. v_z=0.1 근처 exact 의 작은 점프는 다른 발의 모드 변화.

## 7. 비교군 (train_cimpc.py)

| | forward | backward | 명령 |
|---|---|---|---|
| (a) DynaFlow | MJX soft | MJX autodiff | `--backend mjx` |
| (b) | MJX soft | CI-MPC (19) | `--backend cimpc --rho 1e-4` |
| (c) CI-MPC | hard | CI-MPC (19) | `--forward cimpc --backend cimpc --rho 1e-4` |
| (d) DiffSim | hard | de Saxcé (15) | `--forward cimpc --backend diffsim --rho 0.05 [--delta 0.2]` |
| (e) | MJX soft | de Saxcé (15) | `--backend diffsim --rho 0.05` |

(d) 가 논문 설정. (e) 는 Newton warm start 를 MJX 접촉력에서 받는 ablation. 모두 `--n-sub 2` 로 맞춘다
(CIMPC_NOTES 6절: forward cimpc 는 backward n_sub 가 forward 와 같아야 한다).

## 8. 학습 경로 smoke test (GPU, float32, batch 64, n_sub 2, 30 step)

| 설정 | step 0 loss | step 30 loss | 컴파일 | 30 step |
|---|---|---|---|---|
| (d) `--forward cimpc --backend diffsim --rho 0.05` | 0.40973 | 0.243 | ~440 s* | ~6 s |
| (e) `--backend diffsim --rho 0.05` | 0.33979 | 0.280 | ~300 s* | ~4 s |
| (c) `--forward cimpc --backend cimpc --rho 1e-4` | 0.40973 | 0.234 | ~435 s* | ~5 s |

NaN 없음. (c) 와 (d) 는 forward 가 같으므로 step 0 loss 가 bit-identical 이고 이후 gradient 차이로 갈라진다.
step 당 비용은 CI-MPC 와 같다 (Newton 12x12 는 jacfwd 48 tangent 에 비해 무시할 수준). *컴파일 시간은 CPU 에서
검증 스크립트가 동시에 돌던 중의 값이라 과대.

## 9. 학습 테스트 (GPU, float32, B=256, H=16, 2000 step, n_sub 2, hard forward, seed 0)

스크립트 `rho_sweep_results/diffsim/run.sh`, 로그/집계 `rho_sweep_results/diffsim/{*.log, results.json}`,
체크포인트 `checkpoints/2026-10-30-*_{diffsim_rho0.05_delta0.2, diffsim_rho0.05_delta0, cimpc_rho1e-4_fwdhard}/state_{1000,2000}`.
세 런은 forward 가 같아 step 0 loss 가 동일 (0.40676) 하고 gradient 차이로만 갈라진다.

| 설정 | 후반 25% 평균 | 최소 | step 2000 | base_pos | joint_pos | 시간/런 |
|---|---|---|---|---|---|---|
| (d) diffsim rho 0.05, delta 0.2 | **0.02079** | 0.01826 | 0.02109 | 0.00782 | 0.0408 | ~17 min + 컴파일 7 min |
| (d) diffsim rho 0.05, delta 0 | 0.02133 | 0.01870 | 0.02196 | 0.00782 | 0.0414 | 같음 |
| (c) cimpc rho 1e-4 (hard fwd) | 0.02215 | 0.01908 | 0.02231 | 0.00778 | 0.0421 | 같음 |
| 참고: pass1 mjx (MJX fwd, CIMPC_NOTES 6.1) | 0.02613 | | | | | |
| 참고: pass1 cimpc rho 1e-4 (MJX fwd) | 0.02736 | | | | | |

* **DiffSim gradient 로 학습이 된다.** 발산/NaN 없음, 전 구간에서 loss 곡선이 CI-MPC 와 같은 모양 (같은 배치에서 같이 튄다: step 600, 1000, 1900).
* 순위는 delta 0.2 < delta 0 < cimpc 이지만 격차 (2.5%, 4%) 는 CI-MPC 의 시드 std (±0.0017~0.0023, 6.1 절) 안이다.
  **단일 seed 로는 구별 안 됨.** 시드 3개 이상 돌려야 한다.
* delta 는 거의 영향이 없다 (6.1 절의 rho 가설과 같은 이유: 모방 과제는 접촉 모드를 바꿀 필요가 없다).
* hard forward 세 런이 전부 pass1 의 MJX forward 런들보다 낮다 (0.021~0.022 vs 0.026~0.030). 그러나 pass1 은 f_x 버그
  (CIMPC_NOTES 8.8) 가 있을 때 값이고, forward 가 다르면 같은 action 에 다른 궤적이 나오므로 loss 바닥 자체가 다를 수 있다.
  forward 간 비교는 `eval_metrics.py` 의 TRE 로 해야 한다.

## 10. 다음

* rho 스윕 (0.01 / 0.05 / 0.1) 과 delta (0 / 0.2) 로 (d) 학습, eval_metrics.py 로 (a)(c) 와 TRE/SAE 비교.
* board drag 로 갈 때 `make_delassus` / `physical_mask` 는 접촉 수와 무관하므로 `foot_ids` 자리의 geom-pair 일반화
  (접촉별 normal, 두 body Jacobian, per-pair mu) 만 필요. CI-MPC 쪽과 같은 일반화 지점이다.
