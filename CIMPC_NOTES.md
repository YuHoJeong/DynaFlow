# DynaFlow x CI-MPC gradient 교체 — 1단계 기록

목표: DynaFlow 의 동역학 gradient 를 CI-MPC (Kim et al., *IJRR* 2025) 의
relaxed complementarity gradient 로 바꿔도 motion imitation 이 되는지 본다.
그 전에 저자들이 지적한 **"CI-MPC 가 DynaFlow 규모로 GPU 병렬화가 되는가"** 를 먼저 판정한다.

관련 파일: `cimpc_grad.py` (구현), `verify_grad.py` (검증)

---

## 1. 왜 gradient 만 바꾸는가

DynaFlow 에서 동역학이 학습에 관여하는 지점은 **한 군데**다.

- `main.py:143` / `flow_matching.py:31` → `diffsim.batched_rollout(env, q0, dq0, actions)`
- 그 결과가 `main.py:180` 의 `nnx.value_and_grad(loss_fn)` 안으로 들어간다.

그래서 `batched_rollout` 자리에 `custom_vjp` 로 감싼 롤아웃을 넣으면
**forward 는 MJX 그대로, backward 만** 교체된다.
forward 궤적과 loss 값이 baseline 과 bit-identical 이므로,
학습 결과의 차이는 전적으로 gradient 차이에서 온다 (controlled comparison).

부수 효과로, forward solver (Gauss-Seidel + bisection) 를 JAX 로 옮길 필요가 없어진다.
아래 3절에서 보듯 **이게 GPU 병렬화의 최대 난관을 통째로 없앤다.**

### 1.1 요청의 해석 -- "gradient 를 바꿔라" 는 forward 교체를 포함하지 않는다

원 요청은 "DynaFlow 로 imitation 학습, 그 다음 **동역학 gradient** 를 CI-MPC gradient 로
교체" 다. 문자 그대로 gradient 교체이고, 아래 1.2 에서 보듯 CI-MPC 자체가 gradient 기법이라
"CI-MPC gradient 로 바꾼다" 에 forward 교체가 들어갈 여지가 없다.
forward 까지 바꾸려면 "접촉 solver 를 CI-MPC 의 hard contact solver 로 교체" 라는
별개 요청이어야 한다. 현재 구현은 요청을 좁힌 것이 아니다.

### 1.2 정정 -- CI-MPC 원본도 relaxation 은 backward 에만 쓴다

한때 "forward 와 backward 가 같은 relaxed 모델일 때 rho 가 가장 잘 시험된다" 고
정리한 적이 있는데 **틀렸다.** C++ 원본 (`~/workspace/general_contact_TO`) 에서
`rho` / `A_lw_relaxation` 이 등장하는 파일은 `contactSolverGradient.*`,
`contactGradient.cpp` 등 gradient 계열뿐이다. forward 접촉 solver
(per-contact Gauss-Seidel + bisection) 는 **hard contact** 로 lambda 를 정확히 풀고,
그 lambda 를 받아 `[A + rho D]^-1` 로 complementarity 를 완화하는 건 gradient 계산
단계뿐이다. 논문의 아이디어 자체가 "forward 는 물리적으로 정확하게, gradient 만
매끄럽게" 다.

따라서 현재 구조(forward MJX 그대로 + backward CI-MPC gradient)는 **논문과 같은
구조** 다. 원본과 다른 점은 relaxed 여부가 아니라 **forward 접촉 모델이
hard (CI-MPC) 냐 soft (MJX) 냐** 하나다.
6.1 절의 "rho 효과가 안 보인 이유" 후보에서 "backward 만 relaxed 라서" 는 지운다.

### 1.3 forward 를 CI-MPC solver 로 바꾸는 건 dt 문제가 아니다

6.2 절의 "MJX forward 를 dt=0.02 로 키워 backward 에 맞추자" 를 기각한 건 MJX soft
contact 의 시간상수 (~0.01s) 때문이다. 반대로 CI-MPC solver 를 JAX 로 옮겨
dt=0.005 x 4 로 밟는 것은 hard contact 라 dt 에 둔감해서 물리적으로는 가능하다.
그래도 안 하는 이유는 두 가지다.

1. **GPU 배치 부적합** (3.2 절): Gauss-Seidel 은 접촉 간 순차 의존, bisection 은
   데이터 의존 종료조건이라 고정 반복수로 최악 비용을 항상 내야 하고,
   dt=0.005 면 그 비용이 4배다.
2. **통제 변수 상실**: forward 를 바꾸면 궤적·loss 값이 달라져 MJX autodiff 와의
   차이가 gradient 때문인지 접촉 모델 때문인지 분리되지 않는다.
   (처음에 "참조 궤적이 MuJoCo soft contact 로 만들어져서 불일치" 라고 썼는데
   틀렸다. `loadData.py` 를 보면 데이터는 **Isaac 규약** 에서 변환한 것이라
   MJX 와도 이미 접촉 모델이 다르다. hard contact 로 바꾼다고 불일치가
   새로 생기는 게 아니다.)

즉 gradient-only 교체는 "gradient 의 효과" 를 통제된 조건에서 보는 1단계다.
forward 까지 CI-MPC 로 바꾸는 것이 저자의 원래 질문일 가능성은 1.4 절에서 따로 다룬다.

### 1.4 재해석 -- 저자의 "GPU 병렬화" 질문은 forward 교체를 전제한다

gradient-only 교체는 3.2 절에서 보듯 GPU 병렬화 난관을 **회피** 한다.
그런데 저자들이 "CI-MPC 가 DynaFlow 규모로 GPU 병렬화가 되는가부터 확인하라" 고
했다는 건, 병렬화가 문제가 되는 부분 = **forward 접촉 solver** 를 GPU 에 올릴
것을 전제했다는 뜻이다. 즉 원래 의도는 "forward 도 backward 도 CI-MPC 동역학으로
바꾸고, 그래도 DynaFlow 가 학습되는가" 일 가능성이 크다.
gradient-only 결과(6절) 는 그 질문의 절반만 답한 것이다.

과학적으로도 그쪽이 맞다. `D = diag(1/lambda_n^2)` 는 hard contact solver 가
complementarity 를 정확히 만족시키는 lambda 를 전제로 유도된 항이다.
MJX soft contact 의 lambda 에 이 항을 얹는 건 논문이 검증한 조합이 아니다.
6.1 절에서 rho 효과가 안 보인 이유가 "soft contact 와의 상호작용" 일 수 있는
이상, hard contact forward 없이는 rho 를 제대로 시험할 수 없다.

#### 3.2 절의 "나쁨" 판정은 과장이었다

C++ forward solver (`contactSolverGradient.cpp:480-640`) 의 실제 루프 구조:

```
outer (multi_ind)   : 최대 10회, ||dlambda|| < 1e-6 이면 조기 종료
  per-contact (GS)  : fnum = 4
    bisection       : 최대 8회, 1e-6 조기 종료
```

고정 반복수로 펼쳐도 스텝당 10 x 4 x 8 = 320 회의 3x3 연산이다.
fnum=4 라 "접촉 간 순차 의존" 은 길이 4 짜리 체인일 뿐이고, MJX 의 constraint
solver 도 어차피 `while_loop` 기반 반복 solver 다. `vmap` + `fori_loop` 으로
16,384 인스턴스를 돌리는 데 구조적 장벽은 없다. 실제 비용은 재 봐야 하지만
자릿수가 MJX 와 다를 이유가 없다.

#### 전체 교체 시 재사용 가능한 것

- `make_rollout` / `make_batched_rollout` 의 custom_vjp scaffold: forward 함수만
  `diffsim.rollout` 에서 JAX CI-MPC rollout 으로 바꾸면 된다.
- `cimpc_jacobian`: 그대로 backward 로 쓴다. lambda 를 MJX `contact_force` 대신
  자체 solver 출력에서 받으면 되고, 그러면 C++ 처럼 `E_s` 를 lambda 에서 뽑을 수
  있어 `slide_ratio` 튜닝값(4.7 절) 이 없어진다.
- `smooth_acc_and_mass`, `contact_jacobians`, `foot_height`: M, nle, J, phi 를
  MJX 운동학에서 얻는 건 그대로다. 접촉 solver 만 새로 쓴다.

#### 열리는 질문 (구현 전에 정해야 함)

1. forward dt: 논문은 0.025 단일 스텝. hard contact 는 dt 에 둔감하므로 0.02 x 1
   과 0.005 x 4 를 둘 다 시도. 6.2 절의 n_sub 문제는 forward 와 backward 를
   같은 dt 로 두면 자동으로 사라진다.
2. 참조 데이터 재현성: 데이터가 Isaac 에서 왔으므로 (1.3 절) 어느 시뮬레이터든
   불일치는 있다. `eval_metrics.py` 의 SAE 를 새 forward 로 먼저 재서
   hard contact 가 참조 전이를 얼마나 재현하는지 봐야 학습 loss 바닥을 해석할 수 있다.
3. 비교 조건: (a) MJX fwd + MJX grad, (b) MJX fwd + CI-MPC grad [완료],
   (c) CI-MPC fwd + CI-MPC grad, (d) CI-MPC fwd + rho=0. (c) vs (d) 가 rho 의
   진짜 시험이고, (b) vs (c) 가 forward 접촉 모델의 효과다.

---

## 2. 구현 구조

```
make_batched_rollout(env, jacobian_fn) -> (qpos_traj, qvel_traj)
    forward : diffsim.rollout 을 그대로 호출  (재구현 금지 -- 아래 함정 3 참조)
    backward: DDP-style adjoint recursion
                lam_i = f_x[i]^T (lam_{i+1} + output_cotangent_i)

jacobian_fn(env, qpos, qvel, warmstart, u, qpos_next) -> (f_x, f_u)
    f_x : (2nv, 2nv)  = d x_{i+1} / d x_i   (tangent 좌표)
    f_u : (2nv, nu)
```

backward 를 스텝별 `(f_x, f_u)` 의 adjoint recursion 으로 짠 건 의도적이다.
CI-MPC 가 실제로 계산하는 게 정확히 이 형태 (DDP backward pass) 라서,
`jacobian_fn` 만 갈아끼우면 나머지는 손댈 필요가 없다.

### 교체 granularity: substep 이 아니라 control step

`simulation_dt=0.005`, `control_dt=0.02` → `n_substeps=4`.
CI-MPC 논문의 `dt=0.025` 는 substep 이 아니라 **control step** 에 대응한다.
그래서 `(f_x, f_u)` 도 control step 단위로 정의했다.
substep 단위로 하면 gradient 개수가 4배(65,536개)가 되고 dt 도 안 맞는다.

### 함정 1 — quaternion tangent 공간 (`nq=19` vs `nv=18`)

MuJoCo 의 `qpos` 는 nq=19 (free joint quaternion), `qvel` 은 nv=18 이다.
CI-MPC 의 `f_x` 는 nv 차원 tangent 공간에서 정의된다.
둘을 잇는 `E(q) : R^18 -> R^19` 와 left inverse `E^+(q)` 가 필요하다.

```
출력 cotangent 당기기 :  dL/dq_tangent = E(q_next)^T  dL/dqpos_next
입력으로 밀기         :  dL/dqpos_0    = (E^+(q_0))^T dL/dq_tangent_0
```

**이걸 빼먹으면 quaternion 성분에서만 조용히 틀린 gradient 가 나온다.**
`verify_grad.py` test 1 이 `mj_integratePos` 의 수치 야코비안과 대조해서
MuJoCo 규약(quaternion 좌/우 곱, local vs global 각속도)까지 확인한다.

### 함정 2 — 검증용 초기 상태

`qpos0` 는 다리를 편 자세다. 여기에 trunk 높이만 0.27 로 낮추면
**발이 지면에 17.7cm 파묻히고** (`contact dist = -0.177`,
`qfrc_constraint ~ 1.9e4`) 유한차분이 완전히 무의미해진다.
반드시 `nominal_ptarg` 관절각 + 접지 높이(`STAND_Z = 0.2865`) 를 써야 한다.
action 도 정규화된 `[-1,1]` 이 아니라 **raw joint target** (`env.ctrl_range` 안) 이다.

### 함정 3 — forward 를 재구현하지 말 것

`diffsim.rollout` 의 scan 을 복제하면 XLA 가 다르게 융합해서 roundoff 가 달라지고,
접촉 모드 경계 근처에서 그 roundoff 가 O(1e-6) 까지 증폭된다.
`diffsim.rollout` 을 그대로 호출해야 bit-identical 이 보장된다.

---

## 3. GPU 병렬화 판정 (저자들이 물어본 것)

### 3.1 C++ 바인딩 경로 — 느려서가 아니라 통합 비용 때문에 접는다

CI-MPC 코드(`~/workspace/general_contact_TO`)는 C++ / Pinocchio / Eigen / qpSWIFT /
raisim 기반 **단일 스레드 CPU** 다. 논문 Sec 5.3 기준 gradient 1회 = **70 us**
(4접촉 clamping, Ryzen 5 3600X).

DynaFlow 는 `BATCH_SIZE=1024`, `HORIZON=16`:

```
학습 1스텝당 control-step gradient = 1024 x 16 = 16,384 개
16,384 x 70us = 1.15 초 / 학습 스텝  (단일 스레드 CPU)
x 50,000 스텝  = 57,344 초 = 약 16 시간
```

**주의: 이 숫자만 보면 실격이 아니다.** 같은 GPU 에서 잰 plain MJX autodiff 가
`1192 ms/step` (B=1024, H=16, x64) 이라 50,000 스텝에 약 16.6 시간이다.
즉 raw 연산량은 같은 자릿수다. gradient-only 교체라면 forward 는 GPU 에 남고
backward 만 CPU 로 나가므로, 총 소요는 대략 3배 느려지는 수준
(GPU forward ~0.6s + CPU backward 1.15s) 으로 추정된다. 데이터 전송도
16,384 x (36x36 + 36x12) x 8B = 226 MB/step 이라 PCIe 로 ~23ms, 치명적이지 않다.

접는 이유는 속도가 아니라 통합 비용이다:

1. `jax.pure_callback` 이 `vmap` + `scan` 안에 들어가면 XLA 의 비동기 디스패치와
   버퍼 donation 이 막힌다. 위 추정치는 낙관적 하한이다.
2. CI-MPC 코드는 `nq=19, nv=18, nu=12, fnum=4` 와 Hound URDF 에 하드코딩돼 있고
   단일 스레드다. batch 처리 + 스레드 풀 + 재진입 안전성을 새로 얹어야 한다.
3. jit / f32-f64 전환 / multi-GPU 를 전부 포기하게 된다.
4. 경계를 넘나드는 디버깅 비용.

반면 JAX 재구현은 이 전부를 유지하면서 GPU 에서 batch 로 돈다.
**"불가능해서" 가 아니라 "얻는 게 없어서" 접는 쪽이 정확한 표현이다.**

### 3.2 JAX 재구현: 부품별 GPU 적합성

| CI-MPC 구성요소 | batch 적합성 | 근거 |
|---|---|---|
| Delassus `A = J M^-1 J^T` | 좋음 | 12x12 dense, `vmap` 자명 |
| `[A + rho D]^-1` 풀이 | 좋음 | 12x12 batched solve |
| `dA/dxi`, `db/dxi` (kinematic Hessian) | 보통 | `jacfwd` 로 얻음. nv=18 방향 → 메모리 큼 |
| ABA derivatives | 좋음 | MJX 가 이미 제공 |
| 접촉 모드 분류 (separating/clamping/sliding) | 주의 | 데이터 의존 분기 → **고정크기 마스킹 필수** |
| per-contact Gauss-Seidel 반복 | 나쁨 | 접촉 간 순차 의존 |
| bisection (sliding) | 나쁨 | 데이터 의존 종료조건 → 고정 반복수로 최악 비용 상시 지불 |

**핵심**: 아래 두 개("나쁨")는 전부 **forward solver** 다.
gradient 만 교체하면 MJX 가 이미 `lambda` 와 접촉 상태를 주므로 **필요 없다.**
즉 gradient-only 교체는 GPU 병렬화의 난관을 통째로 회피한다.

### 3.3 남는 진짜 제약: 가변 크기

CI-MPC 의 clamping 집합 `c` 와 sliding 집합 `s` 는 샘플마다 크기가 다르다.
JAX 에서는 가변 크기 행렬을 못 쓴다.
→ 항상 `3 x fnum = 12` 크기 시스템을 만들고,
비활성 성분은 마스킹 + 대각 항등원으로 채워 가역성을 유지해야 한다.
`cimpc_grad.cimpc_jacobian` 에 이 방식으로 구현돼 있다.

---

## 4. 검증 결과 (`python verify_grad.py` — 5/5 통과)

flight(무접촉) / contact(4발 접지) 두 regime 에서 모두 돌린다.

| 검사 | flight | contact |
|---|---|---|
| `E^+E = I`, `mj_integratePos` 야코비안 대조 | 1e-19 / 8e-11 | (동일) |
| forward 불변성 (`custom_vjp` == `diffsim.rollout`) | 2e-15 | 1e-14 |
| `custom_vjp` vs plain reverse-mode autodiff | 3e-16 | **3e-4** |
| **유한차분(h 안정 구간) vs `custom_vjp`** | **7e-10 ~ 2e-9** | **3e-9 ~ 5e-9** |

FD 는 16개 probe 전부 조건이 좋아서 하나도 제외되지 않았다.

**판정: 배선이 맞다.** 이제 gradient 를 바꿨을 때 생기는 차이는
배선 버그가 아니라 gradient 차이라고 말할 수 있다.

### 4.1 contact 의 3e-4 차이는 어느 쪽이 틀린 건가

`custom_vjp` 쪽이 맞다. 같은 성분에서

- 유한차분 vs `custom_vjp` : **3e-9**
- 유한차분 vs plain autodiff : **~3e-4**

plain reverse-mode 는 `qacc_warmstart` 체인을 통과해서 미분한다.
warmstart 는 solver 의 초기값일 뿐 물리적 의미가 없고,
수렴이 덜 된 solver 에서만 결과에 남는 경로다.
`mjx_jacobian` 은 warmstart 를 상수로 두고 forward-mode 로 미분해서 이 경로가 없다.

### 4.2 부수적으로 발견한 것 (중요)

**MJX 의 plain reverse-mode autodiff 는 `solver iterations = 1` 에서만 동작한다.**

```
iterations=1    ls=5    plain OK      custom_vjp OK
iterations=10   ls=10   plain 실패    custom_vjp OK
iterations=100  ls=50   plain 실패    custom_vjp OK
```

MJX 의 constraint solver 가 `lax.while_loop` 이라 역전파가 안 된다
(`Reverse-mode differentiation does not work for lax.while_loop`).
DynaFlow 는 `iterations=1` 이라 우연히 통과하고 있을 뿐이다.
`custom_vjp` 는 스텝별 forward-mode 야코비안을 쓰므로 이 제약이 없다.

함의: **CI-MPC 실험에서 forward 정확도를 올리고 싶다면 이 scaffold 가 필수다.**
(수렴 정도가 gradient 에 미치는 영향은 설정에 따라 다르다. 한 구성에서는
`iterations` 1 -> 10 에서 gradient norm 이 79.2562 -> 79.2938 로 바뀌었고,
verify 의 기본 케이스에서는 4자리까지 동일했다. 즉 항상 큰 차이는 아니지만,
`iterations=1` 로 고정될 수밖에 없다는 제약 자체가 문제다.)

### 4.3 검증 스크립트를 쓸 때 주의

- 접촉 구간에서 **단일 h 유한차분은 신뢰할 수 없다.** 같은 성분에서 h 만 바꿔도
  값이 0.55 -> 4.62 -> -84.7 로 튄다. h 가 크면 접촉 모드가 바뀌고(truncation),
  작으면 stiff 한 constraint force 때문에 catastrophic cancellation 이 난다.
  그래서 `robust_fd` 가 여러 h 로 재고 인접 h 끼리 가장 잘 맞는 구간을 채택하며,
  (Richardson 외삽이 아니다 -- 접촉 구간은 h^2 오차 모델이 성립하지 않아 외삽이 위험하다)
  조건이 나쁜 성분은 판정에서 제외한다.
- `jax.config.update("jax_enable_x64", True)` 를 **import 전에** 해야 한다.
  float32 FD 로는 아무것도 판정할 수 없다.
- `mjx_jacobian` 은 첫 컴파일에 수 분 걸린다 (49 tangent x 4 substep).
  `cimpc_grad._ROLLOUT_CACHE` 가 (env, backend) 별로 한 번만 컴파일한다.

## 4.4 실제 학습 shape 벤치마크 (`verify_grad.py --bench`)

RTX (CudaDevice 0), B=1024, H=16, n_substeps=4, **x64**:

| backend | 시간 | 비고 |
|---|---|---|
| plain MJX autodiff | **1192 ms/step** | compile 97s. 50,000 스텝 = 16.6시간 |
| custom_vjp + `mjx_jacobian` | **OOM** | 24.9 GiB 할당 실패 |

**`mjx_jacobian` 은 검증 전용이다. production batch 로 못 올린다.**
`jax.jacfwd` 가 49개 tangent 방향으로 `mjx.Data` 전체를 4 substep 동안
16,384개 인스턴스에 대해 물고 있어서 메모리가 터진다.

대응:
- `cimpc_jacobian` 은 `mjx.step` 전체를 미분하지 않는다. 운동학과 질량행렬만
  `jacfwd` 하므로 훨씬 가볍다 (다만 nested jacfwd 라 여전히 확인 필요).
- 그래도 부족하면 batch 를 청크로 나눠 `jax.lax.map` 으로 돌려 메모리를 상한지운다.
- 실학습은 float32 이므로 위 수치의 대략 절반이다 (x64 검증용 측정값).

## 4.5 `cimpc_jacobian` 상태 (`verify_grad.py --cimpc`)

### 잡은 버그 두 개

첫 실행에서 **flight(접촉 없음) 인데도** `cos(g_mjx, g_cimpc) = 0.494`,
크기 비 0.092 가 나왔다. 접촉이 없으면 `dlambda = 0` 이라 매끄러운 동역학으로
환원되어 MJX 와 거의 같아야 하므로, 접촉과 무관한 부분이 틀렸다는 뜻이었다.

1. **action 을 토크로 취급했다.** DynaFlow 의 action 은 position actuator 의
   **목표 관절각** 이다 (`gaintype=fixed kp=20`, `biastype=affine -kp=-20, -kd=-0.5`).
   `(h + Bu)dt` 처럼 u 를 일반화력으로 넣으면 안 된다.
2. **`qfrc_bias` 부호를 반대로 썼다.** MuJoCo 규약은
   `M qacc + qfrc_bias = qfrc_applied` 라 `qacc = M^-1(tau - h)` 다.
   Go1 에서 `|qfrc_bias| ~ 117` vs `|qfrc_actuator| ~ 0.56` 로 bias 가 200배
   지배적이라 이 부호 하나로 gradient 가 통째로 어긋났다.

둘 다 `d.qacc_smooth` 를 쓰면 한 번에 해결된다
(`M^-1(qfrc_actuator - qfrc_bias) == qacc_smooth` 를 6e-16 로 확인).

추가로 접촉 판정을 투영된 lambda 대신 **발 높이** 기준으로 바꿨다
(CI-MPC Sec 5.2, `f_k < 0`). lambda 기준으로는 최소자승 투영 잔차 때문에
실제 접촉 3개인 상태에서 4개가 활성으로 잡혔다.

### 수정 후 — 매끄러운 부분은 정확히 맞다

`cos(g_mjx, g_cimpc)` / 크기비:

| n_substeps (dt) | flight qvel0 | flight act | contact qvel0 | contact act |
|---|---|---|---|---|
| 4 (0.020) | 0.9960 / 1.010 | 0.9716 / 1.390 | 0.9634 / 0.760 | 0.8540 / 1.052 |
| 2 (0.010) | 0.9999 / 1.006 | 0.9986 / 1.080 | 0.9032 / 2.720 | 0.8510 / 0.727 |
| 1 (0.005) | **1.000000 / 1.0000** | **1.000000 / 1.0000** | 0.5574 / 2.323 | 0.9620 / 0.945 |

**n_substeps=1 flight 에서 정확히 1.000000 이다.**
→ 접촉 없는 부분의 동역학·질량행렬·actuator·tangent 변환이 전부 맞다는 뜻이다.
n_substeps=4 에서 남는 차이(cos 0.996 / 크기비 1.39)는 버그가 아니라
**적분 granularity** 다: `cimpc_jacobian` 은 `dt=0.02` semi-implicit Euler 한 스텝인데
MJX forward 는 `0.005 x 4` 다. position servo 토크가 매 substep 갱신되는 q 에
의존해서 action gradient 에서 특히 벌어진다.

이건 의도적 선택으로 둘 수 있다 (CI-MPC 자체가 `dt=0.025` 단일 스텝이다).
다만 **forward 와 backward 의 적분 granularity 가 다르다는 점은 명시적으로
기록해두고 ablation 대상으로 삼아야 한다.** 없애려면 `cimpc_jacobian` 이
`dt=0.005` 부분 야코비안 4개를 곱하도록 바꾸면 된다.

### contact 는 아직 검증할 기준이 없다

접촉 구간의 차이(cos 0.85~0.96)는 **일부는 의도된 것**이고
(relaxed complementarity gradient 는 MJX 와 다른 게 목적)
**일부는 미완성 때문**이다:

- sliding 접촉 미구현 (모든 활성 접촉을 clamping 취급)
- drift compensation 미구현 (논문 Appendix B)
- lambda 추출이 최소자승 투영이라 잔차가 남는다
  (`sum_k J_k^T lam_k` vs `qfrc_constraint` 상대오차 1.2e-3;
   `nefc=64` 라 발 접촉 외 제약이 섞여 있다)

MJX gradient 는 애초에 정답이 아니므로 이걸로는 접촉 부분을 검증할 수 없다.
**제대로 검증하려면 같은 상태에 대해 C++ `ContactSolverGradient::GetGradient` 의
`d_lambda_lw_dz` 와 대조해야 한다.** 이게 다음 단계의 핵심이다.

## 4.6 C++ 원본 대조 (`cimpc_ref/`) -- 전부 통과

접촉 gradient 를 MJX 로 검증할 수 없으므로 저자들의 C++ 구현과 직접 대조했다.
모델 동등성 문제(Hound vs Go1, 메시가 .dae 라 MuJoCo 가 못 읽음)는
**로봇을 맞추는 대신 공식만 대조** 하는 방식으로 우회했다:
C++ 이 덤프한 M, J, nle, foot_pos 로 eq.(19) 를 numpy 재구성해서 비교한다.

| 검사 | 오차 |
|---|---|
| `lambda == -A^-1 b` (drift 포함) | 1.4e-10 |
| `dlambda/dtau`, rho = 0 / 1e-6 / 1e-4 / 1e-2 | 8e-16 / 5e-16 / 7e-16 / 7e-16 |
| `dlambda/dv` | 7e-16 |
| `dlambda/dq` | 6.5e-11 (유한차분 정밀도 한계) |
| **sliding**: 행 구조 / `lambda` / `dlambda/dtau` / `dlambda/dv` / `dlambda/dq` | 일치 / 1.1e-13 / 5.8e-9 / 5.8e-9 / 5.8e-9 |

### 역산해낸 규약

```
A = J M^-1 J^T
b = J (v + M^-1 (tau - nle) dt),   b[3k+2] += phi_k / dt
lambda = -A^-1 b                   impulse, local-world-aligned
D = diag(1/lambda_n^2)  on 인덱스 3k+2
dlambda/dxi = -[A + rho D]^-1 (dA/dxi lambda + db/dxi)
발 순서 FL,FR,RL,RR / 열 [dq|dv|dtau] / 행 = clamping*3 + sliding*1
```

### 발견 1 -- drift compensation 은 필수다

`cimpc_jacobian` 에서 "선택적 TODO" 로 분류했던 것이 사실상 지배항이었다.
`dlambda/dq` 기여 크기가 **drift 항 24.6 vs `dA/dq*lambda` 항 0.18** 이고,
빼면 상대오차가 1.0 (통째로 틀림) 이다. `lambda` 자체도 10% 틀린다.
`b[normal] += phi_k/dt` 한 줄이며, `cimpc_grad.py` 에 반영했다.

### 발견 2 -- rho D 항 확정

`D = diag(1/lambda_n^2)` 를 normal 인덱스에만 놓는 게 맞다.
모든 rho 에서 기계 정밀도로 일치하고, rho 를 무시하면 오차가 rho 에 비례해 커진다.
**논문의 핵심 기여 항이 구현과 일치한다.**

### 빌드 메모

Pinocchio 2.6.17 + Eigen 3.4.0 설치돼 있음. raisim/qpSWIFT 불필요.
Eigen 3.4 가 rank-3 slice -> rank-2 대입을 거부해서
`contactSolverGradient.cpp` **사본** 에 `.reshape()` 8곳을 넣어 빌드한다
(원본 미수정). `cimpc_ref/build.sh` 가 자동으로 한다.

## 4.7 sliding 구현 + 접촉점 Jacobian 수정

### 발견 -- 접촉 Jacobian 을 잘못된 점에서 계산하고 있었다

`sum_k J_k^T lambda_k == qfrc_constraint` 를 성분별로 보니
**병진 성분(0:3) 은 정확히 맞는데 회전 성분(3:6) 만 어긋났다.**
모멘트 팔이 틀렸다는 뜻이다.

원인: 발 **site(구 중심)** 에서 Jacobian 을 계산했는데
MuJoCo 는 **접촉점(구 표면, 23mm 아래)** 에 힘을 가한다.
힘을 받는 물질점의 Jacobian 은 각속도 블록에 `-skew(p_c - p_body)` 가 들어가므로
접촉점 위치에 의존한다. ("구는 offset 이 world 고정이라 J 가 같다" 는 오판이었다.)

| Jacobian 계산 지점 | `sum J^T f` vs `qfrc_constraint` |
|---|---|
| 구 중심 (site) | 4.6e-03 |
| **접촉점 (`support.jac` at `contact.pos`)** | **4.8e-17** |

MuJoCo 의 구-평면 접촉점은 침투 구간의 중점이다: `z_contact = (z_center - r)/2`.

부수 효과가 크다: 기존에는 순기구학을 `jacfwd` 로 미분해서 J 를 얻었는데
`support.jac` 은 직접 준다. **nested jacfwd 가 사라져 훨씬 싸고 메모리도 덜 쓴다.**

### 발견 -- lambda 를 다음 스텝 상태에서 뽑고 있었다

`_step_from_state` (4 substep 진행 후) 의 접촉력을 쓰면서 J 는 시작 상태에서
계산해 1.4e-4 오차가 남았다. CI-MPC 의 `SolveContactImpulse` 는 **현재** 상태에서
lambda 를 풀므로 `mjx.forward` 여야 맞다. 고친 뒤 **2.1e-17**.

lambda 추출도 최소자승 투영에서 `support.contact_force(..., to_world_frame=True)`
(pyramid 표현을 풀어 발 geom 별로 합산) 로 바꿨다.

### sliding 구현

행/열 구조는 C++ 사양 그대로 (`contactSolverGradient.cpp:2160-2180`):

```
clamping k : J_left 에 3행 J_k      , J_right 에 3행 J_k
sliding  k : J_left 에 1행 J_k[2,:] , J_right 에 1행 E_k^T J_k
A = J_left M^-1 J_right^T           <- sliding 이 섞이면 비대칭
```
풀고 나서 eq.(14) `d(lambda_s)/dxi = E_s d(lambda_n)/dxi` 로 3D 로 되돌린다.
JAX 가 가변 크기를 못 쓰므로 항상 12 크기로 만들고 마스킹한다
(비활성 행/열은 대각 항등원으로 채워 가역성 유지).

**E_s 는 속도에서 뽑는다.** C++ 은 `E = [lam_x/lam_z, lam_y/lam_z, 1]` 로
lambda 에서 뽑지만, 그건 자기 solver 가 lambda 를 원뿔 경계에 정확히 올려놓기
때문이다. MJX 는 그렇지 않아서 최대 소산 방향
`E = [-mu vhat_t_x, -mu vhat_t_y, 1]` 을 쓴다.

**sliding 판정은 "마찰 포화 AND 접선 속도" 두 조건을 다 요구한다.**
어느 하나만으로는 안 된다 -- vx=0 에서도 발 접선속도가 2~5 cm/s 인데
마찰은 포화되지 않아 물리적으로 clamping 이다.

| 측방 속도 | `\|lambda_t\|/(mu lambda_n)` | 판정 |
|---|---|---|
| vx=0.0 | 0.07~0.23 | clamping |
| vx=0.5 | 0.77~0.79 | clamping (보수적) |
| vx=2.0 | 0.93~0.96 | 1개 sliding |
| vx=4.0 | 0.96~0.995 | 전부 sliding |

**주의: MJX 의 soft contact 는 stick/slip 경계가 원래 뚜렷하지 않다.**
C++ 은 hard contact 라 `abs(||lam_t|| - mu lam_n) < 1e-6` 으로 딱 갈린다.
`slide_ratio` (기본 0.95) 는 이 모호함을 어디서 자를지 정하는 튜닝 값이며
**ablation 대상** 으로 삼아야 한다.

### 결과

| | cos(g_mjx, g_cimpc) | 크기비 |
|---|---|---|
| flight, n_substeps=1 | **+1.000000** | 1.0000 |
| contact | +0.8991 | 1.1509 |
| sliding (vx=4) | +0.8638 | 1.1363 |

flight 가 정확히 1.0 이므로 회귀 없음.
contact 는 lambda 추출이 정확해지면서 0.827 -> 0.899 로 올라갔다.

## 4.8 sliding 도 C++ 과 수치 대조 완료

`cimpc_ref/compare.py` [5] 가 clamping 뿐 아니라 sliding 케이스도 수치로 본다.

| 검사 (sliding, contact=[1 1 0 0]) | 오차 |
|---|---|
| 행 개수 = clamping*3 + sliding*1 | 일치 |
| `lambda == -A^-1 b` | 1.1e-13 |
| `dlambda/dtau` | 5.8e-9 |
| `dlambda/dv` | 5.8e-9 |
| `dlambda/dq` | 5.8e-9 |

**중간에 한 번 헛짚었던 것**: `dlambda/dv` 가 2.5e-2 로 틀리게 나왔는데,
구현 문제가 아니라 **검증에 쓴 간이 공식** `-A^-1 Jl` 이 문제였다.
그 식은 `v = 0` 에서 Coriolis 항이 사라진다는 가정에서 나온 것이고,
sliding 테스트 상태는 `v[0] = 1.0` 이라 `d(nle)/dv` 가 살아 있다.
`db/dv` 를 제대로 수치미분하니 5.8e-9 로 맞았다.
(구현은 `jacfwd` 로 전체를 미분하므로 원래부터 이 항을 포함한다.)

## 4.9 최종 상태 -- `verify_grad.py --cimpc` 6/6 통과

| 검사 | 결과 |
|---|---|
| tangent map | PASS |
| forward 불변성 | PASS |
| custom_vjp == plain autodiff | PASS |
| 유한차분 (flight/contact) | PASS, 7e-11 ~ 9e-9 |
| solver iterations 견고성 | PASS |
| **contact impulse 추출** | **PASS, 1.99e-16** (이전 1.24e-3 FAIL) |

CI-MPC vs MJX gradient (차이 측정, pass/fail 아님):

| regime | cos | 크기비 | 접촉 상태 |
|---|---|---|---|
| flight | +0.9716 | 1.3900 | 활성 0개 |
| flight (n_substeps=1) | **+1.000000** | 1.0000 | 활성 0개 |
| contact | +0.8518 | 1.1332 | 활성 3, sliding 0 |
| sliding | +0.2069 | 0.9390 | 활성 4, sliding 4 |

`n_substeps=1` flight 에서 정확히 1.0 이므로 회귀 없음.
sliding 의 cos 가 낮은 것(0.21)은 **버그가 아니라 구조 차이** 다:
CI-MPC 는 sliding 접촉에서 normal 만 미지수로 두고 접선을 `E_s` 로 종속시키는데
MJX 는 3성분을 독립으로 본다. 공식 자체는 위에서 C++ 과 5.8e-9 로 대조됐다.

## 6. 학습 실험 결과 (Go1, B=256, H=16, 후반 25% 스텝 loss 평균)

> 주의: 이 절의 결과는 8.8 절에서 고친 f_x 버그 (quaternion 적분 Lie-group 항 누락) 가
> 있는 상태로 얻었다. 결론이 바뀔 크기는 아니지만 재확인 대상이다.

원본 데이터: `rho_sweep_results/{results_pass1.json, pass2/, nsub/, nsub4/}`.
스크립트: `rho_sweep.py`. forward 는 전 조건에서 **완전히 동일** (MJX, 0.005 x 4).

### 6.1 rho 스윕 -- pass1 (시드 1, 2000스텝) 은 과대 해석이었다

pass1 에서 rho=0 (+17.7%) -> rho=1e-4 (+6.5%) 를 보고 "완화가 돕는다" 고 썼는데,
시드 3개 (pass2, 1200스텝) 로 보니 **구별되지 않는다**:

| 조건 | 평균 | ±std | MJX 대비 |
|---|---|---|---|
| mjx | 0.03060 | 0.00056 | -- |
| rho=0 | 0.03627 | 0.00233 | +18.5% |
| rho=1e-4 | 0.03533 | 0.00172 | +15.4% |
| rho=1e-4, lr x2 | 0.03508 | 0.00187 | +14.6% |

- rho=0 vs rho=1e-4 : 0.6σ -> **차이 없음**
- rho=1e-4 vs lr x2 : 0.2σ -> **격차는 learning-rate 스케일 문제가 아니다**
- lr x2 vs mjx : 4.0σ -> **CI-MPC 가 MJX 보다 나쁜 건 유의하다**
- CI-MPC 조건의 시드 편차가 MJX 의 3~4배 (0.0017~0.0023 vs 0.00056), 세 조건에서 일관

**rho 가 안 먹힌 이유 (가설)**: 논문에서 rho 의 역할은 MPC 최적화가 접촉 모드를
깨고 나올 수 있게 하는 것이다. DynaFlow 는 이미 올바른 접촉을 가진 참조 모션을
모방하는 과제라 접촉 모드를 바꿔야 할 상황 자체가 잘 안 생긴다.
"rho 가 쓸모없다" 가 아니라 "이 과제가 rho 를 시험하지 않는다" 일 수 있다.
검증하려면 참조와 다른 접촉 패턴에서 시작하는 실험이 필요하다.

"backward 만 relaxed 라서" 는 원인 후보가 **아니다** -- CI-MPC 원본도 relaxation 을
gradient 에만 쓴다 (1.2 절). 남는 후보는 위 과제 가설과, MJX soft contact 와의
상호작용 (hard contact 를 전제로 한 `D = diag(1/lambda_n^2)` 가 soft contact 의
lambda 에서 어떻게 작동하는가) 두 가지다.

### 6.2 backward substep 정합 -- 결정적이고, n_sub=2 면 충분

같은 실험 안에서 rho=1e-4 고정, backward 의 substep 수만 바꿈 (시드 3):

| n_sub | 어긋남 (flight cos) | 평균 | ±std | 시간/런 | MJX 대비 |
|---|---|---|---|---|---|
| mjx | -- | 0.03065 | 0.00193 | 492s | -- |
| 1 | 14.5° (0.968) | 0.04335 | 0.00279 | 511s | **+41.4%** |
| 2 | 2.5° (0.999) | 0.03454 | 0.00155 | 854s | **+12.7%** |
| 4 | 0° (1.000) | 0.03393 | 0.00228 | 1347s | **+10.7%** |

- n_sub=1 vs 2 : 4.8σ **유의** -- 14.5° 어긋남이 +41% 를 만든다
- n_sub=2 vs 4 : 0.4σ **차이 없음** -- 2 면 충분, 37% 싸다
- n_sub=4 vs mjx : 1.9σ 경계 -- 완전 정합이면 MJX 와 거의 겹친다

**권장: `cimpc_jacobian(..., n_sub=2)`.**
"forward 를 backward 에 맞추자 (timestep 0.02)" 는 대안은 Go1 발 접촉의 시간상수가
0.01s 라 timestep 이 그보다 커져 물리 자체가 깨지므로 불가.

### 6.3 종합

1. **CI-MPC gradient 로 DynaFlow imitation 이 학습된다.** 27런 전부 발산 없음.
2. MJX autodiff 대비 **10~15% 나쁘고**, learning-rate 스케일 문제가 아니다.
3. **rho 완화 효과는 이 과제에서 검출되지 않았다** (6.1 가설 참조).
4. CI-MPC gradient 는 시드 간 변동이 3~4배 크다.
5. backward 는 forward 의 적분 단위에 맞춰야 하고, `n_sub=2` 면 충분하다.
6. 비용: `n_sub=2` 기준 MJX 의 1.7배. `n_sub=1` 은 MJX 와 같은 속도지만 쓸 이유 없음.

한계: 1200~2000 스텝 (main.py 는 50,001). loss 는 학습 배치 기준 (held-out 아님).

## 7. 학습 결과를 직접 보기 -- 도구

| 파일 | 역할 |
|---|---|
| `train_cimpc.py` | 체크포인트를 **저장하는** 학습 (rho_sweep.py 는 저장 안 함). main.py 와 같은 Adam/EMA, rollout 만 교체 |
| `eval_metrics.py` | 논문 Sec. V-A 의 **TRE / SAE** 를 체크포인트별로 계산해 표로 출력 |
| `tester.py` | 기존 영상 렌더 (render_rhc / render_openloop). CI-MPC 체크포인트도 그대로 사용 가능 -- forward 만 쓰므로 |

### 지표 정의 (eval_metrics.py)

- **SAE**(x_i, x_{i+1}) = ||x_{i+1} - f(x_i, ID(x_i, x_{i+1}))||.
  ID 는 `diffsim.control_step` 을 통해 Adam 으로 푼다 (ctrl_range 안, best 추적).
  best 추적이 없으면 Adam 의 첫 스텝(~lr) 때문에 이미 최적인 초기값에서 멀어져 SAE 가 부풀려진다
  (실측: 0.042 -> best 추적 후 median 0.000).
- **TRE** = 같은 x0 / command / **같은 초기 노이즈** 로 생성한 궤적과 참조 궤적의 MSE.
- 거리 공간은 학습 loss 와 동일: [상대 xy raw, 나머지 35 normalized]. raw 그룹별 RMSE 도 같이 낸다.
- DynaFlow 출력은 MJX 롤아웃이라 SAE 는 solver 정밀도 바닥 (warmstart 차이) 만 남는다.
  **CI-MPC 모델과 MJX 모델이 갈리는 지표는 TRE 다.**

### 실행 순서 (GPU 가 비면)

```bash
# 1. CI-MPC 학습 (6.2절 권장 설정)  -- 20k 스텝, B=256, n_sub=2: 약 5시간
python train_cimpc.py --backend cimpc --rho 1e-4 --n-sub 2 --steps 20000 --batch 256
# 2. 공정 비교용 MJX baseline (같은 batch/steps). 기존 state_50000 은 B=1024/50k 라 조건이 다르다
python train_cimpc.py --backend mjx --steps 20000 --batch 256
# 3. 표
python eval_metrics.py --n-traj 256 \
    --ckpt mjx_paper=checkpoints/2026-08-18-15-07-27_DynaFlow:50000 \
    --ckpt mjx=checkpoints/<MJX_baseline>:20000 \
    --ckpt cimpc=checkpoints/<CIMPC_rho0.0001_nsub2>:20000
# 4. 영상: tester.py 의 checkpoint_name / target_epoch 만 바꿔서 실행
```

CPU 스모크 (`JAX_PLATFORMS=cpu`, n_traj=4) 로 파이프라인은 확인됨:
dataset SAE 0.137 (20 iters), mjx state_50000 TRE 0.00070 / SAE median 0.000.

## 8. JAX hard-contact forward (`cimpc_forward.py`, `verify_forward.py`)

1.4 절의 결론에 따라 forward 도 CI-MPC 로 바꾸기 위한 첫 단계.
C++ `ContactSolverGradient::SolveContactImpulse` (per-contact Gauss-Seidel +
마찰 원뿔 bisection, Hwangbo 방식) 를 고정 반복수 JAX 로 옮겼다.

```
cimpc_forward.solve_contact_impulse(J, MinvJT, v_free, phi, dt, mu)  로봇 무관 solver
cimpc_forward.cimpc_substep / cimpc_control_step                       MJX 운동학 + solver
cimpc_forward.make_rollout / make_batched_rollout                       diffsim.rollout 과 같은 형태
```

M, qacc_smooth (actuator·중력·passive 포함), 발 위치, 접촉점 Jacobian 은 MJX 에서
가져오고 (`smooth_forward` = `mjx.forward` 에서 collision/constraint 를 뺀 것,
qacc_smooth 가 bit-identical), **접촉 solver 만 새로 썼다.**
J 와 phi 는 발 구의 바닥점 (center - r) 에서 계산한다 (C++ 과 같은 규약).

`JAX_PLATFORMS=cpu python verify_forward.py` -- **29/29 통과** (C++ 대조 18, smooth 3,
서기 3 [near-rigid MuJoCo 와 trunk z 3.3mm 이내], 낙하 3, autodiff 1, 데이터셋 재현 1).

### 8.1 C++ 과의 수치 대조 -- 전부 기계 정밀도

C++ 덤프 (M, J, nle, foot_pos) 로 같은 입력을 만들어 solver 만 비교했다.
JAX 는 조기 종료가 안 되므로 `done` 플래그로 "수렴 시점의 값을 유지" 하게 해서
C++ 의 break 와 같은 답이 나오게 했다.

| 상태 | contact / sliding | lambda | v_next |
|---|---|---|---|
| clamping (4발 정지) | 1111 / 0000 | 8e-16 | 6e-16 |
| sliding (vx=1.0) | 1111 / 1111 | 9e-16 | 6e-16 |
| separating (공중) | 0000 / 0000 | 0 | 1e-15 |
| mixed (pitch 0.1) | 1100 / 0000 | 5e-16 | 9e-16 |
| landing (vz=-1) | 1111 / 0000 | 7e-16 | 9e-16 |
| partial (vx=0.3, vy=0.1) | 1111 / 1111 | 8e-16 | 7e-16 |

### 8.2 sliding 해의 물리 검사

서기 중 앞발이 sliding 으로 분류되는 상태에서, 마찰 impulse 가 impulse 후 slip 속도의
반대 방향인지 봤다: `cos(v_t, lambda_t) = -0.994 ~ -0.997`. th 를 그리드로 훑어
D(th)=0 근이 Coulomb 조건 (180deg) 과 일치하는 것도 확인했다. bisection 이 맞는 근을 찾는다.

### 8.3 서기 tip-over -- 버그가 아니라 물리

`nominal_ptarg` 서기를 1초 돌리면 0.4초쯤 앞발이 마찰 포화 (비율 1.0) 로 뒤로
미끄러지며 넘어진다. 원인 후보를 하나씩 제거했다:

- GS 수렴: n_outer 10 vs 200 에서 lambda 차이 0, Signorini residual 1e-9
- MuJoCo C 를 near-rigid (dt=1e-3, solref 0.002, solimp 0.99) 로 돌리면
  **같은 시점 (t=20) 에 앞발이 완전히 뜨고 (Fz=0) pitch -9.2deg 까지 기운다.**
  CI-MPC 는 -9.4deg. 그 뒤 MuJoCo 는 앞발이 다시 붙어 -8.7deg 에서 멈추고,
  CI-MPC 는 앞발이 미끄러져 계속 기운다. lift-off 직후의 민감한 분기다.
- mu=100 이면 넘어지지 않는다 (z 0.20 유지)
- kp=20 servo 가 약해서 MJX soft 도 1초에 8cm 주저앉는다. 이 자세 자체가 marginal 하다.

즉 rigid Coulomb 접촉 + 약한 servo 의 물리적 결과다. 정책이 들어가는 데이터셋
롤아웃 (아래) 에서는 문제가 되지 않는다. 낙하 테스트의 collapse 후 거동이 MJX 와 다른 것도
발 외 링크에 접촉이 없는 모델 (CI-MPC 와 동일) 이라서다.

### 8.4 데이터셋 전이 재현 -- hard contact 가 MJX 보다 낫다 (1.4 절 질문 2 의 답)

128 세그먼트 x 16 스텝, 데이터셋 action 으로 롤아웃, 학습 loss 와 같은 거리:

| forward | loss-like | step16 base_pos | joint_pos | 침투>1mm 비율 |
|---|---|---|---|---|
| MJX (soft, 0.005x4) | 0.0466 | 0.034 | 0.140 | -- |
| CI-MPC n_sub=4 | 0.0282 | 0.020 | 0.106 | 0.000 (min -0.2mm) |
| CI-MPC n_sub=2 | 0.0184 | 0.015 | 0.085 | 0.000 (min -4.8mm) |
| CI-MPC n_sub=1 | 0.0115 | 0.011 | 0.060 | 0.032 (min -4mm) |

한 스텝 전이 오차도 MJX 와 같은 수준이다 (trot: base_pos 0.0008 vs 0.0007).
참조가 Isaac (rigid 접촉) 이라 hard contact 쪽이 더 잘 맞는 게 자연스럽다.
n_sub 가 작을수록 오차가 줄어드는 건 흥미롭지만 n_sub=1 은 침투가 생긴다
(dt=0.02 선형화 오차). **forward 는 n_sub=2 이상 권장.**

### 8.5 autodiff

고정 반복수라 plain reverse-mode 가 통과한다 (MJX 의 `while_loop` 제약 없음).
`norm`, `arctan2`, `arccos` 의 0/경계 gradient NaN 을 double-where 로 막았다.
이걸로 "CI-MPC forward + plain autodiff" 조건도 비교군에 넣을 수 있다.

### 8.6 GPU 비용 -- forward only (H=16, f32, RTX 16GB)

| forward | B=256 (GPU 공유 중, 참고치) | **B=1024 (GPU 단독)** |
|---|---|---|
| MJX (0.005 x 4) | 31 ms | **50 ms** |
| CI-MPC n_sub=1 | 66 ms | 48 ms |
| CI-MPC n_sub=2 | 133 ms | **75 ms** |
| CI-MPC n_sub=4 | 472 ms | 152 ms |

B=256 에서 15배였던 n_sub=4 의 비율이 B=1024 에서 3배로 줄었다. CI-MPC solver 의 비용이
배치가 아니라 **커널 launch 지연** (10 sweep x 4 발 x 8 bisection 의 작은 순차 연산) 에
묶여 있다는 가설이 맞다. 배치를 키우면 MJX 만 비례해서 늘고 CI-MPC 는 거의 그대로다.
**n_sub=2 는 B=1024 에서 MJX 의 1.5배.** 위 수치는 `cimpc_forward.UNROLL = True`
(solver 의 sweep / bisection scan 을 펼쳐 XLA 융합) 가 켜진 상태의 측정이다.
unroll 없는 수치는 따로 재지 않았다.

### 8.7 전체 교체 배선 (`cimpc_grad.make_rollout(forward="cimpc")`, `train_cimpc.py --forward`)

```
cg.make_rollout(env, jacobian_fn, forward="cimpc", fwd_n_sub=2)
    forward : cimpc_forward.make_rollout(env, n_sub=2)        hard contact
    backward: jacobian_fn = partial(cimpc_jacobian, forward="cimpc", n_sub=2)
```

`_cimpc_step_jacobian(forward="cimpc")` 가 바뀌는 점:

- lambda / contact / sliding 을 `hard_contact_impulse` (forward 와 같은 solver, 같은 입력) 에서 받는다.
  MJX `contact_force` 최소자승·투영이 없어진다.
- 접촉점 Jacobian 을 발 바닥점 (`contact_jacobians_bottom`) 에서 계산한다 (forward 와 같은 규약).
- sliding 판정은 solver 플래그 그대로 (alpha in {0,1}), `E_s = [lam_x/lam_n, lam_y/lam_n, 1]`
  을 C++ 처럼 lambda 에서 뽑는다. `slide_lo/hi/vel_eps` 튜닝값이 필요 없다.
- n_sub>1 의 중간 상태를 `mjx.step` 대신 `cimpc_substep` 으로 진행한다.
  **backward n_sub 는 forward n_sub 와 같아야 한다** (train_cimpc 가 강제).

`train_cimpc.py`:

```
--forward mjx   (기본)   diffsim.rollout, backward 는 --backend 대로            [조건 a, b]
--forward cimpc          hard contact forward; --backend cimpc -> CI-MPC grad     [조건 c]
                                              --backend mjx   -> plain autodiff  [조건 d']
--fwd-n-sub N            forward substep (기본 --n-sub 와 같게)
```

1.4 절의 비교 조건 (d) "CI-MPC fwd + rho=0" 은 `--forward cimpc --backend cimpc --rho 0`.
plain autodiff (d') 는 고정 반복수 solver 를 그대로 미분한 것으로, 수렴한 clamping
상태에서는 active-set 의 implicit derivative 와 같으므로 rho=0 과 거의 같아야 한다
(아래 검증이 그걸 확인한다).

검증 `verify_forward.py --wiring`: (a) forward 불변성, (b) custom_vjp(rho=0) vs plain
autodiff 의 cos / 크기비, (c) 유한차분. 결과는 8.8 절.

### 8.8 배선 검증 (`verify_forward.py --wiring`) -- 8/8 통과, 그리고 f_x 버그 하나

forward="cimpc", n_sub=2, H=4. custom_vjp(`cimpc_jacobian`, rho=0) 를 (i) 같은
hard-contact forward 를 plain autodiff 로 미분한 것, (ii) x64 유한차분과 비교했다.
plain autodiff 는 고정 반복수 solver 를 그대로 미분한 것이고, 수렴한 active set 에서는
eq.(19) 의 rho=0 과 같아야 한다.

| | forward 불변성 | cos(custom, plain) | 크기비 | FD 상대오차 (custom / plain) |
|---|---|---|---|---|
| flight | 0 (bit-identical) | 1.000000 | 1.0000 | 3e-10 / 3e-10 |
| contact (4발 clamping) | 0 | 1.000000 (1-cos = 6e-16) | 1.0000 | 6e-8 / 4e-8 |

**발견한 버그 -- f_x 의 위치 블록.** 처음 결과는 contact 에서 custom_vjp 가 FD 와
1.6e-3 (최대 1.5e-2) 어긋났고 plain autodiff 는 4e-8 로 맞았다. 즉 FD 잡음이 아니라
`cimpc_jacobian` 의 오차였다. 원인은 `f_x = [[I + dt dv/dq, dt dv/dv], ...]` 조립이
quaternion 적분 (`quat_integrate`, 지수 사상) 의 Lie-group 항
`d theta_next / d theta = I + O(omega dt)` 를 무시한 것이다. omega ~ 1 rad/s, dt=0.01 이면
각속도 블록에 1e-2 상대오차가 남는다. flight (omega ~ 0) 에서 1e-4, contact 에서 1e-3 로
나타난 게 이것이다. 고친 뒤 실제 적분 함수 `_integrate_pos` 를 tangent 좌표에서
`jacfwd` 로 직접 미분한다 (`Pq = E^+(q_next) dq_next/dtheta`, `Pv = E^+(q_next) dq_next/dv`).

**이 버그는 forward="mjx" 모드에도 있었다.** 4.5~4.9 절의 cos 수치와 6 절의 학습 결과는
옛 f_x 로 얻은 것이다. 오차 크기가 전체 gradient 의 cos 로는 0.99996 수준이라 결론이
뒤집힐 가능성은 낮지만, 6 절을 다시 돌릴 때는 새 f_x 기준으로 봐야 한다.
(`verify_grad.py` 의 유한차분 검사는 `mjx_jacobian` 을 쓰므로 이 버그를 잡지 못했다.
`cimpc_jacobian` 은 MJX 와 cos 만 봤고, MJX 는 정답이 아니라 판정이 불가능했다.
hard-contact forward 가 들어오면서 처음으로 `cimpc_jacobian` 을 정답과 대조할 수 있게
된 것이 이 발견의 배경이다.)

**학습 스모크** (`JAX_PLATFORMS=cpu`, `--forward cimpc --backend cimpc --n-sub 2 --steps 2 --batch 4`):
loss 유한, 체크포인트 `CIMPC_rho0.0001_nsub2_fwdcimpc2/state_2` 저장. end-to-end 통과.

### 8.9 남은 것

1. ~~GPU 시간~~ -- 완료 (8.6 forward, 8.10 학습 스텝, 8.11 판정).
2. 1.4 절 비교 조건 (c)(d) 학습: `--forward cimpc --backend cimpc --rho {0, 1e-4} --n-sub 2`,
   그리고 (d') `--forward cimpc --backend mjx --n-sub 2` (plain autodiff).
3. 새 f_x 로 6 절 (forward=mjx) 재확인.

### 8.10 학습 스텝 시간 -- rollout gradient 전체 (`bench_cimpc.py --full --cond X`)

H=16, f32, `jax.grad` of sum(rollout) (DiT 모델 제외, rollout 의 fwd+bwd 만). GPU 단독.

| 조건 | B=256 | B=1024 | peak GPU (1024) |
|---|---|---|---|
| (a) MJX fwd + MJX autodiff | 159 ms | 336 ms | 3.8 GiB |
| (b) MJX fwd + CI grad n_sub=2 | 300 ms | 1200 ms | 11.4 GiB |
| **(c) CI fwd n2 + CI grad n2** | **357 ms** | **1260 ms** | 11.3 GiB |
| (c) CI fwd n4 + CI grad n4 | 732 ms | GPU OOM (22.7 GiB) | 5.1 GiB (256) |
| (d') CI fwd n2 + plain autodiff | 컴파일 중 **호스트 OOM** (RSS 47 GB) | 동일 | -- |

읽는 법:

- **forward 교체 비용은 작다.** (c) - (b) = 57 ms (B=256), 60 ms (B=1024). forward-only
  측정 (8.6, n_sub=2: 75 ms) 과 일치한다. 저자들이 걱정한 "CI-MPC solver 의 GPU 병렬화" 는
  이걸로 답이 났다: **병목이 아니다.**
- **병목은 backward `cimpc_jacobian` 이다.** (b) 가 (a) 의 1.9배 (256) / 3.6배 (1024) 이고,
  배치에 선형으로 늘며 peak 메모리도 3배다. `A_and_b` 를 2nv+nu=48 방향 `jacfwd` 하는
  비용이고, 이건 논문의 eq.(19) 자체가 요구하는 계산이다 (C++ 도 같은 양을 CPU 에서 한다).
- 학습 계획 (B=256, 20k 스텝) 기준 (c) 는 약 2시간 + 모델 오버헤드. MJX 는 약 1시간. 감당 가능.
- B=1024 는 (b)(c) 모두 1.2 s/step 으로 50k 스텝이면 17시간. 이 규모로 가려면
  `--chunk 256` 이나 backward 의 jacfwd 방향 수를 줄이는 최적화가 필요하다.
- n_sub=4 는 배치 1024 에서 22.7 GiB 단일 할당으로 터진다. `--chunk 256` 으로 우회 가능하지만
  6.2 절대로 n_sub=2 면 충분하므로 쓸 이유가 없다.
- (d') plain autodiff through hard-contact forward 는 XLA 가 scan(H) x substep x
  scan(10 sweep) x scan(8 bisection) 의 역전파 그래프를 컴파일하다 호스트 RAM 47 GB 에서
  죽는다 (`dmesg` 로 확인, 세 번 재현). `jax.checkpoint` 나 sweep/bisection 수 축소로
  풀릴 수 있지만 비교군일 뿐이라 시도하지 않았다. **비교 조건 (d) 는 rho=0 의
  `cimpc_jacobian` 으로 대신한다** (8.8 절에서 plain autodiff 와 cos 1.000000 확인됨).

측정 주의: 호스트 OOM 은 stdout 버퍼링 때문에 앞 조건의 결과까지 지운다.
`--cond` 로 조건마다 프로세스를 분리하고 `python -u` 로 돌려야 한다.

### 8.11 GPU 병렬화 판정 (저자 질문에 대한 답)

1. **된다.** CI-MPC forward solver 는 fnum=4, 고정 반복수 (10 x 4 x 8) 로 펼치면
   `vmap` 에 자명하고, B=1024 에서 MJX forward 의 1.5배 (n_sub=2) 다. 결과는 C++ 과 1e-16.
2. **비용의 주인은 gradient 다.** eq.(19) 의 dA/dx, db/dx 를 만드는 jacfwd 가
   학습 스텝의 대부분이고 MJX autodiff 의 2~4배다. 이건 forward 교체 전에도 같았다 (6.3 절).
3. 실험 규모 (B=256, 20k 스텝) 는 문제없이 돈다.

## 5. 다음 단계

1. ~~접촉 gradient 를 C++ 원본과 대조~~ -- 완료 (4.6, 4.8절)
2. ~~drift compensation~~ -- 완료
3. ~~sliding~~ -- 완료 (4.7절)
4. ~~rho 스윕 / 시드 반복 / lr 스케일 / n_sub 비교~~ -- 완료 (6절)
5. **rho 가 실제로 시험되는 과제 설계.** 참조 모션과 접촉 패턴이 다른 초기 상태
   (초기 자세를 크게 흔들거나, gait 이 다른 command) 에서 rho=0 vs rho>0 비교.
   이게 논문의 주장을 DynaFlow 맥락에서 제대로 시험하는 실험이다.
6. 긴 학습 (수만 스텝) 으로 순위가 유지되는지, held-out loss 로 확인.
7. Go1 에서 결론이 나면 그때 bipedal canonical example 로.

---

## 9. 재검증 (2026-10-29, `verify_cimpc.py`) -- 논문·C++·2D 구현과 다시 대조

stash@{0} 의 forward(hard contact) + backward(eq.19) 를 처음부터 다시 의심하고 검증했다.
기존 스크립트와 겹치지 않는 것만 새로 봤다. 전부 x64 CPU. `JAX_PLATFORMS=cpu python verify_cimpc.py [ABCDE]`.

| 구분 | 검사 | 결과 |
|---|---|---|
| [A] forward 물리 | Signorini `v_n + phi/dt = 0`, Coulomb, `lam_n >= 0`, separating 발 `v_n,next + phi/dt >= 0`, **sliding 해가 eq.(7) 의 전역 최소** (theta 스캔 36001점) -- flight/clamping/sliding/partial/mixed/landing | 20/20 |
| [B] backward rho=0 | `cimpc_jacobian(forward="cimpc")` 의 **f_x, f_u 전체 행렬** vs x64 중앙 FD (tangent 좌표), n_sub=1,2 | clamping 계열 전부 1e-8~1e-10 |
| [C] backward rho>0 | eq.(19) 를 인덱스 기반으로 따로 구현 (J 도 support.jac 대신 물질점 위치의 jacfwd) vs `cimpc_jacobian`, rho = 0 / 1e-4 / 1e-2, dt = 0.02 / 0.01, sliding 포함 | 전부 1e-15 |
| [E] sliding 의 진짜 미분 | sliding 접촉의 (lam_n, theta) 를 모두 미분하는 **정확한 KKT implicit 미분** vs FD | 3e-6 ~ 5e-6 (일치) |
| 기존 `verify_forward.py` | C++ 덤프 6상태 1e-16, near-rigid MuJoCo 3.3mm, 데이터셋 재현 | 37/37 (`cf.UNROLL=False` 로 실행, 아래) |

결론: **forward 와 backward 모두 논문/C++ 과 같은 것을 구현하고 있고, 그 구현은 수치적으로 맞다.** 고칠 버그는 없었다.

### 9.1 발견 1 -- sliding 에서 CI-MPC gradient 는 진짜 미분과 40~60% 다르다 (논문 eq.14 의 근사)

[E] 에서 sliding 접촉의 진짜 미분(theta 포함) 은 FD 와 5e-6 로 맞는데, `cimpc_jacobian` 은 FD 와
0.39 (4발 sliding, slip 1.1 m/s) / 0.60 (1발 sliding, slip 0.56 m/s) 어긋난다 (최대 성분 대비).
KKT 시스템에 "E_s 를 해에서 고정" 하는 가정을 넣으면 `cimpc_jacobian` 과 1e-15 로 일치한다.
즉 차이는 **전적으로 논문 eq.(14) `d lam_s = E_s d lam_n` (마찰 방향 theta 를 상수로 보는 것)** 이고,
C++ 원본도 같은 가정이다 (`E_matrix = lambda.head(2)/lambda(2)`). 구현 버그가 아니라 방법의 성질이다.
slip 이 느릴수록 d theta / d xi ~ 1/|v_t| 가 커져 근사 오차가 커진다.
DynaFlow 학습에서 발이 미끄러지는 구간 (착지 직후, 선회) 의 gradient 가 이만큼 틀어진다는 뜻이다.
rho 효과를 해석할 때 이 오차와 분리해서 봐야 한다. 정확한 sliding 미분이 필요하면 [E] 의
`exact_step_jacobian` (KKT: Signorini 1행 + 정류조건 `D(theta)=0` + 원뿔 `h2=0`) 을 backend 로 올리면 된다.

### 9.2 발견 2 -- hard forward 의 plain autodiff 는 sliding 에서 정답이 아니다

[B] 에서 clamping 은 autodiff == FD (1e-10) 지만, sliding 이 있으면 autodiff 가 FD 와 ~10% 어긋난다
(h 를 1e-4~1e-8 로 바꿔도 FD 는 안정 -> autodiff 쪽이 틀림). bisection 의 bracket 선택은 이산이라
autodiff 는 초기 bracket 끝점 (clamping 해의 투영 theta, theta +- acos(...)) 의 미분만 전파하고
근 theta* 의 이동은 보지 못한다. 따라서
- 8.7/8.8 절의 비교군 (d') `--forward cimpc --backend mjx` 는 sliding 이 있는 학습에서 **유효한 비교군이 아니다**
  (8.8 의 검증은 flight/clamping 뿐이라 이걸 못 봤다). rho=0 비교군은 `--backend cimpc --rho 0` 으로.
  `train_cimpc.py` 가 이 조합에서 경고를 찍는다.
- 2D (`canonex_relaxedynaflow`) 는 원뿔 모서리가 2개뿐이라 닫힌 형태로 풀어 이 문제가 없다. 3D 고유.

### 9.3 발견 3 -- 학습 기본 solver 설정의 정밀도 ([D])

`n_outer=10, n_bisect=8, tol=1e-6` 을 수렴해 (40/50/1e-11) 와 비교: clamping 계열은 `|dlam| ~ 1e-9`,
**sliding 은 `|dlam| 1.2e-3, |dv_next| 9.6e-3`** (각도 bisection 8회 = ~1e-2 rad). GS 는 5~8 sweep 에 1e-16 수렴.
모드 분류는 전부 동일. 학습에 문제될 크기는 아니지만 sliding 정밀도를 올리려면 `n_bisect` 를 16 정도로.

### 9.4 검증하면서 내가 틀렸던 것 (기록)

- 독립 구현의 접촉 Jacobian 을 `d(site - r z)/dq` 로 만들었더니 support.jac 과 접선 행이 정확히 r=0.023 차이.
  구 바닥점은 body 에 고정된 점이 아니라서 omega x (-r z) 가 빠진다. forward (`contact_geometry`) 와
  backward (`contact_jacobians_bottom`) 가 쓰는 **물질점 Jacobian 이 맞다** (C++ frame Jacobian 과 같은 의미).
- sliding 의 "최대 소산" 을 `cos(lam_t, v_t,next) = -1` 로 검사하면 실패한다 (-0.98~-0.99). 아파런트 관성이
  비등방이면 eq.(7) 의 최적점은 마찰이 slip 과 정확히 반대가 아니다. 올바른 검사는 eq.(7) 목적함수의 1D 최소.
- `verify_forward.py` 는 지금 환경 (호스트 RAM 20GB 여유) 에서 [3] 서기 롤아웃 컴파일 중 OOM (RSS 26GB).
  `cimpc_forward.UNROLL=False` 로 돌리면 37/37 통과. 수치는 같다.
