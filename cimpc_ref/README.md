# CI-MPC C++ 원본 대조

`cimpc_grad.cimpc_jacobian` 의 접촉 gradient 는 MJX 로 검증할 수 없다
(MJX gradient 는 애초에 정답이 아니다). 유일한 기준은 논문 저자들의 C++ 구현이다.

## 실행

```bash
./build.sh          # CI-MPC 원본을 건드리지 않고 빌드 (원본 소스는 사본에 패치)
python compare.py
```

전제: Pinocchio (테스트 환경 2.6.17) 와 Eigen 3.3+ 가 설치돼 있을 것.
raisim / qpSWIFT / matplotlib 는 필요 없다 -- `contactSolverGradient` 는
Pinocchio + Eigen 만 쓴다. 원본 `CMakeLists.txt` 는 raisim 을 요구하지만
여기서는 필요한 타깃만 따로 빌드한다.

`CIMPC_SRC` 환경변수로 CI-MPC 경로를 바꿀 수 있다
(기본값 `~/workspace/general_contact_TO/contact_solving_gradient`).

## 빌드 패치

Eigen 3.4 는 rank-3 tensor slice 를 rank-2 텐서에 바로 대입하지 못한다
(원본은 구버전 Eigen 전제). `build.sh` 가 `contactSolverGradient.cpp` **사본** 에
`.reshape(extent2d)` 를 8곳 넣어서 빌드한다. **원본 파일은 수정하지 않는다.**

## 모델 동등성 문제를 어떻게 우회했나

C++ 은 Hound URDF 전용이고 (프레임 인덱스 9,17,25,33 등이 하드코딩),
DynaFlow 는 Go1 MJCF 다. Hound 메시는 `.dae` 라 MuJoCo 가 읽지도 못한다.

그래서 **로봇을 맞추는 대신 공식만 대조한다.**
C++ 이 덤프한 `M`, `J`, `nle`, `foot_pos` 를 그대로 받아서 eq.(19) 를
numpy 로 재구성하고, C++ 의 해석적 gradient 와 비교한다.
`dA/dq`, `db/dq` 는 C++ 을 tangent 방향으로 섭동시켜 수치미분한다.

이러면 **"eq.(19) 를 내가 제대로 이해했는가"** 만 따로 검증된다.
"MJX 로부터 A, b 를 제대로 만드는가" 는 `verify_grad.py` 가 따로 본다.

## 역산해낸 규약

```
A = J M^-1 J^T                          Delassus, (3*fnum, 3*fnum)
b = J (v + M^-1 (tau - nle) dt)         nle 는 Pinocchio 부호 (M a + nle = tau)
b[3k+2] += phi_k / dt                   drift compensation, phi_k = 발 높이
lambda = -A^-1 b                        impulse (force 아님), local-world-aligned
D = diag(1/lambda_n^2)  on 인덱스 3k+2
dlambda/dxi = -[A + rho D]^-1 (dA/dxi lambda + db/dxi)
```

- 발 순서 **FL, FR, RL, RR**
- `dlambda_dz` 열 순서 `[dq | dv | dtau]`, 각 nv=18
- 행 개수 = clamping*3 + sliding*1 (sliding 은 normal 성분만; 접선은 `lambda_s = E_s lambda_n` 로 복원)
- `dt = 0.025`, `mu = 0.8` 은 `contactHeader.h` 의 컴파일 상수

## 결과 (전부 통과)

| 검사 | 오차 |
|---|---|
| `lambda == -A^-1 b` (drift 포함) | 1.4e-10 |
| `dlambda/dtau`, rho = 0 / 1e-6 / 1e-4 / 1e-2 | 8e-16 / 5e-16 / 7e-16 / 7e-16 |
| `dlambda/dv` | 7e-16 |
| `dlambda/dq` | 6.5e-11 (유한차분 정밀도 한계) |
| sliding 행 구조 | 일치 |

### 두 가지 중요한 발견

**1. drift compensation 은 선택 사항이 아니다.**
빼면 `lambda` 가 10% 틀리고, `dlambda/dq` 는 통째로 틀린다
(기여 크기: `db/dq` 의 drift 항 24.6 vs `dA/dq*lambda` 항 0.18).
`cimpc_grad.py` 에 반영했다.

**2. rho D 항의 위치가 확정됐다.**
`D = diag(1/lambda_n^2)` 를 normal 인덱스에만. rho 를 무시하면 오차가
rho 에 비례해 커진다 (1e-6 -> 3.3e-5, 1e-2 -> 3.3e-1).
논문의 핵심 기여 항이 구현과 일치한다.

## 남은 것

`E_s = [mu cos th, mu sin th, 1]^T` 의 `th` 를 C++ 이 어떻게 정하는지 확인해서
sliding 케이스를 채우는 것. `compare.py` 의 [5] 는 아직 행 개수만 본다.
