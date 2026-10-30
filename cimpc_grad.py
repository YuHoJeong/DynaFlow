"""
CI-MPC gradient swap for DynaFlow.

목적
----
DynaFlow 학습 루프에서 동역학 롤아웃의 **backward pass만** 교체할 수 있게 만든다.
forward 는 기존 `diffsim.rollout` (MJX) 을 그대로 쓰고 VJP 만 갈아끼우므로,
forward 궤적과 loss 값은 baseline 과 완전히 동일하다.
따라서 학습 결과의 차이는 전적으로 gradient 차이에서만 온다 (controlled comparison).

구조
----
    make_rollout(env, jacobian_fn) -> rollout_states(q0, dq0, actions)
        returns (qpos_traj, qvel_traj)   # diffsim.batched_rollout 의 .qpos/.qvel 과 동일

        forward : diffsim.control_step 를 scan (MJX, 로직 변경 없음)
        backward: DDP-style adjoint recursion
                    lam_i = f_x[i]^T (lam_{i+1} + output_cotangent_i)
                  여기서 (f_x, f_u) 를 `jacobian_fn` 이 공급한다.

    jacobian_fn(env, qpos, qvel, warmstart, u, qpos_next) -> (f_x, f_u)
        f_x : (2nv, 2nv)  = d x_{i+1} / d x_i     (tangent 좌표)
        f_u : (2nv, nu)   = d x_{i+1} / d u_i

백엔드
------
    mjx_jacobian   : MJX 를 그대로 미분한 baseline.
                     custom_vjp 배선이 맞는지 검증하는 기준이며,
                     이걸 쓰면 plain MJX autodiff 와 gradient 가 일치해야 한다.
    cimpc_jacobian : CI-MPC (Kim et al., IJRR 2025) 의 relaxed complementarity
                     gradient, eq.(19).  clamping-only 구현 (sliding 은 TODO).

좌표계 주의 (중요)
------------------
MuJoCo 의 qpos 는 nq=19 (free joint quaternion 포함), qvel 은 nv=18 이다.
CI-MPC 의 f_x 는 nv 차원 tangent 공간에서 정의된다.
둘 사이를 오가려면 E(q) : R^nv -> R^nq (tangent -> ambient) 와
그 left inverse E^+(q) 가 필요하다.
이 변환을 빼먹으면 quaternion 성분에서 조용히 틀린 gradient 가 나온다.
verify_grad.py 의 test 1 이 이 변환을 mj_integratePos 로 검증한다.
"""

import jax
import jax.numpy as jnp
from functools import partial

import mujoco
from mujoco import mjx
from mujoco.mjx._src import support

import diffsim


# ===========================================================================
# 1. Tangent-space maps   (ambient nq=19  <->  tangent nv=18)
# ===========================================================================

def quat_tangent_matrix(quat):
    """4x3 matrix Q s.t.  d(quat) = 0.5 * Q @ dtheta,  dtheta 는 body-local 각속도.

    quat 는 MuJoCo 규약 (w, x, y, z).
    quat_dot = 0.5 * quat (x) [0, omega_local] 을 행렬로 편 것.
    Q^T Q = I_3 (unit quat 일 때) 이므로 left inverse 는 2 Q^T.
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jnp.array([
        [-x, -y, -z],
        [ w, -z,  y],
        [ z,  w, -x],
        [-y,  x,  w],
    ])


def tangent_to_ambient(qpos):
    """E(q) : (nq, nv).   d(qpos) = E @ d(q_tangent)."""
    nq = qpos.shape[-1]
    nv = nq - 1
    Q = quat_tangent_matrix(qpos[3:7])
    E = jnp.zeros((nq, nv))
    E = E.at[0:3, 0:3].set(jnp.eye(3))
    E = E.at[3:7, 3:6].set(0.5 * Q)
    E = E.at[7:nq, 6:nv].set(jnp.eye(nq - 7))
    return E


def ambient_to_tangent(qpos):
    """E^+(q) : (nv, nq).   E 의 left inverse (E^+ E = I_nv)."""
    nq = qpos.shape[-1]
    nv = nq - 1
    Q = quat_tangent_matrix(qpos[3:7])
    Ep = jnp.zeros((nv, nq))
    Ep = Ep.at[0:3, 0:3].set(jnp.eye(3))
    Ep = Ep.at[3:6, 3:7].set(2.0 * Q.T)
    Ep = Ep.at[6:nv, 7:nq].set(jnp.eye(nq - 7))
    return Ep


def integrate_tangent(qpos, dtheta):
    """qpos (+) dtheta : tangent 방향 섭동을 manifold 위로 올린다.

    유한차분 검증에서 quaternion 의 unit norm 을 깨지 않고 섭동하는 데 쓴다.
    dtheta = 0 근방에서 d/d(dtheta) = E(qpos) 가 되도록 되어 있다.
    """
    nq = qpos.shape[-1]
    quat = qpos[3:7]
    dq = jnp.concatenate([jnp.array([1.0]), 0.5 * dtheta[3:6]])
    quat_new = diffsim.q_mul(quat, dq)
    quat_new = quat_new / jnp.linalg.norm(quat_new)
    return jnp.concatenate([
        qpos[0:3] + dtheta[0:3],
        quat_new,
        qpos[7:nq] + dtheta[6:],
    ])


# ===========================================================================
# 2. Jacobian 백엔드
# ===========================================================================

def _step_from_state(env, qpos, qvel, warmstart, u):
    """control_step 한 번. warmstart 를 명시적으로 주입해서 재현성을 보장한다.

    diffsim.rollout 은 mjx.Data 를 scan carry 로 들고 다니지만,
    mjx.step 이 실제로 참조하는 이전 스텝 정보는 qpos/qvel/ctrl/qacc_warmstart 뿐이다.
    (verify_grad.py test 2b 가 이 가정을 검증한다.)
    """
    d = mjx.make_data(env.mjx_model).replace(
        qpos=qpos, qvel=qvel, qacc_warmstart=warmstart)
    d = diffsim.control_step(env.mjx_model, d, u, env.n_substeps)
    return d


def mjx_jacobian(env, qpos, qvel, warmstart, u, qpos_next):
    """Baseline: MJX 를 그대로 미분해서 tangent 좌표의 (f_x, f_u) 를 만든다.

    ambient Jacobian 을 뽑은 뒤 E / E^+ 로 sandwich:
        f_x = [[ E+_next Jqq E ,  E+_next Jqv ],
               [          Jvq E ,          Jvv ]]
        f_u = [[ E+_next Jqu ], [ Jvu ]]

    NOTE: jacfwd 라 tangent 방향이 nq+nv+nu = 49 개다. 정확하지만 느리다.
          이 백엔드는 검증용이고, 실제 학습에는 plain autodiff 를 쓰면 된다.
    """
    def f(q, v, uu):
        d = _step_from_state(env, q, v, warmstart, uu)
        return d.qpos, d.qvel

    (Jqq, Jqv, Jqu), (Jvq, Jvv, Jvu) = jax.jacfwd(f, argnums=(0, 1, 2))(qpos, qvel, u)

    E = tangent_to_ambient(qpos)              # (nq, nv)
    Epn = ambient_to_tangent(qpos_next)       # (nv, nq)

    f_x = jnp.block([[Epn @ Jqq @ E, Epn @ Jqv],
                     [      Jvq @ E,       Jvv]])
    f_u = jnp.concatenate([Epn @ Jqu, Jvu], axis=0)
    return f_x, f_u


# ---------------------------------------------------------------------------
# CI-MPC gradient  (Kim et al., IJRR 2025, Section 4.2)
# ---------------------------------------------------------------------------
#
#   forward (MJX 가 이미 풀어준 것을 재사용):
#       v_{i+1} = v_i + M^-1 ( (h + B u) dt + J^T lambda )
#       q_{i+1} = q_i (+) v_{i+1} dt                       (semi-implicit Euler)
#
#   contact impulse gradient, eq.(19):
#       d(lambda)/d(xi) = -[A + rho D]^-1 ( dA/d(xi) lambda + db/d(xi) )
#
#       A = J M^-1 J^T                                     (Delassus matrix)
#       b = J M^-1 ( (h + B u) dt + M v_i )                (unconstrained contact vel)
#       D = diag( 1/lambda[l]^2 )  for l in normal indices, 0 otherwise
#       rho = relaxation variable (rho -> 0 이면 Werling et al. 2021 의 hard gradient)
#
#   rho D 항이 이 논문의 핵심 기여다. contact mode 를 깨는 방향의 정보를 gradient 에
#   남겨주기 때문에, 기존 hard-contact gradient 처럼 "현재 접촉 상태를 유지하는"
#   방향으로만 밀지 않는다.  DynaFlow 의 motion imitation 에서도 이게 다를 거라고
#   기대하는 지점.
# ---------------------------------------------------------------------------


def foot_ids(env):
    """발 4개의 site / geom / body id 와 구 반지름, 마찰계수.

    순서는 CI-MPC 와 맞춘다: FL, FR, RL, RR.
    """
    names = ['FL', 'FR', 'RL', 'RR']
    site = jnp.array([mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in names])
    geom = [mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
    body = [int(env.mj_model.geom_bodyid[g]) for g in geom]
    radius = float(env.mj_model.geom_size[geom[0], 0])
    mu = float(env.mj_model.geom_friction[geom[0], 0])
    return site, geom, body, radius, mu


def foot_site_ids(env):
    return foot_ids(env)[0]


def foot_geom_radius(env):
    return foot_ids(env)[3]


def foot_positions(env, qpos, site_ids):
    """구 중심(발 site) 위치 (fnum, 3)."""
    d = mjx.make_data(env.mjx_model).replace(qpos=qpos)
    d = mjx.kinematics(env.mjx_model, d)
    return d.site_xpos[site_ids]


def foot_height(env, qpos, site_ids, foot_radius):
    """CI-MPC Sec 5.2 의 접촉 판정용 발 높이 phi_k = (구 중심 z) - r."""
    return foot_positions(env, qpos, site_ids)[:, 2] - foot_radius


def contact_points(env, qpos, site_ids, foot_radius):
    """MuJoCo 가 실제로 힘을 가하는 접촉점 (fnum, 3).

    구-평면 접촉에서 MuJoCo 의 접촉점은 침투 구간의 중점이다:
        z_contact = (z_center - r) / 2 = phi / 2
    xy 는 구 중심과 같다.
    """
    c = foot_positions(env, qpos, site_ids)
    return c.at[:, 2].set((c[:, 2] - foot_radius) / 2.0)


def contact_jacobians(env, qpos, site_ids=None, foot_radius=None, body_ids=None):
    """J_k = **접촉점에서의** body Jacobian (fnum, 3, nv).

    중요: 구 중심(site)이 아니라 접촉점에서 계산해야 한다.
    힘을 받는 물질점의 Jacobian 은 각속도 블록에 -skew(p_c - p_body) 가 들어가므로
    23mm 차이가 모멘트 팔을 바꾼다. 실측으로 확인:
        접촉점 Jacobian : sum_k J_k^T f_k == qfrc_constraint  오차 4.8e-17
        구 중심 Jacobian:                                     오차 4.6e-03

    부수 효과로 jacfwd 가 사라진다 (기존 구현은 순기구학을 jacfwd 로 미분했다).
    nested jacfwd 가 없어져 훨씬 싸고 메모리도 덜 쓴다.
    """
    if site_ids is None or body_ids is None or foot_radius is None:
        s_, _, b_, r_, _ = foot_ids(env)
        site_ids = s_ if site_ids is None else site_ids
        body_ids = b_ if body_ids is None else body_ids
        foot_radius = r_ if foot_radius is None else foot_radius
    d = mjx.make_data(env.mjx_model).replace(qpos=qpos)
    d = mjx.forward(env.mjx_model, d)
    pts = contact_points(env, qpos, site_ids, foot_radius)
    return jnp.stack([support.jac(env.mjx_model, d, pts[k], body_ids[k])[0].T
                      for k in range(len(body_ids))])


def contact_jacobians_bottom(env, qpos, site_ids, foot_radius, body_ids):
    """발 구의 **바닥점** (center - [0,0,r]) 에서의 J (fnum, 3, nv).

    hard-contact forward (`cimpc_forward`) 의 접촉점 규약. MJX soft contact 의
    접촉점(침투 중점, `contact_jacobians`) 과 다르므로 backward 도 forward 에 맞춰야 한다.
    운동학 + com_pos 만 계산하므로 `contact_jacobians` (mjx.forward) 보다 싸다.
    """
    from mujoco.mjx._src import smooth
    d = mjx.make_data(env.mjx_model).replace(qpos=qpos)
    d = smooth.kinematics(env.mjx_model, d)
    d = smooth.com_pos(env.mjx_model, d)
    pts = d.site_xpos[site_ids].at[:, 2].add(-foot_radius)
    return jnp.stack([support.jac(env.mjx_model, d, pts[k], body_ids[k])[0].T
                      for k in range(len(body_ids))])


def hard_contact_impulse(env, qpos, qvel, u, dt, ids):
    """`cimpc_forward` 의 solver 가 현재 상태에서 푼 lambda (fnum,3), contact, sliding.

    forward 와 같은 입력이므로 forward 가 실제로 적용한 impulse 와 동일하다.
    """
    import cimpc_forward as cf
    m = env.mjx_model
    d = mjx.make_data(m).replace(qpos=qpos, qvel=qvel, ctrl=u)
    d = cf.smooth_forward(m, d)
    M = mjx.full_m(m, d)
    J, phi = cf.contact_geometry(env, d, ids)
    fnum, _, nv = J.shape
    MinvJT = jnp.linalg.solve(M, J.reshape(3 * fnum, nv).T).T.reshape(fnum, 3, nv).transpose(0, 2, 1)
    return cf.solve_contact_impulse(J, MinvJT, qvel + d.qacc_smooth * dt, phi, dt, ids[4])


def foot_contact_impulse(env, qpos, qvel, u, warmstart, site_ids=None,
                         geom_ids=None, body_ids=None, foot_radius=None, dt=None):
    """MJX 가 푼 접촉력에서 발별 impulse lambda (fnum, 3) 를 뽑는다. world 좌표.

    `support.contact_force(..., to_world_frame=True)` 로 pyramid 표현을 풀어
    발 geom 에 맞춰 합산한다. 최소자승 투영보다 정확하다 (위 4.8e-17 검증).
    """
    if site_ids is None or geom_ids is None or body_ids is None or foot_radius is None:
        s_, g_, b_, r_, _ = foot_ids(env)
        site_ids = s_ if site_ids is None else site_ids
        geom_ids = g_ if geom_ids is None else geom_ids
        body_ids = b_ if body_ids is None else body_ids
        foot_radius = r_ if foot_radius is None else foot_radius
    # 스텝을 진행한 뒤가 아니라 **현재 상태** 의 접촉력이어야 한다.
    # CI-MPC 의 SolveContactImpulse 도 현재 (q, v, tau) 에서 lambda 를 푼다.
    # _step_from_state 를 쓰면 4 substep 뒤 상태의 lambda 가 나와서
    # 같은 qpos 에서 계산한 J 와 어긋난다 (실측 1.4e-4 오차).
    d = mjx.make_data(env.mjx_model).replace(
        qpos=qpos, qvel=qvel, ctrl=u, qacc_warmstart=warmstart)
    d = mjx.forward(env.mjx_model, d)
    ncon = d._impl.contact.dim.shape[0]
    geom = d._impl.contact.geom
    lam = jnp.zeros((len(geom_ids), 3), dtype=qpos.dtype)
    for i in range(ncon):
        f = support.contact_force(env.mjx_model, d, i, to_world_frame=True)[:3]
        for k, gid in enumerate(geom_ids):
            hit = ((geom[i, 0] == gid) | (geom[i, 1] == gid)).astype(qpos.dtype)
            lam = lam.at[k].add(hit * f)
    return lam * (env.control_dt if dt is None else dt), d   # force -> impulse


def smooth_acc_and_mass(env, qpos, qvel, u):
    """접촉을 제외한 가속도 qacc_smooth 와 질량행렬 M.

        qacc_smooth = M^-1 (qfrc_actuator - qfrc_bias)     (6e-16 로 검증됨)

    두 가지를 직접 쓰면 안 된다:
      1. DynaFlow 의 action 은 **토크가 아니라 position actuator 의 목표각** 이다
         (gaintype=fixed kp=20, biastype=affine -kp=-20 -kd=-0.5).
      2. qfrc_bias 는 `M qacc + qfrc_bias = qfrc_applied` 규약이라 **부호가 음수** 다.
         Go1 에서 |qfrc_bias| ~ 117 vs |qfrc_actuator| ~ 0.56 로 bias 가 200배
         지배적이라, 부호를 틀리면 gradient 가 통째로 어긋난다.
    """
    d = mjx.make_data(env.mjx_model).replace(qpos=qpos, qvel=qvel, ctrl=u)
    d = mjx.forward(env.mjx_model, d)
    return d.qacc_smooth, mjx.full_m(env.mjx_model, d)


def _smoothstep(x, lo, hi):
    """[lo, hi] 구간에서 0 -> 1 로 매끄럽게 (C^1) 올라간다."""
    t = jnp.clip((x - lo) / jnp.maximum(hi - lo, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _cimpc_step_jacobian(env, qpos, qvel, warmstart, u, dt, ids, rho,
                         contact_eps, slide_lo, slide_hi, slide_vel_eps,
                         forward="mjx"):
    """semi-implicit Euler 한 스텝(길이 dt)의 (f_x, f_u). tangent 좌표.

    forward="mjx"  : lambda 를 MJX soft contact 에서 뽑고 (contact_force), 접촉점은
                     침투 중점, clamping/sliding 은 alpha 로 연속 전이 (아래 주석).
    forward="cimpc": lambda 를 `cimpc_forward` 의 hard-contact solver 에서 받는다.
                     접촉점은 발 바닥점, sliding 판정은 solver 의 플래그 그대로,
                     E_s 는 C++ 처럼 lambda 에서 뽑는다. slide_* 튜닝값이 필요 없다.

    CI-MPC eq.(19). 규약은 전부 C++ 원본과 대조해서 확정했다 (`cimpc_ref/`):

        A = J_left M^-1 J_right^T
        b = J_left (v + qacc_smooth dt),  b[3k+2] += phi_k / dt   (drift compensation)
        lambda = -A^-1 b                                          impulse, world 좌표
        D = diag(1/lambda_n^2)  on 인덱스 3k+2
        dlambda/dxi = -[A + rho D]^-1 (dA/dxi lambda + db/dxi)
    """
    site_ids, geom_ids, body_ids, foot_radius, mu = ids
    nv, nu = env.mj_model.nv, env.mj_model.nu
    fnum = len(geom_ids)
    n = 3 * fnum
    dt_arr = jnp.asarray(dt, dtype=qpos.dtype)

    if forward == "cimpc":
        jac_fn = contact_jacobians_bottom
        lam, contact, sliding = hard_contact_impulse(env, qpos, qvel, u, dt, ids)
        lam_n = lam[:, 2]
        active = contact.astype(lam.dtype)
        alpha = sliding.astype(lam.dtype)
        # C++: E = [lam_x/lam_n, lam_y/lam_n, 1]. hard solver 는 원뿔 경계에 정확히 올려놓는다.
        E = jnp.concatenate([lam[:, :2] / jnp.maximum(lam_n, contact_eps)[:, None],
                             jnp.ones((fnum, 1), dtype=lam.dtype)], axis=-1)
    else:
        jac_fn = contact_jacobians
        lam, _ = foot_contact_impulse(env, qpos, qvel, u, warmstart,
                                      site_ids, geom_ids, body_ids, foot_radius, dt)
        lam_n = lam[:, 2]
        active = (lam_n > contact_eps).astype(lam.dtype)           # (fnum,)

    # --- clamping <-> sliding 을 **연속** 으로 섞는다 -----------------------
    # C++ 은 자기 solver 가 lambda 를 마찰 원뿔 경계에 정확히 올려놓으므로
    # abs(||lambda_t|| - mu lambda_n) < 1e-6 으로 딱 갈린다.
    # MJX 는 soft contact 라 경계가 뚜렷하지 않다 (실측: 측방 0.5 m/s 에서
    # 포화 비율 0.78 로 애매, 4 m/s 에서 0.96~0.995).
    # 하드 임계값을 쓰면 "어디서 자르나" 가 임의적일 뿐 아니라 배치 안에서
    # gradient 가 불연속으로 점프한다. 그래서 alpha in [0,1] 로 연속 전이시킨다.
    # alpha=0 은 순수 clamping, alpha=1 은 순수 sliding 이며 두 극단에서
    # C++ 과 대조된 구조를 정확히 재현한다.
        J_now = contact_jacobians(env, qpos, site_ids, foot_radius, body_ids)   # (fnum,3,nv)
        v_t = jnp.einsum('kdv,v->kd', J_now, qvel)[:, :2]
        v_t_norm = jnp.linalg.norm(v_t, axis=-1)
        lam_t_norm = jnp.linalg.norm(lam[:, :2], axis=-1)
        ratio = lam_t_norm / jnp.maximum(mu * lam_n, contact_eps)
        alpha = (_smoothstep(ratio, slide_lo, slide_hi)
                 * _smoothstep(v_t_norm, 0.0, slide_vel_eps) * active)     # (fnum,)

        # E_k: 최대 소산 방향 lambda_t = -mu lambda_n vhat_t 에서 나온다.
        # C++ 은 E = [lam_x/lam_z, lam_y/lam_z, 1] 로 lambda 에서 뽑지만,
        # 그건 자기 solver 가 원뿔 경계에 정확히 올려놓기 때문이다. MJX 는 아니므로
        # 속도에서 뽑는 게 안전하다 (두 방식은 경계 위에서 일치한다).
        vhat = v_t / jnp.maximum(v_t_norm, 1e-12)[:, None]
        E = jnp.concatenate([-mu * vhat, jnp.ones((fnum, 1), dtype=lam.dtype)], axis=-1)

    # 행 가중치: 접선 2행은 (1-alpha) 로 줄어들고 normal 행은 그대로.
    # alpha=1 이면 접선 행이 사라져 sliding 구조(normal 1행)가 된다.
    w = jnp.stack([active * (1.0 - alpha), active * (1.0 - alpha), active], axis=-1)
    w = w.reshape(-1)                                              # (n,)

    lam_vec = lam.reshape(-1) * w

    def left_right(J):
        """CI-MPC 행/열 구조 (`contactSolverGradient.cpp:2160-2180`).

        clamping k : 왼쪽 3행 J_k        , 오른쪽 3행 J_k
        sliding  k : 왼쪽 1행 J_k[2,:]   , 오른쪽 1행 E_k^T J_k
        -> sliding 이 섞이면 A 가 비대칭이다.
        alpha 로 두 구조를 연속으로 섞는다.
        """
        Jl = (J * w.reshape(fnum, 3)[:, :, None]).reshape(n, nv)
        JE = jnp.einsum('kd,kdv->kv', E, J)                        # (fnum, nv)
        Jr = Jl.reshape(fnum, 3, nv)
        normal_row = ((1.0 - alpha)[:, None] * J[:, 2, :]
                      + alpha[:, None] * JE) * active[:, None]
        Jr = Jr.at[:, 2, :].set(normal_row)
        return Jl, Jr.reshape(n, nv)

    def A_and_b(theta_q, vv, uu):
        q = integrate_tangent(qpos, theta_q)
        J = jac_fn(env, q, site_ids, foot_radius, body_ids)
        acc_s, M = smooth_acc_and_mass(env, q, vv, uu)
        Jl, Jr = left_right(J)
        A = Jl @ jnp.linalg.solve(M, Jr.T)
        b = Jl @ (vv + acc_s * dt_arr)
        # drift compensation (Sec 5.2 / Appendix B): v_n = 0 대신 v_n + phi_k/dt = 0.
        # C++ 대조 결과 이 항이 dlambda/dq 를 지배한다 (24.6 vs 0.18). 필수다.
        phi = foot_height(env, q, site_ids, foot_radius)
        b = b.at[2::3].add(phi / dt_arr * active)
        return A, b

    zero_q = jnp.zeros(nv, dtype=qpos.dtype)
    (dA_dq, dA_dv, dA_du), (db_dq, db_dv, db_du) = jax.jacfwd(
        A_and_b, argnums=(0, 1, 2))(zero_q, qvel, u)
    A, _ = A_and_b(zero_q, qvel, u)

    inv_lam2 = active / jnp.maximum(lam_n, contact_eps) ** 2
    D = jnp.diag(jnp.stack([jnp.zeros_like(inv_lam2), jnp.zeros_like(inv_lam2),
                            inv_lam2], axis=-1).reshape(-1))

    # 비활성/약화된 행은 대각 항등원으로 채워 가역성을 유지하고 결과를 지운다.
    S = (A + rho * D) * jnp.outer(w, w) + jnp.diag(1.0 - w)

    def solve_dlam(dA_dxi, db_dxi):
        rhs = jnp.einsum('ijk,j->ik', dA_dxi, lam_vec) + db_dxi
        return -jnp.linalg.solve(S, rhs * w[:, None]) * w[:, None]

    dlam_dq, dlam_dv, dlam_du = (solve_dlam(dA_dq, db_dq),
                                 solve_dlam(dA_dv, db_dv),
                                 solve_dlam(dA_du, db_du))

    def expand(dl):
        """sliding 성분은 normal 만 풀었으므로 eq.(14) 로 3D 복원:
             d(lambda_s)/dxi = E_s d(lambda_n)/dxi
        alpha 로 clamping 해와 연속으로 섞는다."""
        dl3 = dl.reshape(fnum, 3, -1)
        dn = dl3[:, 2, :]
        return ((1.0 - alpha)[:, None, None] * dl3
                + alpha[:, None, None] * E[:, :, None] * dn[:, None, :]).reshape(n, -1)

    dlam_dq, dlam_dv, dlam_du = expand(dlam_dq), expand(dlam_dv), expand(dlam_du)
    lam_full = (((1.0 - alpha)[:, None] * lam + alpha[:, None] * E * lam_n[:, None])
                * active[:, None]).reshape(-1)

    # ---- f_x, f_u 조립 ----
    def v_next_smooth(theta_q, vv, uu):
        q = integrate_tangent(qpos, theta_q)
        J = jac_fn(env, q, site_ids, foot_radius, body_ids).reshape(n, nv)
        acc_s, M = smooth_acc_and_mass(env, q, vv, uu)
        return vv + acc_s * dt_arr + jnp.linalg.solve(M, J.T @ lam_full)

    dvn_dq, dvn_dv, dvn_du = jax.jacfwd(v_next_smooth, argnums=(0, 1, 2))(zero_q, qvel, u)

    _, M0 = smooth_acc_and_mass(env, qpos, qvel, u)
    J0 = jac_fn(env, qpos, site_ids, foot_radius, body_ids).reshape(n, nv)
    Minv_JT0 = jnp.linalg.solve(M0, J0.T)

    dvn_dq = dvn_dq + Minv_JT0 @ dlam_dq
    dvn_dv = dvn_dv + Minv_JT0 @ dlam_dv
    dvn_du = dvn_du + Minv_JT0 @ dlam_du

    # 위치 갱신은 q_next = integrate(q, v_next dt) 다. 병진/관절은 q + dt v 지만
    # quaternion 은 지수 사상이라 d(theta_next)/d(theta) = I + O(omega dt) 의
    # Lie-group 항이 붙는다. `I + dt dvn_dq` 로 근사하면 omega ~ 1 rad/s, dt=0.01 에서
    # 1e-2 상대오차가 각속도 블록에 남는다 (실측: contact FD 대비 1.6e-3, plain
    # autodiff 는 4e-8). 그래서 실제 적분 함수를 tangent 좌표에서 직접 미분한다.
    from mujoco.mjx._src import forward as mjx_forward
    v_next0 = v_next_smooth(zero_q, qvel, u)
    jnt_type = env.mjx_model.jnt_type

    def pos_step(theta_q, vn):
        q = integrate_tangent(qpos, theta_q)
        return mjx_forward._integrate_pos(jnt_type, q, vn, dt_arr)

    Dq_theta, Dq_v = jax.jacfwd(pos_step, argnums=(0, 1))(zero_q, v_next0)   # (nq,nv) x2
    Ep_next = ambient_to_tangent(pos_step(zero_q, v_next0))                    # (nv,nq)
    Pq = Ep_next @ Dq_theta                     # d theta_next / d theta   (v_next 고정)
    Pv = Ep_next @ Dq_v                         # d theta_next / d v_next  (~ dt I)

    f_x = jnp.block([[Pq + Pv @ dvn_dq, Pv @ dvn_dv],
                     [          dvn_dq,      dvn_dv]])
    f_u = jnp.concatenate([Pv @ dvn_du, dvn_du], axis=0)
    return f_x, f_u


def cimpc_jacobian(env, qpos, qvel, warmstart, u, qpos_next,
                   rho=1e-4, contact_eps=1e-9,
                   slide_lo=0.7, slide_hi=0.98, slide_vel_eps=2e-2,
                   n_sub=None, forward="mjx"):
    """CI-MPC relaxed gradient. forward 와 같은 substep 으로 쪼개서 연쇄곱한다.

    forward 는 `simulation_dt=0.005` 를 `n_substeps=4` 번 밟는데,
    backward 를 `control_dt=0.02` 한 스텝으로 두면 적분 granularity 가 어긋난다.
    실측: flight(접촉 없음) 에서 n_substeps=1 이면 MJX 와 cos=1.000000 인데
          n_substeps=4 면 cos=0.9716, action gradient 크기비가 1.39 까지 벌어졌다.
    CI-MPC 원 논문은 dt=0.025 단일 스텝이지만 여기서 그걸 따를 이유가 없으므로
    forward 에 맞춘다.

        f_x = f_x^(n-1) ... f_x^(0),   f_u = sum_j (뒤쪽 f_x 들) f_u^(j)

    n_sub=1 로 두면 예전처럼 control_dt 한 스텝으로 계산한다 (4배 싸다).

    forward="cimpc" 이면 lambda / 접촉 상태 / 중간 substep 상태를 전부
    `cimpc_forward` 의 hard-contact 동역학에서 가져온다. 이때 n_sub 는 forward 의
    n_sub 와 같아야 한다 (같은 semi-implicit Euler 스텝의 정확한 구조 미분이 된다).
    """
    n_sub = env.n_substeps if n_sub is None else int(n_sub)
    dt = env.control_dt / n_sub
    nv, nu = env.mj_model.nv, env.mj_model.nu
    ids = foot_ids(env)

    q, v, ws = qpos, qvel, warmstart
    f_x = jnp.eye(2 * nv, dtype=qpos.dtype)
    f_u = jnp.zeros((2 * nv, nu), dtype=qpos.dtype)
    for j in range(n_sub):
        fx, fu = _cimpc_step_jacobian(env, q, v, ws, u, dt, ids, rho,
                                      contact_eps, slide_lo, slide_hi, slide_vel_eps,
                                      forward=forward)
        f_x = fx @ f_x
        f_u = fx @ f_u + fu
        if j < n_sub - 1:
            if forward == "cimpc":
                import cimpc_forward as cf
                q, v, _ = cf.cimpc_substep(env, q, v, u, dt, ids)
            else:
                d = mjx.make_data(env.mjx_model).replace(
                    qpos=q, qvel=v, ctrl=u, qacc_warmstart=ws)
                d = mjx.step(env.mjx_model, d)
                q, v, ws = d.qpos, d.qvel, d.qacc_warmstart
    return f_x, f_u


# ===========================================================================
# 3. custom_vjp rollout
# ===========================================================================

# (env, jacobian_fn) 당 컴파일된 rollout 을 재사용한다.
# mjx_jacobian 의 jacfwd 그래프는 컴파일이 매우 비싸서 (49 tangent x 4 substep),
# 캐시가 없으면 make_rollout 을 부를 때마다 몇 분씩 다시 컴파일한다.
_ROLLOUT_CACHE = {}


def make_rollout(env, jacobian_fn=None, forward="mjx", fwd_n_sub=None):
    """(qpos_traj, qvel_traj) 를 돌려주는 rollout 을 만든다.

    jacobian_fn=None 이면 custom_vjp 없이 순수 autodiff (baseline).
    jacobian_fn 을 주면 backward 만 그 함수가 공급하는 (f_x, f_u) 로 대체된다.

    forward="mjx"   : diffsim.rollout (MJX soft contact, 0.005 x 4)
    forward="cimpc" : cimpc_forward.make_rollout (hard contact, control_dt / fwd_n_sub)
                      jacobian_fn 은 partial(cimpc_jacobian, forward="cimpc", n_sub=fwd_n_sub)
                      여야 forward 와 같은 동역학을 미분한다.
    """
    key = (id(env), jacobian_fn, forward, fwd_n_sub)
    if key in _ROLLOUT_CACHE:
        return _ROLLOUT_CACHE[key][1]
    nv = env.mj_model.nv
    nq = env.mj_model.nq

    ws0 = mjx.make_data(env.mjx_model).qacc_warmstart

    if forward == "cimpc":
        import cimpc_forward as cf
        _hard = cf.make_rollout(env, n_sub=fwd_n_sub)

        def _forward(q0, dq0, action_traj):
            qs, vs, _ = _hard(q0, dq0, action_traj)
            # hard contact 에는 warmstart 가 없다. jacobian_fn(forward="cimpc") 는 쓰지 않는다.
            return qs, vs, jnp.zeros((qs.shape[0], nv), dtype=qs.dtype)
    elif forward == "mjx":
        def _forward(q0, dq0, action_traj):
            """forward 는 diffsim.rollout 을 **그대로** 호출한다.

            scan 을 복제하면 XLA 가 다르게 융합해서 roundoff 가 달라지고,
            접촉 모드 경계 근처에서는 그 roundoff 가 O(1e-6) 까지 증폭된다.
            재사용해야 baseline 과 bit-identical 이 보장된다.

            ws_t[i] = 스텝 i 에 **들어갈 때의** qacc_warmstart.
            """
            traj = diffsim.rollout(env, q0, dq0, action_traj)
            ws_t = jnp.concatenate([ws0[None], traj.qacc_warmstart[:-1]], axis=0)
            return traj.qpos, traj.qvel, ws_t
    else:
        raise ValueError(f"forward={forward!r} (mjx | cimpc)")

    if jacobian_fn is None:
        def rollout_states(q0, dq0, action_traj):
            qpos_t, qvel_t, _ = _forward(q0, dq0, action_traj)
            return qpos_t, qvel_t
        out = jax.jit(rollout_states)
        _ROLLOUT_CACHE[key] = (env, out)   # env 참조를 잡아둬서 id 재사용을 막는다
        return out

    @jax.custom_vjp
    def rollout_states(q0, dq0, action_traj):
        qpos_t, qvel_t, _ = _forward(q0, dq0, action_traj)
        return qpos_t, qvel_t

    def _fwd(q0, dq0, action_traj):
        qpos_t, qvel_t, ws_t = _forward(q0, dq0, action_traj)
        # 각 스텝의 "이전" 상태: x_0, x_1, ..., x_{H-1}
        qpos_prev = jnp.concatenate([q0[None], qpos_t[:-1]], axis=0)
        qvel_prev = jnp.concatenate([dq0[None], qvel_t[:-1]], axis=0)
        res = (qpos_prev, qvel_prev, ws_t, action_traj, qpos_t)
        return (qpos_t, qvel_t), res

    def _bwd(res, ct):
        qpos_prev, qvel_prev, ws_t, action_traj, qpos_next = res
        ct_qpos, ct_qvel = ct                                   # (H, nq), (H, nv)

        f_x, f_u = jax.vmap(partial(jacobian_fn, env))(
            qpos_prev, qvel_prev, ws_t, action_traj, qpos_next)  # (H,2nv,2nv), (H,2nv,nu)

        # 출력 cotangent 를 ambient -> tangent 로 당긴다:  dL/dq_t = E^T dL/dqpos
        E_next = jax.vmap(tangent_to_ambient)(qpos_next)         # (H, nq, nv)
        ct_tan = jnp.concatenate(
            [jnp.einsum('hqv,hq->hv', E_next, ct_qpos), ct_qvel], axis=-1)  # (H, 2nv)

        def rev(lam, xs):
            fx_i, fu_i, ct_i = xs
            tot = lam + ct_i
            return fx_i.T @ tot, fu_i.T @ tot

        lam0, g_u = jax.lax.scan(
            rev, jnp.zeros(2 * nv), (f_x, f_u, ct_tan), reverse=True)

        # tangent -> ambient:  dL/dqpos = (E^+)^T dL/dq_t
        Ep0 = ambient_to_tangent(qpos_prev[0])                   # (nv, nq)
        g_q0 = Ep0.T @ lam0[:nv]
        g_v0 = lam0[nv:]
        return g_q0, g_v0, g_u

    rollout_states.defvjp(_fwd, _bwd)
    out = jax.jit(rollout_states)
    _ROLLOUT_CACHE[key] = (env, out)       # env 참조를 잡아둬서 id 재사용을 막는다
    return out


def make_batched_rollout(env, jacobian_fn=None, chunk=None, forward="mjx", fwd_n_sub=None):
    """DynaFlow 의 diffsim.batched_rollout 자리에 그대로 들어가는 batch 버전.

    chunk 를 주면 배치를 쪼개 순차 처리해서 peak 메모리를 묶는다
    (cimpc_jacobian + B=1024 는 chunk=256 이 필요하다).

    사용법 (main.py:143 교체):
        rollout_fn = cimpc_grad.make_batched_rollout(env, cimpc_grad.cimpc_jacobian,
                                                     chunk=256)
        pred_qpos, pred_qvel = rollout_fn(q0, dq0, actions)
        pred_states = jnp.concatenate([pred_qpos, pred_qvel], axis=-1)[:, :-1, :]
    """
    key = (id(env), jacobian_fn, "batched", chunk, forward, fwd_n_sub)
    if key in _ROLLOUT_CACHE:
        return _ROLLOUT_CACHE[key][1]
    single = make_rollout(env, jacobian_fn, forward=forward, fwd_n_sub=fwd_n_sub)
    vmapped = jax.vmap(single, in_axes=(0, 0, 0))

    if chunk is None:
        out = jax.jit(vmapped)
    else:
        # 배치를 청크로 쪼개 순차 처리해서 peak 메모리를 상한 지운다.
        # cimpc_jacobian 은 B=1024 를 한 번에 하면 7 GiB 단일 할당이 필요해 OOM 이다
        # (B=256 은 1.5 GiB). lax.map 은 내부가 scan 이라 backward 도 청크 단위로
        # 순차 실행되므로 peak 이 청크 크기로 묶인다.
        def chunked(q0, dq0, actions):
            B = q0.shape[0]
            if B % chunk != 0:
                raise ValueError(f"batch {B} 가 chunk {chunk} 로 나눠떨어지지 않는다")
            nc = B // chunk
            resh = lambda x: x.reshape((nc, chunk) + x.shape[1:])
            qp, qv = jax.lax.map(lambda a: vmapped(*a),
                                 (resh(q0), resh(dq0), resh(actions)))
            return (qp.reshape((B,) + qp.shape[2:]),
                    qv.reshape((B,) + qv.shape[2:]))
        out = jax.jit(chunked)

    _ROLLOUT_CACHE[key] = (env, out)
    return out
