"""
CI-MPC hard-contact forward dynamics in JAX.

목적
----
DynaFlow 의 forward 시뮬레이터(MJX soft contact)를 CI-MPC (Kim et al., IJRR 2025)
의 **hard contact solver** 로 바꾼다. `cimpc_grad.cimpc_jacobian` 이 backward 를
담당하므로, 이 모듈이 붙으면 forward 도 backward 도 CI-MPC 동역학이 된다.

알고리즘 (C++ `ContactSolverGradient::SolveContactImpulse` 를 그대로 옮김)
---------------------------------------------------------------------------
    v_free = v + M^-1 (tau - nle) dt                    접촉 없는 다음 속도
    P_k    = J_k M^-1 J_k^T                             (3x3) per-contact Delassus
    outer 반복 (최대 n_outer):
      contact k 순서대로 (Gauss-Seidel):
        c_k  = J_k (v_free + sum_{j!=k} M^-1 J_j^T lambda_j)
        c_k' = c_k + [0, 0, phi_k/dt]                   drift compensation
        c_k'[2] > 0            -> separating, lambda_k = 0
        lambda_k = -P_k^-1 c_k'                          clamping (발이 멈춤)
        ||lambda_t|| > mu lambda_n -> sliding:
            원뿔 표면 lambda(th) = r(th) [cos th, sin th, 1/mu],
            r(th) 는 normal 속도 0 (Signorini) 이 되도록 정한다.
            D(th) = (P lambda + c)^T zeta(th), zeta = P[2] x grad(h2)
            D(th) = 0 을 bisection (n_bisect) 으로 푼다.
    v_next = v_free + sum_k M^-1 J_k^T lambda_k
    q_next = integrate(q, v_next dt)                     semi-implicit Euler

JAX 제약
--------
C++ 은 1e-6 수렴 시 조기 종료하지만 JAX 는 고정 반복수여야 한다.
`done` 플래그로 조기 종료 **시점의 값을 그대로 유지** 하도록 해서
C++ 과 같은 답이 나오게 했다 (verify_forward.py 가 C++ 덤프와 대조).
fnum=4 라 GS 체인은 길이 4, bisection 은 8회라 스텝당 고정 비용은 작다.

MJX 에서 가져오는 것: M, qacc_smooth (actuator/중력/passive 포함), 발 위치,
접촉점 Jacobian. **접촉 solver 만 새로 쓴다.**

접촉점 규약
-----------
J 와 phi 는 발 구의 **바닥점** (center - [0,0,r]) 에서 계산한다.
C++ 은 발 프레임 원점에서 계산하고 그 높이를 phi 로 쓰므로 같은 규약이다.
MJX 의 soft contact 접촉점(침투 중점, `cimpc_grad.contact_points`) 과는 다르다.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from mujoco import mjx
from mujoco.mjx._src import support, smooth, passive as mjx_passive
from mujoco.mjx._src import forward as mjx_forward

import diffsim
import cimpc_grad as cg

# 고정 반복수 scan 을 펼쳐서 XLA 가 반복 사이의 작은 연산들을 융합하게 한다.
# GPU 에서는 이 solver 가 (배치와 무관하게) 커널 launch 지연에 묶이므로 중요하다.
UNROLL = True


# ===========================================================================
# 1. 접촉 impulse solver (순수 선형대수, 로봇 무관)
# ===========================================================================

def _norm(x):
    """||x|| 인데 x=0 에서 gradient 가 NaN 이 되지 않는다 (double-where)."""
    s = jnp.sum(x * x)
    safe = jnp.where(s > 0, s, 1.0)
    return jnp.where(s > 0, jnp.sqrt(safe), 0.0)


def _cone_point(P2, ci_n, th, mu):
    """원뿔 표면 위에서 normal 속도가 0 이 되는 점. r(th)[cos, sin, 1/mu]."""
    d = jnp.array([jnp.cos(th), jnp.sin(th), 1.0 / mu])
    denom = P2 @ d
    r = -ci_n / denom
    return r * d, r


def _D_of(P, ci, lam, coeff):
    """bisection 의 목적함수 D(th) = (P lam + ci)^T (P[2] x grad h2)."""
    grad_h2 = coeff * lam
    zeta = jnp.cross(P[2], grad_h2)
    return (P @ lam + ci) @ zeta


def _sliding_bisection(P, ci, ci_, lam_c, mu, coeff, n_bisect, tol):
    """C++ 의 sliding 분기. lam_c 는 clamping 해 (원뿔 밖)."""
    # sliding 분기가 선택되지 않는 경우에도 계산은 하므로 (JAX where),
    # NaN 이 나오지 않게 입력을 안전한 값으로 막는다.
    ci_n = jnp.minimum(ci_[2], -1e-12)
    lam_t_norm = jnp.maximum(_norm(lam_c[:2]), 1e-12)

    # arctan2(0,0) 의 gradient 는 NaN -> 접선 성분이 0 이면 각도 0 으로 둔다
    t_zero = (lam_c[0] == 0) & (lam_c[1] == 0)
    th1 = jnp.arctan2(jnp.where(t_zero, 0.0, lam_c[1]),
                      jnp.where(t_zero, 1.0, lam_c[0]))
    lam1, r1 = _cone_point(P[2], ci_n, th1, mu)
    D1 = _D_of(P, ci, lam1, coeff)

    arg = r1 / lam_t_norm
    # arccos 는 |arg|=1 에서 gradient 가 발산 -> 정의역 밖이면 안전한 값으로 대체
    out = jnp.abs(arg) >= 1.0
    th_bound = jnp.where(out, 2.0, jnp.arccos(jnp.where(out, 0.0, arg)))

    th2 = th1 + th_bound
    lam2, _ = _cone_point(P[2], ci_n, th2, mu)
    D2 = _D_of(P, ci, lam2, coeff)

    flip = D1 * D2 > 0
    th2b = th1 - th_bound
    lam2b, _ = _cone_point(P[2], ci_n, th2b, mu)
    D2b = _D_of(P, ci, lam2b, coeff)
    th2 = jnp.where(flip, th2b, th2)
    lam2 = jnp.where(flip, lam2b, lam2)
    D2 = jnp.where(flip, D2b, D2)

    def body(carry, _):
        th1, th2, D1, D2, lam, done = carry
        th_mid = 0.5 * (th1 + th2)
        lam_mid, _ = _cone_point(P[2], ci_n, th_mid, mu)
        D_mid = _D_of(P, ci, lam_mid, coeff)
        conv = _norm(lam - lam_mid) < tol
        lam_new = jnp.where(done, lam, lam_mid)
        # C++: lamb_temp 는 갱신된 뒤 break 하므로 conv 순간의 mid 값이 답이다.
        # 그 다음 bracket 갱신은 done 이 아닐 때만.
        upd = jnp.logical_not(done | conv)
        same = D_mid * D1 > 0
        th1n = jnp.where(upd & same, th_mid, th1)
        D1n = jnp.where(upd & same, D_mid, D1)
        th2n = jnp.where(upd & jnp.logical_not(same), th_mid, th2)
        D2n = jnp.where(upd & jnp.logical_not(same), D_mid, D2)
        return (th1n, th2n, D1n, D2n, lam_new, done | conv), None

    init = (th1, th2, D1, D2, lam2, jnp.array(False))
    (_, _, _, _, lam, _), _ = lax.scan(body, init, None, length=n_bisect, unroll=UNROLL)
    return lam


def solve_contact_impulse(J, MinvJT, v_free, phi, dt, mu,
                          n_outer=10, n_bisect=8, tol=1e-6, lam0=None):
    """per-contact Gauss-Seidel + bisection (C++ SolveContactImpulse 와 동일).

    J      : (fnum, 3, nv)   접촉점 Jacobian (world 정렬)
    MinvJT : (fnum, nv, 3)   M^-1 J_k^T
    v_free : (nv,)           접촉 없는 다음 속도
    phi    : (fnum,)         발 높이 (음수 = 침투)
    반환: lambda (fnum,3) impulse, contact (fnum,) bool, sliding (fnum,) bool
    """
    fnum = J.shape[0]
    dtype = v_free.dtype
    P = jnp.einsum('kav,kvb->kab', J, MinvJT)                      # (fnum,3,3)
    coeff = jnp.array([2.0, 2.0, -2.0 * mu ** 2], dtype=dtype)
    lam0 = jnp.zeros((fnum, 3), dtype=dtype) if lam0 is None else lam0

    def solve_one(k, lam):
        dv = jnp.einsum('jvb,jb->v', MinvJT, lam) - MinvJT[k] @ lam[k]
        ci = J[k] @ (v_free + dv)
        ci_ = ci.at[2].add(phi[k] / dt)
        sep = ci_[2] > 0
        lam_c = -jnp.linalg.solve(P[k], ci_)
        slide = (_norm(lam_c[:2]) - mu * lam_c[2]) > 0
        lam_s = _sliding_bisection(P[k], ci, ci_, lam_c, mu, coeff, n_bisect, tol)
        lam_k = jnp.where(sep, jnp.zeros(3, dtype), jnp.where(slide, lam_s, lam_c))
        return lam.at[k].set(lam_k)

    def sweep(carry, _):
        lam, done = carry
        new = lam
        for k in range(fnum):
            new = solve_one(k, new)
        conv = _norm(new - lam) < tol
        new = jnp.where(done, lam, new)
        return (new, done | conv), None

    (lam, _), _ = lax.scan(sweep, (lam0, jnp.array(False)), None, length=n_outer, unroll=UNROLL)

    # 최종 상태 분류 (C++ 과 동일 규약)
    lam_n = lam[:, 2]
    contact = lam_n > 0
    lam_t = jax.vmap(_norm)(lam[:, :2])
    sliding = (jnp.abs(lam_t - mu * lam_n) < tol) & contact
    return lam, contact, sliding


# ===========================================================================
# 2. MJX 에서 M, qacc_smooth, J, phi 를 얻는 부분
# ===========================================================================

def smooth_forward(m, d):
    """mjx.forward 에서 constraint / collision 을 뺀 것. qacc_smooth 까지만 계산."""
    d = smooth.kinematics(m, d)
    d = smooth.com_pos(m, d)
    d = smooth.crb(m, d)
    d = smooth.factor_m(m, d)
    d = smooth.transmission(m, d)
    d = d.tree_replace({
        '_impl.actuator_velocity': d._impl.actuator_moment @ d.qvel,
    })
    d = smooth.com_vel(m, d)
    d = mjx_passive.passive(m, d)
    d = smooth.rne(m, d)
    d = mjx_forward.fwd_actuation(m, d)
    d = mjx_forward.fwd_acceleration(m, d)
    return d


def contact_geometry(env, d, ids):
    """발 바닥점에서의 J (fnum,3,nv) 와 높이 phi (fnum,)."""
    site_ids, _, body_ids, radius, _ = ids
    pts = d.site_xpos[site_ids].at[:, 2].add(-radius)             # 구 바닥점
    phi = pts[:, 2]
    J = jnp.stack([support.jac(env.mjx_model, d, pts[k], body_ids[k])[0].T
                   for k in range(len(body_ids))])
    return J, phi


def cimpc_substep(env, qpos, qvel, u, dt, ids, n_outer=10, n_bisect=8, tol=1e-6):
    """CI-MPC hard contact 한 스텝 (길이 dt).  반환 (qpos_next, qvel_next, aux)."""
    m = env.mjx_model
    mu = ids[4]
    d = mjx.make_data(m).replace(qpos=qpos, qvel=qvel, ctrl=u)
    d = smooth_forward(m, d)
    M = mjx.full_m(m, d)
    v_free = qvel + d.qacc_smooth * dt
    J, phi = contact_geometry(env, d, ids)
    fnum, _, nv = J.shape
    MinvJT = jnp.linalg.solve(M, J.reshape(3 * fnum, nv).T)           # (nv, 3fnum)
    MinvJT = MinvJT.T.reshape(fnum, 3, nv).transpose(0, 2, 1)         # (fnum, nv, 3)
    lam, contact, sliding = solve_contact_impulse(
        J, MinvJT, v_free, phi, dt, mu, n_outer, n_bisect, tol)
    v_next = v_free + jnp.einsum('kvb,kb->v', MinvJT, lam)
    q_next = mjx_forward._integrate_pos(m.jnt_type, qpos, v_next, dt)
    aux = dict(lam=lam, contact=contact, sliding=sliding, phi=phi)
    return q_next, v_next, aux


def cimpc_control_step(env, qpos, qvel, u, n_sub=None, ids=None, **kw):
    """control_dt 동안 n_sub 번 substep. diffsim.control_step 의 hard-contact 판."""
    n_sub = env.n_substeps if n_sub is None else int(n_sub)
    dt = env.control_dt / n_sub
    ids = cg.foot_ids(env) if ids is None else ids
    aux = None
    for _ in range(n_sub):
        qpos, qvel, aux = cimpc_substep(env, qpos, qvel, u, dt, ids, **kw)
    return qpos, qvel, aux


def make_rollout(env, n_sub=None, **kw):
    """(q0, dq0, action_traj) -> (qpos_traj, qvel_traj, aux_traj).

    diffsim.rollout 과 같은 형태 (스텝 i 의 출력 = x_{i+1}).
    """
    ids = cg.foot_ids(env)
    step = partial(cimpc_control_step, env, n_sub=n_sub, ids=ids, **kw)

    def rollout(q0, dq0, action_traj):
        def f(carry, u):
            q, v = carry
            q, v, aux = step(q, v, u)
            return (q, v), (q, v, aux)
        _, (qs, vs, aux) = lax.scan(f, (q0, dq0), action_traj)
        return qs, vs, aux
    return jax.jit(rollout)


def make_batched_rollout(env, n_sub=None, **kw):
    single = make_rollout(env, n_sub=n_sub, **kw)
    return jax.jit(jax.vmap(single, in_axes=(0, 0, 0)))
