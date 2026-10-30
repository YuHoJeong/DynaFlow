"""
DiffSim gradient (de Saxcé full-cone central-path relaxation) for DynaFlow -- "diffsim" backend.

참조
----
* 논문: "Differentiable Contact Dynamics for Legged Locomotion: Policy Learning and
  Contact-Implicit Trajectory Optimization" (T-RO 2026 투고, ~/Downloads/DiffSim.pdf)
  Sec. V (eq. 12-16, Alg. 1), App. A-B.
* C++ 원본: ~/workspace/DiffSim/aQuadrupedBPTT/lib/contactSolverGradient.cpp
  `GetGradientRelaxedV3` (mode_of_relaxation == 5).
* 2D 재현: ~/workspace/canonex_relaxedynaflow/contact_diffsim.py

CI-MPC (cimpc_grad.py) 와 같은 비대칭 구조: forward 는 hard contact 그대로, backward 만 바꾼다.
차이는 **무엇을 완화하느냐**:

    CI-MPC  : Signorini 만 완화 (lam_n sig_n = rho). 마찰 원뿔은 hard, separating 접촉은 제외,
              sliding 은 E_s 고정 (eq.14 근사). -> 접선 감도 없음, 모드 전이 방향 정보 없음.
    DiffSim : de Saxcé 보정된 원뿔 상보성 전체를 central path 를 따라 완화
                  lam~ o sig~ = rho e,  lam~ = (mu lam_n, lam_t),  sig~ = (sig_n/mu + |sig_t|, sig_t)
              모든 접촉(separating 포함)이 항상 3행씩 들어가는 3fnum x 3fnum 고정 크기 시스템.
              평가점 (lam*, sig*) 는 hard 해에서 시작한 Newton (Alg. 1) 으로 central path 위에 올린다.

한 substep (dt) 의 backward:

    1. hard 해 lam_h, Delassus (A, b):  sig = A lam + b,  b_n += phi/dt (+ delta, eq.16)
    2. Newton (Alg. 1):  (M_lam + M_sig A + beta |F| I) dlam = -F,   F = lam~ o sig~ - rho e
    3. 물리성 검사 (lam_n > 0, |lam_t| < mu lam_n, sig_n > sigma_n_min, 잔차 수렴) -> 실패 접촉은 dlam_k = 0
    4. 감도 eq.(15):  dlam/dxi = -(M_lam + M_sig A)^-1 M_sig (dA/dxi lam* + db/dxi)
    5. (f_x, f_u) 조립은 cimpc_grad 와 동일 (v_next = v_free + M^-1 J^T lam*, 위치 적분은 정확히 미분)

좌표 규약: lam, sig 는 접촉별 [x, y, z] = [T, T, N] (cimpc_forward 와 동일). 논문/App. B 는 [N, T] 순서지만
같은 블록의 재배열이고, C++ 원본도 [T, T, N] 이다.

원본과의 차이 (의도적, verify_diffsim.py 가 각각 확인):
  * 후진 line search 가 기본 ON (C++ 은 DIFFSIM_V3_LS_ENABLE 기본 OFF). 2D 재현에서 원뿔 경계 근처 clamping
    시작점이 spurious root 로 튀는 것이 확인돼 켜 두었다. n_linesearch=0 이면 C++ 과 같다.
  * spurious root: C++ 는 (옵션) 그 스텝의 f_x, f_u 전체를 0 으로 둔다 (BPTT chain 절단). 여기서는 2D 재현처럼
    해당 접촉의 dlam_k 만 0 (논문 VIII-D "sets the corresponding contact gradient to zero").
  * 수렴/절단 판정은 상대 기준 |F_k| <= cut_tol * max(rho, |lam~_k| |sig~_k|). 학습은 float32 라
    C++ 의 절대 1e-10 은 도달할 수 없다.
  * lam_eval: 논문 VI-A / C++ 는 df/dx 의 나머지 인자 (M^-1 J^T lam, dJ^T/dq lam) 도 lam* 에서 평가한다 ("relaxed", 기본).
    2D 재현은 lam_h 를 썼다 ("hard"). 옵션으로 둘 다 지원.
  * rho 단위: F 는 impulse x velocity. 논문의 rho ~ 1 은 45 kg, dt = 25 ms (lam_n ~ 2.8 N s) 기준.
    Go1 (12 kg) 의 substep lam_n 은 dt=0.005 에서 ~0.15, 0.01 에서 ~0.3 N s 이므로 같은 완화 정도는 rho ~ 0.05-0.1.
    clamping 접촉에서 sig_n* ~ rho / lam_n 이 "완화된 접촉 속도" 다.

알려진 구조적 사실 (구현 버그가 아님, verify_diffsim.py [A]/[E]):
  * hard forward 의 sliding 해 (Hwangbo bisection = 이방성 Mapp 에서의 eq.(7) 최소점) 는 de Saxcé 의
    등방 MDP (lam_t = -mu lam_n sig_t/|sig_t|) 를 정확히 만족하지 않는다. 그래서 sliding 접촉에서는
    |F(lam_h)| 이 rho=0 에서도 0 이 아니고, Newton 은 lam_h 에서 O(rho) 가 아니라 그 불일치만큼 떨어진 root 로 간다.
    원본 (RaiSim forward + de Saxcé backward) 도 같은 상황이다.
"""

import jax
import jax.numpy as jnp
from functools import partial

from mujoco import mjx
from mujoco.mjx._src import forward as mjx_forward

import cimpc_grad as cg
from cimpc_grad import (integrate_tangent, ambient_to_tangent, foot_ids, foot_height,
                        contact_jacobians, contact_jacobians_bottom, smooth_acc_and_mass,
                        hard_contact_impulse, foot_contact_impulse)

_EPS_T = 1e-12      # C++ kEpsT: |sig_t| 가 이보다 작으면 sig_t/|sig_t| := 0  (App. B 의 규약)


# ===========================================================================
# 1. Jordan 대수 블록 (접촉별 3x3, [T, T, N] 순서)
# ===========================================================================

def _row_norm(x):
    return jnp.sqrt(jnp.sum(x * x, axis=-1))


def _unit_rows(x, eps=_EPS_T):
    n = _row_norm(x)
    safe = jnp.where(n > eps, n, 1.0)
    return jnp.where((n > eps)[:, None], x / safe[:, None], 0.0), n


def central_path_residual(lam, sig, mu, rho):
    """F = lam~ o sig~ - rho e.  lam, sig: (fnum, 3) -> (fnum, 3).   (C++ compute_F_perfoot 와 동일)

        F_t = mu lam_n sig_t + y0 lam_t
        F_n = mu lam_n y0 + lam_t . sig_t - rho,     y0 = sig_n / mu + |sig_t|
    """
    lT, lN = lam[:, :2], lam[:, 2]
    sT, sN = sig[:, :2], sig[:, 2]
    y0 = sN / mu + _row_norm(sT)
    FT = mu * lN[:, None] * sT + y0[:, None] * lT
    FN = mu * lN * y0 + jnp.sum(lT * sT, axis=-1) - rho
    return jnp.concatenate([FT, FN[:, None]], axis=-1)


def arrow_blocks(lam, sig, mu):
    """M_lam = L(sig~) C,  M_sig = L(lam~) D(sig)   (App. B / C++ LyC, LxD).  (fnum,3,3) x 2."""
    fnum = lam.shape[0]
    lT, lN = lam[:, :2], lam[:, 2]
    sT, sN = sig[:, :2], sig[:, 2]
    unit, sTn = _unit_rows(sT)
    y0 = sN / mu + sTn
    I2 = jnp.eye(2, dtype=lam.dtype)
    Ml = jnp.zeros((fnum, 3, 3), lam.dtype)
    Ml = Ml.at[:, :2, :2].set(y0[:, None, None] * I2)
    Ml = Ml.at[:, :2, 2].set(mu * sT)
    Ml = Ml.at[:, 2, :2].set(sT)
    Ml = Ml.at[:, 2, 2].set(sN + mu * sTn)
    Ms = jnp.zeros((fnum, 3, 3), lam.dtype)
    Ms = Ms.at[:, :2, :2].set(lT[:, :, None] * unit[:, None, :] + mu * lN[:, None, None] * I2)
    Ms = Ms.at[:, :2, 2].set(lT / mu)
    Ms = Ms.at[:, 2, :2].set(mu * lN[:, None] * unit + lT)
    Ms = Ms.at[:, 2, 2].set(lN)
    return Ml, Ms


def block_diag(B):
    """(fnum,3,3) -> (3fnum, 3fnum)."""
    fnum = B.shape[0]
    return jnp.einsum('kij,kl->kilj', B, jnp.eye(fnum, dtype=B.dtype)).reshape(3 * fnum, 3 * fnum)


# ===========================================================================
# 2. Alg. 1: 정규화 Newton 으로 central path 위의 평가점
# ===========================================================================

def relaxed_point(lam0, A, b, mu, rho, beta=0.1, n_newton=10, n_linesearch=8, newton_tol=1e-8):
    """lam0 (fnum,3) 에서 시작, (A, b) 고정, sig = A lam + b 를 소거한 lam 만의 root finding.

        (M_lam + M_sig A + beta |F| I) dlam = -F

    반복수는 고정 (vmap/jit), |F| < newton_tol * max(rho sqrt(fnum), |F(lam0)|) 이면 갱신을 멈춘다.
    n_linesearch > 0 이면 |F| 를 줄이는 가장 큰 2^-k 스텝 (없으면 가장 작은 것).
    반환: lam* (fnum,3), sig* (fnum,3), F* (fnum,3), |F| 이력 (n_newton,), 사용한 반복수.
    """
    fnum = lam0.shape[0]
    n = 3 * fnum
    dtype = lam0.dtype
    eye = jnp.eye(n, dtype=dtype)
    rho = jnp.asarray(rho, dtype)

    def F_of(lam_flat):
        sig = (A @ lam_flat + b).reshape(fnum, 3)
        return central_path_residual(lam_flat.reshape(fnum, 3), sig, mu, rho).reshape(-1)

    def JF_of(lam_flat):
        sig = (A @ lam_flat + b).reshape(fnum, 3)
        Ml, Ms = arrow_blocks(lam_flat.reshape(fnum, 3), sig, mu)
        return block_diag(Ml) + block_diag(Ms) @ A

    lam_flat0 = lam0.reshape(-1)
    nF0 = jnp.linalg.norm(F_of(lam_flat0))
    tol = newton_tol * jnp.maximum(rho * jnp.sqrt(float(fnum)), nF0)
    alphas = 0.5 ** jnp.arange(n_linesearch + 1, dtype=dtype)

    def body(i, carry):
        lam, hist, n_it = carry
        F = F_of(lam)
        nF = jnp.linalg.norm(F)
        JF = JF_of(lam) + beta * nF * eye
        dlam = -jnp.linalg.solve(JF, F)
        if n_linesearch > 0:
            nF_try = jax.vmap(lambda a: jnp.linalg.norm(F_of(lam + a * dlam)))(alphas)
            dec = nF_try < nF
            k = jnp.where(jnp.any(dec), jnp.argmax(dec), n_linesearch)
            alpha = alphas[k]
        else:
            alpha = jnp.ones((), dtype)
        conv = nF < tol
        lam = jnp.where(conv, lam, lam + alpha * dlam)
        return lam, hist.at[i].set(nF), n_it + jnp.where(conv, 0, 1)

    lam, hist, n_it = jax.lax.fori_loop(
        0, n_newton, body, (lam_flat0, jnp.zeros(n_newton, dtype), jnp.int32(0)))
    sig = (A @ lam + b).reshape(fnum, 3)
    lam = lam.reshape(fnum, 3)
    return lam, sig, central_path_residual(lam, sig, mu, rho), hist, n_it


def physical_mask(lam, sig, F, mu, rho, cut_tol=1e-3, sigma_n_min=-0.1, cone_tol=1e-6):
    """접촉별 bool: 수렴한 물리적 root 인가 (논문 VIII-D, C++ DIFFSIM_V3_SPURIOUS_CUT 의 조건 + 잔차)."""
    lT, lN = lam[:, :2], lam[:, 2]
    sT, sN = sig[:, :2], sig[:, 2]
    sTn = _row_norm(sT)
    lam_tilde = jnp.sqrt((mu * lN) ** 2 + jnp.sum(lT * lT, -1))
    sig_tilde = jnp.sqrt((sN / mu + sTn) ** 2 + sTn ** 2)
    scale = jnp.maximum(rho, lam_tilde * sig_tilde)
    finite = jnp.all(jnp.isfinite(lam), -1) & jnp.all(jnp.isfinite(sig), -1) & jnp.all(jnp.isfinite(F), -1)
    return ((lN > 0.0) & (_row_norm(lT) - mu * lN < cone_tol) & (sN > sigma_n_min)
            & (_row_norm(F) <= cut_tol * scale) & finite)


# ===========================================================================
# 3. Delassus (A, b) 와 한 substep 의 (f_x, f_u)
# ===========================================================================

def make_delassus(env, qpos, dt, ids, delta=0.0, forward="cimpc"):
    """A_and_b(theta, v, u) -> (A, b).   sig = A lam + b 가 다음 접촉 속도 (normal 은 drift 보정 phi/dt 포함).
    delta 는 상수로 더해지므로 db 에 d(delta) 항이 생기지 않는다 (논문 V-C)."""
    site_ids, _, body_ids, r, _ = ids
    jac_fn = contact_jacobians_bottom if forward == "cimpc" else contact_jacobians
    nv = env.mj_model.nv
    n = 3 * len(body_ids)
    dt_arr = jnp.asarray(dt, dtype=qpos.dtype)

    def A_and_b(theta, vv, uu):
        q = integrate_tangent(qpos, theta)
        J = jac_fn(env, q, site_ids, r, body_ids).reshape(n, nv)
        acc_s, M = smooth_acc_and_mass(env, q, vv, uu)
        A = J @ jnp.linalg.solve(M, J.T)
        b = J @ (vv + acc_s * dt_arr)
        phi = foot_height(env, q, site_ids, r)
        b = b.at[2::3].add(phi / dt_arr + delta)
        return A, b

    return A_and_b, jac_fn


def _assemble(env, qpos, qvel, u, dt, ids, jac_fn, lam_full, dlam_dq, dlam_dv, dlam_du):
    """dlambda/dxi 와 적용 impulse lam_full (n,) 로 한 스텝의 (f_x, f_u). cimpc_grad 와 동일."""
    site_ids, _, body_ids, r, _ = ids
    nv = env.mj_model.nv
    n = lam_full.shape[0]
    dt_arr = jnp.asarray(dt, dtype=qpos.dtype)
    zero_q = jnp.zeros(nv, dtype=qpos.dtype)

    def v_next_smooth(theta, vv, uu):
        q = integrate_tangent(qpos, theta)
        J = jac_fn(env, q, site_ids, r, body_ids).reshape(n, nv)
        acc_s, M = smooth_acc_and_mass(env, q, vv, uu)
        return vv + acc_s * dt_arr + jnp.linalg.solve(M, J.T @ lam_full)

    dvn_dq, dvn_dv, dvn_du = jax.jacfwd(v_next_smooth, argnums=(0, 1, 2))(zero_q, qvel, u)
    _, M0 = smooth_acc_and_mass(env, qpos, qvel, u)
    J0 = jac_fn(env, qpos, site_ids, r, body_ids).reshape(n, nv)
    Minv_JT0 = jnp.linalg.solve(M0, J0.T)
    dvn_dq = dvn_dq + Minv_JT0 @ dlam_dq
    dvn_dv = dvn_dv + Minv_JT0 @ dlam_dv
    dvn_du = dvn_du + Minv_JT0 @ dlam_du

    v_next0 = v_next_smooth(zero_q, qvel, u)
    jnt_type = env.mjx_model.jnt_type

    def pos_step(theta, vn):
        return mjx_forward._integrate_pos(jnt_type, integrate_tangent(qpos, theta), vn, dt_arr)

    Dq_theta, Dq_v = jax.jacfwd(pos_step, argnums=(0, 1))(zero_q, v_next0)
    Ep_next = ambient_to_tangent(pos_step(zero_q, v_next0))
    Pq = Ep_next @ Dq_theta
    Pv = Ep_next @ Dq_v
    f_x = jnp.block([[Pq + Pv @ dvn_dq, Pv @ dvn_dv],
                     [          dvn_dq,      dvn_dv]])
    f_u = jnp.concatenate([Pv @ dvn_du, dvn_du], axis=0)
    return f_x, f_u


def _diffsim_step_jacobian(env, qpos, qvel, warmstart, u, dt, ids, rho, delta=0.0, beta=0.1,
                           n_newton=10, n_linesearch=8, newton_tol=1e-8, cut_tol=1e-3,
                           sigma_n_min=-0.1, cone_tol=1e-6, cut_spurious=True,
                           lam_eval="relaxed", forward="cimpc", return_info=False):
    """semi-implicit Euler 한 스텝 (길이 dt) 의 (f_x, f_u), tangent 좌표. 논문 eq.(15) + Alg. 1.

    forward="cimpc": lam_h 는 hard solver (cimpc_forward) 해, 접촉점은 발 바닥점.
    forward="mjx"  : lam_h 는 MJX soft contact 의 발별 impulse (Newton 의 warm start 로만 쓴다),
                     접촉점은 침투 중점 (cimpc_grad 의 mjx 경로와 동일).
    """
    site_ids, geom_ids, body_ids, r, mu = ids
    nv = env.mj_model.nv
    fnum = len(body_ids)
    n = 3 * fnum

    if forward == "cimpc":
        lam_h, contact, sliding = hard_contact_impulse(env, qpos, qvel, u, dt, ids)
    elif forward == "mjx":
        lam_h, _ = foot_contact_impulse(env, qpos, qvel, u, warmstart,
                                        site_ids, geom_ids, body_ids, r, dt)
        contact = lam_h[:, 2] > 0
        sliding = jnp.zeros(fnum, dtype=bool)
    else:
        raise ValueError(f"forward={forward!r}")

    A_and_b, jac_fn = make_delassus(env, qpos, dt, ids, delta, forward)
    zero_q = jnp.zeros(nv, dtype=qpos.dtype)
    A, b = A_and_b(zero_q, qvel, u)

    lam_s, sig_s, F, hist, n_it = relaxed_point(lam_h, A, b, mu, rho, beta, n_newton,
                                                n_linesearch, newton_tol)
    ok = physical_mask(lam_s, sig_s, F, mu, rho, cut_tol, sigma_n_min, cone_tol)
    if not cut_spurious:
        ok = jnp.ones(fnum, dtype=bool)
    # 절단된 접촉은 hard 해로 되돌린다 (NaN 전파 방지 + 적용 impulse 는 forward 와 같게)
    lam_use = jnp.where(ok[:, None], lam_s, lam_h)
    sig_use = jnp.where(ok[:, None], sig_s, (A @ lam_h.reshape(-1) + b).reshape(fnum, 3))

    Ml, Ms = arrow_blocks(lam_use, sig_use, mu)
    Ms_bd = block_diag(Ms)
    S = block_diag(Ml) + Ms_bd @ A
    ok3 = jnp.repeat(ok, 3)
    S = jnp.where(ok3[:, None], S, jnp.eye(n, dtype=qpos.dtype))

    (dA_dq, dA_dv, dA_du), (db_dq, db_dv, db_du) = jax.jacfwd(
        A_and_b, argnums=(0, 1, 2))(zero_q, qvel, u)
    lam_vec = lam_use.reshape(-1)

    def solve_dlam(dA, db):
        rhs = Ms_bd @ (jnp.einsum('ijk,j->ik', dA, lam_vec) + db)
        rhs = jnp.where(ok3[:, None], rhs, 0.0)
        return -jnp.linalg.solve(S, rhs)

    dlam_dq, dlam_dv, dlam_du = solve_dlam(dA_dq, db_dq), solve_dlam(dA_dv, db_dv), solve_dlam(dA_du, db_du)

    if lam_eval == "relaxed":
        lam_full = lam_vec
    elif lam_eval == "hard":
        lam_full = lam_h.reshape(-1)
    else:
        raise ValueError(f"lam_eval={lam_eval!r}")

    f_x, f_u = _assemble(env, qpos, qvel, u, dt, ids, jac_fn, lam_full, dlam_dq, dlam_dv, dlam_du)
    if return_info:
        info = dict(lam_h=lam_h, contact=contact, sliding=sliding, lam=lam_s, sig=sig_s, F=F,
                    hist=hist, n_it=n_it, ok=ok, A=A, b=b, S=S, Ml=Ml, Ms=Ms,
                    dlam_dq=dlam_dq, dlam_dv=dlam_dv, dlam_du=dlam_du, lam_full=lam_full)
        return f_x, f_u, info
    return f_x, f_u


def relaxed_substep(env, qpos, qvel, u, dt, ids, rho, delta=0.0, beta=0.1, n_newton=10,
                    n_linesearch=8, newton_tol=1e-8):
    """검증용 '완화 동역학' 한 스텝: hard 해 -> Newton -> lam* 를 **적용** 한 다음 상태.
    diffsim 의 (f_x, f_u) (lam_eval="relaxed") 는 이 사상의 정확한 미분이어야 한다 (verify_diffsim.py [D])."""
    site_ids, _, body_ids, r, mu = ids
    nv = env.mj_model.nv
    n = 3 * len(body_ids)
    lam_h, contact, sliding = hard_contact_impulse(env, qpos, qvel, u, dt, ids)
    A_and_b, jac_fn = make_delassus(env, qpos, dt, ids, delta, "cimpc")
    A, b = A_and_b(jnp.zeros(nv, dtype=qpos.dtype), qvel, u)
    lam_s, sig_s, F, _, _ = relaxed_point(lam_h, A, b, mu, rho, beta, n_newton, n_linesearch, newton_tol)
    J = jac_fn(env, qpos, site_ids, r, body_ids).reshape(n, nv)
    acc_s, M = smooth_acc_and_mass(env, qpos, qvel, u)
    v_next = qvel + acc_s * dt + jnp.linalg.solve(M, J.T @ lam_s.reshape(-1))
    q_next = mjx_forward._integrate_pos(env.mjx_model.jnt_type, qpos, v_next, dt)
    return q_next, v_next, dict(lam=lam_s, lam_h=lam_h, sig=sig_s, F=F, contact=contact, sliding=sliding)


# ===========================================================================
# 4. control step 단위 (substep 연쇄) -- cimpc_grad.make_rollout 의 jacobian_fn 규약
# ===========================================================================

def diffsim_jacobian(env, qpos, qvel, warmstart, u, qpos_next,
                     rho=1e-2, delta=0.0, beta=0.1, n_newton=10, n_linesearch=8,
                     newton_tol=1e-8, cut_tol=1e-3, sigma_n_min=-0.1, cone_tol=1e-6,
                     cut_spurious=True, lam_eval="relaxed", n_sub=None, forward="cimpc"):
    """DiffSim gradient. cimpc_grad.cimpc_jacobian 과 같은 substep 연쇄.

        f_x = f_x^(n-1) ... f_x^(0),   f_u = sum_j (뒤쪽 f_x 들) f_u^(j)

    forward="cimpc" 면 중간 substep 상태를 hard-contact 동역학 (cimpc_forward) 으로 재생성한다
    (n_sub 는 forward 의 n_sub 와 같아야 한다). forward="mjx" 면 mjx.step 으로.
    """
    n_sub = env.n_substeps if n_sub is None else int(n_sub)
    dt = env.control_dt / n_sub
    nv, nu = env.mj_model.nv, env.mj_model.nu
    ids = foot_ids(env)
    kw = dict(rho=rho, delta=delta, beta=beta, n_newton=n_newton, n_linesearch=n_linesearch,
              newton_tol=newton_tol, cut_tol=cut_tol, sigma_n_min=sigma_n_min, cone_tol=cone_tol,
              cut_spurious=cut_spurious, lam_eval=lam_eval, forward=forward)

    q, v, ws = qpos, qvel, warmstart
    f_x = jnp.eye(2 * nv, dtype=qpos.dtype)
    f_u = jnp.zeros((2 * nv, nu), dtype=qpos.dtype)
    for j in range(n_sub):
        fx, fu = _diffsim_step_jacobian(env, q, v, ws, u, dt, ids, **kw)
        f_x = fx @ f_x
        f_u = fx @ f_u + fu
        if j < n_sub - 1:
            if forward == "cimpc":
                import cimpc_forward as cf
                q, v, _ = cf.cimpc_substep(env, q, v, u, dt, ids)
            else:
                d = mjx.make_data(env.mjx_model).replace(qpos=q, qvel=v, ctrl=u, qacc_warmstart=ws)
                d = mjx.step(env.mjx_model, d)
                q, v, ws = d.qpos, d.qvel, d.qacc_warmstart
    return f_x, f_u


def make_rollout(env, forward="cimpc", fwd_n_sub=None, **kw):
    """cimpc_grad.make_rollout 에 diffsim_jacobian 을 꽂은 것. kw 는 diffsim_jacobian 의 인자."""
    n_sub = kw.pop("n_sub", fwd_n_sub)
    jac = partial(diffsim_jacobian, forward=forward, n_sub=n_sub, **kw)
    return cg.make_rollout(env, jac, forward=forward, fwd_n_sub=fwd_n_sub)


def make_batched_rollout(env, chunk=None, forward="cimpc", fwd_n_sub=None, **kw):
    n_sub = kw.pop("n_sub", fwd_n_sub)
    jac = partial(diffsim_jacobian, forward=forward, n_sub=n_sub, **kw)
    return cg.make_batched_rollout(env, jac, chunk=chunk, forward=forward, fwd_n_sub=fwd_n_sub)
