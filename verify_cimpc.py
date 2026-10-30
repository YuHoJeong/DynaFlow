"""cimpc_forward + cimpc_grad(forward="cimpc") 독립 재검증 (x64, CPU).

    JAX_PLATFORMS=cpu python verify_cimpc.py

기존 verify_forward.py / verify_grad.py 와 겹치지 않는 것만 본다.

 [A] 물리 조건: solver 해가 Signorini(v_n + phi/dt = 0), Coulomb(||lam_t|| <= mu lam_n),
     최대 소산(sliding 마찰이 slip 반대) 을 만족하는가. 접촉 모드 전부 (flight / clamping /
     sliding / partial / mixed / landing).
 [B] rho=0 analytic (f_x, f_u) vs x64 중앙 유한차분 **전체 행렬** (tangent 좌표),
     그리고 plain autodiff(= 수렴한 solver 의 정확한 미분). n_sub=1, 2.
     clamping 은 FD 와 일치해야 하고, sliding 은 논문 eq.(14) 의 "E_s 고정" 근사 때문에
     정확한 미분과 차이가 있어야 한다 -> 그 크기를 잰다.
 [C] rho>0: eq.(19) 를 인덱스 기반으로 따로 구현해 cimpc_jacobian 과 대조.
     (cimpc_grad 는 가중치 w/alpha 마스킹으로 축소 시스템을 만들고, 여기서는 행을 직접
     고른다. J 도 support.jac 대신 발 위치의 jacfwd 로 얻는다.)

solver 는 반복수를 올리고 tol 을 낮춰 **수렴한** 해를 쓴다 (기본 8회 bisection 은
각도 1e-2 rad 정밀도라 FD 기준이 못 된다). 학습용 기본값의 효과는 [D] 에서 따로 잰다.
"""
import os, sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from mujoco import mjx
from mujoco.mjx._src import forward as mjx_forward

import diffsim
import cimpc_grad as cg
import cimpc_forward as cf
cf.UNROLL = False          # 반복수를 올렸으므로 scan 을 펼치지 않는다 (컴파일 메모리)

NQ, NV, NU, FNUM = 19, 18, 12, 4
STAND_Z = 0.2865
RED, GRN, YEL, RST = "\033[91m", "\033[92m", "\033[93m", "\033[0m"
results = []


def check(name, err, tol):
    ok = bool(np.isfinite(err)) and err <= tol
    results.append((name, ok))
    print(f"  [{GRN+'PASS'+RST if ok else RED+'FAIL'+RST}] {name:<58s} {err:.3e}  (tol {tol:.0e})")
    return ok


def set_solver(n_outer, n_bisect, tol):
    """cimpc_forward 의 solver 기본값을 바꾼다 (cimpc_grad.hard_contact_impulse 도 이걸 쓴다)."""
    cf.solve_contact_impulse.__defaults__ = (n_outer, n_bisect, tol, None)
    cf.cimpc_substep.__defaults__ = (n_outer, n_bisect, tol)


def stand_state(env, height=STAND_Z, pitch=0.0, vel=None, joint_noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    q = np.zeros(NQ); q[2] = height
    q[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
    q[7:] = np.asarray(env.nominal_ptarg) + joint_noise * rng.standard_normal(12)
    v = np.zeros(NV)
    if vel is not None:
        v[:len(vel)] = vel
    return jnp.array(q), jnp.array(v)


def make_regimes(env, ids):
    """(q, v, u) 와 기대 모드. 서기 상태는 hard forward 로 5스텝 안정화한 것."""
    u_nom = jnp.array(env.nominal_ptarg, dtype=jnp.float64)
    q0, v0 = stand_state(env)
    q_s, v_s = q0, v0
    for _ in range(5):
        q_s, v_s, _ = cf.cimpc_control_step(env, q_s, v_s, u_nom, n_sub=2, ids=ids)
    rng = np.random.default_rng(11)
    du = lambda: 0.02 * jnp.array(rng.standard_normal(12))
    R = {}
    R["flight"]   = (q0.at[2].add(0.3), v0, u_nom + du())
    R["clamping"] = (q_s, v_s, u_nom + du())
    R["sliding"]  = (q_s, v_s.at[0].set(1.0), u_nom + du())            # 4발 sliding (C++ 'sliding' 덤프와 같은 상황)
    R["partial"]  = (q_s, v_s.at[0].set(0.3).at[1].set(0.1), u_nom + du())
    q_m, v_m = stand_state(env, pitch=0.1)
    R["mixed"]    = (q_m, v_m, u_nom + du())                          # 뒷발만 접촉
    q_l, v_l = stand_state(env, height=STAND_Z + 0.004, vel=[0, 0, -1.0])
    R["landing"]  = (q_l, v_l, u_nom + du())                          # phi>0 이지만 phi_next<0 -> 접촉
    return R


# ---------------------------------------------------------------------------
# 공통: tangent 좌표의 (f_x, f_u) 를 FD / autodiff 로
# ---------------------------------------------------------------------------
def make_step(env, ids, n_sub):
    def step(q, v, u):
        qn, vn, aux = cf.cimpc_control_step(env, q, v, u, n_sub=n_sub, ids=ids)
        return qn, vn, aux
    return jax.jit(step)


def fd_jacobian(step, q, v, u, h=1e-6):
    """중앙 FD. 섭동은 tangent 좌표, 출력은 E^+(q_next) 로 tangent 로 당긴다 (cimpc_grad 규약)."""
    qn0, vn0, aux0 = step(q, v, u)
    Ep = np.asarray(cg.ambient_to_tangent(qn0))
    mode0 = (np.asarray(aux0["contact"]), np.asarray(aux0["sliding"]))
    cols = []
    mode_ok = True
    def col(fp, fm):
        qp, vp, ap = fp; qm, vm, am = fm
        nonlocal mode_ok
        for a in (ap, am):
            mode_ok &= bool(np.all(np.asarray(a["contact"]) == mode0[0]) and np.all(np.asarray(a["sliding"]) == mode0[1]))
        dq = Ep @ (np.asarray(qp) - np.asarray(qm)) / (2 * h)
        dv = (np.asarray(vp) - np.asarray(vm)) / (2 * h)
        return np.concatenate([dq, dv])
    for j in range(NV):
        e = jnp.zeros(NV).at[j].set(h)
        cols.append(col(step(cg.integrate_tangent(q, e), v, u), step(cg.integrate_tangent(q, -e), v, u)))
    for j in range(NV):
        e = jnp.zeros(NV).at[j].set(h)
        cols.append(col(step(q, v + e, u), step(q, v - e, u)))
    for j in range(NU):
        e = jnp.zeros(NU).at[j].set(h)
        cols.append(col(step(q, v, u + e), step(q, v, u - e)))
    Jm = np.stack(cols, axis=1)
    return Jm[:, :2 * NV], Jm[:, 2 * NV:], mode_ok, mode0


def autodiff_jacobian(step, q, v, u):
    """수렴한 고정 반복수 solver 의 정확한 미분 (plain jacfwd)."""
    def f(theta, vv, uu):
        qn, vn, _ = step(cg.integrate_tangent(q, theta), vv, uu)
        return qn, vn
    qn0, vn0, _ = step(q, v, u)
    Ep = cg.ambient_to_tangent(qn0)
    (Jqq, Jqv, Jqu), (Jvq, Jvv, Jvu) = jax.jacfwd(f, argnums=(0, 1, 2))(jnp.zeros(NV), v, u)
    f_x = jnp.block([[Ep @ Jqq, Ep @ Jqv], [Jvq, Jvv]])
    f_u = jnp.concatenate([Ep @ Jqu, Jvu], axis=0)
    return np.asarray(f_x), np.asarray(f_u)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ---------------------------------------------------------------------------
# [C]/[E] 공통: 기하 헬퍼와 (dlambda/dxi -> f_x, f_u) 조립
# ---------------------------------------------------------------------------
def _geom(env, ids, q):
    site_ids, geom_ids, body_ids, r, mu = ids
    m = env.mjx_model

    def foot_bottom(qq):                                   # (4,3) 구 바닥점
        d = mjx.make_data(m).replace(qpos=qq)
        d = mjx.kinematics(m, d)
        return d.site_xpos[site_ids].at[:, 2].add(-r)

    def J_at(theta):                                       # (4,3,nv) 접촉점 Jacobian
        """현재 최저점에 있는 **물질점** 의 Jacobian. 구 바닥점은 body 에 고정되지 않으므로
        site - r*z 를 그냥 미분하면 접선 행에 omega x (-r z) 가 빠진다 (실측 0.023 = r 차이).
        forward (support.jac) 와 같은 규약: 현재 바닥점을 body 좌표로 고정하고 그 점의 속도를 본다."""
        qq = cg.integrate_tangent(q, theta)
        d0 = mjx.kinematics(m, mjx.make_data(m).replace(qpos=qq))
        pts = d0.site_xpos[site_ids].at[:, 2].add(-r)                       # (4,3) 현재 바닥점 (world)
        bid = jnp.array(body_ids)
        p_local = jnp.einsum('kji,kj->ki', d0.xmat[bid], pts - d0.xpos[bid])  # R^T (p - x)

        def pts_of(delta):
            dd = mjx.kinematics(m, mjx.make_data(m).replace(qpos=cg.integrate_tangent(qq, delta)))
            return dd.xpos[bid] + jnp.einsum('kij,kj->ki', dd.xmat[bid], p_local)
        return jax.jacfwd(pts_of)(jnp.zeros(NV))

    def acc_M(qq, vv, uu):
        return cg.smooth_acc_and_mass(env, qq, vv, uu)
    return foot_bottom, J_at, acc_M


def _assemble(env, ids, q, v, u, dt, lam_full, dlam_q, dlam_v, dlam_u):
    """dlambda/dxi (4,3,·) 와 고정 lambda 로 한 스텝의 (f_x, f_u). tangent 좌표."""
    m = env.mjx_model
    foot_bottom, J_at, acc_M = _geom(env, ids, q)
    z = jnp.zeros(NV)

    def vnext(theta, vv, uu):
        qq = cg.integrate_tangent(q, theta)
        J = J_at(theta)
        acc, M = acc_M(qq, vv, uu)
        return vv + acc * dt + jnp.linalg.solve(M, jnp.einsum('kcv,kc->v', J, lam_full))

    dvn_q, dvn_v, dvn_u = jax.jacfwd(vnext, argnums=(0, 1, 2))(z, v, u)
    J0 = J_at(z).reshape(12, NV)
    _, M0 = acc_M(q, v, u)
    MinvJT = jnp.linalg.solve(M0, J0.T)                    # (nv, 12)
    dvn_q = dvn_q + MinvJT @ dlam_q.reshape(12, -1)
    dvn_v = dvn_v + MinvJT @ dlam_v.reshape(12, -1)
    dvn_u = dvn_u + MinvJT @ dlam_u.reshape(12, -1)

    vn0 = vnext(z, v, u)
    def pos_step(theta, vn):
        return mjx_forward._integrate_pos(m.jnt_type, cg.integrate_tangent(q, theta), vn, dt)
    Dq_th, Dq_v = jax.jacfwd(pos_step, argnums=(0, 1))(z, vn0)
    Ep = cg.ambient_to_tangent(pos_step(z, vn0))
    Pq, Pv = Ep @ Dq_th, Ep @ Dq_v
    f_x = jnp.block([[Pq + Pv @ dvn_q, Pv @ dvn_v], [dvn_q, dvn_v]])
    f_u = jnp.concatenate([Pv @ dvn_u, dvn_u], axis=0)
    return np.asarray(f_x), np.asarray(f_u)


# ---------------------------------------------------------------------------
# [C] eq.(19) 독립 구현 (한 스텝, dt): 행을 직접 고르는 축소 시스템, E_s 고정
# ---------------------------------------------------------------------------
def indep_step_jacobian(env, ids, q, v, u, dt, rho):
    foot_bottom, J_at, acc_M = _geom(env, ids, q)
    lam, contact, sliding = cg.hard_contact_impulse(env, q, v, u, dt, ids)
    lam, contact, sliding = np.asarray(lam), np.asarray(contact), np.asarray(sliding)
    E = np.concatenate([lam[:, :2] / np.where(contact[:, None], lam[:, 2:3], 1.0), np.ones((FNUM, 1))], axis=1)

    rows = []                                              # (k, 성분)
    for k in range(FNUM):
        if not contact[k]:
            continue
        rows += [(k, 2)] if sliding[k] else [(k, 0), (k, 1), (k, 2)]
    nr = len(rows)
    lam_red = jnp.array([lam[k, c] for k, c in rows])
    is_n = jnp.array([c == 2 for k, c in rows])
    E_j = jnp.array(E)

    def Ab(theta, vv, uu):
        qq = cg.integrate_tangent(q, theta)
        J = J_at(theta)
        acc, M = acc_M(qq, vv, uu)
        phi = foot_bottom(qq)[:, 2]
        Jl = jnp.stack([J[k, c] for k, c in rows]) if nr else jnp.zeros((0, NV))
        Jr = jnp.stack([(E_j[k] @ J[k]) if (sliding[k] and c == 2) else J[k, c] for k, c in rows]) if nr else jnp.zeros((0, NV))
        A = Jl @ jnp.linalg.solve(M, Jr.T)
        b = Jl @ (vv + acc * dt) + jnp.array([phi[k] / dt if c == 2 else 0.0 for k, c in rows]) if nr else jnp.zeros(0)
        return A, b

    z = jnp.zeros(NV)
    A0, b0 = Ab(z, v, u)
    resid = float(jnp.abs(A0 @ lam_red + b0).max()) if nr else 0.0

    if nr:
        (dA_dq, dA_dv, dA_du), (db_dq, db_dv, db_du) = jax.jacfwd(Ab, argnums=(0, 1, 2))(z, v, u)
        D = jnp.diag(jnp.where(is_n, 1.0 / lam_red ** 2, 0.0))
        S = A0 + rho * D
        def dlam(dA, db):
            return -jnp.linalg.solve(S, jnp.einsum('ijk,j->ik', dA, lam_red) + db)   # (nr, n)
        dl = [dlam(dA_dq, db_dq), dlam(dA_dv, db_dv), dlam(dA_du, db_du)]
    else:
        dl = [jnp.zeros((0, NV)), jnp.zeros((0, NV)), jnp.zeros((0, NU))]

    def full(dlr):                                         # (nr, n) -> (4,3,n), eq.(14)
        out = jnp.zeros((FNUM, 3, dlr.shape[1]))
        for i, (k, c) in enumerate(rows):
            if sliding[k]:
                out = out.at[k].set(E_j[k][:, None] * dlr[i][None, :])
            else:
                out = out.at[k, c].set(dlr[i])
        return out
    dlam_q, dlam_v, dlam_u = (full(d) for d in dl)
    lam_full = jnp.array(lam * contact[:, None])
    f_x, f_u = _assemble(env, ids, q, v, u, dt, lam_full, dlam_q, dlam_v, dlam_u)
    return f_x, f_u, dict(lam=lam, contact=contact, sliding=sliding, resid=resid)


# ---------------------------------------------------------------------------
# [E] 정확한 implicit 미분: sliding 의 theta 도 미분한다 (eq.14 의 'E_s 고정' 근사 없음)
#     접촉 k 의 세 방정식:
#       clamping   : J_k v_next + [0,0,phi_k/dt] = 0
#       sliding    : [J_k v_next]_n + phi_k/dt = 0,   D_k = (J_k v_next).(P_k[2] x grad h2(lam_k)) = 0,
#                    h2(lam_k) = |lam_t|^2 - mu^2 lam_n^2 = 0        (solver 의 bisection 이 푸는 조건 그대로)
#       separating : lam_k = 0
#     dlam/dxi = -(dR/dlam)^-1 dR/dxi.  FD 와 일치해야 한다 (= 진짜 정답).
# ---------------------------------------------------------------------------
def exact_step_jacobian(env, ids, q, v, u, dt, fix_E=False):
    """fix_E=True: sliding 의 (D=0, h2=0) 두 행을 lam_t - E lam_n = 0 (E 는 해에서 고정) 으로 바꾼다.
    이건 논문 eq.(14) 의 근사와 정확히 같은 가정이므로 cimpc_jacobian 과 일치해야 한다."""
    mu = ids[4]
    foot_bottom, J_at, acc_M = _geom(env, ids, q)
    lam, contact, sliding = cg.hard_contact_impulse(env, q, v, u, dt, ids)
    lam = jnp.asarray(lam); contact = np.asarray(contact); sliding = np.asarray(sliding)
    coeff = jnp.array([2.0, 2.0, -2.0 * mu ** 2])
    E_fix = jax.lax.stop_gradient(lam[:, :2] / jnp.where(contact[:, None], lam[:, 2:3], 1.0))

    def R(lam_flat, theta, vv, uu):
        L = lam_flat.reshape(FNUM, 3)
        qq = cg.integrate_tangent(q, theta)
        J = J_at(theta)
        acc, M = acc_M(qq, vv, uu)
        phi = foot_bottom(qq)[:, 2]
        MinvJT = jnp.linalg.solve(M, J.reshape(12, NV).T).T.reshape(FNUM, 3, NV).transpose(0, 2, 1)
        vn = vv + acc * dt + jnp.einsum('kvb,kb->v', MinvJT, L)
        vc = jnp.einsum('kcv,v->kc', J, vn)                             # (4,3) 다음 접촉 속도
        P = jnp.einsum('kav,kvb->kab', J, MinvJT)
        out = []
        for k in range(FNUM):
            if not contact[k]:
                out.append(L[k])
            elif not sliding[k]:
                out.append(vc[k] + jnp.array([0.0, 0.0, phi[k] / dt]))
            elif fix_E:
                out.append(jnp.concatenate([jnp.array([vc[k, 2] + phi[k] / dt]), L[k, :2] - E_fix[k] * L[k, 2]]))
            else:
                zeta = jnp.cross(P[k, 2], coeff * L[k])
                out.append(jnp.stack([vc[k, 2] + phi[k] / dt,
                                      vc[k] @ zeta,
                                      L[k, 0] ** 2 + L[k, 1] ** 2 - mu ** 2 * L[k, 2] ** 2]))
        return jnp.concatenate(out)

    z = jnp.zeros(NV)
    lf = lam.reshape(-1)
    resid = float(jnp.abs(R(lf, z, v, u)).max())
    dR_dlam = jax.jacfwd(R, argnums=0)(lf, z, v, u)
    dR_q, dR_v, dR_u = jax.jacfwd(R, argnums=(1, 2, 3))(lf, z, v, u)
    sol = lambda B: (-jnp.linalg.solve(dR_dlam, B)).reshape(FNUM, 3, -1)
    f_x, f_u = _assemble(env, ids, q, v, u, dt, lam, sol(dR_q), sol(dR_v), sol(dR_u))
    return f_x, f_u, dict(resid=resid, contact=contact, sliding=sliding)


# ---------------------------------------------------------------------------
def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "ABCDE"
    env = diffsim.Env()
    ids = cg.foot_ids(env)
    mu = ids[4]
    ws0 = jnp.zeros(NV)
    set_solver(40, 50, 1e-11)
    R = make_regimes(env, ids)
    dt = env.control_dt / 2
    print(f"jax {jax.__version__} x64={jax.config.jax_enable_x64} devices={jax.devices()}  mu={mu}")

    # ---------------- [A] 물리 조건 ----------------
    if "A" in only:
        print(f"\n{YEL}[A] hard-contact solver 해의 물리 조건 (dt=0.01){RST}")

        for name, (q, v, u) in R.items():
            qn, vn, aux = cf.cimpc_substep(env, q, v, u, dt, ids)
            lam, con, sl, phi = (np.asarray(aux[k]) for k in ("lam", "contact", "sliding", "phi"))
            d = mjx.make_data(env.mjx_model).replace(qpos=q, qvel=v, ctrl=u); d = cf.smooth_forward(env.mjx_model, d)
            J, _ = cf.contact_geometry(env, d, ids)
            vc = np.asarray(jnp.einsum('kcv,v->kc', J, vn))                 # 다음 접촉 속도
            print(f"  {name:<9s} contact={con.astype(int)} sliding={sl.astype(int)}  phi={np.array2string(phi, precision=4)}  "
                  f"lam_n={np.array2string(lam[:, 2], precision=4)}")
            if con.any():
                sig = np.abs(vc[con, 2] + phi[con] / dt).max()
                check(f"{name}: Signorini  v_n + phi/dt = 0 (active)", sig, 1e-9)
                cone = (np.linalg.norm(lam[:, :2], axis=1) - mu * lam[:, 2]).max()
                check(f"{name}: Coulomb    ||lam_t|| <= mu lam_n", max(cone, 0.0), 1e-9)
                check(f"{name}: lam_n >= 0 (active)", max(-lam[con, 2].min(), 0.0), 0.0)
            if sl.any():
                # eq.(7): 원뿔 표면 + Signorini 를 만족하는 1-매개 곡선 lam(th) 위에서 v^T Mapp v 가 최소인가 (theta 스캔)
                M = mjx.full_m(env.mjx_model, d)
                MinvJT = jnp.linalg.solve(M, J.reshape(12, NV).T).T.reshape(FNUM, 3, NV).transpose(0, 2, 1)
                P = np.asarray(jnp.einsum('kav,kvb->kab', J, MinvJT))
                v_free = np.asarray(v + d.qacc_smooth * dt)
                gaps, coss = [], []
                for k in range(FNUM):
                    if not sl[k]:
                        continue
                    dv = np.asarray(jnp.einsum('jvb,jb->v', MinvJT, lam)) - np.asarray(MinvJT[k]) @ lam[k]
                    ci = np.asarray(J[k]) @ (v_free + dv); ci_n = ci[2] + phi[k] / dt
                    Pinv = np.linalg.inv(P[k])
                    def obj(l): vv = P[k] @ l + ci; return vv @ Pinv @ vv
                    th = np.linspace(-np.pi, np.pi, 20001)
                    dvec = np.stack([np.cos(th), np.sin(th), np.full_like(th, 1 / mu)], 1)
                    r = -ci_n / (dvec @ P[k][2]); ok = r > 0
                    J_th = np.array([obj(r[i] * dvec[i]) if ok[i] else np.inf for i in range(len(th))])
                    gaps.append((obj(lam[k]) - J_th.min()) / abs(J_th.min()))
                    coss.append(float(lam[k, :2] @ vc[k, :2]) / (np.linalg.norm(lam[k, :2]) * np.linalg.norm(vc[k, :2]) + 1e-300))
                print(f"    sliding 발 cos(lam_t, v_t_next) = {np.array2string(np.array(coss), precision=4)}  (등방 Mapp 에서만 -1)")
                check(f"{name}: eq.(7) 최적성  obj(lam) - min_th obj <= 0 (sliding)", float(max(gaps)), 1e-6)
            if (~con).any():
                check(f"{name}: separating 발의 lam = 0", float(np.abs(lam[~con]).max()), 0.0)
                check(f"{name}: separating 발은 v_n,next + phi/dt >= 0", max(-(vc[~con, 2] + phi[~con] / dt).max(), 0.0), 1e-12)

    # ---------------- [B] rho=0 vs FD / autodiff ----------------
    if "B" in only:
        print(f"\n{YEL}[B] rho=0 analytic (f_x,f_u) vs 중앙 FD 전체 행렬 / plain autodiff{RST}")
        for n_sub in (1, 2):
            step = make_step(env, ids, n_sub)
            print(f"  --- n_sub={n_sub} (dt={env.control_dt / n_sub}) ---")
            for name, (q, v, u) in R.items():
                t0 = time.time()
                fx_a, fu_a = cg.cimpc_jacobian(env, q, v, ws0, u, None, rho=0.0, n_sub=n_sub, forward="cimpc")
                fx_a, fu_a = np.asarray(fx_a), np.asarray(fu_a)
                fx_fd, fu_fd, mode_ok, (con, sl) = fd_jacobian(step, q, v, u)
                fx_ad, fu_ad = autodiff_jacobian(step, q, v, u)
                e_fd = max(rel(fx_a, fx_fd), rel(fu_a, fu_fd))
                e_ad = max(rel(fx_a, fx_ad), rel(fu_a, fu_ad))
                e_fd_ad = max(rel(fx_ad, fx_fd), rel(fu_ad, fu_fd))
                tag = f"{name}[c={con.astype(int)} s={sl.astype(int)}]"
                print(f"  {tag:<40s} FD-vs-autodiff {e_fd_ad:.1e}  (모드 불변={mode_ok}, {time.time() - t0:.0f}s)")
                if not mode_ok:
                    print(f"    FD 섭동 중 접촉 모드가 바뀜 -> 이 regime 은 판정에서 제외")
                    continue
                if sl.any():
                    # sliding: (i) plain autodiff 는 bisection 의 bracket 선택을 미분하므로 정답이 아니다 (FD 와 어긋남을 기록)
                    #          (ii) analytic 은 eq.(14) 'E_s 고정' 근사 -> FD(정답) 와 차이가 있어야 한다. 크기 기록, 판정은 [E].
                    print(f"    autodiff vs FD {e_fd_ad:.2e}  <- bisection 통과 autodiff 는 정답이 아님")
                    print(f"    analytic vs FD {e_fd:.2e}  <- eq.(14) 'E_s 고정' 근사의 크기 (판정은 [E])")
                else:
                    check(f"{name} n_sub={n_sub}: analytic vs FD", e_fd, 1e-6)
                    check(f"{name} n_sub={n_sub}: analytic vs autodiff", e_ad, 1e-8)

    # ---------------- [C] rho>0 독립 구현 ----------------
    if "C" in only:
        print(f"\n{YEL}[C] eq.(19) 독립 구현 vs cimpc_jacobian (n_sub=1, dt={env.control_dt}){RST}")
        for name, (q, v, u) in R.items():
            if name not in ("flight", "clamping", "mixed"):
                continue
            for rho in (0.0, 1e-4, 1e-2):
                fx_i, fu_i, info = indep_step_jacobian(env, ids, q, v, u, env.control_dt, rho)
                fx_a, fu_a = cg.cimpc_jacobian(env, q, v, ws0, u, None, rho=rho, n_sub=1, forward="cimpc")
                e = max(rel(fx_a, fx_i), rel(fu_a, fu_i))
                if rho == 0.0:
                    print(f"  {name:<9s} contact={info['contact'].astype(int)} sliding={info['sliding'].astype(int)}  "
                          f"|A lam + b| = {info['resid']:.1e}")
                    if info["contact"].any():
                        check(f"{name}: 축소 시스템 잔차 A lam + b = 0", info["resid"], 1e-8)
                check(f"{name} rho={rho:g}: 독립 eq.(19) vs cimpc_jacobian", e, 1e-9)
            # rho 가 실제로 gradient 를 바꾸는지 (접촉 있을 때)
            if info["contact"].any():
                fx0, _ = cg.cimpc_jacobian(env, q, v, ws0, u, None, rho=0.0, n_sub=1, forward="cimpc")
                fx2, _ = cg.cimpc_jacobian(env, q, v, ws0, u, None, rho=1e-2, n_sub=1, forward="cimpc")
                print(f"    rho=1e-2 가 f_x 를 바꾸는 크기: {rel(fx2, fx0):.2e}")

        print(f"\n{YEL}[C'] 같은 비교, substep dt={dt} (sliding 포함){RST}")
        for name, (q, v, u) in R.items():
            for rho in (0.0, 1e-4, 1e-2):
                fx_i, fu_i, info = indep_step_jacobian(env, ids, q, v, u, dt, rho)
                fx_a, fu_a = cg._cimpc_step_jacobian(env, q, v, ws0, u, dt, ids, rho, 1e-9, 0.7, 0.98, 2e-2, forward="cimpc")
                e = max(rel(np.asarray(fx_a), fx_i), rel(np.asarray(fu_a), fu_i))
                if rho == 0.0:
                    print(f"  {name:<9s} contact={info['contact'].astype(int)} sliding={info['sliding'].astype(int)}  |A lam + b| = {info['resid']:.1e}")
                    if info["contact"].any():
                        check(f"{name} dt={dt}: 축소 시스템 잔차", info["resid"], 1e-8)
                check(f"{name} dt={dt} rho={rho:g}: 독립 eq.(19) vs cimpc", e, 1e-9)

    # ---------------- [E] 정확한 implicit 미분 vs FD vs analytic ----------------
    if "E" in only:
        print(f"\n{YEL}[E] 한 substep (dt={dt}): 정확한 implicit 미분(theta 포함) vs FD vs analytic(E_s 고정){RST}")
        def sub(qq, vv, uu):
            return cf.cimpc_substep(env, qq, vv, uu, dt, ids)
        sub = jax.jit(sub)
        q_s, v_s, u_s = R["sliding"]
        q2, v2, _ = cf.cimpc_substep(env, q_s, v_s, u_s, dt, ids)
        cases = {"clamping": R["clamping"], "sliding(4발, slip 1.1 m/s)": (q_s, v_s, u_s),
                 "sliding(1발, slip 0.56 m/s)": (q2, v2, u_s), "partial": R["partial"]}
        for name, (q, v, u) in cases.items():
            fx_fd, fu_fd, mode_ok, (con, sl) = fd_jacobian(sub, q, v, u)
            fx_e, fu_e, info = exact_step_jacobian(env, ids, q, v, u, dt)
            # cimpc_jacobian 은 control step 단위라 한 substep 만 보려면 내부 함수를 직접 호출
            fx_a, fu_a = cg._cimpc_step_jacobian(env, q, v, ws0, u, dt, ids, 0.0, 1e-9, 0.7, 0.98, 2e-2, forward="cimpc")
            fx_a, fu_a = np.asarray(fx_a), np.asarray(fu_a)
            e_exact = max(rel(fx_e, fx_fd), rel(fu_e, fu_fd))
            e_anal = max(rel(fx_a, fx_fd), rel(fu_a, fu_fd))
            print(f"  {name:<30s} c={con.astype(int)} s={sl.astype(int)}  |R|={info['resid']:.1e}  모드불변={mode_ok}")
            check(f"{name}: KKT 잔차", info["resid"], 1e-9)
            check(f"{name}: 정확한 implicit 미분 vs FD", e_exact, 1e-5)
            if sl.any():
                fx_f, fu_f, _ = exact_step_jacobian(env, ids, q, v, u, dt, fix_E=True)
                e_fix = max(rel(fx_a, fx_f), rel(fu_a, fu_f))
                print(f"    analytic(E_s 고정) vs FD = {e_anal:.2e}   <- 논문 eq.(14) 근사의 크기")
                check(f"{name}: KKT 에 E_s 고정 가정을 넣으면 analytic 과 일치", e_fix, 1e-9)
            else:
                check(f"{name}: analytic vs FD (sliding 없음 -> 일치해야)", e_anal, 1e-6)

    # ---------------- [D] 학습 기본 solver 설정의 영향 ----------------
    if "D" in only:
        print(f"\n{YEL}[D] 학습 기본값 (n_outer=10, n_bisect=8, tol=1e-6) 의 해 정밀도{RST}")
        for name, (q, v, u) in R.items():
            set_solver(40, 50, 1e-11)
            _, vn_ref, aux_ref = cf.cimpc_substep(env, q, v, u, dt, ids)
            set_solver(10, 8, 1e-6)
            _, vn_def, aux_def = cf.cimpc_substep(env, q, v, u, dt, ids)
            dl = float(jnp.abs(aux_def["lam"] - aux_ref["lam"]).max()); dv = float(jnp.abs(vn_def - vn_ref).max())
            same = bool(jnp.all(aux_def["sliding"] == aux_ref["sliding"]) & jnp.all(aux_def["contact"] == aux_ref["contact"]))
            print(f"  {name:<9s} |dlam|={dl:.2e}  |dv_next|={dv:.2e}  모드 동일={same}  "
                  f"(sliding={np.asarray(aux_def['sliding']).astype(int)})")
        set_solver(40, 50, 1e-11)

    print(f"\n{'=' * 70}")
    n_ok = sum(o for _, o in results)
    for nm, o in results:
        if not o:
            print(f"  {RED}FAIL{RST}  {nm}")
    print(f"  {n_ok}/{len(results)} 통과")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
