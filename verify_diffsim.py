"""diffsim_grad.py (de Saxcé full-cone central-path gradient) 검증, x64 CPU.

    JAX_PLATFORMS=cpu python verify_diffsim.py [ABCDEF]

 [A] Newton (Alg. 1): hard 해에서의 잔차 |F(lam_h)|, rho 별 수렴 / 물리적 root, |lam* - lam_h| 의 rho 비례.
     line search 유무 (C++ 기본 OFF, 여기 기본 ON) 비교.
 [B] C++ GetGradientRelaxedV3 의 F / M_lam / M_sig / Newton 반복을 numpy 로 직역해 대조 (랜덤 + 실제 입력).
 [C] 감도 식 (15): 명시적 블록 (M_lam + M_sig A, M_sig (dA lam + db)) vs 잔차 R(lam, xi) 의 autodiff.
     dlam/dxi 와 (f_x, f_u) 조립까지 독립 경로 (verify_cimpc._assemble) 와 대조.
 [D] end-to-end: 완화 동역학 (lam* 적용) 의 중앙 FD vs diffsim step Jacobian. IFT 가 맞으면 일치해야 한다.
     delta > 0 도 포함 (delta 는 b 에만 들어가고 db 에는 없음).
 [E] rho -> 0 극한: hard step 의 FD 와 대조. clamping / flight / mixed / landing 에서는 정확한 fixed-mode
     감도 (= FD) 를 회복해야 한다. sliding 은 de Saxcé MDP root 와 solver 해 (이방성 Mapp) 가 달라
     차이가 남는다 -> 크기 기록.
 [F] 연속성 스캔 (논문 Fig. 3 유사): 서 있는 로봇의 base v_z 를 스캔하며 clamping -> separating 전이에서
     d lam_n / d v_z 가 rho = 0 (불연속) 과 rho > 0 (연속) 에서 어떻게 다른지, delta 의 효과.
"""
import os, sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from mujoco import mjx

import diffsim
import cimpc_grad as cg
import cimpc_forward as cf
import diffsim_grad as dg
import verify_cimpc as vc
from verify_cimpc import (check, rel, make_regimes, stand_state, set_solver, fd_jacobian,
                          make_step, exact_step_jacobian, NQ, NV, NU, FNUM, YEL, RST)
cf.UNROLL = False

results = vc.results


def fd_step_jacobian(step, q, v, u, h=1e-6):
    """임의의 step(q,v,u)->(qn,vn,aux) 의 중앙 FD (tangent 좌표). 모드 검사 없음."""
    qn0, vn0, _ = step(q, v, u)
    Ep = np.asarray(cg.ambient_to_tangent(qn0))
    cols = []
    def col(fp, fm):
        qp, vp, _ = fp; qm, vm, _ = fm
        return np.concatenate([Ep @ (np.asarray(qp) - np.asarray(qm)) / (2 * h),
                               (np.asarray(vp) - np.asarray(vm)) / (2 * h)])
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
    return Jm[:, :2 * NV], Jm[:, 2 * NV:]


# ---------------------------------------------------------------------------
# [B] C++ 직역 (contactSolverGradient.cpp GetGradientRelaxedV3, 12 = fnum*3, [T,T,N])
# ---------------------------------------------------------------------------
def cpp_F_blocks(lam, sig, mu, rho, kEpsT=1e-12):
    F = np.zeros(12); Ml = np.zeros((12, 12)); Ms = np.zeros((12, 12))
    I2 = np.eye(2)
    for i in range(4):
        sT = sig[i*3:i*3+2]; sN = sig[i*3+2]; lT = lam[i*3:i*3+2]; lN = lam[i*3+2]
        raw = np.linalg.norm(sT)
        sTn = raw if raw > kEpsT else kEpsT
        unit = sT / sTn if raw > kEpsT else np.zeros(2)
        y0 = sN / mu + sTn
        F[i*3:i*3+2] = mu * lN * sT + y0 * lT
        F[i*3+2] = mu * lN * y0 + lT @ sT - rho
        LxD = np.zeros((3, 3)); LyC = np.zeros((3, 3))
        LxD[:2, :2] = np.outer(lT, unit) + mu * lN * I2
        LxD[:2, 2] = lT / mu
        LxD[2, :2] = mu * lN * unit + lT
        LxD[2, 2] = lN
        LyC[:2, :2] = y0 * I2
        LyC[:2, 2] = mu * sT
        LyC[2, :2] = sT
        LyC[2, 2] = sN + mu * sTn
        Ms[i*3:i*3+3, i*3:i*3+3] = LxD
        Ml[i*3:i*3+3, i*3:i*3+3] = LyC
    return F, Ml, Ms


def cpp_newton(lam_warm, A, b, mu, rho, beta=0.1, kmax=20, tol=1e-10):
    lam = lam_warm.copy(); sig = A @ lam + b
    iters = 0
    for it in range(kmax):
        F, Ml, Ms = cpp_F_blocks(lam, sig, mu, rho)
        JF = Ml + Ms @ A
        nF = np.linalg.norm(F)
        if nF < tol:
            break
        if beta > 0:
            JF = JF + beta * nF * np.eye(12)
        delta = np.linalg.solve(JF, -F)
        lam = lam + delta
        sig = b + A @ lam
        iters = it + 1
    return lam, sig, iters


# ---------------------------------------------------------------------------
def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "ABCDEF"
    env = diffsim.Env()
    ids = cg.foot_ids(env)
    mu = ids[4]
    ws0 = jnp.zeros(NV)
    set_solver(40, 50, 1e-11)
    R = make_regimes(env, ids)
    dt = env.control_dt / 2
    z = jnp.zeros(NV)
    print(f"jax {jax.__version__} x64={jax.config.jax_enable_x64} devices={jax.devices()}  mu={mu}  dt={dt}")

    def delassus(q, v, u, delta=0.0):
        A_and_b, _ = dg.make_delassus(env, q, dt, ids, delta, "cimpc")
        return A_and_b, A_and_b(z, v, u)

    # ---------------- [A] Newton ----------------
    if "A" in only:
        print(f"\n{YEL}[A] Alg. 1 Newton: hard 해에서의 잔차, rho 별 수렴 / 물리성{RST}")
        newton = jax.jit(lambda lam, A, b, rho, ls: dg.relaxed_point(lam, A, b, mu, rho, 0.1, 40, ls, 1e-14),
                         static_argnums=(4,))
        for name, (q, v, u) in R.items():
            lam_h, con, sl = cg.hard_contact_impulse(env, q, v, u, dt, ids)
            _, (A, b) = delassus(q, v, u)
            sig_h = (A @ lam_h.reshape(-1) + b).reshape(FNUM, 3)
            F0 = np.asarray(dg.central_path_residual(lam_h, sig_h, mu, 0.0))
            con, sl = np.asarray(con), np.asarray(sl)
            nF0 = np.linalg.norm(F0, axis=1)
            scale = np.maximum(np.abs(np.asarray(lam_h)).max(1) * np.abs(np.asarray(sig_h)).max(1), 1e-300)
            print(f"  {name:<9s} c={con.astype(int)} s={sl.astype(int)}  |F_k(lam_h)| = {np.array2string(nF0, precision=2)}"
                  f"   (/|lam||sig| = {np.array2string(nF0 / scale, precision=2)})")
            if (~sl).any():
                check(f"{name}: rho=0 에서 clamping/separating 발은 F(lam_h)=0", float(nF0[~sl].max()), 1e-9)
            if sl.any():
                print(f"    sliding 발의 F_t != 0: solver 해 (이방성 eq.7 최적) 가 de Saxcé 등방 MDP 를 안 만족 (구조적, 원본도 동일)")
            prev = None
            for rho in (1e-6, 1e-4, 1e-2, 1e-1):
                row = []
                for ls in (8, 0):
                    lam_s, sig_s, F, hist, n_it = newton(lam_h, A, b, rho, ls)
                    ok = np.asarray(dg.physical_mask(lam_s, sig_s, F, mu, rho, 1e-6, -0.1, 1e-6))
                    nF = float(np.linalg.norm(np.asarray(F)))
                    dl = float(np.abs(np.asarray(lam_s) - np.asarray(lam_h)).max())
                    row.append((ls, int(n_it), nF, ok, dl))
                    tag = "LS" if ls else "noLS"
                    check(f"{name} rho={rho:g} {tag}: Newton 수렴 |F| < 1e-10", nF, 1e-10)
                    check(f"{name} rho={rho:g} {tag}: 모든 접촉이 물리적 root", 0.0 if ok.all() else 1.0, 0.0)
                print(f"    rho={rho:<5g} LS: it={row[0][1]:2d} |F|={row[0][2]:.1e} ok={row[0][3].astype(int)} |lam*-lam_h|={row[0][4]:.2e}"
                      f"  | noLS: it={row[1][1]:2d} |F|={row[1][2]:.1e} ok={row[1][3].astype(int)} |lam*-lam_h|={row[1][4]:.2e}"
                      + (f"  ratio/rho={row[0][4] / rho:.2f}" if not sl.any() else ""))

    # ---------------- [B] C++ 직역 대조 ----------------
    if "B" in only:
        print(f"\n{YEL}[B] C++ GetGradientRelaxedV3 직역 (numpy) vs diffsim_grad{RST}")
        rng = np.random.default_rng(3)
        for trial in range(3):
            lam = rng.standard_normal((4, 3)); lam[:, 2] = np.abs(lam[:, 2]) + 0.1
            sig = rng.standard_normal((4, 3)) * 0.5
            if trial == 2:
                sig[1, :2] = 0.0                     # sig_t = 0 규약 (unit = 0)
            rho = 10.0 ** rng.uniform(-4, 0)
            Fc, Mlc, Msc = cpp_F_blocks(lam.reshape(-1), sig.reshape(-1), mu, rho)
            Fj = np.asarray(dg.central_path_residual(jnp.array(lam), jnp.array(sig), mu, rho)).reshape(-1)
            Ml, Ms = dg.arrow_blocks(jnp.array(lam), jnp.array(sig), mu)
            check(f"trial{trial}: F (compute_F_perfoot; C++ 는 |sig_t| 를 1e-12 로 클램프)", rel(Fj, Fc), 1e-11)
            check(f"trial{trial}: M_lam (LyC)", rel(np.asarray(dg.block_diag(Ml)), Mlc), 1e-11)
            check(f"trial{trial}: M_sig (LxD)", rel(np.asarray(dg.block_diag(Ms)), Msc), 1e-11)
        # Newton: 실제 (A, b, lam_h) 에서, line search 없이 (C++ 기본) 1 스텝과 수렴점
        for name in ("clamping", "partial", "mixed"):
            q, v, u = R[name]
            lam_h, _, _ = cg.hard_contact_impulse(env, q, v, u, dt, ids)
            _, (A, b) = delassus(q, v, u)
            An, bn, lh = np.asarray(A), np.asarray(b), np.asarray(lam_h).reshape(-1)
            for rho in (1e-4, 1e-2):
                lam1_c, _, _ = cpp_newton(lh, An, bn, mu, rho, 0.1, 1, 0.0)
                lam1_j, _, _, _, _ = dg.relaxed_point(lam_h, A, b, mu, rho, 0.1, 1, 0, 0.0)
                check(f"{name} rho={rho:g}: Newton 1 스텝 (beta=0.1, LS 없음; LU 구현 차이)", rel(np.asarray(lam1_j).reshape(-1), lam1_c), 1e-10)
                lamc, _, itc = cpp_newton(lh, An, bn, mu, rho, 0.1, 30, 1e-13)
                lamj, _, Fj, _, itj = dg.relaxed_point(lam_h, A, b, mu, rho, 0.1, 30, 0, 1e-14)
                check(f"{name} rho={rho:g}: Newton 수렴점 (C++ it={itc}, jax it={int(itj)})", rel(np.asarray(lamj).reshape(-1), lamc), 1e-9)

    # ---------------- [C] 감도 식 vs autodiff ----------------
    if "C" in only:
        print(f"\n{YEL}[C] eq.(15) 명시적 블록 vs 잔차 R(lam, xi) 의 autodiff, 그리고 (f_x,f_u) 조립 독립 경로{RST}")
        rho = 1e-3
        for name, (q, v, u) in R.items():
            A_and_b, (A, b) = delassus(q, v, u)
            fx, fu, info = dg._diffsim_step_jacobian(env, q, v, ws0, u, dt, ids, rho, n_newton=40, newton_tol=1e-14,
                                                    return_info=True)
            lam_s = jax.lax.stop_gradient(info["lam"])
            ok = np.asarray(info["ok"])
            min_sT = float(np.linalg.norm(np.asarray(info["sig"])[:, :2], axis=1).min())
            print(f"  {name:<9s} ok={ok.astype(int)} n_it={int(info['n_it'])} |F|={float(np.linalg.norm(np.asarray(info['F']))):.1e}  min|sig_t*|={min_sT:.1e}")
            if not ok.all():
                print("    절단된 접촉이 있어 autodiff 대조는 건너뜀"); continue

            def Rfun(lam_flat, theta, vv, uu):
                A_, b_ = A_and_b(theta, vv, uu)
                sig = (A_ @ lam_flat + b_).reshape(FNUM, 3)
                return dg.central_path_residual(lam_flat.reshape(FNUM, 3), sig, mu, rho).reshape(-1)

            lf = lam_s.reshape(-1)
            dR_dlam = jax.jacfwd(Rfun, 0)(lf, z, v, u)
            dR_q, dR_v, dR_u = jax.jacfwd(Rfun, (1, 2, 3))(lf, z, v, u)
            check(f"{name}: M_lam + M_sig A  vs  dR/dlam (autodiff)", rel(info["S"], dR_dlam), 1e-10)
            dl_q = -jnp.linalg.solve(dR_dlam, dR_q); dl_v = -jnp.linalg.solve(dR_dlam, dR_v); dl_u = -jnp.linalg.solve(dR_dlam, dR_u)
            e = max(rel(info["dlam_dq"], dl_q), rel(info["dlam_dv"], dl_v), rel(info["dlam_du"], dl_u))
            check(f"{name}: dlam/dxi  명시적 vs autodiff IFT", e, 1e-9)
            fx_i, fu_i = vc._assemble(env, ids, q, v, u, dt, lam_s, dl_q.reshape(FNUM, 3, -1),
                                      dl_v.reshape(FNUM, 3, -1), dl_u.reshape(FNUM, 3, -1))
            check(f"{name}: (f_x, f_u) vs 독립 조립 (verify_cimpc._assemble)", max(rel(fx, fx_i), rel(fu, fu_i)), 1e-9)

    # ---------------- [D] end-to-end IFT (완화 동역학 FD) ----------------
    if "D" in only:
        print(f"\n{YEL}[D] 완화 동역학 (lam* 적용) 의 중앙 FD vs diffsim step Jacobian (lam_eval=relaxed){RST}")
        cases = [(name, 1e-3, 0.0) for name in R] + [("clamping", 1e-2, 0.2), ("partial", 1e-2, 0.2), ("flight", 1e-3, 0.2)]
        for name, rho, delta in cases:
            q, v, u = R[name]
            step = jax.jit(lambda qq, vv, uu: dg.relaxed_substep(env, qq, vv, uu, dt, ids, rho, delta, 0.1, 40, 8, 1e-14))
            t0 = time.time()
            fx_fd, fu_fd = fd_step_jacobian(step, q, v, u)
            fx, fu, info = dg._diffsim_step_jacobian(env, q, v, ws0, u, dt, ids, rho, delta=delta, n_newton=40,
                                                    newton_tol=1e-14, return_info=True)
            ok = np.asarray(info["ok"])
            e = max(rel(fx, fx_fd), rel(fu, fu_fd))
            print(f"  {name:<9s} rho={rho:g} delta={delta:g} ok={ok.astype(int)} ({time.time() - t0:.0f}s)")
            check(f"{name} rho={rho:g} delta={delta:g}: analytic vs FD(완화 동역학)", e, 1e-5)

    # ---------------- [E] rho -> 0 극한 ----------------
    if "E" in only:
        print(f"\n{YEL}[E] rho -> 0 (1e-10, delta=0): hard step 의 FD / cimpc(rho=0) / 정확한 implicit 미분과 대조{RST}")
        sub = jax.jit(lambda qq, vv, uu: cf.cimpc_substep(env, qq, vv, uu, dt, ids))
        for name, (q, v, u) in R.items():
            fx_fd, fu_fd, mode_ok, (con, sl) = fd_jacobian(sub, q, v, u)
            fx, fu, info = dg._diffsim_step_jacobian(env, q, v, ws0, u, dt, ids, 1e-10, n_newton=60, newton_tol=1e-14,
                                                    return_info=True)
            fx_c, fu_c = cg._cimpc_step_jacobian(env, q, v, ws0, u, dt, ids, 0.0, 1e-9, 0.7, 0.98, 2e-2, forward="cimpc")
            e_fd = max(rel(fx, fx_fd), rel(fu, fu_fd))
            e_c = max(rel(fx, fx_c), rel(fu, fu_c))
            ok = np.asarray(info["ok"])
            print(f"  {name:<9s} c={con.astype(int)} s={sl.astype(int)} 모드불변={mode_ok} ok={ok.astype(int)} n_it={int(info['n_it'])} "
                  f"|F|={float(np.linalg.norm(np.asarray(info['F']))):.1e}  |lam*-lam_h|={float(np.abs(np.asarray(info['lam']) - np.asarray(info['lam_h'])).max()):.1e}")
            if not mode_ok:
                print("    FD 중 모드 변화 -> 제외"); continue
            if sl.any():
                fx_e, fu_e, _ = exact_step_jacobian(env, ids, q, v, u, dt)
                e_e = max(rel(fx, fx_e), rel(fu, fu_e))
                print(f"    diffsim(rho->0) vs FD(hard)        = {e_fd:.2e}   <- 등방 MDP root 와 solver 해의 차이")
                print(f"    diffsim(rho->0) vs cimpc(rho=0)     = {e_c:.2e}")
                print(f"    diffsim(rho->0) vs 정확한 implicit  = {e_e:.2e}")
            else:
                check(f"{name}: diffsim(rho->0) vs FD(hard)", e_fd, 1e-5)
                check(f"{name}: diffsim(rho->0) vs cimpc(rho=0)", e_c, 1e-5)

    # ---------------- [F] 연속성 스캔 ----------------
    if "F" in only:
        print(f"\n{YEL}[F] base v_z 스캔: FL 발 d lam_n / d v_z  (clamping -> separating 전이){RST}")
        q_s, v_s, u_s = R["clamping"]
        vzs = np.linspace(-0.2, 0.3, 26)
        settings = [(1e-3, 0.0), (1e-2, 0.0), (5e-2, 0.0), (1e-2, 0.1)]
        jac = {s: jax.jit(lambda qq, vv, uu, rho=s[0], delta=s[1]: dg._diffsim_step_jacobian(
            env, qq, vv, ws0, uu, dt, ids, rho, delta=delta, n_newton=40, newton_tol=1e-14, return_info=True))
            for s in settings}
        hard = jax.jit(lambda qq, vv, uu: cg.hard_contact_impulse(env, qq, vv, uu, dt, ids)[0])
        hdr = "   v_z    lam_h,n   exact(FD)  " + "  ".join(f"rho={r:g},d={d:g}" for r, d in settings)
        print(hdr)
        rows = {s: [] for s in settings}
        allok = True
        for vz in vzs:
            v = v_s.at[2].set(float(vz))
            h = 1e-6
            lp = hard(q_s, v.at[2].add(h), u_s)[0, 2]; lm = hard(q_s, v.at[2].add(-h), u_s)[0, 2]
            ex = float((lp - lm) / (2 * h)); ln = float(hard(q_s, v, u_s)[0, 2])
            vals = []
            for s in settings:
                _, _, info = jac[s](q_s, v, u_s)
                allok &= bool(np.asarray(info["ok"]).all())
                val = float(info["dlam_dv"][2, 2])
                rows[s].append(val); vals.append(val)
            print(f"  {vz:+.3f}  {ln:8.4f}  {ex:9.4f}  " + "  ".join(f"{x:13.4f}" for x in vals))
        check("스캔 전 구간에서 모든 접촉이 물리적 root", 0.0 if allok else 1.0, 0.0)
        for s in settings:
            jumps = np.abs(np.diff(rows[s])).max()
            print(f"    rho={s[0]:g} delta={s[1]:g}: 인접 샘플 간 최대 변화 {jumps:.4f}")

    n_ok = sum(ok for _, ok in results); n = len(results)
    print(f"\n==== {n_ok}/{n} PASS ====")
    for name_, ok_ in results:
        if not ok_:
            print(f"  FAIL: {name_}")


if __name__ == "__main__":
    main()
