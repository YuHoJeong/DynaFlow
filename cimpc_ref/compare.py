"""CI-MPC C++ 원본과 eq.(19) 구현을 대조한다.

목적
----
`cimpc_grad.cimpc_jacobian` 의 접촉 gradient 는 MJX 로 검증할 수 없다
(MJX gradient 는 애초에 정답이 아니다). 유일한 기준은 논문 저자들의 C++ 구현이다.

여기서는 모델 동등성 문제(Hound vs Go1)를 우회한다.
C++ 이 덤프한 M, J, nle, foot_pos 만 가지고 eq.(19) 를 numpy 로 재구성해서
C++ 의 해석적 gradient 와 맞는지 본다. 즉 **공식이 맞는지** 를 검증한다.
(MJX 로부터 A, b 를 만드는 부분은 verify_grad.py 가 따로 본다.)

역산해낸 규약
-------------
    A = J M^-1 J^T                                  Delassus, (3*fnum, 3*fnum)
    b = J (v + M^-1 (tau - nle) dt)                 nle 는 Pinocchio 부호 (M a + nle = tau)
    b[3k+2] += phi_k / dt                           drift compensation, phi_k = 발 높이
    lambda = -A^-1 b                                impulse (force 아님), local-world-aligned
    D = diag(1/lambda_n^2)  on normal 인덱스 3k+2
    dlambda/dxi = -[A + rho D]^-1 (dA/dxi lambda + db/dxi)

    발 순서 FL, FR, RL, RR.  dlambda_dz 열: [dq | dv | dtau], 각 nv=18.
    행 개수 = clamping*3 + sliding*1 (sliding 은 normal 성분만).

실행:
    ./build.sh && python compare.py
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_dump import parse

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "build", "dump_gradient")
NV, FNUM = 18, 4
RED, GRN, YEL, RST = "\033[91m", "\033[92m", "\033[93m", "\033[0m"

# 하네스 주석에 있던 "4발 clamping" 자세
Q_CLAMP = [0, 0, 0.51 - 0.025, 0, 0, 0, 1,
           -0.05, 0.7854, -1.5708, 0.05, 0.7854, -1.5708,
           -0.05, 0.7854, -1.5708, 0.05, 0.7854, -1.5708]


def write_state(path, q, v, tau, rho):
    with open(path, "w") as f:
        for arr in (q, v, tau):
            f.write(" ".join(repr(float(x)) for x in arr) + "\n")
        f.write(repr(float(rho)) + "\n")


def run(state, idx=None, eps=None):
    cmd = [EXE, state] + ([str(idx), repr(eps)] if idx is not None else [])
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    tmp = os.path.join(HERE, "build", "_last_dump.txt")
    open(tmp, "w").write(out)
    return parse(tmp)


def A_b_of(bl, v, tau):
    """C++ 덤프로부터 Delassus A 와 unconstrained 접촉속도 b 를 재구성."""
    M, J, nle, fp = bl["mass_matrix"], bl["foot_jacobian"], bl["nle"].ravel(), bl["foot_pos"]
    dt = bl["meta"][0, 0]
    Minv = np.linalg.inv(M)
    A = J @ Minv @ J.T
    b = J @ (v + Minv @ (tau - nle) * dt)
    for k in range(FNUM):
        b[3 * k + 2] += fp[2, k] / dt          # drift compensation
    return A, b


def D_of(lam_flat):
    """D = diag(1/lambda_n^2), normal 인덱스에만."""
    d = np.zeros(3 * FNUM)
    for k in range(FNUM):
        d[3 * k + 2] = 1.0 / lam_flat[3 * k + 2] ** 2
    return np.diag(d)


def report(name, err, tol):
    ok = bool(np.isfinite(err) and err < tol)
    print(f"  [{GRN + 'PASS' + RST if ok else RED + 'FAIL' + RST}] {name:<40s} err={err:.3e} (tol={tol:.0e})")
    return ok


def rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(b).max(), 1e-30))


def main():
    if not os.path.exists(EXE):
        print(f"{RED}dump_gradient 가 없다. 먼저 ./build.sh 를 실행할 것.{RST}")
        return 1

    sdir = os.path.join(HERE, "build")
    v = np.zeros(NV)
    tau = np.zeros(NV)
    ok = True

    # ---- 1. b 정의 (drift compensation 포함) ----------------------------
    print(f"\n{YEL}[1] b 정의: lambda = -A^-1 b 가 성립하는가{RST}")
    st = os.path.join(sdir, "s_rho0.txt")
    write_state(st, Q_CLAMP, v, tau, 0.0)
    base = run(st)
    dt = base["meta"][0, 0]
    lam = base["lambda_lw"].reshape(-1, order="F")
    A0, b0 = A_b_of(base, v, tau)
    ok &= report("lambda == -A^-1 b  (drift 포함)", rel(-np.linalg.solve(A0, b0), lam), 1e-8)

    b_nd = b0.copy()
    for k in range(FNUM):
        b_nd[3 * k + 2] -= base["foot_pos"][2, k] / dt
    print(f"        drift 항 제외 시: err={rel(-np.linalg.solve(A0, b_nd), lam):.3e}"
          f"  {YEL}<- drift 는 선택 사항이 아니다{RST}")

    # ---- 2. rho 스윕: D 의 위치와 형태 ----------------------------------
    print(f"\n{YEL}[2] rho 완화항: dlambda/dtau = -[A + rho D]^-1 J M^-1 dt{RST}")
    for rho in [0.0, 1e-6, 1e-4, 1e-2]:
        st = os.path.join(sdir, f"s_rho{rho:g}.txt")
        write_state(st, Q_CLAMP, v, tau, rho)
        bl = run(st)
        M, J = bl["mass_matrix"], bl["foot_jacobian"]
        lam_r = bl["lambda_lw"].reshape(-1, order="F")
        Minv = np.linalg.inv(M)
        A = J @ Minv @ J.T
        pred = -np.linalg.solve(A + rho * D_of(lam_r), J @ Minv) * dt
        cpp = bl["dlambda_dz"][:, 2 * NV:3 * NV]
        e_ign = rel(-np.linalg.solve(A, J @ Minv) * dt, cpp)
        ok &= report(f"rho={rho:g}", rel(pred, cpp), 1e-12)
        if rho > 0:
            print(f"        (rho 무시하면 err={e_ign:.3e})")

    # ---- 3. dlambda/dv -------------------------------------------------
    print(f"\n{YEL}[3] dlambda/dv = -A^-1 J   (v=0 이라 Coriolis 항 소멸){RST}")
    J0 = base["foot_jacobian"]
    ok &= report("dlambda/dv", rel(-np.linalg.solve(A0, J0), base["dlambda_dz"][:, NV:2 * NV]), 1e-12)

    # ---- 4. dlambda/dq: A, b 를 tangent 방향으로 수치미분 --------------
    print(f"\n{YEL}[4] dlambda/dq = -A^-1 (dA/dq lambda + db/dq){RST}")
    print(f"        (A, b 를 C++ 을 tangent 방향으로 섭동시켜 수치미분)")
    eps = 1e-6
    dA = np.zeros((3 * FNUM, 3 * FNUM, NV))
    db = np.zeros((3 * FNUM, NV))
    st0 = os.path.join(sdir, "s_rho0.txt")
    for i in range(NV):
        Ap, bp = A_b_of(run(st0, i, eps), v, tau)
        Am, bm = A_b_of(run(st0, i, -eps), v, tau)
        dA[:, :, i] = (Ap - Am) / (2 * eps)
        db[:, i] = (bp - bm) / (2 * eps)
    rhs = np.einsum("ijk,j->ik", dA, lam) + db
    pred_dq = -np.linalg.solve(A0, rhs)
    cpp_dq = base["dlambda_dz"][:, 0:NV]
    ok &= report("dlambda/dq", rel(pred_dq, cpp_dq), 1e-8)

    p_hess = -np.linalg.solve(A0, np.einsum("ijk,j->ik", dA, lam))
    p_b = -np.linalg.solve(A0, db)
    print(f"        기여 크기:  dA/dq*lambda = {np.abs(p_hess).max():7.4f}"
          f"   db/dq = {np.abs(p_b).max():7.4f}")

    # ---- 5. sliding 케이스 (구조 + 수치) --------------------------------
    print(f"\n{YEL}[5] sliding 케이스{RST}")
    q_sl = list(Q_CLAMP); q_sl[3:7] = [0.0, 0.0249974, 0.0, 0.9996875]
    v_sl = np.zeros(NV); v_sl[0] = 1.0
    st = os.path.join(sdir, "s_slide.txt")
    write_state(st, q_sl, v_sl, tau, 0.0)
    bl = run(st)
    cs = bl["contact_state"].ravel().astype(int)
    ss = bl["sliding_state"].ravel().astype(int)
    exp_rows = int(cs.sum()) * 3 - int(ss.sum()) * 2
    print(f"        contact={cs} sliding={ss} -> 행 {bl['dlambda_dz'].shape[0]} (예상 {exp_rows})")
    ok &= report("행 개수 = clamping*3 + sliding*1",
                 abs(bl["dlambda_dz"].shape[0] - exp_rows), 0.5)

    lam_s = bl["lambda_lw"]
    # E_k = [lam_x/lam_z, lam_y/lam_z, 1]  (C++ 규약; solver 가 원뿔 경계에 올려놓음)
    E = np.zeros((FNUM, 3))
    for k in range(FNUM):
        if cs[k]:
            E[k] = [lam_s[0, k] / lam_s[2, k], lam_s[1, k] / lam_s[2, k], 1.0]

    def A_b_mixed(b_, v_):
        """clamping/sliding 이 섞인 A, b. sliding 은 왼쪽 normal 1행, 오른쪽 E^T J 1행."""
        M, J, nle, fp = b_["mass_matrix"], b_["foot_jacobian"], b_["nle"].ravel(), b_["foot_pos"]
        Minv = np.linalg.inv(M)
        L, R, drift = [], [], []
        for k in range(FNUM):
            if not cs[k]:
                continue
            Jk = J[3 * k:3 * k + 3, :]
            if ss[k]:
                L.append(Jk[2:3, :]); R.append((E[k] @ Jk)[None, :]); drift.append(fp[2, k] / dt)
            else:
                L.append(Jk); R.append(Jk); drift += [0.0, 0.0, fp[2, k] / dt]
        Jl, Jr = np.vstack(L), np.vstack(R)
        return Jl @ Minv @ Jr.T, Jl @ (v_ + Minv @ (tau - nle) * dt) + np.array(drift), Jl, Minv

    As, bs, Jls, Minvs = A_b_mixed(bl, v_sl)
    u = []                                   # 미지수: sliding 은 lambda_n 만
    for k in range(FNUM):
        if cs[k]:
            u += ([lam_s[2, k]] if ss[k] else list(lam_s[:, k]))
    u = np.array(u)
    ns = As.shape[0]
    cpp_s = bl["dlambda_dz"]
    print(f"        A 비대칭도 |A-A^T|max = {np.abs(As - As.T).max():.3e}")
    ok &= report("lambda == -A^-1 b  (sliding)", rel(-np.linalg.solve(As, bs), u), 1e-8)
    ok &= report("dlambda/dtau (sliding)",
                 rel(-np.linalg.solve(As, Jls @ Minvs) * dt, cpp_s[:, 2 * NV:3 * NV]), 1e-7)

    # dlambda/dv 와 dlambda/dq: b 를 수치미분한다.
    # 주의: 이 상태는 v != 0 이라 db/dv != Jl 이다 (Coriolis 항).
    #       v=0 에서만 쓰는 간이식 -A^-1 Jl 을 그대로 쓰면 2.5e-2 로 틀린다.
    dA = np.zeros((ns, ns, NV)); dbq = np.zeros((ns, NV)); dbv = np.zeros((ns, NV))
    for i in range(NV):
        Ap, bp, _, _ = A_b_mixed(run(st, i, eps), v_sl)
        Am, bm, _, _ = A_b_mixed(run(st, i, -eps), v_sl)
        dA[:, :, i] = (Ap - Am) / (2 * eps); dbq[:, i] = (bp - bm) / (2 * eps)
        vp = v_sl.copy(); vp[i] += eps
        vm = v_sl.copy(); vm[i] -= eps
        stp = os.path.join(sdir, "s_slide_vp.txt"); write_state(stp, q_sl, vp, tau, 0.0)
        stm = os.path.join(sdir, "s_slide_vm.txt"); write_state(stm, q_sl, vm, tau, 0.0)
        _, bvp, _, _ = A_b_mixed(run(stp), vp)
        _, bvm, _, _ = A_b_mixed(run(stm), vm)
        dbv[:, i] = (bvp - bvm) / (2 * eps)
    ok &= report("dlambda/dv (sliding, Coriolis 포함)",
                 rel(-np.linalg.solve(As, dbv), cpp_s[:, NV:2 * NV]), 1e-7)
    ok &= report("dlambda/dq (sliding)",
                 rel(-np.linalg.solve(As, np.einsum("ijk,j->ik", dA, u) + dbq), cpp_s[:, 0:NV]), 1e-7)

    print(f"\n{'=' * 62}\n  {GRN + '전부 통과' + RST if ok else RED + '실패 있음' + RST}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
