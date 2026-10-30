"""
cimpc_grad.py 검증 스크립트.

배선(custom_vjp + tangent 좌표 변환)이 맞는지를 먼저 증명해야
CI-MPC gradient 를 갈아끼우는 게 의미가 있다.
test 1~4 가 통과하면, gradient 를 바꿨을 때 생기는 차이가
배선 버그가 아니라 정말 gradient 차이라고 말할 수 있다.

두 가지 regime 에서 모두 검증한다:
    flight  : 접촉 없음. 완전히 매끄러워서 FD 가 엄격하게 맞아야 한다.
    contact : 4발 접지. MJX 의 contact solve 는 piecewise smooth 라
              FD 오차가 더 크지만, 여기서 맞아야 실제 의미가 있다.

주의: action 은 raw joint position target 이다 (env.ctrl_range 안).
      정규화된 [-1,1] 값을 그냥 넣으면 로봇이 무너지면서 궤적이 chaotic 해지고
      FD 자체가 무의미해진다.  반드시 env.nominal_ptarg 근방에서 검증할 것.

NOTE: mjx_jacobian 은 jacfwd 를 49개 tangent 방향 x 4 substep 으로 펼치므로
      첫 컴파일에 수 분이 걸린다. cimpc_grad._ROLLOUT_CACHE 가 (env, backend) 별로
      한 번만 컴파일하도록 캐싱한다.

실행:
    python verify_grad.py              # 배선 검증 (mjx 백엔드)
    python verify_grad.py --cimpc      # CI-MPC 백엔드 비교까지
    python verify_grad.py --bench      # 실제 학습 shape (B=1024,H=16) 타이밍
"""

import sys
import time
import argparse

import numpy as np
import jax

# 유한차분 검증은 float32 로는 신뢰할 수 없다. 반드시 x64 를 먼저 켠다.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import mujoco
from mujoco import mjx

import diffsim
import cimpc_grad as cg


RED, GRN, YEL, CYA, RST = "\033[91m", "\033[92m", "\033[93m", "\033[96m", "\033[0m"

# 접촉 regime 의 높이는 Go1 의 nominal 자세(nominal_ptarg)에서 발이 지면에
# 살짝 닿는 값이다. trunk z=0.30 에서 발 중심 z=0.0352, 발 반경 0.023 이므로
# 접지 높이는 0.2878, 여기서 1.3mm 만 파묻히게 잡는다.
# 이 값을 잘못 잡으면(예: qpos0 의 편 다리 자세에 z=0.27) 발이 17cm 파묻혀
# qfrc_constraint 가 1e4 규모가 되고 FD 가 완전히 무의미해진다.
STAND_Z = 0.2865

REGIMES = {
    # 접촉 없음. 완전히 매끄러워 FD 가 엄격히 맞아야 한다.
    "flight":  dict(height=0.80,    act_noise=0.02, quat_noise=0.02,  joint_noise=0.03, seed=11),
    # 4발 접지. 얕은 접촉이라야 FD 가 의미를 갖는다.
    "contact": dict(height=STAND_Z, act_noise=0.01, quat_noise=0.003, joint_noise=0.01, seed=22),
}


def report(name, err, tol):
    ok = bool(np.isfinite(err) and err < tol)
    tag = f"{GRN}PASS{RST}" if ok else f"{RED}FAIL{RST}"
    print(f"    [{tag}] {name:<44s} err={err:.3e}  (tol={tol:.0e})")
    return ok


def rel_err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    den = max(np.abs(a).max(), np.abs(b).max(), 1e-12)
    return float(np.abs(a - b).max() / den)


def make_case(env, rng, regime, H):
    """검증용 (qpos, qvel, actions, w_q, w_v).

    actions 는 nominal 자세 근방의 raw joint target 이다.
    """
    cfg = REGIMES[regime]
    k = jax.random.split(rng, 5)
    nq, nv, nu = env.mj_model.nq, env.mj_model.nv, env.mj_model.nu

    nominal = jnp.array(env.nominal_ptarg, dtype=jnp.float64)
    qpos = jnp.array(env.mj_model.qpos0, dtype=jnp.float64)
    qpos = qpos.at[2].set(cfg["height"]).at[7:].set(nominal)
    quat = qpos[3:7] + cfg["quat_noise"] * jax.random.normal(k[0], (4,), dtype=jnp.float64)
    qpos = qpos.at[3:7].set(quat / jnp.linalg.norm(quat))
    qpos = qpos.at[7:].add(cfg["joint_noise"] * jax.random.normal(k[1], (nu,), dtype=jnp.float64))

    qvel = 0.02 * jax.random.normal(k[2], (nv,), dtype=jnp.float64)
    qvel = qvel.at[0].add(cfg.get("base_vx", 0.0))     # sliding regime 용 측방 속도

    actions = (nominal[None]
               + cfg["act_noise"] * jax.random.normal(k[3], (H, nu), dtype=jnp.float64))

    w_q = jax.random.normal(k[4], (H, nq), dtype=jnp.float64)
    w_v = jax.random.normal(jax.random.fold_in(k[4], 1), (H, nv), dtype=jnp.float64)
    return qpos, qvel, actions, w_q, w_v


FD_HS = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7)


def robust_fd(f):
    """중심차분을 여러 h 로 구하고 인접 h 끼리 가장 잘 맞는 구간을 채택한다.

    NOTE: 이건 Richardson 외삽이 **아니다**.
          Richardson 외삽은 D(h) = f' + c h^2 + O(h^4) 라는 걸 이용해
          (4 D(h/2) - D(h)) / 3 으로 h^2 항을 소거하는 기법이다.
          여기서 하는 건 그냥 **h 안정 구간 탐색** 이다: 여러 h 로 재보고
          값이 수렴해 있는 구간을 고른다. 접촉 구간에서는 h^2 오차 모델
          자체가 성립하지 않아서(모드 전환이 있음) 외삽이 오히려 위험하다.

    접촉 구간에서는 단일 h 유한차분을 신뢰할 수 없다.
    h 가 크면 접촉 모드가 바뀌고 (truncation), 작으면 stiff 한 constraint force
    때문에 catastrophic cancellation 이 난다.  실측상 같은 성분에서 h 만 바꿔도
    값이 0.55 -> 4.62 -> -84.7 로 튄다.

    returns (value, cond)  -- cond 는 인접 h 간 불일치. 작을수록 믿을 만하다.
    """
    vals = np.array([(f(h) - f(-h)) / (2 * h) for h in FD_HS])
    diffs = np.abs(np.diff(vals))
    b = int(np.argmin(diffs))
    return 0.5 * (vals[b] + vals[b + 1]), float(diffs[b])


def loss_of(fn, w_q, w_v):
    def L(q, v, a):
        qp, qv = fn(q, v, a)
        return jnp.sum(qp * w_q) + jnp.sum(qv * w_v)
    return L


# ---------------------------------------------------------------------------
# Test 1 : tangent map  (regime 무관)
# ---------------------------------------------------------------------------
def test_tangent_map(env, rng):
    print(f"\n{YEL}[1] tangent map  E(q) / E^+(q){RST}")
    qpos, _, _, _, _ = make_case(env, rng, "contact", 2)
    nv = env.mj_model.nv
    ok = True

    E = cg.tangent_to_ambient(qpos)
    Ep = cg.ambient_to_tangent(qpos)
    ok &= report("E^+ @ E == I_nv", rel_err(Ep @ E, jnp.eye(nv)), 1e-10)

    E_ad = jax.jacfwd(lambda d: cg.integrate_tangent(qpos, d))(
        jnp.zeros(nv, dtype=jnp.float64))
    ok &= report("E == d(integrate_tangent)/d(dtheta)|_0", rel_err(E, E_ad), 1e-10)

    # MuJoCo 규약 (quaternion 좌/우 곱, local vs global 각속도) 이 맞는지.
    # 이게 틀리면 gradient 가 quaternion 성분에서만 조용히 어긋난다.
    h = 1e-6
    E_mj = np.zeros((env.mj_model.nq, nv))
    for i in range(nv):
        v = np.zeros(nv); v[i] = 1.0
        qp = np.array(qpos, np.float64).copy()
        qm = np.array(qpos, np.float64).copy()
        mujoco.mj_integratePos(env.mj_model, qp, v, h)
        mujoco.mj_integratePos(env.mj_model, qm, v, -h)
        E_mj[:, i] = (qp - qm) / (2 * h)
    ok &= report("E == mj_integratePos jacobian", rel_err(E, E_mj), 1e-6)
    return ok


# ---------------------------------------------------------------------------
# Test 2 : forward 불변성
# ---------------------------------------------------------------------------
def test_forward(env, rng, H=8):
    print(f"\n{YEL}[2] forward 불변성  (custom_vjp 가 forward 를 바꾸지 않는가){RST}")
    ok = True
    for regime in REGIMES:
        print(f"  {CYA}{regime}{RST}")
        qpos, qvel, actions, _, _ = make_case(env, jax.random.fold_in(rng, REGIMES[regime]["seed"]), regime, H)

        traj = diffsim.rollout(env, qpos, qvel, actions)
        for name, backend in [("no-vjp", None), ("custom_vjp(mjx)", cg.mjx_jacobian)]:
            qp, qv = cg.make_rollout(env, backend)(qpos, qvel, actions)
            ok &= report(f"{name} forward == diffsim.rollout",
                         max(rel_err(qp, traj.qpos), rel_err(qv, traj.qvel)), 1e-12)

        # mjx.step 이 참조하는 이전 스텝 정보가 qpos/qvel/warmstart 뿐인가?
        # 이 가정이 깨지면 mjx_jacobian 이 carried Data 기준 미분과 달라진다.
        init = mjx.make_data(env.mjx_model).replace(qpos=qpos, qvel=qvel)
        d1 = diffsim.control_step(env.mjx_model, init, actions[0], env.n_substeps)
        d2 = diffsim.control_step(env.mjx_model, d1, actions[1], env.n_substeps)
        d2r = cg._step_from_state(env, d1.qpos, d1.qvel, d1.qacc_warmstart, actions[1])
        ok &= report("warmstart 재구성 == carried Data",
                     max(rel_err(d2.qpos, d2r.qpos), rel_err(d2.qvel, d2r.qvel)), 1e-12)
    return ok


# ---------------------------------------------------------------------------
# Test 3 : custom_vjp(mjx backend) == plain MJX autodiff
# ---------------------------------------------------------------------------
def test_vjp_vs_autodiff(env, rng, H=6):
    """custom_vjp(mjx_jacobian) vs plain MJX reverse-mode autodiff.

    flight 에서는 완전히 일치해야 한다.
    contact 에서는 O(1e-4) 차이가 난다 -- 버그가 아니라 서로 다른 미분 경로다:
      - plain reverse-mode 는 qacc_warmstart 체인을 통과해서 미분한다.
        warmstart 는 solver 의 초기값일 뿐이라 이 경로는 물리적 의미가 없고,
        iterations=1 (DynaFlow 현재 설정) 처럼 수렴이 덜 된 solver 에서만 남는다.
      - mjx_jacobian 은 warmstart 를 상수로 두고 forward-mode 로 미분한다.
    test 4 의 유한차분은 FD 가 잘 조건화된 성분에서 custom_vjp 쪽 손을 들어준다.
    """
    print(f"\n{YEL}[3] custom_vjp(mjx_jacobian) vs plain MJX autodiff{RST}")
    ok = True
    for regime in REGIMES:
        print(f"  {CYA}{regime}{RST}")
        qpos, qvel, actions, w_q, w_v = make_case(
            env, jax.random.fold_in(rng, REGIMES[regime]["seed"]), regime, H)

        Lb = loss_of(cg.make_rollout(env, None), w_q, w_v)
        Lc = loss_of(cg.make_rollout(env, cg.mjx_jacobian), w_q, w_v)
        gb = jax.grad(Lb, argnums=(0, 1, 2))(qpos, qvel, actions)
        gc = jax.grad(Lc, argnums=(0, 1, 2))(qpos, qvel, actions)

        # contact 에서는 warmstart 경로 차이만큼 벌어지므로 상한만 확인한다.
        tol = 1e-7 if regime == "flight" else 1e-2
        E = cg.tangent_to_ambient(qpos)
        ok &= report("dL/d(actions)", rel_err(gb[2], gc[2]), tol)
        ok &= report("dL/d(qvel0)", rel_err(gb[1], gc[1]), tol)
        ok &= report("dL/d(qpos0) (tangent 사영)", rel_err(E.T @ gb[0], E.T @ gc[0]), tol)
    return ok


def test_solver_iterations(env_unused, rng, H=4):
    """solver iterations 를 올렸을 때 어느 backward 가 살아남는가.

    MJX 의 constraint solver 는 lax.while_loop 이라 reverse-mode 미분이 안 된다.
    iterations=1 (DynaFlow 현재 설정) 에서만 우연히 통과할 뿐이다.
    custom_vjp 는 스텝별 forward-mode jacobian 을 쓰므로 제약이 없다.
    -> CI-MPC 실험에서 forward 정확도를 올리고 싶다면 이 scaffold 가 필수다.
    """
    print(f"\n{YEL}[5] solver iterations 에 대한 backward 견고성{RST}")
    ok = True
    for iters, ls in [(1, 5), (10, 10), (100, 50)]:
        env = diffsim.Env()
        env.mj_model.opt.iterations = iters
        env.mj_model.opt.ls_iterations = ls
        env.mjx_model = mjx.put_model(env.mj_model)
        qpos, qvel, actions, w_q, w_v = make_case(
            env, jax.random.fold_in(rng, REGIMES["contact"]["seed"]), "contact", H)
        line = f"    iterations={iters:<4d} ls={ls:<3d}"
        for name, backend in [("plain", None), ("custom_vjp", cg.mjx_jacobian)]:
            try:
                g = jax.grad(loss_of(cg.make_rollout(env, backend), w_q, w_v),
                             argnums=2)(qpos, qvel, actions)
                line += f"  {name}={GRN}OK{RST}(|g|={float(jnp.linalg.norm(g)):.4f})"
            except Exception:
                line += f"  {name}={RED}실패{RST}"
                if name == "custom_vjp":
                    ok = False
        print(line)
    print(f"    {YEL}-> plain reverse-mode 는 iterations>1 에서 죽는다 (while_loop).{RST}")
    return ok


# ---------------------------------------------------------------------------
# Test 4 : 유한차분 (ground truth)
# ---------------------------------------------------------------------------
def test_finite_difference(env, rng, backend, name, H=4):
    """h 안정 구간 탐색 방식 FD 를 ground truth 로 삼는다.

    FD 가 잘 조건화된 성분(인접 h 간 불일치가 작은 것)만 판정에 쓴다.
    접촉 구간에서 단일 h FD 를 그대로 믿으면 멀쩡한 gradient 도 FAIL 이 난다.
    """
    print(f"\n{YEL}[4] 유한차분(h 안정 구간 탐색) vs custom_vjp ({name}){RST}")
    ok = True
    for regime in REGIMES:
        print(f"  {CYA}{regime}{RST}")
        qpos, qvel, actions, w_q, w_v = make_case(
            env, jax.random.fold_in(rng, REGIMES[regime]["seed"]), regime, H)

        L = loss_of(cg.make_rollout(env, None), w_q, w_v)   # FD 는 항상 진짜 forward 로
        g_q, g_v, g_a = jax.grad(
            loss_of(cg.make_rollout(env, backend), w_q, w_v),
            argnums=(0, 1, 2))(qpos, qvel, actions)

        nv = env.mj_model.nv
        cond_tol = 1e-6      # FD 신뢰 기준 (인접 h 간 불일치)
        tol = 1e-6 if regime == "flight" else 1e-5

        def check(label, probes):
            nonlocal ok
            errs, n_used, n_skip = [], 0, 0
            for perturb, analytic in probes:
                fd, cond = robust_fd(perturb)
                scale = max(abs(fd), abs(analytic), 1.0)
                if cond / scale > cond_tol:
                    n_skip += 1
                    continue
                n_used += 1
                errs.append(abs(fd - analytic) / scale)
            if n_used == 0:
                print(f"    [{YEL}SKIP{RST}] {label:<38s} FD 조건 불량으로 전부 제외")
                return
            ok &= report(f"{label}  ({n_used}개 사용, {n_skip}개 제외)",
                         float(max(errs)), tol)
    
        check("dL/d(actions)", [
            (lambda h, i=i, j=j: L(qpos, qvel, actions.at[i, j].add(h)), float(g_a[i, j]))
            for i, j in [(0, 0), (0, 5), (H // 2, 3), (H - 1, 11)]])

        check("dL/d(qvel0)", [
            (lambda h, j=j: L(qpos, qvel.at[j].add(h), actions), float(g_v[j]))
            for j in [0, 2, 4, 7, 13]])

        # tangent 공간에서 섭동해야 quaternion 이 안 깨진다. E/E^+ 의 실질 검증.
        g_tan = cg.tangent_to_ambient(qpos).T @ g_q
        check("dL/d(q0_tangent)", [
            (lambda h, j=j: L(cg.integrate_tangent(
                qpos, h * jnp.zeros(nv, dtype=jnp.float64).at[j].set(1.0)), qvel, actions),
             float(g_tan[j]))
            for j in [0, 2, 3, 4, 5, 8, 15]])
    return ok


# ---------------------------------------------------------------------------
# Test 5 : contact impulse 추출 정합성 (CI-MPC 백엔드의 전제)
# ---------------------------------------------------------------------------
def test_contact_extraction(env, rng):
    print(f"\n{YEL}[5] contact impulse 추출  (CI-MPC 백엔드 전제){RST}")
    qpos, qvel, actions, _, _ = make_case(env, rng, "contact", 2)
    u = actions[0]
    ws = jnp.zeros(env.mj_model.nv, dtype=jnp.float64)
    sids = cg.foot_site_ids(env)

    lam, d = cg.foot_contact_impulse(env, qpos, qvel, u, ws, sids)
    J = cg.contact_jacobians(env, qpos, sids).reshape(-1, env.mj_model.nv)
    recon = J.T @ lam.reshape(-1) / env.control_dt

    n_act = int(np.sum(np.asarray(lam[:, 2]) > 1e-6))
    print(f"      활성 접촉 {n_act}/4,  normal impulse = "
          f"{np.array2string(np.asarray(lam[:,2]), precision=4)}")
    ok = report("sum_k J_k^T lam_k == qfrc_constraint",
                rel_err(recon, d.qfrc_constraint), 1e-3)
    if not ok:
        print(f"      {YEL}-> 발 이외 접촉이 있거나 site 대응이 틀렸다는 뜻.{RST}")
    return ok


# ---------------------------------------------------------------------------
# CI-MPC gradient 비교 (pass/fail 이 아니라 '얼마나 다른가')
# ---------------------------------------------------------------------------
def compare_cimpc(env, rng, H=4):
    print(f"\n{YEL}[6] CI-MPC gradient vs MJX gradient  (차이 측정){RST}")
    print(f"    CI-MPC gradient 는 의도적으로 다른 gradient 다. FD 와 일치할 이유가 없다.")
    print(f"    flight 는 접촉이 없어 dlambda=0 이므로 cos 가 1 에 가까워야 한다 (회귀 검사).")
    # sliding regime 은 여기서만 본다. tests 1~4 는 flight/contact 만.
    regimes = dict(REGIMES)
    regimes["sliding"] = dict(REGIMES["contact"], base_vx=4.0, seed=33)
    for regime in regimes:
        REGIMES.setdefault(regime, regimes[regime])
        qpos, qvel, actions, w_q, w_v = make_case(
            env, jax.random.fold_in(rng, regimes[regime]["seed"]), regime, H)
        try:
            gm = jax.grad(loss_of(cg.make_rollout(env, cg.mjx_jacobian), w_q, w_v),
                          argnums=2)(qpos, qvel, actions)
            gc = jax.grad(loss_of(cg.make_rollout(env, cg.cimpc_jacobian), w_q, w_v),
                          argnums=2)(qpos, qvel, actions)
            a, b = np.asarray(gm).ravel(), np.asarray(gc).ravel()
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
            ratio = float(np.linalg.norm(b) / (np.linalg.norm(a) + 1e-30))
            lam, _ = cg.foot_contact_impulse(env, qpos, qvel, actions[0],
                                             jnp.zeros(env.mj_model.nv, dtype=jnp.float64))
            mu = cg.foot_ids(env)[4]
            ln = np.asarray(lam[:, 2]); lt = np.linalg.norm(np.asarray(lam[:, :2]), axis=-1)
            act = ln > 1e-9
            Jn = cg.contact_jacobians(env, qpos)
            vt = np.linalg.norm(np.asarray(jnp.einsum('kdv,v->kd', Jn, qvel)[:, :2]), axis=-1)
            sl = act & (lt >= 0.95 * mu * np.maximum(ln, 1e-9)) & (vt > 1e-2)
            print(f"  {CYA}{regime:<8s}{RST} cos(g_mjx, g_cimpc) = {cos:+.4f}   "
                  f"|g_cimpc|/|g_mjx| = {ratio:.4f}"
                  f"   활성={act.astype(int)} sliding={sl.astype(int)}")
        except Exception as e:
            print(f"  {CYA}{regime:<8s}{RST} {RED}실패{RST}: {type(e).__name__}: {str(e)[:120]}")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def bench(env, rng, B=1024, H=16):
    print(f"\n{YEL}[bench] 실제 학습 shape  B={B}, H={H}, n_substeps={env.n_substeps}{RST}")
    print(f"        학습 1스텝당 control-step gradient {B*H:,}개")
    nv, nu = env.mj_model.nv, env.mj_model.nu
    qpos = jnp.tile(jnp.array(env.mj_model.qpos0, dtype=jnp.float64).at[2].set(0.27), (B, 1))
    qvel = jnp.zeros((B, nv), dtype=jnp.float64)
    actions = jnp.tile(jnp.array(env.nominal_ptarg, dtype=jnp.float64), (B, H, 1))

    for name, backend in [("plain MJX autodiff", None),
                          ("custom_vjp + mjx_jacobian", cg.mjx_jacobian)]:
        fn = cg.make_batched_rollout(env, backend)

        def L(a):
            qp, qv = fn(qpos, qvel, a)
            return jnp.sum(qp) + jnp.sum(qv)
        try:
            g = jax.jit(jax.grad(L))
            t0 = time.time(); g(actions).block_until_ready()
            compile_t = time.time() - t0
            t0 = time.time()
            for _ in range(3):
                g(actions).block_until_ready()
            dt = (time.time() - t0) / 3
            print(f"        {name:<28s} {dt*1000:9.1f} ms/step   (compile {compile_t:.1f}s)")
        except Exception as e:
            print(f"        {name:<28s} {RED}실패{RST}: {type(e).__name__}: {str(e)[:100]}")

    n = B * H
    print(f"\n        {YEL}참고 — CI-MPC C++ (Pinocchio, 단일스레드) = 70us/gradient (논문 Sec 5.3){RST}")
    print(f"              {n:,} x 70us = {n*70e-6:.2f} s/step  ->  "
          f"50,000 step = {n*70e-6*50000/3600:.0f} 시간 ({n*70e-6*50000/86400:.0f}일)")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cimpc", action="store_true", help="CI-MPC 백엔드 비교까지")
    ap.add_argument("--bench", action="store_true", help="타이밍만 측정")
    args = ap.parse_args()

    print(f"jax {jax.__version__}  mujoco {mujoco.__version__}  x64={jax.config.jax_enable_x64}")
    print(f"devices: {jax.devices()}")

    env = diffsim.Env()
    print(f"nq={env.mj_model.nq} nv={env.mj_model.nv} nu={env.mj_model.nu}  "
          f"sim_dt={env.simulation_dt} control_dt={env.control_dt} "
          f"n_substeps={env.n_substeps}")

    rng = jax.random.PRNGKey(0)

    if args.bench:
        bench(env, rng)
        return

    results = [
        ("tangent map",   test_tangent_map(env, jax.random.fold_in(rng, 1))),
        ("forward",       test_forward(env, jax.random.fold_in(rng, 2))),
        ("vjp==autodiff", test_vjp_vs_autodiff(env, jax.random.fold_in(rng, 3))),
        ("FD",            test_finite_difference(
            env, jax.random.fold_in(rng, 4), cg.mjx_jacobian, "mjx_jacobian")),
        ("solver 견고성",  test_solver_iterations(env, jax.random.fold_in(rng, 7))),
    ]

    if args.cimpc:
        results.append(("contact 추출", test_contact_extraction(env, jax.random.fold_in(rng, 5))))
        compare_cimpc(env, jax.random.fold_in(rng, 6))

    print(f"\n{'='*66}")
    n_ok = sum(1 for _, o in results if o)
    for name, o in results:
        print(f"  {GRN+'PASS'+RST if o else RED+'FAIL'+RST}  {name}")
    print(f"  {n_ok}/{len(results)} 통과")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
