"""cimpc_forward.py 검증.

    JAX_PLATFORMS=cpu python verify_forward.py            # 정확성 (x64)
    python verify_forward.py --bench                      # GPU 시간 측정 (f32)

검사
----
 1. C++ 원본 대조: `cimpc_ref/build/dump_gradient` 가 덤프한 M, J, nle, foot_pos 로
    같은 (v_free, P, phi) 를 만들어 `solve_contact_impulse` 를 돌리고
    C++ 의 lambda / contact_state / sliding_state / v_next 와 비교한다.
    clamping, sliding, separating, mixed, landing, partial 여섯 상태.
 2. `smooth_forward` 의 qacc_smooth == `mjx.forward` 의 qacc_smooth (Go1)
 3. Go1 서기 0.3초: 발 침투 없음, near-rigid MuJoCo C 와 trunk 높이 5mm 이내
 4. Go1 낙하: 착지 impulse 가 침투 없이 처리되고 접촉이 유지되는지
 5. 고정 반복수라 plain autodiff 가 통과하는지 (NaN 없음)
 6. 데이터셋 전이 재현 (32 세그먼트 x 16 스텝): CI-MPC n_sub=4 가 MJX 보다 나쁘지 않아야

서기 테스트에 대한 주의: 이 자세(kp=20 servo)는 어느 시뮬레이터에서도 주저앉고,
rigid contact 에서는 0.4초쯤 앞발이 마찰 포화로 미끄러져 넘어진다.
버그가 아니라 물리다 (near-rigid MuJoCo 도 같은 시점에 앞발이 뜨며 -9deg 기울고,
mu=100 이면 넘어지지 않는다). 그래서 0.3초 구간만 near-rigid MuJoCo 와 대조한다.
"""
import os
import sys
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--bench", action="store_true")
ap.add_argument("--B", type=int, default=1024)
ap.add_argument("--H", type=int, default=16)
ap.add_argument("--wiring", action="store_true", help="[7] custom_vjp 배선 검사만")
args = ap.parse_args()

import jax
if not args.bench:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cimpc_ref"))
from parse_dump import parse

import diffsim
import cimpc_grad as cg
import cimpc_forward as cf

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "cimpc_ref", "build", "dump_gradient")
NV, NQ, FNUM = 18, 19, 4
STAND_Z = 0.2865
RED, GRN, RST = "\033[91m", "\033[92m", "\033[0m"
results = []


def check(name, err, tol):
    ok = err < tol
    results.append(ok)
    print(f"  [{GRN+'PASS' if ok else RED+'FAIL'}{RST}] {name}: {err:.3e} (tol {tol:.0e})")
    return ok


# ---------------------------------------------------------------------------
# 1. C++ 대조
# ---------------------------------------------------------------------------
Q_STAND = [0, 0, 0.485, 0, 0, 0, 1,
           -0.05, 0.7854, -1.5708, 0.05, 0.7854, -1.5708,
           -0.05, 0.7854, -1.5708, 0.05, 0.7854, -1.5708]


def write_state(path, q, v, tau, rho=0.0):
    with open(path, "w") as f:
        for arr in (q, v, tau):
            f.write(" ".join(repr(float(x)) for x in arr) + "\n")
        f.write(repr(float(rho)) + "\n")


def cpp_dump(q, v, tau, tag):
    path = os.path.join(HERE, "cimpc_ref", "build", f"_fwd_{tag}.txt")
    write_state(path, q, v, tau)
    out = subprocess.run([EXE, path], capture_output=True, text=True).stdout
    dump = path.replace(".txt", "_dump.txt")
    open(dump, "w").write(out)
    return parse(dump)


def test_cpp():
    print("\n[1] C++ SolveContactImpulse 대조")
    if not os.path.exists(EXE):
        print("  dump_gradient 없음 -- cimpc_ref/build.sh 먼저"); results.append(False); return
    pitch = 0.1
    q_tilt = list(Q_STAND); q_tilt[2] = 0.50
    q_tilt[3:7] = [0.0, np.sin(pitch / 2), 0.0, np.cos(pitch / 2)]      # xyzw (Pinocchio)
    q_air = list(Q_STAND); q_air[2] = 0.6
    z = [0.0] * NV
    v_slide = list(z); v_slide[0] = 1.0
    v_land = list(z); v_land[2] = -1.0
    v_partial = list(z); v_partial[0] = 0.3; v_partial[1] = 0.1
    cases = {
        "clamping":   (Q_STAND, z, z),
        "sliding":    (Q_STAND, v_slide, z),
        "separating": (q_air, z, z),
        "mixed":      (q_tilt, z, z),
        "landing":    (Q_STAND, v_land, z),
        "partial":    (Q_STAND, v_partial, z),
    }
    solve = jax.jit(cf.solve_contact_impulse, static_argnames=("n_outer", "n_bisect"))
    for tag, (q, v, tau) in cases.items():
        bl = cpp_dump(q, v, tau, tag)
        M, J, nle, fp = bl["mass_matrix"], bl["foot_jacobian"], bl["nle"].ravel(), bl["foot_pos"]
        dt, mu = bl["meta"][0, 0], bl["meta"][0, 1]
        v = np.array(v, float); tau = np.array(tau, float)
        Minv = np.linalg.inv(M)
        v_free = v + Minv @ (tau - nle) * dt
        Jk = J.reshape(FNUM, 3, NV)
        MinvJT = np.stack([Minv @ Jk[k].T for k in range(FNUM)])
        phi = fp[2, :]
        lam, contact, sliding = solve(jnp.array(Jk), jnp.array(MinvJT), jnp.array(v_free),
                                      jnp.array(phi), dt, mu)
        lam = np.asarray(lam)
        v_next = v_free + sum(MinvJT[k] @ lam[k] for k in range(FNUM))
        lam_cpp = bl["lambda_lw"].T
        v_cpp = bl["xnext"].ravel()[NQ:]
        cs, ss = bl["contact_state"].ravel().astype(int), bl["sliding_state"].ravel().astype(int)
        print(f"  -- {tag}: C++ contact={cs.tolist()} sliding={ss.tolist()}  "
              f"JAX contact={np.asarray(contact).astype(int).tolist()} "
              f"sliding={np.asarray(sliding).astype(int).tolist()}")
        scale = max(np.abs(lam_cpp).max(), 1e-12)
        check(f"{tag}: lambda", np.abs(lam - lam_cpp).max() / scale, 1e-5)
        check(f"{tag}: v_next", np.abs(v_next - v_cpp).max() / max(np.abs(v_cpp).max(), 1e-12), 1e-5)
        check(f"{tag}: state flags", float((np.asarray(contact).astype(int) != cs).sum()
                                           + (np.asarray(sliding).astype(int) != ss).sum()), 0.5)


# ---------------------------------------------------------------------------
# Go1 상태
# ---------------------------------------------------------------------------
def stand_state(env, height=STAND_Z, seed=0, joint_noise=0.0):
    rng = np.random.default_rng(seed)
    q = np.zeros(NQ); q[2] = height; q[3] = 1.0
    q[7:] = np.asarray(env.nominal_ptarg) + joint_noise * rng.standard_normal(12)
    return jnp.array(q), jnp.zeros(NV)


def test_smooth(env):
    print("\n[2] smooth_forward == mjx.forward (qacc_smooth, M)")
    q, v = stand_state(env, joint_noise=0.05)
    u = jnp.array(env.nominal_ptarg, dtype=q.dtype)
    d0 = mjx.make_data(env.mjx_model).replace(qpos=q, qvel=v + 0.1, ctrl=u)
    d_full = mjx.forward(env.mjx_model, d0)
    d_sm = cf.smooth_forward(env.mjx_model, d0)
    check("qacc_smooth", float(jnp.abs(d_full.qacc_smooth - d_sm.qacc_smooth).max()), 1e-9)
    check("full_m", float(jnp.abs(mjx.full_m(env.mjx_model, d_full)
                                  - mjx.full_m(env.mjx_model, d_sm)).max()), 1e-12)
    check("site_xpos", float(jnp.abs(d_full.site_xpos - d_sm.site_xpos).max()), 1e-12)


def _mujoco_near_rigid_stance(env, n_steps):
    """MuJoCo C 를 near-rigid 접촉으로 설정해 같은 서기를 돌린다 (CI-MPC forward 의 기준).

    MJX soft contact (solref 0.01, 1cm 침투) 는 hard contact 의 기준이 못 된다.
    dt=1e-3, solref=(0.002,1), solimp=(0.99,0.99,0.001), Newton 100 iter 로
    침투 <0.1mm 짜리 근사 rigid 접촉을 만든다.
    """
    import mujoco
    m = mujoco.MjModel.from_xml_path("robots/go1/scene.xml")
    m.opt.timestep = 1e-3; m.opt.iterations = 100; m.opt.ls_iterations = 50
    for n in ["FL", "FR", "RL", "RR"]:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        m.geom_solref[g] = [0.002, 1.0]; m.geom_solimp[g] = [0.99, 0.99, 0.001, 0.5, 2]
    d = mujoco.MjData(m)
    q0, _ = stand_state(env)
    d.qpos[:] = np.asarray(q0); d.ctrl[:] = np.asarray(env.nominal_ptarg)
    out = []
    for _ in range(n_steps):
        for _ in range(20):
            mujoco.mj_step(m, d)
        out.append(d.qpos.copy())
    return np.stack(out)


def test_stand(env):
    print("\n[3] Go1 서기 0.3초 (15 control step, n_sub=4) -- near-rigid MuJoCo 와 대조")
    # 주의: 이 자세는 kp=20 servo 가 약해서 어느 시뮬레이터에서도 주저앉는다 (MJX 도 1초에 8cm).
    # 그리고 rigid contact 에서는 0.4초쯤 앞발이 마찰 포화로 미끄러져 넘어질 수 있다
    # (near-rigid MuJoCo 도 같은 시점에 앞발이 완전히 뜨며 -9deg 까지 기운다).
    # 그래서 그 이전 구간 (0.3초) 의 trunk 높이 궤적만 비교한다.
    H = 15
    q0, v0 = stand_state(env)
    u = jnp.tile(jnp.array(env.nominal_ptarg, dtype=q0.dtype), (H, 1))
    qs, vs, aux = cf.make_rollout(env, n_sub=4)(q0, v0, u)
    ref = _mujoco_near_rigid_stance(env, H)
    soft = diffsim.rollout(env, q0, v0, u).qpos
    phi = np.asarray(aux["phi"])
    dz_rigid = np.abs(np.asarray(qs[:, 2]) - ref[:, 2]).max()
    dz_soft = np.abs(np.asarray(qs[:, 2]) - np.asarray(soft[:, 2])).max()
    print(f"  trunk z @t=10: CI-MPC {float(qs[9,2]):.4f}  near-rigid MuJoCo {ref[9,2]:.4f}  MJX soft {float(soft[9,2]):.4f}")
    print(f"  |dz| max vs near-rigid = {dz_rigid:.4f},  vs MJX soft = {dz_soft:.4f}")
    check("발 침투 (phi >= -1e-3)", float(max(0.0, -phi.min())), 1e-3)
    check("near-rigid MuJoCo 와 trunk z 차이 < 5mm", float(dz_rigid), 5e-3)
    check("NaN 없음", float(jnp.isnan(qs).any() | jnp.isnan(vs).any()), 0.5)


def test_drop(env):
    print("\n[4] Go1 낙하 (10cm 위에서, 0.5초)")
    # 발 외 링크는 접촉이 없는 모델이라 (CI-MPC 와 동일) collapse 이후 거동은 MJX 와 다르다.
    # 여기서는 착지 impulse 가 침투 없이 처리되는지만 본다.
    q0, v0 = stand_state(env, height=STAND_Z + 0.10)
    u = jnp.tile(jnp.array(env.nominal_ptarg, dtype=q0.dtype), (25, 1))
    ro = cf.make_rollout(env, n_sub=4)
    qs, vs, aux = ro(q0, v0, u)
    phi = np.asarray(aux["phi"])
    contact = np.asarray(aux["contact"]).any(axis=1)
    first_c = int(np.argmax(contact))
    fz = np.asarray(aux["lam"][:, :, 2]).sum(-1) / (env.control_dt / 4)
    print(f"  첫 접촉 스텝 {first_c}, 최대 수직력 {fz.max():.0f} N (mg = {float(env.mj_model.body_subtreemass[1]*9.81):.0f} N), phi 최소 {phi.min():.2e}")
    check("착지 후 침투 (phi >= -1e-3)", float(max(0.0, -phi.min())), 1e-3)
    check("착지 후 접촉 유지", float(1.0 - contact[first_c:].mean()), 1e-9)
    check("NaN 없음", float(jnp.isnan(qs).any() | jnp.isnan(vs).any()), 0.5)


def test_dataset(env):
    print("\n[6] 데이터셋 전이 재현 (32 세그먼트, H=16): CI-MPC n_sub=4 가 MJX 보다 나쁘지 않아야")
    import loadData
    ds = loadData.SequenceDataset(horizon=16, max_path_length=250, action_scale=1.0,
                                  cache_path="cache/go1_locomotion.pkl")
    obs = np.asarray(ds.fields.observations, dtype=np.float64)
    act = np.asarray(ds.fields.actions, dtype=np.float64)
    rng = np.random.default_rng(0); N, H = 32, 16
    eps = rng.integers(0, obs.shape[0], N); t0s = rng.integers(0, 250 - H - 1, N)
    Q0 = jnp.array(obs[eps, t0s, :19]); V0 = jnp.array(obs[eps, t0s, 19:])
    U = jnp.array(np.stack([act[e, t:t + H] for e, t in zip(eps, t0s)]))
    TRUE = np.stack([obs[e, t + 1:t + H + 1] for e, t in zip(eps, t0s)])
    W = np.array([1, 1, 1, 50, 50, 50, 50] + [10] * 12 + [50, 50, 50, 5, 5, 20] + [0.1] * 12, float)

    def loss_like(pred):   # 학습 loss 와 같은 거리 [xy raw, 나머지 normalized]
        pn = np.asarray(ds.normalizer.normalize(pred, 'observations'))
        tn = np.asarray(ds.normalizer.normalize(TRUE, 'observations'))
        p = np.concatenate([pred[..., :2] - pred[..., :1, :2], pn[..., 2:]], -1)
        t = np.concatenate([TRUE[..., :2] - TRUE[..., :1, :2], tn[..., 2:]], -1)
        return float(np.mean((p - t) ** 2 * W))

    ref = diffsim.batched_rollout(env, Q0, V0, U)
    l_mjx = loss_like(np.concatenate([np.asarray(ref.qpos), np.asarray(ref.qvel)], -1))
    res = {}
    for n_sub in (4, 2, 1):
        qs, vs, aux = cf.make_batched_rollout(env, n_sub=n_sub)(Q0, V0, U)
        res[n_sub] = loss_like(np.concatenate([np.asarray(qs), np.asarray(vs)], -1))
        phi = np.asarray(aux["phi"])
        print(f"  CI-MPC n_sub={n_sub}: loss-like {res[n_sub]:.5f}   phi min {phi.min():.1e}, 침투>1mm 비율 {(phi < -1e-3).mean():.3f}")
    print(f"  MJX            : loss-like {l_mjx:.5f}")
    check("CI-MPC n_sub=4 <= MJX (loss-like)", res[4] / l_mjx, 1.0)


def test_autodiff(env):
    print("\n[5] plain autodiff 통과 (고정 반복수)")
    q0, v0 = stand_state(env)
    u = jnp.tile(jnp.array(env.nominal_ptarg, dtype=q0.dtype), (4, 1))
    ro = cf.make_rollout(env, n_sub=2)
    L = lambda u: jnp.sum(ro(q0, v0, u)[0] ** 2)
    g = jax.grad(L)(u)
    print(f"  |g| = {float(jnp.linalg.norm(g)):.4e}")
    check("grad finite", float(jnp.logical_not(jnp.isfinite(g).all())), 0.5)


def test_wiring(env):
    """[7] cimpc_grad.make_rollout(forward="cimpc") 배선.

    (a) forward 불변성: custom_vjp 롤아웃 == cimpc_forward.make_rollout (bit-identical)
    (b) custom_vjp (cimpc_jacobian, rho=0, forward="cimpc") vs plain autodiff through
        hard-contact forward. 같은 동역학이므로 flight 는 정확히 1.0, clamping 접촉은
        active-set 의 implicit derivative 라 매우 가까워야 한다.
    (c) 유한차분 (x64, h=1e-6) vs custom_vjp
    """
    print("\n[7] custom_vjp 배선 (forward=cimpc, n_sub=2)")
    from functools import partial
    H, n_sub = 4, 2
    hard = cf.make_rollout(env, n_sub=n_sub)
    plain = cg.make_rollout(env, None, forward="cimpc", fwd_n_sub=n_sub)
    for rho in (0.0,):
        jac = partial(cg.cimpc_jacobian, rho=rho, n_sub=n_sub, forward="cimpc")
        custom = cg.make_rollout(env, jac, forward="cimpc", fwd_n_sub=n_sub)

    q_air, v_air = stand_state(env, height=STAND_Z + 0.3)
    q_st, v_st = stand_state(env)
    u_nom = jnp.tile(jnp.array(env.nominal_ptarg, dtype=q_st.dtype), (5, 1))
    qs, vs, _ = hard(q_st, v_st, u_nom)             # 5 스텝 진행해 발이 접지된 상태
    q_c, v_c = qs[-1], vs[-1]
    rng = np.random.default_rng(3)
    W = jnp.array(rng.standard_normal((H, 19)))
    Wv = jnp.array(rng.standard_normal((H, 18)))

    def L(fn, q0, v0, u):
        qp, vp = fn(q0, v0, u)
        return jnp.sum(qp * W) + jnp.sum(vp * Wv)

    for tag, (q0, v0) in {"flight": (q_air, v_air), "contact": (q_c, v_c)}.items():
        u = jnp.tile(jnp.array(env.nominal_ptarg, dtype=q0.dtype), (H, 1)) + 0.02 * jnp.array(rng.standard_normal((H, 12)))
        qp_c, vp_c = custom(q0, v0, u)
        qp_h, vp_h, _ = hard(q0, v0, u)
        check(f"{tag}: forward 불변성", float(jnp.abs(qp_c - qp_h).max() + jnp.abs(vp_c - vp_h).max()), 1e-15)

        g_c = jax.grad(L, argnums=(1, 2, 3))(custom, q0, v0, u)
        g_p = jax.grad(L, argnums=(1, 2, 3))(plain, q0, v0, u)
        gc = jnp.concatenate([g.ravel() for g in g_c]); gp = jnp.concatenate([g.ravel() for g in g_p])
        cos = float(gc @ gp / (jnp.linalg.norm(gc) * jnp.linalg.norm(gp)))
        ratio = float(jnp.linalg.norm(gc) / jnp.linalg.norm(gp))
        print(f"  {tag}: custom_vjp vs plain autodiff  cos={cos:.6f}  |g| 비={ratio:.4f}")
        check(f"{tag}: cos > 0.999", 1.0 - cos, 1e-3)
        check(f"{tag}: |g| 비 within 1%", abs(ratio - 1.0), 1e-2)

        # 유한차분: action 성분 6개, qvel 성분 3개.
        # 접촉 구간의 단일 h FD 는 신뢰할 수 없다 (CIMPC_NOTES 4.3 절: 모드 전환 / stiff phi/dt).
        # 그래서 contact 는 "custom_vjp 가 plain autodiff 보다 FD 에서 더 멀지 않다" 로 본다.
        h = 1e-6
        e_c, e_p = [], []
        f0 = lambda uu: float(L(custom, q0, v0, uu))
        for (t, j) in [(0, 1), (0, 5), (1, 2), (2, 8), (3, 11), (1, 0)]:
            e = jnp.zeros_like(u).at[t, j].set(h)
            fd = (f0(u + e) - f0(u - e)) / (2 * h)
            e_c.append(abs(fd - float(g_c[2][t, j])) / max(abs(fd), 1e-8))
            e_p.append(abs(fd - float(g_p[2][t, j])) / max(abs(fd), 1e-8))
        for j in (0, 2, 7):
            e = jnp.zeros_like(v0).at[j].set(h)
            fd = (float(L(custom, q0, v0 + e, u)) - float(L(custom, q0, v0 - e, u))) / (2 * h)
            e_c.append(abs(fd - float(g_c[1][j])) / max(abs(fd), 1e-8))
            e_p.append(abs(fd - float(g_p[1][j])) / max(abs(fd), 1e-8))
        mc, mp = float(np.median(e_c)), float(np.median(e_p))
        print(f"  {tag}: FD 상대오차 median  custom_vjp={mc:.2e}  plain autodiff={mp:.2e}  (max custom {max(e_c):.2e})")
        if tag == "flight":
            check(f"{tag}: FD vs custom_vjp (median < 1e-4)", mc, 1e-4)
        else:
            check(f"{tag}: FD 오차 custom_vjp <= 1.5 x plain", mc / max(mp, 1e-12), 1.5)


def bench(env):
    print(f"\n[bench] B={args.B} H={args.H} f32  ({jax.devices()[0]})")
    q0, v0 = stand_state(env)
    B, H = args.B, args.H
    q0 = jnp.tile(q0[None], (B, 1)).astype(jnp.float32)
    v0 = jnp.tile(v0[None], (B, 1)).astype(jnp.float32)
    u = jnp.tile(jnp.array(env.nominal_ptarg, dtype=jnp.float32), (B, H, 1))

    def timeit(name, fn, n=5):
        t0 = time.time(); out = fn(); jax.block_until_ready(out)
        tc = time.time() - t0
        ts = []
        for _ in range(n):
            t0 = time.time(); out = fn(); jax.block_until_ready(out); ts.append(time.time() - t0)
        print(f"  {name:28s} compile {tc:6.1f}s   run {1e3*np.median(ts):8.1f} ms")

    timeit("MJX batched_rollout (fwd)", lambda: diffsim.batched_rollout(env, q0, v0, u).qpos)
    for n_sub in (1, 2, 4):
        ro = cf.make_batched_rollout(env, n_sub=n_sub)
        timeit(f"CI-MPC fwd n_sub={n_sub}", lambda: ro(q0, v0, u)[0])


if __name__ == "__main__":
    from mujoco import mjx
    env = diffsim.Env()
    if args.bench:
        bench(env)
    elif args.wiring:
        test_wiring(env)
        print(f"\n{sum(results)}/{len(results)} PASS")
    else:
        test_cpp()
        test_smooth(env)
        test_stand(env)
        test_drop(env)
        test_autodiff(env)
        test_dataset(env)
        test_wiring(env)
        n = sum(results)
        print(f"\n{'='*60}\n{n}/{len(results)} PASS")
