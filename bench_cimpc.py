"""cimpc_jacobian 의 실제 학습 shape 메모리/속도 측정.

verify_grad.py 는 검증용이라 x64 를 강제하지만, 실학습은 float32 다.
여기서는 float32 기본으로 배치 크기를 늘려가며 어디서 터지는지 본다.

실행:
    python bench_cimpc.py               # float32, B 스윕
    python bench_cimpc.py --f64         # float64
    python bench_cimpc.py --b 1024      # 특정 배치만
    python bench_cimpc.py --full        # 1.4절 비교 조건 (a)(b)(c)(d') 의 rollout gradient 시간
"""
import argparse
import time
import os

import numpy as np
import jax

ap = argparse.ArgumentParser()
ap.add_argument("--f64", action="store_true")
ap.add_argument("--b", type=int, default=None)
ap.add_argument("--horizon", type=int, default=16)
ap.add_argument("--full", action="store_true", help="forward 교체 조건까지 포함")
ap.add_argument("--chunk", type=int, default=None)
ap.add_argument("--cond", type=str, default=None,
                help="--full 에서 조건 하나만: a | b | c2 | d | c4  (컴파일 OOM 분리용)")
args = ap.parse_args()
if args.f64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import diffsim
import cimpc_grad as cg

GRN, RED, YEL, RST = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def measure(env, backend, name, B, H, dtype, forward="mjx", fwd_n_sub=None):
    nv, nu = env.mj_model.nv, env.mj_model.nu
    qpos = jnp.tile(jnp.array(env.mj_model.qpos0, dtype=dtype).at[2].set(0.2865), (B, 1))
    qpos = qpos.at[:, 7:].set(jnp.array(env.nominal_ptarg, dtype=dtype))
    qvel = jnp.zeros((B, nv), dtype=dtype)
    actions = jnp.tile(jnp.array(env.nominal_ptarg, dtype=dtype), (B, H, 1))

    fn = cg.make_batched_rollout(env, backend, chunk=args.chunk, forward=forward, fwd_n_sub=fwd_n_sub)

    def L(a):
        qp, qv = fn(qpos, qvel, a)
        return jnp.sum(qp) + jnp.sum(qv)

    try:
        g = jax.jit(jax.grad(L))
        t0 = time.time()
        g(actions).block_until_ready()
        tc = time.time() - t0
        t0 = time.time()
        for _ in range(3):
            g(actions).block_until_ready()
        dt = (time.time() - t0) / 3
        peak = 0
        try:
            st = jax.devices()[0].memory_stats()
            peak = st.get("peak_bytes_in_use", 0) / 2**30
        except Exception:
            pass
        print(f"    {name:<26s} {GRN}OK{RST}  {dt*1000:8.1f} ms/step  "
              f"(compile {tc:5.1f}s, peak {peak:5.2f} GiB)")
        return True
    except Exception as e:
        msg = str(e).split("\n")[0][:80]
        print(f"    {name:<26s} {RED}실패{RST}  {type(e).__name__}: {msg}")
        return False


def main():
    dtype = jnp.float64 if args.f64 else jnp.float32
    env = diffsim.Env()
    H = args.horizon
    print(f"dtype={np.dtype(dtype).name}  H={H}  n_substeps={env.n_substeps}")
    print(f"devices: {jax.devices()}")
    try:
        print(f"GPU 메모리: {jax.devices()[0].memory_stats()['bytes_limit']/2**30:.1f} GiB")
    except Exception:
        pass

    batches = [args.b] if args.b else [64, 256, 1024]
    for B in batches:
        print(f"\n{YEL}B={B}  (control-step gradient {B*H:,}개/스텝){RST}")
        if args.full:
            from functools import partial
            jac_m = partial(cg.cimpc_jacobian, n_sub=2)
            jac_c2 = partial(cg.cimpc_jacobian, n_sub=2, forward="cimpc")
            jac_c4 = partial(cg.cimpc_jacobian, n_sub=4, forward="cimpc")
            conds = {"a": ("(a) MJX fwd + MJX autodiff", None, "mjx", None),
                     "b": ("(b) MJX fwd + CI grad n2", jac_m, "mjx", None),
                     "c2": ("(c) CI fwd n2 + CI grad n2", jac_c2, "cimpc", 2),
                     "d": ("(d') CI fwd n2 + autodiff", None, "cimpc", 2),
                     "c4": ("(c) CI fwd n4 + CI grad n4", jac_c4, "cimpc", 4)}
            keys = [args.cond] if args.cond else list(conds)
            for k in keys:
                name, backend, fwd, fns = conds[k]
                measure(env, backend, name, B, H, dtype, forward=fwd, fwd_n_sub=fns)
            continue
        for name, backend in [("plain MJX autodiff", None),
                              ("cimpc_jacobian", cg.cimpc_jacobian),
                              ("mjx_jacobian", cg.mjx_jacobian)]:
            measure(env, backend, name, B, H, dtype)


if __name__ == "__main__":
    main()
