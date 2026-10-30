"""CI-MPC gradient 의 rho 스윕 학습.

목적
----
DynaFlow 의 동역학 gradient 를 CI-MPC relaxed gradient 로 바꿨을 때
motion imitation 이 되는지, 그리고 완화 변수 rho 가 얼마나 영향을 주는지 본다.

rho -> 0 이면 Werling et al. (2021) 의 hard-contact gradient 와 같아지므로
rho 를 0 부터 올리며 성능 곡선을 보는 게 가장 깔끔한 ablation 이다.

forward 는 전 조건에서 **완전히 동일** 하다 (MJX). backward 만 바뀐다.
따라서 loss 곡선 차이는 전적으로 gradient 차이에서 온다.

실행:
    python rho_sweep.py --steps 2000 --batch 256
    python rho_sweep.py --only mjx          # baseline 만
    python rho_sweep.py --n-sub 1           # substep 정합 끄기 (4배 빠름)
"""
import argparse
import json
import os
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

import module
import loadData
import diffsim
import cimpc_grad as cg
from diffsim import state_to_obs
from config import PREFIX

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--horizon", type=int, default=16)
ap.add_argument("--chunk", type=int, default=None, help="배치 청크 (메모리 상한)")
ap.add_argument("--n-sub", type=int, default=None, help="backward substep 수 (기본: forward 와 일치)")
ap.add_argument("--n-subs", type=int, nargs="*", default=None,
                help="여러 n_sub 를 한 번에 비교. 주면 --rhos 의 첫 값으로 고정하고 "
                     "n_sub 만 바꿔 돌린다 (적분 정합 비용/효과 측정용).")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--seeds", type=int, nargs="*", default=[0])
ap.add_argument("--log-every", type=int, default=50)
ap.add_argument("--only", type=str, default=None, help="한 조건만")
ap.add_argument("--rhos", type=float, nargs="*", default=[0.0, 1e-6, 1e-4, 1e-2])
ap.add_argument("--lr-scaled", type=float, default=None,
                help="주면 rho=<lr-scaled> 조건을 lr 2배로 하나 더 돌린다. "
                     "CI-MPC gradient 크기가 MJX 의 약 절반이라, 격차가 "
                     "gradient 품질 차이인지 단순 스케일 차이인지 가른다.")
ap.add_argument("--out", type=str, default=os.path.join(PREFIX, "rho_sweep_results"))
args = ap.parse_args()

COMMAND_MASK_PROB = 0.1
D_MODEL, N_HEADS, DEPTH = 384, 3, 3

# 손실 가중치는 main.py 와 동일하게 둔다 (조건 간 비교가 목적).
WEIGHTS = jnp.array([1, 1, 1, 50, 50, 50, 50,
                     10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                     50, 50, 50, 5, 5, 20,
                     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


def build_loss(env, dataset, rollout_fn, horizon):
    """main.py:100-175 의 loss_fn 과 동일. rollout 만 갈아끼운다."""
    step_w = jnp.concatenate([jnp.array([5.0]), jnp.ones(horizon - 3), jnp.array([2.5])])
    weights = WEIGHTS[None, None, :] * step_w[None, :, None]

    def loss_fn(model, batch, rng):
        traj, cond, command = batch
        B = traj.shape[0]
        rng, key_t, key_s0, key_o, key_mask = jax.random.split(rng, 5)

        mask = jax.random.uniform(key_mask, shape=(B,)) < COMMAND_MASK_PROB
        command = jnp.where(mask[:, None], command.at[:, 0].set(0.0), command)

        t = jax.random.uniform(key_t, shape=(B, 1))
        true_norm = traj[..., env.act_dim:]
        true_unnorm = dataset.normalizer.unnormalize(true_norm, 'observations')
        true_states = jnp.concatenate([true_norm[..., :2], true_unnorm[..., 2:]], axis=-1)

        noise = jax.random.uniform(key_o, shape=cond[0].shape) * 2 - 1
        scales = jnp.zeros_like(noise)
        scales = scales.at[:, 7:19].set(0.05).at[:, 19:22].set(0.1).at[:, 22:25].set(0.25)
        true_states = true_states.at[:, 0].add(noise * scales)

        obs = state_to_obs(true_states[:, 0, :19], true_states[:, 0, 19:])
        s0 = jax.random.normal(key_s0, shape=true_norm.shape)
        st = t[:, None, :] * true_norm + (1 - t)[:, None, :] * s0

        actions = env.scale_action(model(st, t, jnp.concatenate([obs, command], axis=-1)))

        q0, dq0 = jnp.split(true_states[:, 0], [env.mj_model.nq], axis=-1)
        pred_qpos, pred_qvel = rollout_fn(q0, dq0, actions)
        pred_states = jnp.concatenate([pred_qpos, pred_qvel], axis=-1)[:, :-1, :]

        pred_norm = dataset.normalizer.normalize(pred_states, 'observations')
        pred_obs = jnp.concatenate([pred_states[..., :2], pred_norm[..., 2:]], axis=-1)

        se = (pred_obs - true_norm[:, 1:, :]) ** 2
        loss = jnp.mean(se * weights)

        err = (true_states[:, 1:, :] - pred_states) ** 2
        info = {"rmse_state": jnp.mean(jnp.sqrt(err), axis=(0, 1))}
        return loss, (info, rng)

    return loss_fn


def run_one(tag, rollout_fn, env, dataset, data_stream, args, seed, lr):
    """한 조건 x 한 시드 학습. loss 이력을 돌려준다."""
    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    model = module.DiT1d(in_dim=env.state_dim, out_dim=env.act_dim,
                         attr_dim=env.obs_dim + 4, d_model=D_MODEL,
                         n_heads=N_HEADS, depth=DEPTH, rngs=nnx.Rngs(default=sub))
    model.initialize_weights()
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(50), optax.adam(lr)))

    loss_fn = build_loss(env, dataset, rollout_fn, args.horizon)

    @nnx.jit
    def train_step(model, optimizer, batch, rng):
        (loss, (info, rng)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch, rng)
        gnorm = optax.global_norm(grads)
        optimizer.update(grads)
        return loss, info, gnorm, rng

    rng = jax.random.PRNGKey(seed)
    hist = {"step": [], "loss": [], "gnorm": [], "rmse_base_pos": [], "rmse_joint_pos": []}
    t0 = time.time()
    for step in range(args.steps):
        batch = jax.tree_util.tree_map(lambda x: jnp.array(x), next(data_stream))
        loss, info, gnorm, rng = train_step(model, optimizer, batch, rng)
        if step % args.log_every == 0 or step == args.steps - 1:
            l, g = float(loss), float(gnorm)
            r = np.asarray(info["rmse_state"])
            hist["step"].append(step); hist["loss"].append(l); hist["gnorm"].append(g)
            hist["rmse_base_pos"].append(float(r[0:3].mean()))
            hist["rmse_joint_pos"].append(float(r[7:19].mean()))
            if not np.isfinite(l):
                print(f"    step {step:5d}  loss=NaN -> 중단")
                break
            print(f"    step {step:5d}  loss={l:10.5f}  |g|={g:9.3f}  "
                  f"base_pos={r[0:3].mean():.4f}  joint_pos={r[7:19].mean():.4f}  "
                  f"({time.time()-t0:6.1f}s)")
    hist["wall_time"] = time.time() - t0
    return hist


def main():
    os.makedirs(args.out, exist_ok=True)
    env = diffsim.Env()
    n_sub = env.n_substeps if args.n_sub is None else args.n_sub
    print(f"batch={args.batch} horizon={args.horizon} steps={args.steps} "
          f"chunk={args.chunk} n_sub={n_sub} (forward n_substeps={env.n_substeps}) "
          f"seeds={args.seeds}")

    dataset = loadData.SequenceDataset(
        horizon=args.horizon, max_path_length=250, action_scale=1.0,
        cache_path=os.path.join(PREFIX, 'cache', 'go1_locomotion.pkl'))

    # (태그, backend, lr). backend 는 조건당 한 번만 만들어 시드 간 재컴파일을 막는다.
    conds = []
    if args.n_subs:
        # n_sub 비교 모드: rho 는 고정하고 backward substep 수만 바꾼다.
        # forward 는 항상 n_substeps=4 로 동일하므로, n_sub 가 작을수록
        # 적분 단위가 어긋난다 (실측 cos: n_sub=4 -> 1.000, 2 -> 0.999, 1 -> 0.968).
        # 그 어긋남이 실제 loss 에 영향을 주는지 보는 게 목적이다.
        rho = args.rhos[0]
        if args.only in (None, "mjx"):
            conds.append(("mjx", None, args.lr))
        for ns in args.n_subs:
            conds.append((f"rho={rho:g},n_sub={ns}",
                          partial(cg.cimpc_jacobian, rho=rho, n_sub=ns), args.lr))
    else:
        if args.only in (None, "mjx"):
            conds.append(("mjx", None, args.lr))
        for rho in args.rhos:
            tag = f"rho={rho:g}"
            if args.only in (None, tag):
                conds.append((tag, partial(cg.cimpc_jacobian, rho=rho, n_sub=n_sub), args.lr))
        if args.lr_scaled is not None:
            rho = args.lr_scaled
            conds.append((f"rho={rho:g},lr2x",
                          partial(cg.cimpc_jacobian, rho=rho, n_sub=n_sub), args.lr * 2))

    results = {}
    for tag, backend, lr in conds:
        rollout_fn = cg.make_batched_rollout(env, backend, chunk=args.chunk)
        for seed in args.seeds:
            key = f"{tag}|seed{seed}"
            print(f"\n{'='*70}\n[{key}]  lr={lr:g}  "
                  f"backend={'plain MJX autodiff' if backend is None else 'CI-MPC'}")
            np.random.seed(42 + seed)     # 시드별로 데이터 순서도 바꾼다
            stream = loadData.make_data_stream(dataset, args.batch)
            try:
                results[key] = run_one(tag, rollout_fn, env, dataset, stream,
                                       args, seed, lr)
                results[key]["tag"] = tag
                results[key]["seed"] = seed
                results[key]["lr"] = lr
            except Exception as e:
                print(f"    실패: {type(e).__name__}: {str(e)[:200]}")
                results[key] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            with open(os.path.join(args.out, "results.json"), "w") as f:
                json.dump({"args": vars(args), "results": results}, f, indent=2)

    # --- 요약: 조건별로 시드 평균 +- 표준편차 ---------------------------
    print(f"\n{'='*70}\n요약 -- 후반 25% 스텝의 loss 평균 (시드 간 평균 +- 표준편차)")
    by_tag = {}
    for key, h in results.items():
        if "error" in h:
            continue
        L, S = np.array(h["loss"]), np.array(h["step"])
        late = L[S >= args.steps * 0.75].mean()
        by_tag.setdefault(h["tag"] + (f" lr={h['lr']:g}" if h["lr"] != args.lr else ""),
                          []).append(late)
    base = None
    print(f"  {'조건':<20s} {'loss':>10s} {'±std':>9s} {'n':>3s} {'MJX 대비':>10s}")
    for tag, vals in by_tag.items():
        v = np.array(vals)
        if tag == "mjx":
            base = v.mean()
    for tag, vals in by_tag.items():
        v = np.array(vals)
        rel = f"{v.mean()/base:6.3f}배" if base else "-"
        print(f"  {tag:<20s} {v.mean():10.5f} {v.std():9.5f} {len(v):3d} {rel:>10s}")
    print(f"\n결과 저장: {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
