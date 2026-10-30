"""CI-MPC gradient 로 DynaFlow 를 학습하고 **체크포인트를 저장** 한다.

rho_sweep.py 는 loss 곡선만 남기고 모델을 저장하지 않는다. 학습 결과를 직접 보려면
(eval_metrics.py 의 TRE/SAE 표, tester.py 의 영상) orbax 체크포인트가 필요하다.
main.py 와 같은 구조 (Adam + clip 50, EMA 0.995) 에 rollout 만 갈아끼웠다.

실행:
    # CI-MPC gradient, rho=1e-4, backward substep 2 (6.2절 권장)
    python train_cimpc.py --backend cimpc --rho 1e-4 --n-sub 2 --steps 20000 --batch 256

    # 같은 설정의 MJX baseline (공정 비교용: 같은 batch/steps)
    python train_cimpc.py --backend mjx --steps 20000 --batch 256

    # forward 도 hard contact (CI-MPC 전체 교체), backward CI-MPC gradient, n_sub=2
    python train_cimpc.py --forward cimpc --backend cimpc --rho 1e-4 --n-sub 2 --steps 20000 --batch 256
    # hard contact forward + plain autodiff. 주의: sliding 접촉에서는 bisection 을 통과한 autodiff 가
    # 정답 미분이 아니다 (verify_cimpc.py [B], CIMPC_NOTES 9절). rho=0 비교군은 --backend cimpc --rho 0 을 쓸 것.
    python train_cimpc.py --forward cimpc --backend mjx --n-sub 2 --steps 20000 --batch 256

저장 위치: checkpoints/<시각>_<name>/state_<step>   (EMA 상태, tester.py/eval_metrics.py 호환)
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import datetime
import json
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx
import orbax.checkpoint as ocp

import module
import loadData
import diffsim
import cimpc_grad as cg
from diffsim import state_to_obs
from config import PREFIX

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["mjx", "cimpc"], default="cimpc")
ap.add_argument("--rho", type=float, default=1e-4)
ap.add_argument("--n-sub", type=int, default=2,
                help="backward substep 수. --forward mjx 면 forward 는 항상 4 (0.005 x 4)")
ap.add_argument("--forward", choices=["mjx", "cimpc"], default="mjx",
                help="forward 동역학. cimpc = hard contact (cimpc_forward.py). "
                     "이때 forward substep 수는 --fwd-n-sub (기본: --n-sub 와 같게)")
ap.add_argument("--fwd-n-sub", type=int, default=None)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--horizon", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--chunk", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ema-decay", type=float, default=0.995)
ap.add_argument("--ema-start", type=int, default=1000)
ap.add_argument("--save-every", type=int, default=1000)
ap.add_argument("--log-every", type=int, default=100)
ap.add_argument("--name", type=str, default=None)
ap.add_argument("--ckpt-root", type=str, default=os.path.join(PREFIX, "checkpoints"))
args = ap.parse_args()

COMMAND_MASK_PROB = 0.1
D_MODEL, N_HEADS, DEPTH = 384, 3, 3
WEIGHTS = jnp.array([1, 1, 1, 50, 50, 50, 50,
                     10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                     50, 50, 50, 5, 5, 20,
                     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


def build_loss(env, dataset, rollout_fn, horizon):
    """main.py:100-175 와 동일. rollout 만 갈아끼운다."""
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


def main():
    fwd_n_sub = args.fwd_n_sub or args.n_sub
    if args.forward == "cimpc" and args.backend == "cimpc" and fwd_n_sub != args.n_sub:
        raise ValueError("--forward cimpc 에서는 backward n_sub 가 forward 와 같아야 한다 "
                         f"(--n-sub {args.n_sub} vs --fwd-n-sub {fwd_n_sub})")
    if args.forward == "cimpc" and args.backend == "mjx":
        print("[경고] --forward cimpc --backend mjx: hard-contact solver 의 plain autodiff 는 sliding 접촉에서 "
              "bisection 의 bracket 선택을 미분하므로 정답 미분이 아니다 (FD 와 ~10% 어긋남, verify_cimpc.py [B]). "
              "rho=0 비교군은 --backend cimpc --rho 0 을 권장한다.")
    name = args.name or ((f"CIMPC_rho{args.rho:g}_nsub{args.n_sub}" if args.backend == "cimpc"
                          else "MJX_baseline")
                         + (f"_fwdcimpc{fwd_n_sub}" if args.forward == "cimpc" else ""))
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.join(args.ckpt_root, f"{stamp}_{name}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"save_dir: {save_dir}")

    env = diffsim.Env()
    dataset = loadData.SequenceDataset(
        horizon=args.horizon, max_path_length=250, action_scale=1.0,
        cache_path=os.path.join(PREFIX, "cache", "go1_locomotion.pkl"))
    np.random.seed(42 + args.seed)
    stream = loadData.make_data_stream(dataset, args.batch)

    backend = None if args.backend == "mjx" else partial(
        cg.cimpc_jacobian, rho=args.rho, n_sub=args.n_sub, forward=args.forward)
    rollout_fn = cg.make_batched_rollout(env, backend, chunk=args.chunk,
                                         forward=args.forward,
                                         fwd_n_sub=fwd_n_sub if args.forward == "cimpc" else None)
    loss_fn = build_loss(env, dataset, rollout_fn, args.horizon)

    key = jax.random.PRNGKey(args.seed)
    key, sub = jax.random.split(key)
    model = module.DiT1d(in_dim=env.state_dim, out_dim=env.act_dim,
                         attr_dim=env.obs_dim + 4, d_model=D_MODEL,
                         n_heads=N_HEADS, depth=DEPTH, rngs=nnx.Rngs(default=sub))
    model.initialize_weights()
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(50), optax.adam(args.lr)))

    @nnx.jit
    def train_step(model, optimizer, batch, rng):
        (loss, (info, rng)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch, rng)
        optimizer.update(grads)
        return loss, info, rng

    checkpointer = ocp.StandardCheckpointer()
    rng = jax.random.PRNGKey(args.seed)
    ema_state = None
    hist = []
    t0 = time.time()
    for step in range(args.steps + 1):
        batch = jax.tree_util.tree_map(lambda x: jnp.array(x), next(stream))
        loss, info, rng = train_step(model, optimizer, batch, rng)

        _, state = nnx.split(model)
        if ema_state is None or step < args.ema_start:   # ema_start=0 이어도 첫 스텝은 복사
            ema_state = state
        else:
            ema_state = jax.tree_util.tree_map(
                lambda e, o: args.ema_decay * e + (1 - args.ema_decay) * o, ema_state, state)

        if step % args.log_every == 0:
            l = float(loss); r = np.asarray(info["rmse_state"])
            hist.append({"step": step, "loss": l,
                         "rmse_base_pos": float(r[0:3].mean()),
                         "rmse_joint_pos": float(r[7:19].mean())})
            print(f"step {step:6d}  loss={l:9.5f}  base_pos={r[0:3].mean():.4f}  "
                  f"joint_pos={r[7:19].mean():.4f}  ({time.time()-t0:7.0f}s)")
            if not np.isfinite(l):
                print("loss=NaN -> 중단"); break
            with open(os.path.join(save_dir, "history.json"), "w") as f:
                json.dump(hist, f)

        if step % args.save_every == 0 and step > 0:
            checkpointer.save(os.path.join(save_dir, f"state_{step}"), ema_state)
            print(f"  saved state_{step}")

    checkpointer.wait_until_finished()
    print(f"완료: {save_dir}")


if __name__ == "__main__":
    main()
