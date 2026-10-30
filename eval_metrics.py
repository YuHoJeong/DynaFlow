"""DynaFlow 논문 (Sec. V-A) 의 TRE / SAE 를 체크포인트별로 계산해서 표로 만든다.

정의 (논문 eq.7 근방)
--------------------
SAE(x_i, x_{i+1}) = || x_{i+1} - f(x_i, ID(x_i, x_{i+1})) ||
    ID(x_i, x_{i+1}) = argmin_u || x_{i+1} - f(x_i, u) ||,  u in ctrl_range
    ID 는 미분가능 시뮬레이터 f (diffsim.control_step) 를 통해 gradient descent 로 푼다.
    -> 물리적 일관성. 0 이면 동역학적으로 도달 가능한 전이.
    DynaFlow 의 출력은 MJX 롤아웃 자체이므로 ID solver 정밀도 수준의 값이 나와야 정상이다.
    데이터셋 자체의 SAE 는 "데이터의 본질적 비일관성" 이다 (Isaac 에서 수집됨).

TRE(X_hat, X_tilde) = MSE 를 궤적 전체 (H-1 스텝 x 37 차원) 에 대해 평균
    같은 초기상태 x0 / 같은 command / **같은 초기 Gaussian 노이즈** 로 생성한 궤적과 참조 궤적.
    -> 분포 유사도. 체크포인트끼리 비교할 때 노이즈 seed 를 고정해야 의미가 있다.

거리 공간
---------
두 지표 모두 학습 loss 와 같은 공간에서 잰다: [상대 xy (raw), 나머지 35차원 normalized].
raw qpos/qvel 은 차원별 스케일이 극단적으로 달라서 (위치 m vs 관절속도 rad/s)
Euclidean 거리가 의미를 갖지 않는다. 해석을 돕기 위해 raw 단위의 그룹별 RMSE 도 같이 낸다.

실행
----
    # MJX baseline 체크포인트 하나
    python eval_metrics.py --ckpt mjx=checkpoints/2026-08-18-15-07-27_DynaFlow:50000

    # 여러 개 비교 (이름=경로:epoch)
    python eval_metrics.py \\
        --ckpt mjx=checkpoints/2026-08-18-15-07-27_DynaFlow:50000 \\
        --ckpt cimpc=checkpoints/<cimpc 학습 결과>:<epoch> \\
        --n-traj 256 --num-steps 1

    CPU 스모크 테스트:  JAX_PLATFORMS=cpu python eval_metrics.py --n-traj 4 --id-iters 5 ...
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx
import orbax.checkpoint as ocp
from mujoco import mjx

import module
import loadData
import diffsim
from diffsim import state_to_obs
from flow_matching import DynaFlow
from config import PREFIX

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", action="append", default=[],
                help="이름=디렉토리:epoch  (여러 번 지정 가능)")
ap.add_argument("--n-traj", type=int, default=256, help="평가할 참조 궤적 수")
ap.add_argument("--horizon", type=int, default=16)
ap.add_argument("--num-steps", type=int, default=1,
                help="flow Euler 스텝 수. 1 이면 배포와 같은 단발 예측 (main.py 의 render_* 와 동일)")
ap.add_argument("--id-iters", type=int, default=200, help="SAE 역동역학 gradient descent 반복 수")
ap.add_argument("--id-lr", type=float, default=0.01)
ap.add_argument("--chunk", type=int, default=64, help="SAE 계산 시 전이 배치 청크")
ap.add_argument("--seed", type=int, default=0, help="참조 궤적 선택 + 초기 노이즈 seed (체크포인트 간 공유)")
ap.add_argument("--skip-sae", action="store_true")
ap.add_argument("--out", type=str, default=os.path.join(PREFIX, "eval_results"))
args = ap.parse_args()

D_MODEL, N_HEADS, DEPTH = 384, 3, 3
GROUPS = {  # raw 37차원 상태의 그룹 (해석용 RMSE)
    "base_pos": slice(0, 3), "base_ori": slice(3, 7), "joint_pos": slice(7, 19),
    "base_linvel": slice(19, 22), "base_angvel": slice(22, 25), "joint_vel": slice(25, 37),
}


# ---------------------------------------------------------------------------
# 데이터 / 모델
# ---------------------------------------------------------------------------
def load_model(env, ckpt_dir, epoch):
    """tester.py 와 같은 방식으로 EMA 상태를 복원한다."""
    abstract = nnx.eval_shape(lambda: module.DiT1d(
        in_dim=env.state_dim, out_dim=env.act_dim, attr_dim=env.obs_dim + 4,
        d_model=D_MODEL, n_heads=N_HEADS, depth=DEPTH, rngs=nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract)
    # 저장 당시(GPU) 의 sharding 메타데이터 대신 현재 디바이스로 복원한다.
    # 이걸 안 주면 CPU 나 다른 GPU 에서 "Topology mismatch" 로 실패한다.
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    abstract_state = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding), abstract_state)
    state = ocp.StandardCheckpointer().restore(
        os.path.join(ckpt_dir, f"state_{epoch}"), abstract_state)
    return nnx.merge(graphdef, state)


def pick_reference(dataset, n, horizon, seed):
    """참조 궤적 n 개를 고정 seed 로 뽑는다. 체크포인트 간에 같은 궤적을 쓴다."""
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)
    batch = [dataset[int(i)] for i in idxs]
    trajs = jnp.stack([b.trajectories for b in batch])          # (n, H, 12+37)
    cmds = jnp.stack([b.command for b in batch])                 # (n, 4)
    obs_norm = trajs[..., 12:]                                   # [rel xy raw, 35 normed]
    return obs_norm, cmds, idxs


def to_raw_state(obs_norm, normalizer):
    """학습 loss 공간 [rel xy raw, normed 35] -> raw 상태 (qpos 19 + qvel 18).
    main.py 의 true_states 구성과 동일."""
    unnorm = normalizer.unnormalize(obs_norm, "observations")
    return jnp.concatenate([obs_norm[..., :2], unnorm[..., 2:]], axis=-1)


def to_loss_space(raw, normalizer):
    """raw 상태 -> 학습 loss 공간 (main.py 의 pred_obs 구성과 동일)."""
    normed = normalizer.normalize(raw, "observations")
    return jnp.concatenate([raw[..., :2], normed[..., 2:]], axis=-1)


# ---------------------------------------------------------------------------
# TRE
# ---------------------------------------------------------------------------
def generate(model, env, normalizer, ref_norm, cmds, num_steps, seed):
    """같은 x0, 같은 command, 같은 초기 노이즈로 생성. (n, H, 37) loss 공간 + 액션."""
    fm = DynaFlow(model, env, normalizer)
    x0_raw = to_raw_state(ref_norm[:, 0], normalizer)
    x1, actions = fm.sample(jax.random.PRNGKey(seed), x0_raw, cmds,
                            horizon=ref_norm.shape[1], num_steps=num_steps)
    return x1, actions


def tre(gen_norm, ref_norm, normalizer):
    """궤적별 TRE (loss 공간 MSE, 초기상태 제외) 와 raw 그룹별 RMSE."""
    diff = gen_norm[:, 1:] - ref_norm[:, 1:]
    per_traj = jnp.mean(diff ** 2, axis=(1, 2))                   # (n,)
    g_raw = to_raw_state(gen_norm, normalizer)[:, 1:]
    r_raw = to_raw_state(ref_norm, normalizer)[:, 1:]
    groups = {k: float(jnp.sqrt(jnp.mean((g_raw[..., s] - r_raw[..., s]) ** 2)))
              for k, s in GROUPS.items()}
    return np.asarray(per_traj), groups


# ---------------------------------------------------------------------------
# SAE  (역동역학을 gradient descent 로)
# ---------------------------------------------------------------------------
def make_sae_fn(env, normalizer, iters, lr):
    """전이 배치 (x_i, x_{i+1}) -> SAE. loss 공간 거리. u 는 ctrl_range 안에서 최적화."""
    nq = env.mj_model.nq
    lo, hi = env.ctrl_range[:, 0], env.ctrl_range[:, 1]

    def step_state(x_raw, u):
        q, v = x_raw[:nq], x_raw[nq:]
        d = mjx.make_data(env.mjx_model).replace(qpos=q, qvel=v)
        d = diffsim.control_step(env.mjx_model, d, u, env.n_substeps)
        return jnp.concatenate([d.qpos, d.qvel])

    def dist(u, x_raw, x_next_norm):
        pred = to_loss_space(step_state(x_raw, u)[None], normalizer)[0]
        return jnp.sum((pred - x_next_norm) ** 2)

    opt = optax.adam(lr)

    def solve_one(x_raw, x_next_norm, u_init):
        # 최선값을 추적한다. Adam 은 gradient 가 ~0 이어도 첫 스텝이 ~lr 크기라
        # 이미 최적인 초기값(모델 자신의 액션)에서 오히려 멀어질 수 있다.
        # best 를 들고 가면 SAE 가 반복 수에 대해 단조감소하고, iters=0 이면
        # 초기 액션 그대로의 오차 (= 재시뮬레이션 바닥값) 가 된다.
        def body(carry, _):
            u, st, best_d, best_u = carry
            d, g = jax.value_and_grad(dist)(u, x_raw, x_next_norm)
            better = d < best_d
            best_d = jnp.where(better, d, best_d)
            best_u = jnp.where(better, u, best_u)
            upd, st = opt.update(g, st, u)
            u = jnp.clip(u + upd, lo, hi)
            return (u, st, best_d, best_u), None
        d0 = dist(u_init, x_raw, x_next_norm)
        (u, _, best_d, best_u), _ = jax.lax.scan(
            body, (u_init, opt.init(u_init), d0, u_init), None, length=iters)
        d_last = dist(u, x_raw, x_next_norm)
        best_d = jnp.minimum(best_d, d_last)
        best_u = jnp.where(d_last <= best_d, u, best_u)
        return jnp.sqrt(best_d), best_u

    return jax.jit(jax.vmap(solve_one))


def sae(sae_fn, traj_norm, normalizer, u_init, chunk):
    """traj_norm (n, H, 37) 의 모든 전이에 대해 SAE. u_init (n, H-1, 12) 는 초기 액션."""
    raw = to_raw_state(traj_norm, normalizer)
    x = raw[:, :-1].reshape(-1, raw.shape[-1])
    xn = traj_norm[:, 1:].reshape(-1, traj_norm.shape[-1])
    u0 = u_init.reshape(-1, u_init.shape[-1])
    out = []
    for s in range(0, x.shape[0], chunk):
        e, _ = sae_fn(x[s:s + chunk], xn[s:s + chunk], u0[s:s + chunk])
        out.append(np.asarray(e))
    return np.concatenate(out).reshape(traj_norm.shape[0], -1)   # (n, H-1)


# ---------------------------------------------------------------------------
def stats(a):
    a = np.asarray(a).ravel()
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std()), "max": float(a.max())}


def fmt_row(name, tre_s, sae_s, groups=None):
    t = f"{tre_s['mean']:.5f} / {tre_s['median']:.5f}" if tre_s else "—"
    s = f"{sae_s['mean']:.5f} / {sae_s['median']:.5f}" if sae_s else "—"
    g = ""
    if groups:
        g = " | " + " | ".join(f"{groups[k]:.4f}" for k in GROUPS)
    return f"| {name} | {t} | {s}{g} |"


def main():
    os.makedirs(args.out, exist_ok=True)
    env = diffsim.Env()
    dataset = loadData.SequenceDataset(
        horizon=args.horizon, max_path_length=250, action_scale=1.0,
        cache_path=os.path.join(PREFIX, "cache", "go1_locomotion.pkl"))
    normalizer = dataset.normalizer

    ref_norm, cmds, idxs = pick_reference(dataset, args.n_traj, args.horizon, args.seed)
    ref_actions = None   # 데이터셋 액션은 Isaac 규약이라 ID 초기값으로 쓰지 않는다
    print(f"참조 궤적 {args.n_traj}개, horizon={args.horizon}, num_steps={args.num_steps}, "
          f"seed={args.seed}  devices={jax.devices()}")

    sae_fn = None if args.skip_sae else make_sae_fn(env, normalizer, args.id_iters, args.id_lr)
    nominal = jnp.broadcast_to(env.nominal_ptarg, (args.n_traj, args.horizon - 1, env.act_dim))

    results = {"args": vars(args), "ref_indices": idxs.tolist(), "rows": {}}

    # --- 참조 데이터셋 자체의 SAE (본질적 비일관성) ---------------------
    if sae_fn is not None:
        t0 = time.time()
        e = sae(sae_fn, ref_norm, normalizer, nominal, args.chunk)
        results["rows"]["dataset"] = {"sae": stats(e), "tre": None, "groups": None}
        print(f"[dataset]  SAE mean={e.mean():.5f} median={np.median(e):.5f}  ({time.time()-t0:.0f}s)")

    # --- 체크포인트별 -------------------------------------------------------
    for spec in args.ckpt:
        name, rest = spec.split("=", 1)
        ckpt_dir, epoch = rest.rsplit(":", 1)
        ckpt_dir = ckpt_dir if os.path.isabs(ckpt_dir) else os.path.join(PREFIX, ckpt_dir)
        t0 = time.time()
        model = load_model(env, ckpt_dir, int(epoch))
        gen_norm, actions = generate(model, env, normalizer, ref_norm, cmds,
                                     args.num_steps, args.seed)
        per_traj, groups = tre(gen_norm, ref_norm, normalizer)
        row = {"tre": stats(per_traj), "groups": groups, "sae": None,
               "ckpt": f"{ckpt_dir}:{epoch}"}
        msg = f"[{name}]  TRE mean={per_traj.mean():.5f} median={np.median(per_traj):.5f}"
        if sae_fn is not None:
            e = sae(sae_fn, gen_norm, normalizer, actions[:, :-1], args.chunk)
            row["sae"] = stats(e)
            msg += f"  SAE mean={e.mean():.5f} median={np.median(e):.5f}"
        print(msg + f"  ({time.time()-t0:.0f}s)")
        results["rows"][name] = row

    # --- 표 ---------------------------------------------------------------
    hdr = "| 조건 | TRE mean / median | SAE mean / median | " + " | ".join(GROUPS) + " |"
    sep = "|" + "---|" * (3 + len(GROUPS))
    lines = [hdr, sep]
    for name, r in results["rows"].items():
        lines.append(fmt_row(name, r["tre"], r["sae"], r["groups"]))
    table = "\n".join(lines)
    print("\n" + table)
    print("\n(그룹 열은 raw 단위 RMSE: base_pos[m], base_ori[quat], joint_pos[rad], "
          "base_linvel[m/s], base_angvel[rad/s], joint_vel[rad/s])")

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out, "metrics.md"), "w") as f:
        f.write(table + "\n")
    print(f"\n저장: {args.out}/metrics.{{json,md}}")


if __name__ == "__main__":
    main()
