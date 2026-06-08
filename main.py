import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx
import orbax.checkpoint as ocp

import module
import loadData
import diffsim
from utils import Timer, Logger, save_code_snapshot
from diffsim import state_to_obs
import datetime

from config import PREFIX


EXPERIMENT_NAME = "DynaFlow"


start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


# Paths
save_dir = os.path.join(PREFIX, 'checkpoints', f'{start_time}_{EXPERIMENT_NAME}')

log_dir = save_dir
print("Base path :", PREFIX)
print("Save path :", save_dir)
print("Log path :", log_dir)


files_to_save = [
    os.path.join(PREFIX, 'main.py'),
    os.path.join(PREFIX, 'module.py'),
    os.path.join(PREFIX, 'loadData.py'),
    os.path.join(PREFIX, 'diffsim.py'),
    os.path.join(PREFIX, 'utils.py'),
    os.path.join(PREFIX, 'robots/go1/go1.xml')
]
save_code_snapshot(files_to_save, save_dir)

# Hyperparameters
HORIZON = 16

seed = 0
BATCH_SIZE = 1024
LR = 2e-4
EMA_DECAY = 0.995
EMA_START = 1000
MAX_STEPS = 50001

LOG_FREQ = 100 # loss logging
RECORD_FREQ = 1000 # video
SAVE_FREQ = 1000 # save weight

np.random.seed(42)

cache_path = os.path.join(PREFIX, 'cache', 'go1_locomotion.pkl')
dataset = loadData.SequenceDataset(
    horizon=HORIZON,
    max_path_length=250,
    action_scale=1.0,
    cache_path=cache_path
)
data_stream = loadData.make_data_stream(dataset, BATCH_SIZE)

env = diffsim.Env()
print(f"sim_dt: {env.simulation_dt}\ncontrol_dt: {env.control_dt}")

key = jax.random.PRNGKey(42)  # PRNG seed for reproducibility.
key, subkey = jax.random.split(key)  # Split the JAX PRNG key for initialization.

# DiT configuration
D_MODEL = 384
N_HEADS = 3
DEPTH = 3
COMMAND_MASK_PROB = 0.1

model = module.DiT1d(
    in_dim=env.state_dim,
    out_dim=env.act_dim,
    attr_dim = env.obs_dim + 4,
    d_model = D_MODEL,
    n_heads = N_HEADS,
    depth=DEPTH,
    rngs=nnx.Rngs(default=subkey)
)
model.initialize_weights()



optimizer = nnx.Optimizer(model, optax.chain(
    optax.clip_by_global_norm(50),
    optax.adam(LR)
))

@nnx.jit  # Automatic state management
def train_step(model, optimizer, batch, rng: nnx.Rngs):

  def loss_fn(model: nnx.Module, batch, rng: nnx.Rngs):
    # unwrap batch
    traj, cond, command = batch
    B, H, _ = traj.shape

    rng, key_t, key_s0, key_o, key_mask = jax.random.split(rng,5)

    command_mask = jax.random.uniform(key_mask, shape=(B,)) < COMMAND_MASK_PROB
    command_with_gait_zeroed = command.at[:, 0].set(0.0)
    command = jnp.where(command_mask[:, None], command_with_gait_zeroed, command)

    # sample flow time t
    t = jax.random.uniform(key_t, shape=(B, 1))
    true_norm = traj[..., env.act_dim:]
    true_unnorm = dataset.normalizer.unnormalize(true_norm, 'observations')
    true_states = jnp.concatenate([true_norm[...,:2], true_unnorm[...,2:]], axis=-1)

    # for init state randomization
    state_noise = jax.random.uniform(key_o, shape=cond[0].shape) * 2 - 1
    noise_scales = jnp.zeros_like(state_noise)
    noise_scales = noise_scales.at[:, 7:19].set(0.05)
    noise_scales = noise_scales.at[:, 19:22].set(0.1)
    noise_scales = noise_scales.at[:, 22:25].set(0.25)

    state_noise = state_noise * noise_scales
    true_states = true_states.at[:, 0].add(state_noise)

    obs = state_to_obs(true_states[:,0, :19], true_states[:,0, 19:])

    # sample initial state trajectory
    s0 = jax.random.normal(key_s0, shape=true_norm.shape)
    st = t[:, None, :] * true_norm + (1 - t)[:, None, :] * s0

    # inference action trajectory
    attr = jnp.concatenate([obs, command], axis=-1)
    actions_norm = model(st, t, attr)
    # actions = actions_norm * ACTION_SCALE
    actions = env.scale_action(actions_norm)

    # rollout
    q0, dq0 = jnp.split(true_states[:, 0], [env.mj_model.nq], axis=-1)
    rollout_traj = diffsim.batched_rollout(env, q0, dq0, actions)
    pred_states = jnp.concatenate([rollout_traj.qpos, rollout_traj.qvel], axis=-1)[:,:-1,:]

    pred_norm = dataset.normalizer.normalize(pred_states, 'observations')
    pred_obs = jnp.concatenate([pred_states[...,:2], pred_norm[...,2:]], axis=-1)

    target = true_norm[:,1:,:]
    se = (pred_obs - target) ** 2

    weights = jnp.array([1, 1, 1, 50, 50, 50, 50,
                         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                         50, 50, 50, 5, 5, 20,
                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])[None, None, :]

    step_weights = jnp.concatenate([
        jnp.array([5.0]),
        jnp.ones(HORIZON-3),
        jnp.array([2.5])
    ])[None, :, None]
    weights = weights * step_weights

    wse = se * weights
    loss = jnp.mean(wse)

    targ_states = true_states[:,1:,:]
    err = (targ_states - pred_states) ** 2
    rmse_dim = jnp.mean(jnp.sqrt(err), axis=(0,1))
    info = {
        "wse":         jnp.mean(wse, axis=(0,1)),
        "rmse_state":  rmse_dim,
    }

    next_rng = rng

    return loss, (info, next_rng)

  # update
  (loss, (info, rng)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch, rng)
  optimizer.update(grads)

  return loss, info, rng

timer = Timer()
logger = Logger(log_dir)


rng = jax.random.PRNGKey(seed)

ema_state = None

for epoch in range(MAX_STEPS):
    # print('Epoch %d' % epoch)
    batch = next(data_stream)
    batch = jax.tree_util.tree_map(lambda x: jnp.array(x), batch)

    loss, info, rng = train_step(model=model, optimizer=optimizer, batch=batch, rng=rng)

    # Logging
    if epoch % LOG_FREQ == 0:
        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(epoch))
        print('{:<40} {:>8.4f}'.format("Loss: ", loss))
        print('{:<40} {:>8.4f}'.format("Time elapsed in this iteration: ", timer()))
        print('----------------------------------------------------\n')

        total_wse_sum = jnp.sum(info["wse"])
        loss_contribution = info["wse"] / (total_wse_sum + 1e-8)

        logger.scalar('train/loss', float(loss), epoch)

        logger.scalar('error/base_pos', float(jnp.mean(info["rmse_state"][0:3])), epoch)
        logger.scalar('error/base_ori', float(jnp.mean(info["rmse_state"][3:7])), epoch)
        logger.scalar('error/joint_pos', float(jnp.mean(info["rmse_state"][7:19])), epoch)
        logger.scalar('error/base_linvel', float(jnp.mean(info["rmse_state"][19:22])), epoch)
        logger.scalar('error/base_angvel', float(jnp.mean(info["rmse_state"][22:25])), epoch)
        logger.scalar('error/joint_vel', float(jnp.mean(info["rmse_state"][25:37])), epoch)

        logger.scalar('contribution/base_pos', float(jnp.sum(loss_contribution[0:3])), epoch)
        logger.scalar('contribution/base_ori', float(jnp.sum(loss_contribution[3:7])), epoch)
        logger.scalar('contribution/joint_pos', float(jnp.sum(loss_contribution[7:19])), epoch)
        logger.scalar('contribution/base_linvel', float(jnp.sum(loss_contribution[19:22])), epoch)
        logger.scalar('contribution/base_angvel', float(jnp.sum(loss_contribution[22:25])), epoch)
        logger.scalar('contribution/joint_vel', float(jnp.sum(loss_contribution[25:37])), epoch)

    _, state = nnx.split(model)

    if epoch < EMA_START:
        ema_state = state
    else:
        # ema_state ← α·ema_state + (1−α)·online_state
        ema_state = jax.tree_util.tree_map(
            lambda e, o: EMA_DECAY * e + (1 - EMA_DECAY) * o,
            ema_state, state
        )

    if epoch % SAVE_FREQ == 0:
        # save
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(save_dir + f'/state_{epoch}', ema_state)
        print("Saved checkpoint to {}".format(save_dir))

    # Evaluation / Rendering
    if epoch % RECORD_FREQ == 0:
        if epoch % SAVE_FREQ == 0:
            # load
            # Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
            abstract_model = nnx.eval_shape(lambda: module.DiT1d(
                                                in_dim=env.state_dim,
                                                out_dim=env.act_dim,
                                                attr_dim = env.obs_dim + 4,
                                                d_model = D_MODEL,
                                                n_heads = N_HEADS,
                                                depth=DEPTH,
                                                rngs=nnx.Rngs(0)
                                                )
                                            )
            graphdef, abstract_state = nnx.split(abstract_model)
            state_restored = checkpointer.restore(save_dir + f'/state_{epoch}', abstract_state)
            testmodel = nnx.merge(graphdef, state_restored)
            print("Loaded checkpoint from {}".format(save_dir))

        else:
            testmodel = model
            print("Evaluate current model")

        print('----------------------EVALUATING-----------------------')
        q0 = np.array([
            0.0, 0.0, 0.27,  # base pos
            1.0, 0.0, 0.0, 0.0,  # base orientation (quat)
            0.0, 0.9, -1.8,  # FR: abduction, hip, knee
            0.0, 0.9, -1.8,  # FL
            0.0, 0.9, -1.8,  # RR
            0.0, 0.9, -1.8  # RL
        ])
        dq0 = np.zeros(env.mjx_model.nv)

        rng, rhc_rng = jax.random.split(rng)

        video_name = save_dir + f"/rollout/trot/{epoch}.mp4"
        env.render_openloop(
            testmodel,
            q0, dq0,
            200,
            v_command=[1, 1.0, 0, 0],
            rng=rhc_rng,
            name=video_name,
            window=16,
        )

        video_name = save_dir + f"/rollout/bound/{epoch}.mp4"
        env.render_openloop(
            testmodel,
            q0, dq0,
            200,
            v_command=[2, 1.0, 0, 0],
            rng=rhc_rng,
            name=video_name,
            window=16,
        )
        print('------------------------------------------------------\n')


        print('----------------------EVALUATING-----------------------')
        q0 = np.array([
            0.0, 0.0, 0.27,  # base pos
            1.0, 0.0, 0.0, 0.0,  # base orientation (quat)
            0.0, 0.9, -1.8,  # FR: abduction, hip, knee
            0.0, 0.9, -1.8,  # FL
            0.0, 0.9, -1.8,  # RR
            0.0, 0.9, -1.8  # RL
        ])
        dq0 = np.zeros(env.mjx_model.nv)

        rng, rhc_rng = jax.random.split(rng)

        video_name = save_dir + f"/rhc/trot/{epoch}.mp4"
        env.render_rhc(
            testmodel,
            q0, dq0,
            200,
            v_command=[1, 1.5, 0, 0],
            rng=rhc_rng,
            name=video_name,
        )

        video_name = save_dir + f"/rhc/bound/{epoch}.mp4"
        env.render_rhc(
            testmodel,
            q0, dq0,
            200,
            v_command=[2, 1.5, 0, 0],
            rng=rhc_rng,
            name=video_name,
        )
        print('------------------------------------------------------\n')
