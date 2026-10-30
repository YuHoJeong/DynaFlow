import os

# JAX가 GPU 메모리의 75%를 선점하면 MuJoCo 렌더러(EGL)와 충돌하므로 비활성화
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import flax.nnx as nnx
import orbax.checkpoint as ocp
import module
import diffsim
from config import PREFIX


checkpoint_name = '2026-08-18-15-07-27_DynaFlow'
target_epoch = 50000

load_dir = os.path.join(PREFIX, 'checkpoints', checkpoint_name)

env = diffsim.Env()

abstract_model = nnx.eval_shape(lambda: module.DiT1d(
    in_dim=env.state_dim,
    out_dim=env.act_dim,
    attr_dim=env.obs_dim + 4,
    d_model=384,
    n_heads=3,
    depth=3,
    rngs=nnx.Rngs(0)
))
graphdef, abstract_state = nnx.split(abstract_model)
state_restored = ocp.StandardCheckpointer().restore(load_dir + f'/state_{target_epoch}', abstract_state)
model = nnx.merge(graphdef, state_restored)
print(f"Loaded checkpoint: {load_dir}/state_{target_epoch}")

q0 = np.array([
    0.0, 0.0, 0.27,
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8
])
dq0 = np.zeros(env.mjx_model.nv)
rng = jax.random.PRNGKey(0)

command = [1, 1.0, 0, 0]

env.render_rhc(
    model, q0, dq0, 500,
    rng=rng,
    name=load_dir + f"/test_rhc_{target_epoch}.mp4",
    v_command=command
)

env.render_openloop(
    model, q0, dq0, 500,
    rng=rng,
    name=load_dir + f"/test_openloop_{target_epoch}.mp4",
    v_command=command,
    window=16
)
