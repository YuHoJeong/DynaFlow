import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

import diffsim
from diffsim import state_to_obs


class DynaFlow:
    """ Dynamics-embedded Flow Matching """

    def __init__(self, action_model, env, normalizer):
        self.action_model = action_model
        self.env = env
        self.normalizer = normalizer
        self.jit_infer = jax.jit(lambda noise, t, attr: self.action_model(noise, t, attr))

    def x1_prediction(self,
                      xt: jnp.ndarray,
                      init_state: jnp.ndarray,
                      t: jnp.ndarray,
                      command: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q0, dq0 = jnp.split(init_state, [self.env.mj_model.nq], axis=-1)
        obs = state_to_obs(q0, dq0)
        attr = jnp.concatenate([obs, command], axis=-1)

        actions_norm = self.jit_infer(xt, t, attr)
        actions = self.env.scale_action(actions_norm)

        rollout_traj = diffsim.batched_rollout(self.env, q0, dq0, actions)
        pred_states = jnp.concatenate([rollout_traj.qpos, rollout_traj.qvel], axis=-1)[:, :-1, :]
        pred_states = jnp.concatenate([init_state[:, None, :], pred_states], axis=1)

        pred_norm = self.normalizer.normalize(pred_states, 'observations')
        pred_x1 = jnp.concatenate([pred_states[..., :2], pred_norm[..., 2:]], axis=-1)

        return pred_x1, actions

    def get_velocity(self,
                     xt: jnp.ndarray,
                     init_state: jnp.ndarray,
                     t: jnp.ndarray,
                     command: jnp.ndarray) -> jnp.ndarray:
        pred_x1, _ = self.x1_prediction(xt, init_state, t, command)
        t = t.reshape(-1, 1, 1)
        return (pred_x1 - xt) / (1.0 - t + 1e-8)

    def sample(self,
               rng: jax.random.PRNGKey,
               init_state: jnp.ndarray,
               command: jnp.ndarray,
               horizon: int,
               num_steps: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        B = init_state.shape[0]
        rng, noise_rng = jax.random.split(rng)
        x0 = jax.random.normal(noise_rng, (B, horizon, self.env.state_dim))
        dt = 1.0 / num_steps

        def euler_step_fn(carry, i):
            xt, _ = carry
            t = jnp.ones((B,)) * i * dt
            pred_x1, actions = self.x1_prediction(xt, init_state, t.reshape(-1, 1), command)
            velocity = (pred_x1 - xt) / (1.0 - t.reshape(-1, 1, 1) + 1e-8)
            return (xt + velocity * dt, actions), None

        (x1, actions), _ = jax.lax.scan(euler_step_fn, (x0, jnp.zeros((B, horizon, self.env.act_dim))), jnp.arange(num_steps))
        return x1, actions
