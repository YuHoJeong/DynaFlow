import jax
import jax.numpy as jnp

import mujoco
from mujoco import mjx
import cv2
from pathlib import Path
from functools import partial

from tqdm import tqdm


def q_inv(q: jnp.ndarray) -> jnp.ndarray:
    return q * jnp.array([1., -1., -1., -1.])

def q_mul(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jnp.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)

def quat_to_rotmat(q: jnp.ndarray) -> jnp.ndarray:
    """(..., 4) wxyz → (..., 3, 3)"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2 = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return jnp.stack([
        jnp.stack([1-2*y2-2*z2, 2*xy-2*wz,   2*xz+2*wy ], axis=-1),
        jnp.stack([2*xy+2*wz,   1-2*x2-2*z2, 2*yz-2*wx  ], axis=-1),
        jnp.stack([2*xz-2*wy,   2*yz+2*wx,   1-2*x2-2*y2], axis=-1),
    ], axis=-2)

def state_to_obs(qpos: jnp.ndarray, qvel: jnp.ndarray) -> jnp.ndarray:
    base_quat = qpos[..., 3:7]
    joint_pos = qpos[..., 7:19]
    base_ang_vel_local = qvel[..., 3:6]
    joint_vel = qvel[..., 6:18]

    siny_cosp = 2 * (base_quat[..., 0] * base_quat[..., 3] + base_quat[..., 1] * base_quat[..., 2])
    cosy_cosp = 1 - 2 * (base_quat[..., 2] ** 2 + base_quat[..., 3] ** 2)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)
    yaw_quat = jnp.stack([jnp.cos(yaw/2), jnp.zeros_like(yaw), jnp.zeros_like(yaw), jnp.sin(yaw/2)], axis=-1)

    rot_wb = quat_to_rotmat(base_quat)
    rot_bw = jnp.transpose(rot_wb, (0, 2, 1))
    gravity = -rot_bw[:, :, 2]

    return jnp.concatenate([
        gravity,
        joint_pos,
        base_ang_vel_local * 0.25,
        joint_vel * 0.05,
    ], axis=-1)


class Env:
    def __init__(self):
        self.mj_model = mujoco.MjModel.from_xml_path('robots/go1/scene.xml')
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)
        self.simulation_dt = self.mj_model.opt.timestep
        self.control_dt = 0.02
        self.n_substeps = int(self.control_dt / self.simulation_dt)

        self.renderer = mujoco.Renderer(self.mj_model, width=1280, height=720)
        self.renderer.close()
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.mj_model, self.camera)
        self.camera.lookat[:] = [0.0, 0.0, 0.5]
        self.camera.distance = 4.0
        self.camera.elevation = -10
        self.camera.azimuth = 90

        self.dof = 18
        self.state_dim = 37
        self.act_dim = 12
        self.obs_dim = 30

        max_ptarg = jnp.array([1.5, 3.0, 1.5, 1.5, 3.0, 1.5, 1.5, 3.0, 1.5, 1.5, 3.0, 1.5])
        min_ptarg = jnp.array([-1.0, -1.0, -6.0, -1.0, -1.0, -6.0, -1.0, -1.0, -6.0, -1.0, -1.0, -6.0])
        self.nominal_ptarg = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
        self.act_range = jnp.array([1.25, 2, 3.5, 1.25, 2, 3.5, 1.25, 2, 3.5, 1.25, 2, 3.5])
        self.ctrl_range = jnp.stack([min_ptarg, max_ptarg], axis=1)
        self.trunk_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'trunk')


    def render_trajectory(self, qpos_traj: jnp.ndarray, qvel_traj: jnp.ndarray,
                          name="replays/true_state_replay.mp4", frame_per_sec=50,
                          vis_contact=False, vis_perturbation=False):
        T = qpos_traj.shape[0]

        save_dir = Path("./videos")
        video_path = save_dir / name
        video_path.parent.mkdir(parents=True, exist_ok=True)

        self.renderer = mujoco.Renderer(self.mj_model, width=1280, height=720)

        scene_option = mujoco.MjvOption()
        if vis_contact:
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        if vis_perturbation:
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, frame_per_sec, (1280, 720))

        for t in range(T):
            self.mj_data.qpos[:] = qpos_traj[t]
            self.mj_data.qvel[:] = qvel_traj[t]
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self.camera.lookat[:] = self.mj_data.xpos[self.trunk_id]
            self.renderer.update_scene(self.mj_data, camera=self.camera, scene_option=scene_option)
            video_writer.write(cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR))

        video_writer.release()
        self.renderer.close()
        print(f"Saved rollout video at: {video_path}")

    def render_rhc(self, model, q0, dq0, horizon, rng,
                    v_command=[1., 1., 0., 0.], name="./videos/test/rhc.mp4", vis_contact=False):
        mjx_data = mjx.make_data(self.mjx_model).replace(qpos=q0, qvel=dq0)

        jit_step = jax.jit(partial(control_step, mjx_model=self.mjx_model, n_substeps=self.n_substeps))
        jit_infer = jax.jit(lambda noise, t, attr: model(noise, t, attr))

        video_path = Path(name)
        video_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 50, (1280, 720))

        self.renderer = mujoco.Renderer(self.mj_model, width=1280, height=720)
        scene_option = mujoco.MjvOption()
        if vis_contact:
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        for i in tqdm(range(horizon), desc="Rendering.."):
            mj_data = mjx.get_data(self.mj_model, mjx_data)

            obs = state_to_obs(mj_data.qpos[None, :], mj_data.qvel[None, :])
            t = jnp.array([0.])
            rng, s_key = jax.random.split(rng)
            s0 = jax.random.normal(s_key, shape=(1, 16, 37))
            command = jnp.array([v_command])
            attr = jnp.concatenate([obs, command], axis=-1)

            normed_actions = jit_infer(s0, t, attr)
            action = self.scale_action(normed_actions[0, 0, :])
            mjx_data = jit_step(data=mjx_data, u=action)

            mj_data = mjx.get_data(self.mj_model, mjx_data)
            self.camera.lookat[:] = mj_data.xpos[self.trunk_id]
            self.renderer.update_scene(mj_data, camera=self.camera, scene_option=scene_option)
            video_writer.write(cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR))

        video_writer.release()
        self.renderer.close()
        print(f"Saved rollout video at: {video_path}")

    def render_openloop(self, model, q0, dq0, horizon, rng,
                        v_command=[1., 1., 0., 0.], window=6,
                        name="./videos/test/rhc.mp4", vis_contact=False):
        mjx_data = mjx.make_data(self.mjx_model).replace(qpos=q0, qvel=dq0)

        jit_step = jax.jit(partial(control_step, mjx_model=self.mjx_model, n_substeps=self.n_substeps))
        jit_infer = jax.jit(lambda noise, t, attr: model(noise, t, attr))

        video_path = Path(name)
        video_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 50, (1280, 720))

        self.renderer = mujoco.Renderer(self.mj_model, width=1280, height=720)
        scene_option = mujoco.MjvOption()
        if vis_contact:
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        actions = None
        for i in tqdm(range(horizon), desc="Rendering.."):
            mj_data = mjx.get_data(self.mj_model, mjx_data)

            obs = state_to_obs(mj_data.qpos[None, :], mj_data.qvel[None, :])
            t = jnp.array([0.])
            rng, s_key = jax.random.split(rng)
            s0 = jax.random.normal(s_key, shape=(1, 16, 37))
            command = jnp.array([v_command])
            attr = jnp.concatenate([obs, command], axis=-1)

            if i % window == 0:
                normed_actions = jit_infer(s0, t, attr)
                actions = self.scale_action(normed_actions[0, :, :])
            mjx_data = jit_step(data=mjx_data, u=actions[i % window])

            mj_data = mjx.get_data(self.mj_model, mjx_data)
            self.camera.lookat[:] = mj_data.xpos[self.trunk_id]
            self.renderer.update_scene(mj_data, camera=self.camera, scene_option=scene_option)
            video_writer.write(cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR))

        video_writer.release()
        self.renderer.close()
        print(f"Saved rollout video at: {video_path}")

    def scale_action(self, normalized_action):
        return normalized_action * self.act_range + self.nominal_ptarg



def control_step(mjx_model: mjx.Model, data: mjx.Data, u: jnp.ndarray, n_substeps: int):
    data = data.replace(ctrl=u)
    final_data = jax.lax.fori_loop(0, n_substeps, lambda i, d: mjx.step(mjx_model, d), data)
    return final_data


@partial(jax.jit, static_argnames=("env",))
def rollout(env: Env, q0: jnp.ndarray, dq0: jnp.ndarray, action_traj: jnp.ndarray):
    def step_fn(data: mjx.Data, u: jnp.ndarray):
        next_data = control_step(env.mjx_model, data, u, env.n_substeps)
        return next_data, next_data

    init_data = mjx.make_data(env.mjx_model).replace(qpos=q0, qvel=dq0)
    _, traj = jax.lax.scan(step_fn, init_data, action_traj)
    return traj


@partial(jax.jit, static_argnames=("env",))
def batched_rollout(env: Env, q0: jnp.ndarray, dq0: jnp.ndarray, action_traj: jnp.ndarray):
    return jax.vmap(rollout, in_axes=(None, 0, 0, 0))(env, q0, dq0, action_traj)
