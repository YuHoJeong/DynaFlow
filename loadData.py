import os
import collections
from typing import Iterator

import time
from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import pickle as pkl


Batch = namedtuple('Batch', 'trajectories conditions command')


class SequenceDataset:
    def __init__(self, horizon=64, max_path_length=1000,
                 max_n_episodes=100000, termination_penalty=0,
                 action_scale=1.0, cache_path=None):
        self.horizon = horizon
        self.max_path_length = max_path_length

        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pkl.load(f)
            self.normalizer = cached_data['normalizer']
            self.indices = cached_data['indices']
            self.fields = cached_data['fields']
            self.n_episodes = cached_data['n_episodes']
            self.path_lengths = cached_data['path_lengths']
            print(f"Loaded {self.n_episodes} episodes, {len(self.indices)} samples")
        else:
            print("Building dataset from scratch...")
            itr = read_dataset(maxlen=max_path_length, max_episodes=max_n_episodes)

            fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
            for episode in itr:
                fields.add_path(convert_to_jax(episode))
            fields.finalize()

            start_time = time.time()
            self.normalizer = DatasetNormalizer(fields, fields['path_lengths'])
            self.indices = self._make_indices(fields.path_lengths, horizon)
            self.fields = fields
            self.n_episodes = fields.n_episodes
            self.path_lengths = fields.path_lengths
            self._normalize_fields()
            print(f"Built dataset in {time.time() - start_time:.1f}s")
            print(fields)

            if cache_path:
                print(f"Saving cache to {cache_path}")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pkl.dump({
                        'normalizer': self.normalizer,
                        'indices': self.indices,
                        'fields': self.fields,
                        'n_episodes': self.n_episodes,
                        'path_lengths': self.path_lengths,
                    }, f)

    def _normalize_fields(self, keys=('observations',)):
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer.normalize(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def _make_indices(self, path_lengths, horizon):
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            for start in range(max_start):
                indices.append((i, start, start + horizon))
        return np.array(indices)

    def get_conditions(self, observations):
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        unnormed_xy = self.fields.observations[path_ind, start:end, 0:2]
        normed_others = self.fields.normed_observations[path_ind, start:end, 2:]
        unnormed_xy = unnormed_xy - unnormed_xy[0:1,0:2]

        observations = np.concatenate([unnormed_xy, normed_others], axis=-1)
        actions = self.fields.actions[path_ind, start:end]
        trajectories = np.concatenate([actions, observations], axis=-1)

        conditions = self.get_conditions(observations)

        command = self.fields.commands[path_ind, start, :]
        batch = Batch(trajectories, conditions, command)

        return batch


class ReplayBuffer:

    def __init__(self, max_n_episodes, max_path_length, termination_penalty):
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=int), # np.int >> int
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty

    def __repr__(self):
        return '[ datasets/buffer ] Fields:\n' + '\n'.join(
            f'    {key}: {val.shape}'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths'}.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def add_path(self, path):
        path_length = len(path['observations'])
        # print(path_length)
        # print(self.max_path_length)
        assert path_length <= self.max_path_length

        if path['terminals'].any():
            assert (path['terminals'][-1] == True) and (not path['terminals'][:-1].any())

        ## if first path added, set keys based on contents
        self._add_keys(path)

        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict: self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

        ## penalize early termination
        if path['terminals'].any() and self.termination_penalty is not None:
            assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
            self._dict['commands'][self._count, path_length - 1] += self.termination_penalty

        ## record path length
        self._dict['path_lengths'][self._count] = path_length

        ## increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        ## remove extra slots
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')

class LimitsNormalizer:
    """Maps [xmin, xmax] → [-1, 1] per dimension."""

    def __init__(self, X: np.ndarray):
        self.mins = X.min(axis=0).astype(np.float32)
        self.maxs = X.max(axis=0).astype(np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = (x - self.mins) / (self.maxs - self.mins)
        return 2 * x - 1

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        x = jnp.clip(x, -1, 1)
        x = (x + 1) / 2.0
        return x * (self.maxs - self.mins) + self.mins


class DatasetNormalizer:

    def __init__(self, dataset, path_lengths):
        flat = flatten(dataset, path_lengths)
        self.normalizers = {key: LimitsNormalizer(val) for key, val in flat.items()}

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

def convert_to_jax(x):
    if isinstance(x, np.ndarray):
        return jnp.array(x)
    elif isinstance(x, dict):
        return {k: convert_to_jax(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(convert_to_jax(v) for v in x)
    else:
        return x

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

def flatten(dataset, path_lengths):
    '''
        flattens dataset of { key: [ n_episodes x max_path_lenth x dim ] }
            to { key : [ (n_episodes * sum(path_lengths)) x dim ]}
    '''
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        flattened[key] = np.concatenate([
            x[:length]
            for x, length in zip(xs, path_lengths)
        ], axis=0)
    return flattened


def make_data_stream(dataset, batch_size: int, shuffle: bool = True) -> Iterator:
    """
    A pure-Python infinite generator over your SequenceDataset,
    returning JAX arrays on the correct device.
    """
    ds_size = len(dataset)
    idxs = np.arange(ds_size)
    while True:
        if shuffle:
            np.random.shuffle(idxs)
        for start in range(0, ds_size, batch_size):
            batch_idxs = idxs[start:start+batch_size]
            batch = [dataset[i] for i in batch_idxs]
            # stack trajectories into a jnp array [B, H, T]
            trajs = jnp.stack([b.trajectories for b in batch], axis=0)
            # stack each key in conditions
            cond = {k: jnp.stack([b.conditions[k] for b in batch], axis=0)
                    for k in batch[0].conditions}
            cmds = jnp.stack([b.command for b in batch], axis=0)
            yield trajs, cond, cmds

def read_dataset(maxlen=250, max_episodes=None):
    file_paths = [
        'datasets/go1_locomotion/trot.pkl',
        'datasets/go1_locomotion/bound.pkl',
    ]
    dataset = {}
    keys = ['actions', 'observations', 'commands']

    total_episodes = 0
    for file_path in file_paths:
        path = os.path.expanduser(file_path)
        with open(path, 'rb') as f:
            loaded_data = pkl.load(f)
        for key in keys:
            data = loaded_data[key]
            dim = data.shape[-1]

            reshaped_data = data.reshape(-1, 250, dim)

            # Limit episodes if max_episodes is specified
            if max_episodes is not None:
                remaining_episodes = max_episodes - total_episodes
                if remaining_episodes <= 0:
                    break
                if reshaped_data.shape[0] > remaining_episodes:
                    reshaped_data = reshaped_data[:remaining_episodes]

            truncated_data = reshaped_data[:, :maxlen, :]
            flattened_data = truncated_data.reshape(-1, dim)

            if key in dataset:
                dataset[key] = np.concatenate([dataset[key], flattened_data], axis=0)
            else:
                dataset[key] = flattened_data

        total_episodes += reshaped_data.shape[0]
        if max_episodes is not None and total_episodes >= max_episodes:
            break

    final_n_steps = dataset['actions'].shape[0]
    dataset['terminals'] = np.zeros(final_n_steps, dtype=bool)
    dataset['timeouts'] = np.zeros(final_n_steps, dtype=bool)
    dataset['timeouts'][maxlen - 1::maxlen] = True

    generate_pos = True
    include_clock = False
    seperate_gait = True

    if not include_clock:
        dataset['observations'] = np.concatenate([dataset['observations'][:, 0:13], dataset['observations'][:, 18:]],
                                                 axis=-1)
        print(dataset['observations'].shape)
    if not generate_pos:
        dataset['observations'] = dataset['observations'][:, 2:]
    if not seperate_gait:
        dataset['commands'][:, 0] = -1

    # isaac to mjx convention
    joint_pos = dataset['observations'][:, 13:25]
    reordered_joint_pos = reorder_joints(joint_pos)
    joint_vel = dataset['observations'][:, 25:37]
    reordered_joint_vel = reorder_joints(joint_vel)
    base_ori = np.concatenate([dataset['observations'][:, 6:7], dataset['observations'][:, 3:6]], axis=-1)

    base_pos = dataset['observations'][:, 0:3]
    base_linvel = dataset['observations'][:, 7:10]
    base_angvel_w = dataset['observations'][:, 10:13]
    rot_wb = quat_to_rotmat(base_ori)
    base_angvel = np.einsum('...ij,...j->...i', rot_wb.transpose(0, 2, 1), base_angvel_w)

    observations = np.concatenate([base_pos, base_ori,
                                   reordered_joint_pos,
                                   base_linvel, base_angvel, reordered_joint_vel], axis=-1)

    dataset['observations'] = observations
    dataset['actions'] = reorder_joints(dataset['actions'])

    nominal_joint_pos = np.array([
        -0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5
    ])
    dataset['actions'] = dataset['actions'] * 0.25
    dataset['actions'][:, [0, 3, 6, 9]] = dataset['actions'][:, [0, 3, 6, 9]] * 0.5
    dataset['actions'] = nominal_joint_pos + dataset['actions']

    N = dataset['commands'].shape[0]
    data_ = collections.defaultdict(list)

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        final_timestep = dataset['timeouts'][i]

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1

def reorder_joints(joints: np.ndarray) -> np.ndarray:
    """ from isaac convention to others """
    FL = joints[..., 0:3]
    FR = joints[..., 3:6]
    RL = joints[..., 6:9]
    RR = joints[..., 9:12]
    reordered = np.concatenate([FR, FL, RR, RL], axis=-1)
    return reordered

def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]

    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    R11 = 1 - 2 * y2 - 2 * z2
    R12 = 2 * xy - 2 * wz
    R13 = 2 * xz + 2 * wy

    R21 = 2 * xy + 2 * wz
    R22 = 1 - 2 * x2 - 2 * z2
    R23 = 2 * yz - 2 * wx

    R31 = 2 * xz - 2 * wy
    R32 = 2 * yz + 2 * wx
    R33 = 1 - 2 * x2 - 2 * y2

    return np.stack([
        np.stack([R11, R12, R13], axis=-1),
        np.stack([R21, R22, R23], axis=-1),
        np.stack([R31, R32, R33], axis=-1)
    ], axis=1)