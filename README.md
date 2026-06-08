# DynaFlow Code Release
[[project page]](https://sowoolee.github.io/dynaflow/)  [[paper]](https://arxiv.org/pdf/2509.19804)

This repository contains the official implementation of 

**DynaFlow:** Dynamics-embedded Flow Matching for Physically Consistent Motion Generation from State-only Demonstrations


---

## Environment Setup

### 1. Create conda environment

```bash
conda create -n dynaflow python=3.10
conda activate dynaflow
```

### 2. Install MuJoCo & MJX

```bash
pip install mujoco-mjx
```

### 3. Install JAX with CUDA support

```bash
pip install -U "jax[cuda12]"
```

### 4. Install remaining dependencies

```bash
pip install -U flax optax orbax-checkpoint
pip install tensorboard
pip install opencv-python
pip install tqdm pygame
```

For TensorBoard logging:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you encounter `cuSolver internal error`, fix with:

```bash
pip install nvidia-cublas-cu12==12.9.0.13
```

### Verified versions

| Package | Version |
|---|---|
| jax  | 0.6.2 |
| flax | 0.10.7 |
| mujoco-mjx | 3.3.3 |
| optax | 0.2.5 |
| orbax-checkpoint | 0.11.18 |

---

## Dataset

```bash
bash download_dataset.sh
```

Downloads trot and bound locomotion data from Google Drive and places them under `datasets/go1_locomotion/`:

```
datasets/go1_locomotion/
├── trot.pkl
└── bound.pkl
```

On first run, a processed cache is saved to `cache/` to skip preprocessing on subsequent runs.

---

## Train

```bash
python main.py
```

Checkpoints and TensorBoard logs are saved to `checkpoints/`. To monitor training:

```bash
python utils.py
```

---

## Test

Set `checkpoint_name` and `target_epoch` in `tester.py`, then:

```bash
python tester.py
```

---

## Set your own Env

Place your robot's MJCF scene file under `robots/<robot_name>/scene.xml`, then update `diffsim.Env` in `diffsim.py`:

- Change the `from_xml_path` to point to your scene file
- Update `dof`, `state_dim`, `act_dim`, `obs_dim` to match your robot
- Adjust joint-related values (limits, nominal configurations, ...) and other robot-specific params accordingly