# Go2X5 VBC Development Log

Last updated: 2026-06-01

This document is the long-running development record for reproducing and adapting the Visual Whole-Body Control (VBC) pipeline to Unitree Go2 + ARX-X5. Keep it append-only where possible: update the current status table, then add a dated entry under "Progress Log" for every meaningful training run, config change, bug fix, or evaluation.

## Project Scope

The project aims to reproduce the VBC framework on:

- Robot: Unitree Go2 quadruped + ARX-X5 arm
- Simulator: Isaac Gym
- Low-level policy: locomotion + whole-body stability + EE target tracking
- High-level policy: object pick / lift task using low-level policy as a controller
- Base code: `https://github.com/BoZhiStudying233/visual-wholebody-control-go2x5`
- Current remote repository: `git@github.com:lemonoscar/VBC-gx.git`

The original VBC codebase is B1 + Z1 oriented. Go2X5 requires URDF, DOF, observation, control-gain, target-sampling, reward, and high-level task adaptation.

## Current Repository State

- Branch: `main`
- Last checked commit: `a191dd7 Respect WANDB_MODE in low-level training`
- Working tree at last check: clean
- Current important local path: `/home/lemon/research/Issac/visual-wholebody-control-go2x5`
- Remote training path used so far: `~/xhq_workload/VBC-gx`

Important pushed commits:

- `bda22b2 Align Go2X5 low-level stable config`
- `a191dd7 Respect WANDB_MODE in low-level training`

## Current Training Status

### Low-Level Go2X5 Stable Base

Main run:

```text
low-level/logs/go2x5-low/go2x5_stable_base_v1/
```

Available checkpoints include:

```text
model_7600.pt
model_10000.pt
model_17600.pt
```

Latest downloaded checkpoint at last check:

```text
low-level/logs/go2x5-low/go2x5_stable_base_v1/model_17600.pt
```

This checkpoint was load-tested locally in `vwc_go2x5`.

Observed training trend:

- Early training around iteration 39 was unstable: short episode length and negative reward.
- Around iteration 2500, episode length improved to around 366 and reward became positive.
- Around iteration 5000, episode length was around 444 and reward around 9.
- Around iteration 10000, episode length was around 462, but some penalties remained high.
- `Dones: 0.00` in logs is rounded display; it does not mean no reset happened.

Current interpretation:

- The low-level model is good enough for visual inspection and high-level smoke testing.
- It is not yet proven to be final for high-level manipulation.
- If high-level collapses early, run `go2x5_ftlift` fine-tuning before blaming the high-level reward.

### Low-Level Go2X5 FtLift

Status: planned / optional.

Purpose:

- Fine-tune from a stable low-level checkpoint for better high-level transfer.
- Bias low-level toward low EE targets, forward reaching, support during arm movement, and lift-like disturbances.
- This is not a standard VBC paper step. It is a pragmatic adaptation for Go2X5 high-level transfer.

Recommended starting point requested:

```text
go2x5_stable_base_v1/model_7600.pt
```

Recommended output run:

```text
low-level/logs/go2x5-low/go2x5_ftlift_from_stable7600_v1/
```

### High-Level Go2X5

Status: not yet trained successfully in the current confirmed route.

Important blocker before training:

```yaml
high-level/data/cfg/go2x5_pickmulti.yaml
```

currently points to an old low-level path:

```yaml
low_policy_path: "../low-level/logs/go2x5-low/go2x5_b1style_20260418/model_11800.pt"
```

Before high-level training, create a copied config and point `low_policy_path` to the actual low-level checkpoint being evaluated.

Recommended first high-level route:

1. Use `go2x5_stable_base_v1/model_17600.pt` or a finished `go2x5_ftlift_from_stable7600_v1` checkpoint.
2. Smoke test with a small environment count.
3. Train teacher on fixed table height `--table_height 0.25`.
4. Only after fixed-table success, remove `--table_height` for randomized table height.
5. Train student / BC only after teacher quality is acceptable.

## Environment Notes

### Local Environment

Local conda env:

```text
vwc_go2x5
```

Local GPU noted by user:

```text
RTX 4060
```

Important local fix:

- Old `torch 1.10.2 + cu113` failed on RTX 40-series with:

```text
RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)
```

- Upgraded local env to:

```text
torch==2.4.1+cu121
torchvision==0.19.1+cu121
torchaudio==2.4.1+cu121
```

Validation command:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

### Remote Environment

Remote conda env used:

```text
b1z1
```

Remote path:

```text
~/xhq_workload/VBC-gx
```

Known remote issue:

- `git pull` inside the conda env may fail with OpenSSL mismatch.
- Workaround:

```bash
cd ~/xhq_workload/VBC-gx
conda deactivate
unset LD_LIBRARY_PATH
unset PYTHONPATH
GIT_PAGER=cat git pull origin main
git --no-pager log -1 --oneline
```

Then reactivate the training env.

### Mandatory GPU Visibility Rule

All training and replay commands should restrict the process to the first visible GPU:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
```

Inside the process, still use:

```bash
--sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0
```

## Low-Level Design Summary

The Go2X5 low-level task is not a grasping policy. It is a whole-body low-level controller that stabilizes the robot, tracks velocity commands, and tracks EE target commands.

Robot DOF:

- Total DOF: 20
- Legs: 12
- Arm: 6
- Gripper: 2
- Policy actions: 18, excluding gripper

Current low-level convention:

- Leg policy action is the important learned control.
- Arm branch exists structurally but arm behavior is primarily IK / PD controlled in the environment.
- In `ManipLoco.step()`, arm actions are zeroed before control:

```python
actions[:, 12:] = 0.
```

Observation design:

- Base proprio without gait commands: `num_proprio = 66`
- Privileged observation: `num_priv = 18`
- History length: `10`
- Total observation without gait commands: `66 * 11 + 18 = 744`
- With `--observe_gait_commands`, proprio gains 5 gait features and total observation becomes 799.

Important: Training and playback must both include `--observe_gait_commands` if the checkpoint was trained with gait observations.

## Current Go2X5 Low-Level Config Facts

Key values in `low-level/legged_gym/envs/manip_loco/go2x5_config.py`:

```text
num_envs = 6144
num_proprio = 66
num_observations = 744 without gait commands
arm.base_offset = [0.0, 0.0, 0.08]
goal_ee.ranges.pos_p = [-0.7, pi / 3]
base_height_target = 0.28
max_contact_force = 200
feet_height_target = 0.08
max_iterations = 45000
save_interval = 200
```

Important corrections already reflected in current code:

- Observation history uses leg actions only, not all 18 actions.
- Go2X5 arm base offset is config-driven.
- X5 joints are named to avoid PD gain substring collision with leg joints.
- EE quaternion normalization guards against zero-norm NaN.
- Observation buffer has NaN protection.
- Goal pitch lower bound avoids invalid below-ground sampling.
- `train.py` respects `WANDB_MODE`.

## Commands

### Common Low-Level Environment Setup

Local:

```bash
cd /home/lemon/research/Issac/visual-wholebody-control-go2x5
conda activate vwc_go2x5

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PYTHONPATH"
```

Remote:

```bash
cd ~/xhq_workload/VBC-gx
conda activate b1z1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PYTHONPATH"
```

### Train Go2X5 Stable Low-Level From Scratch

```bash
cd low-level/legged_gym/scripts

python train.py --headless --task go2x5 \
  --proj_name go2x5-low --exptid go2x5_stable_base_v1 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

### Resume Go2X5 Stable Low-Level

```bash
cd low-level/legged_gym/scripts

python train.py --headless --task go2x5 \
  --proj_name go2x5-low --exptid go2x5_stable_base_v1 \
  --resumeid go2x5_stable_base_v1 --checkpoint -1 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

### Train Go2X5 FtLift From Stable 7600

Use this when high-level transfer needs a more manipulation-aware low-level.

```bash
cd low-level/legged_gym/scripts

python train.py --headless --task go2x5_ftlift \
  --proj_name go2x5-low --exptid go2x5_ftlift_from_stable7600_v1 \
  --resumeid go2x5_stable_base_v1 --checkpoint 7600 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

Resume:

```bash
python train.py --headless --task go2x5_ftlift \
  --proj_name go2x5-low --exptid go2x5_ftlift_from_stable7600_v1 \
  --resumeid go2x5_ftlift_from_stable7600_v1 --checkpoint -1 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

### tmux Remote Training

Create / attach:

```bash
tmux new -s go2x5_ftlift
tmux attach -t go2x5_ftlift
```

Detach without stopping training:

```text
Ctrl+b
d
```

### Local GUI Replay: Flat Zero-Velocity

The repo's `play.py` has `--headless` defaulted to `True`, so GUI replay uses an import wrapper that sets `args.headless=False`.

```bash
cd /home/lemon/research/Issac/visual-wholebody-control-go2x5
conda activate vwc_go2x5

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PYTHONPATH"
export _ISAACGYM_LIBRARY_PATH_BOOTSTRAPPED=1

cd low-level/legged_gym/scripts

python -c "import isaacgym; import play as p; p.EXPORT_POLICY=False; p.SAVE_ACTOR_HIST_ENCODER=False; p.RECORD_FRAMES=False; p.MOVE_CAMERA=False; args=p.get_args(); args.headless=False; p.play(args)" \
  --task go2x5 \
  --proj_name go2x5-low --exptid go2x5_stable_base_v1 \
  --checkpoint 7600 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands --flat_terrain
```

Viewer notes:

- Press `F` for free camera.
- Avoid numeric keys `1` to `8` when `num_envs=1`; they can trigger an index error in the viewer camera selector.
- Key `0` is safe.

### Local GUI Replay: Complex Terrain

Remove `--flat_terrain`:

```bash
python -c "import isaacgym; import play as p; p.EXPORT_POLICY=False; p.SAVE_ACTOR_HIST_ENCODER=False; p.RECORD_FRAMES=False; p.MOVE_CAMERA=False; args=p.get_args(); args.headless=False; p.play(args)" \
  --task go2x5 \
  --proj_name go2x5-low --exptid go2x5_stable_base_v1 \
  --checkpoint 7600 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

### W&B Offline Sync

Current local W&B runs:

```text
low-level/legged_gym/envs/logs/wandb/offline-run-20260531_024919-7qix21n0
low-level/legged_gym/envs/logs/wandb/offline-run-20260530_230740-ihrya8px
```

Sync latest stable run:

```bash
cd /home/lemon/research/Issac/visual-wholebody-control-go2x5
conda activate vwc_go2x5

wandb sync low-level/legged_gym/envs/logs/wandb/offline-run-20260531_024919-7qix21n0
```

If partially synced:

```bash
wandb sync --include-synced --append \
  low-level/legged_gym/envs/logs/wandb/offline-run-20260531_024919-7qix21n0
```

Known W&B issue:

- Offline runs may contain symlinks to remote source paths.
- If sync fails with missing `files/manip_loco/b1z1_config.py` or `files/manip_loco/manip_loco.py`, replace the broken symlinks with real files from the current repo.

## High-Level Go2X5 Training Route

### Prepare a High-Level Config

Do not overwrite the baseline config. Copy it:

```bash
cd /home/lemon/research/Issac/visual-wholebody-control-go2x5/high-level

cp data/cfg/go2x5_pickmulti.yaml data/cfg/go2x5_pickmulti_stable17600.yaml
```

Set low-level path:

```bash
sed -i 's#low_policy_path:.*#low_policy_path: "../low-level/logs/go2x5-low/go2x5_stable_base_v1/model_17600.pt"#' \
  data/cfg/go2x5_pickmulti_stable17600.yaml
```

For FtLift, point to the latest checkpoint under:

```text
../low-level/logs/go2x5-low/go2x5_ftlift_from_stable7600_v1/
```

### High-Level Smoke Test

```bash
cd /home/lemon/research/Issac/visual-wholebody-control-go2x5
conda activate vwc_go2x5

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PWD/high-level:$PYTHONPATH"

cd high-level

python train_multistate.py \
  --task Go2X5PickMulti \
  --config data/cfg/go2x5_pickmulti_stable17600.yaml \
  --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 \
  --headless --debug --num_envs 34 --timesteps 500 \
  --experiment_dir go2x5-pick-multi-teacher \
  --wandb_name smoke_go2x5_hl_stable17600 \
  --roboinfo --observe_gait_commands \
  --small_value_set_zero --rand_control --stop_pick \
  --table_height 0.25
```

### High-Level Fixed Table Teacher

```bash
python train_multistate.py \
  --task Go2X5PickMulti \
  --config data/cfg/go2x5_pickmulti_stable17600.yaml \
  --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 \
  --headless --num_envs 256 --timesteps 60000 \
  --experiment_dir go2x5-pick-multi-teacher \
  --wandb_name go2x5_teacher_table025_stable17600_v1 \
  --wandb --wandb_project go2x5-high \
  --roboinfo --observe_gait_commands \
  --small_value_set_zero --rand_control --stop_pick \
  --table_height 0.25
```

Resume:

```bash
python train_multistate.py \
  --task Go2X5PickMulti \
  --config data/cfg/go2x5_pickmulti_stable17600.yaml \
  --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 \
  --headless --num_envs 256 --timesteps 60000 \
  --experiment_dir go2x5-pick-multi-teacher \
  --wandb_name go2x5_teacher_table025_stable17600_v1 \
  --wandb --wandb_project go2x5-high \
  --roboinfo --observe_gait_commands \
  --small_value_set_zero --rand_control --stop_pick \
  --table_height 0.25 \
  --resume
```

## Evaluation Criteria

### Low-Level Visual Checks

Flat zero-velocity:

- Base does not sink or crouch excessively.
- Roll / pitch remain stable.
- Feet do not drag or jitter violently.
- Arm target movement does not destabilize the robot.
- No repeated early termination.

Complex terrain:

- Robot can stand and move without immediate foot collision.
- Body does not oscillate aggressively.
- Feet clear terrain enough to avoid continuous dragging.
- Robot can recover from mild terrain-induced body motion.

### High-Level Checks

Early smoke:

- Environment starts without tensor shape mismatch.
- Low-level policy loads successfully.
- Robot does not fall before attempting reach / pick.
- Fixed table height task produces nonzero reach / lift progress.

Teacher training:

- Track total success rate.
- Track object-specific success rate.
- Watch whether failures are grasp failures or base-stability failures.
- If base fails before grasp, return to low-level / FtLift.

## Known Issues and Fixes

### Isaac Gym Dynamic Libraries

Problem:

```text
libpython3.8.so.1.0 not found
libmem_filesys.so not found
carb::gym::Gym acquire failed
```

Fix:

- Ensure conda lib and Isaac Gym binding/plugin directories are in `LD_LIBRARY_PATH`.
- `train.py` and `play.py` now bootstrap these paths before importing Isaac Gym.

### W&B Online Timeout / SSL

Problem:

```text
wandb.errors.errors.CommError: Run initialization has timed out
```

Fix:

```bash
export WANDB_MODE=offline
```

`train.py` now respects `WANDB_MODE`.

### RTX 40-Series With Old PyTorch

Problem:

```text
RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)
```

Cause:

- `torch 1.10.2 + cu113` does not support Ada / RTX 40-series well enough.

Fix:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
rm -rf ~/.cache/torch_extensions/py38_cu113
```

### GUI Replay Wrapper Globals

When importing `play.py` as a module, the globals below are not created by `if __name__ == "__main__"`:

```text
EXPORT_POLICY
SAVE_ACTOR_HIST_ENCODER
RECORD_FRAMES
MOVE_CAMERA
```

The wrapper command manually sets them to `False`.

### GUI Camera Numeric-Key Crash

Problem:

```text
IndexError: index 1 is out of bounds for dimension 0 with size 1
```

Cause:

- `play.py` uses `num_envs = 1`.
- Viewer still registers numeric keys `0` to `8`.
- Pressing `1` to `8` attempts to look at a non-existent environment.

Workaround:

- Press `F` for free camera.
- Avoid numeric keys `1` to `8`.
- Use key `0` only.

Permanent fix candidate:

- Guard `lookat(i)` with `if i < self.num_envs`.

### High-Level Config Path Mismatch

Current `go2x5_pickmulti.yaml` still points to an old low-level checkpoint path. Always create a copied high-level config for each experiment and set `low_policy_path` explicitly.

## Decision Log

### 2026-05-30 to 2026-05-31

- Chose `go2x5_stable_base_v1` as first stable low-level run.
- Trained / downloaded checkpoints up to `model_17600.pt`.
- Decided `model_7600.pt` is a useful mid-training reference for replay.
- Local replay required PyTorch upgrade to CUDA 12.1 wheel.
- W&B offline sync required repairing broken source-file symlinks in the run directory.

### 2026-06-01

- Clarified that `go2x5_ftlift` is optional and not part of the original paper route.
- Decided to run FtLift from `model_7600.pt` if high-level transfer needs it.
- Clarified Go2X5 high-level route:
  - use real low-level checkpoint path,
  - smoke test,
  - fixed table height teacher,
  - randomized table height,
  - student / BC only after teacher works.

## Next Actions

1. Start remote `tmux` session for `go2x5_ftlift_from_stable7600_v1`.
2. Train FtLift from `model_7600.pt`.
3. Sync / inspect W&B or local logs.
4. Replay FtLift checkpoint on flat zero-velocity and complex terrain.
5. Create copied high-level YAML pointing to the selected low-level checkpoint.
6. Run Go2X5 high-level smoke test with fixed table height `0.25`.
7. If smoke is stable, launch fixed-table high-level teacher training.
8. If high-level base stability is poor, continue low-level FtLift or revisit reward/domain randomization.

## Appendix: What Not To Do

- Do not train Go2X5 high-level with the stale `go2x5_b1style_20260418/model_11800.pt` path.
- Do not omit `--observe_gait_commands` when loading checkpoints trained with gait observations.
- Do not interpret `Dones: 0.00` as no resets; the value is rounded.
- Do not compare high-level success before confirming that the low-level robot stays upright in high-level conditions.
- Do not use local `torch 1.10.2 + cu113` on RTX 40-series.
- Do not start long training outside `tmux` on the remote server.
