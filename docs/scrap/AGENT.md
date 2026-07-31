# AGENT.md

本文件给后续进入本仓库的 agent / 开发者使用，记录仓库结构、环境、训练入口、常用命令和当前约束。不要把这里当成论文结论；它是当前复现工作的工程状态说明。

## 仓库概况

- 本地路径：`/home/lemon/research/Issac/visual-wholebody-control-go2x5`
- 远端仓库：`git@github.com:lemonoscar/VBC-gx.git`
- 当前主分支：`main`
- 最近已推送的关键提交：`e116862 Update Go2X5 low-level URDF and training config`
- 原始 codebase：`https://github.com/BoZhiStudying233/visual-wholebody-control-go2x5`
- 目标平台：Unitree Go2 + ARX-X5 机械臂，复现 / 迁移 Visual Whole-Body Control 框架。

主要目录：

- `low-level/`：Isaac Gym + rsl_rl 的低层 whole-body / locomotion 训练。
- `low-level/legged_gym/envs/manip_loco/`：B1Z1 与 Go2X5 的低层环境和配置。
- `low-level/resources/robots/go2x5/`：Go2X5 机器人资源。当前 low-level 使用 `go2_x5.urdf`。
- `high-level/`：基于 skrl 的高层 pick-multi teacher / student 训练。
- `third_party/isaacgym/`：Isaac Gym Python 绑定。
- `third_party/rsl_rl/`：低层 PPO 训练库。
- `third_party/skrl/`：高层 PPO / DAgger 训练库。
- `docs/`：开发过程记录。当前存在 `development_log.md` 和 `development_log_zh.md`。

## 当前 Go2X5 Low-Level 基础配置

关键文件：

- `low-level/legged_gym/envs/manip_loco/go2x5_config.py`
- `low-level/legged_gym/envs/manip_loco/manip_loco.py`
- `low-level/legged_gym/envs/rewards/maniploco_rewards.py`
- `low-level/resources/robots/go2x5/go2_x5.urdf`
- `low-level/legged_gym/scripts/train.py`
- `low-level/legged_gym/scripts/play.py`
- `low-level/legged_gym/scripts/visualize_go2x5_config.py`

当前重要状态：

- low-level policy 已重构为 leg-only：
  - `num_actions = 12`
  - `num_torques = 12`
  - policy 只输出 12 维腿部动作。
  - 手臂由 IK / 位置目标驱动，不再由 PPO policy 直接输出 6 维臂动作。
- 观测维度：
  - `num_proprio = 66`
  - `history_len = 10`
  - `num_priv = 18`
  - `num_observations = 744`
- 当前 Go2X5 asset：
  - `asset.file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2x5/go2_x5.urdf'`
  - `asset.gripper_name = "arm_eef_link"`
  - `arm.base_offset = [0.085, 0.0, 0.094]`
- 当前身体高度：
  - `init_state.pos = [0.0, 0.0, 0.32]`
  - `rewards.base_height_target = 0.33`
- 当前 EE 目标球：
  - `sphere_center.x_offset = 0.22`
  - `sphere_center.z_invariant_offset = 0.37`
  - `pos_l = [0.20, 0.50]`
  - `pos_p = [-0.6, pi/3]`
- 当前训练轮数：
  - `runner.max_iterations = 45000`
  - `save_interval = 200`
- 当前训练强度仍偏大：
  - `domain_rand.friction_range = [0.3, 3.0]`
  - `domain_rand.added_mass_range = [0, 15]`
  - `domain_rand.added_com_range_x/y/z = [-0.15, 0.15]`
  - `domain_rand.push_robots = True`
  - `rewards.scales.collision = -10`
  - `rewards.scales.hip_pos = -0.3`
  - `rewards.scales.torques = -2.5e-5`
  - `rewards.scales.work = -0.003`

注意：当前配置已经完成基础形态修复，但不应视为最终最优训练方案。最近训练显示 reward 容易卡在较低水平，主要被 collision、hip_pos、roll、torques、work 等项压住。后续建议改成 staged training。

## 本地环境

本地机器路径：

```bash
cd /home/lemon/research/Issac/visual-wholebody-control-go2x5
```

常用 conda 环境：

```bash
conda activate vwc_go2x5
```

本地曾遇到的问题：

- RTX 4060 不能稳定使用旧版 `torch==1.10.2+cu113`，会出现 `nvrtc: invalid value for --gpu-architecture`。
- 本地已建议升级到：
  - `torch==2.4.1+cu121`
  - `torchvision==0.19.1+cu121`
  - `torchaudio==2.4.1+cu121`
- Isaac Gym 导入顺序敏感：必须先导入 `isaacgym`，再导入 `torch`。
- `train.py`、`play.py`、`visualize_go2x5_config.py` 已加入 Isaac Gym 动态库路径自举逻辑，正常情况下无需手动设置 `_ISAACGYM_LIBRARY_PATH_BOOTSTRAPPED`。

本地基础环境变量：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PYTHONPATH"
```

## 远端服务器环境

远端仓库路径：

```bash
/data4/duanzhibo/xhq_workload/VBC-gx
```

常用 conda 环境：

```bash
conda activate b1z1
```

远端 GPU 约束：

- 所有训练 / 回放命令都必须显式限制可见 GPU。
- 使用第一张卡：

```bash
export CUDA_VISIBLE_DEVICES=0
```

- 使用第二张卡：

```bash
export CUDA_VISIBLE_DEVICES=1
```

注意：当设置 `CUDA_VISIBLE_DEVICES=1` 后，进程内部看到的第一张可见卡仍是 `cuda:0`。因此命令中仍然使用：

```bash
--sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0
```

远端曾遇到的问题：

- 服务器 conda 环境内 `git pull` 可能因 OpenSSL 版本冲突失败：

```text
OpenSSL version mismatch. Built against 30000020, you have 30500060
```

处理方式：

```bash
conda deactivate
cd /data4/duanzhibo/xhq_workload/VBC-gx
git pull
conda activate b1z1
```

或者使用系统 git，避免被 conda 动态库污染。

## Low-Level 常用命令

### 1. 环境 smoke test

```bash
cd /data4/duanzhibo/xhq_workload/VBC-gx
conda activate b1z1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PYTHONPATH"

cd low-level/legged_gym/tests
python test_env.py --task go2x5 --exptid smoke --headless \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands --debug
```

期望看到：

- `num_actions: 12`
- `num_torques: 12`
- `num_dofs: 20`
- `num_bodies: 28`
- `EE Gripper index: 25`

### 2. 可视化当前 Go2X5 配置

本地 GUI：

```bash
cd /home/lemon/research/Issac/visual-wholebody-control-go2x5
conda activate vwc_go2x5

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PYTHONPATH"

cd low-level/legged_gym/scripts
python visualize_go2x5_config.py \
  --task go2x5 --flat_terrain \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands --max_iterations 20000
```

可视化标记：

- 黄色球：当前 EE 目标点。
- 深蓝球：实际 EE 位置，即 `arm_eef_link`。
- 青色 / 蓝绿色球：目标球采样中心，不是实际末端。
- 绿色点：世界原点 / 参考点。
- 红色点：目标轨迹历史。

### 3. 从头训练 Go2X5 low-level

单卡训练：

```bash
cd /data4/duanzhibo/xhq_workload/VBC-gx
conda activate b1z1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PYTHONPATH"

cd low-level/legged_gym/scripts
python train.py --headless --task go2x5 \
  --proj_name go2x5-low --exptid go2x5_stage1_candidate_v1 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

输出位置：

```text
low-level/logs/go2x5-low/<exptid>/model_<iteration>.pt
low-level/legged_gym/envs/logs/wandb/offline-run-*
```

### 4. tmux 中同时使用第一张和第二张卡

low-level 训练本身是单 GPU 进程。要同时用两张卡，启动两个独立实验。

```bash
tmux new -s go2x5_low
```

Pane 1 使用第一张卡：

```bash
cd /data4/duanzhibo/xhq_workload/VBC-gx
conda activate b1z1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PYTHONPATH"
cd low-level/legged_gym/scripts
python train.py --headless --task go2x5 \
  --proj_name go2x5-low --exptid go2x5_stage1_seed1 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

Pane 2 使用第二张卡：

```bash
cd /data4/duanzhibo/xhq_workload/VBC-gx
conda activate b1z1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PYTHONPATH"
cd low-level/legged_gym/scripts
python train.py --headless --task go2x5 \
  --proj_name go2x5-low --exptid go2x5_stage1_seed2 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

tmux 操作：

```bash
# detach
Ctrl-b d

# reconnect
tmux attach -t go2x5_low

# split pane
Ctrl-b %
Ctrl-b "
```

### 5. 恢复训练

例如从 `model_12000.pt` 恢复：

```bash
cd /data4/duanzhibo/xhq_workload/VBC-gx
conda activate b1z1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PYTHONPATH"

cd low-level/legged_gym/scripts
python train.py --headless --task go2x5 \
  --proj_name go2x5-low --exptid go2x5_stage1_resume12000 \
  --resume --resumeid go2x5_stage1_seed1 --checkpoint 12000 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

### 6. 回放 low-level checkpoint

GUI 回放：

```bash
cd /home/lemon/research/Issac/visual-wholebody-control-go2x5
conda activate vwc_go2x5

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PYTHONPATH"

cd low-level/legged_gym/scripts
python play.py \
  --task go2x5 \
  --proj_name go2x5-low --exptid go2x5_stage1_seed1 \
  --checkpoint 12000 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

若遇到 `NameError: EXPORT_POLICY is not defined`，说明通过非标准 `python -c` 方式调用了 `play()`，优先直接运行 `python play.py ...`。

### 7. W&B 离线同步

训练建议默认：

```bash
export WANDB_MODE=offline
```

同步时不要直接 sync 整个 `wandb/` 目录，要指定单个 offline run：

```bash
wandb login
wandb sync low-level/legged_gym/envs/logs/wandb/offline-run-YYYYMMDD_HHMMSS-xxxxxxx
```

如果 sync 报缺少 `files/manip_loco/b1z1_config.py`，说明该 offline run 记录了不存在的 symlink / saved file。优先查看本地 `wandb-summary.json`、`wandb-history` 或使用已有控制台日志；也可以补齐缺失文件后再 sync。

## High-Level 常用命令

Go2X5 high-level 入口：

- `high-level/train_multistate.py`
- `high-level/play_multistate.py`
- 任务名：`Go2X5PickMulti`
- 配置：`high-level/data/cfg/go2x5_pickmulti.yaml`

注意：

- 当前 high-level 的成功率曾长期为 0，主要怀疑 low-level 不够稳定，不建议在 low-level 未修复前投入大量 high-level 训练。
- `high-level/data/cfg/go2x5_pickmulti.yaml` 中的 `low_policy_path` 需要人工确认，必须指向当前要评估的 Go2X5 low-level checkpoint。
- 原始 high-level 继承了大量 B1Z1 命名和逻辑，文件名中出现 `b1z1_*` 不一定代表实际任务不是 Go2X5。

示例训练：

```bash
cd /data4/duanzhibo/xhq_workload/VBC-gx/high-level
conda activate b1z1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/../third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/../third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/../third_party/isaacgym/python:$PWD/../third_party/skrl:$PWD:$PYTHONPATH"

python train_multistate.py \
  --task Go2X5PickMulti \
  --config data/cfg/go2x5_pickmulti.yaml \
  --rl_device cuda:0 --sim_device cuda:0 \
  --timesteps 60000 --headless \
  --experiment_dir go2x5-pick-multi-teacher \
  --wandb --wandb_project go2x5-pick-multi-teacher \
  --wandb_name go2x5_teacher_candidate \
  --roboinfo --observe_gait_commands \
  --small_value_set_zero --rand_control --stop_pick \
  --table_height 0.25
```

## 当前训练判断与约束

当前 Go2X5 low-level 仍不建议盲目继续 45000 轮满强度训练。最近观察到：

- `Mean reward` 卡在较低水平。
- `Dones = 0.00` 说明不是频繁倒地，但策略质量并不好。
- `collision`、`hip_pos`、`roll`、`torques`、`work` 等负项很大。
- `tracking_lin_vel`、`walking_dof`、`tracking_ee_world` 有正向学习，但整体被不合适的惩罚和强随机化压住。

建议下一步训练路线：

1. Stage 1：稳定站立 / 行走
   - 关闭 push。
   - 缩小 friction、mass、COM、motor strength 随机化。
   - 放松 collision、hip_pos、torques、work、roll。
   - 保留轻量 EE 跟踪，不让手臂目标压过稳定性。
2. Stage 2：增强 EE 目标追踪
   - 增大 `tracking_ee_world` 权重。
   - 扩大 EE 目标球采样范围。
3. Stage 3：鲁棒性 fine-tune
   - 加回 push。
   - 加强 domain randomization。
   - 加复杂地形。
4. Stage 4：进入 high-level pick-multi
   - high-level 只应使用已稳定回放的 low-level checkpoint。

## 代码编辑约束

- 不要随意回退用户已有修改。
- 不要使用 `git reset --hard` 或 `git checkout --` 回退文件，除非用户明确要求。
- 修改训练配置前，先说明会影响已有 checkpoint 的可比性。
- 修改 low-level policy 维度、观测维度或 URDF DOF 后，旧 checkpoint 通常不可直接复用。
- 训练命令必须显式设置 `CUDA_VISIBLE_DEVICES`。
- 服务器上使用 `CUDA_VISIBLE_DEVICES=1` 时，命令内部仍写 `cuda:0`。
- Isaac Gym 相关命令优先离线 W&B：`WANDB_MODE=offline`。
- 训练日志和模型不要轻易加入 git。
- `low-level/logs/`、`high-level/*/checkpoints/`、`wandb/`、压缩包通常不应提交。

## 快速排查表

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| `libpython3.8.so.1.0` 找不到 | conda lib 未进入动态库路径 | 设置 `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:..."` |
| `PyTorch was imported before isaacgym` | 导入顺序错误 | 先 `import isaacgym`，再导入 torch |
| W&B 初始化 90 秒超时 | 网络 / SSL 问题 | `export WANDB_MODE=offline` |
| `nvrtc invalid value for --gpu-architecture` | torch CUDA 版本太旧，不支持本机 GPU | 使用 `torch==2.4.1+cu121` |
| 机器人渲染散架 / 腿部不可见 | Go2 DAE 被错误改成 `Y_UP` | 保持 Go2-X5-lab 官方 `Z_UP`；历史 `Y_UP` 结论已证伪 |
| 回放单环境按快捷键报 index out of bounds | viewer lookat index 超出 env 数量 | 当前已在 `base_task.py` 中做 modulo clamp |
| high-level success rate 全 0 | low-level 不稳或 high-level 低层路径错误 | 先回放 low-level，再确认 `low_policy_path` |
| PhysX CUDA illegal memory access | GPU PhysX 接触对 / 碰撞复杂度 / 长时训练不稳定 | 从最近 checkpoint 恢复，减少 env 或接触复杂度 |
