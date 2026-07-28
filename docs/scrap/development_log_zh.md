# Go2X5 VBC 中文开发记录

最后更新：2026-06-01

本文档用于长期维护 Go2 + ARX-X5 复现 Visual Whole-Body Control 的开发进展。后续每次训练、配置修改、bug 修复、模型评估，都应在本文档中追加记录，避免只依赖聊天记录或临时命令。

建议维护方式：

- 顶部“当前状态”保持最新。
- 每次实验在“阶段记录”中追加日期、命令、checkpoint、现象和结论。
- 遇到新问题时补充到“踩坑与修复”。
- 改动训练路线时同步更新“下一步计划”。

## 项目目标

本项目基于 Visual Whole-Body Control 思路，将原本面向 B1 + Z1 的代码迁移到：

- 机器人：宇树 Go2 四足机器人 + 方舟无限 ARX-X5 机械臂
- 仿真器：Isaac Gym
- low-level：底盘运动、全身稳定、末端执行器目标跟踪
- high-level：桌面物体抓取 / 抬起任务
- 原始参考仓库：`https://github.com/BoZhiStudying233/visual-wholebody-control-go2x5`
- 当前远端仓库：`git@github.com:lemonoscar/VBC-gx.git`

原始 VBC 工作主要面向 B1 + Z1。Go2X5 复现时需要适配 URDF、DOF、观测维度、PD 增益、末端目标采样、奖励项和 high-level 任务接口。

## 当前仓库状态

- 当前分支：`main`
- 最近确认提交：`a191dd7 Respect WANDB_MODE in low-level training`
- 最近确认工作区：干净
- 本地路径：`/home/lemon/research/Issac/visual-wholebody-control-go2x5`
- 远端服务器路径：`~/xhq_workload/VBC-gx`

已经推送的重要提交：

- `bda22b2 Align Go2X5 low-level stable config`
- `a191dd7 Respect WANDB_MODE in low-level training`

## 当前训练状态

### Go2X5 Low-Level Stable Base

主训练 run：

```text
low-level/logs/go2x5-low/go2x5_stable_base_v1/
```

已经下载到本地的重要 checkpoint：

```text
model_7600.pt
model_10000.pt
model_17600.pt
```

当前最新 checkpoint：

```text
low-level/logs/go2x5-low/go2x5_stable_base_v1/model_17600.pt
```

本地已经验证该 checkpoint 可以在 `vwc_go2x5` 环境中正常 `torch.load`。

训练趋势摘要：

- iteration 39 左右：`Mean reward` 为负，`Mean episode length` 很短，训练初期不稳定。
- iteration 2500 左右：episode length 提升到约 366，reward 变为正。
- iteration 5000 左右：episode length 约 444，reward 约 9。
- iteration 10000 左右：episode length 约 462，但 collision、base height、roll 等惩罚仍偏大。
- 日志中的 `Dones: 0.00` 是显示保留两位小数后的结果，不代表完全没有 reset。

当前判断：

- `go2x5_stable_base_v1` 已经可以用于可视化检查和 high-level smoke test。
- 还不能直接认为它是最终 low-level。
- 如果 high-level 中机器人伸臂后容易倒，应优先考虑 low-level 迁移问题，而不是只调 high-level reward。

### Go2X5 FtLift Low-Level

状态：计划执行，可选增强步骤。

定位：

`go2x5_ftlift` 不是原论文中的标准步骤，也不是任务书明确要求的固定步骤。它是当前仓库中为了 Go2X5 high-level 抓取迁移而设计的 low-level fine-tune。

目的：

- 让机器人更适应 high-level 抓取时的低位伸臂。
- 增强机械臂前伸、低目标跟踪和抬物扰动下的底盘支撑。
- 让 low-level 的训练分布更接近 high-level 抓取环境。

它不是用来学习抓取策略的。抓取策略仍由 high-level teacher 学习。

当前计划：

- 从 `go2x5_stable_base_v1/model_7600.pt` 开始 fine-tune。
- 输出 run 名称：

```text
go2x5_ftlift_from_stable7600_v1
```

输出路径：

```text
low-level/logs/go2x5-low/go2x5_ftlift_from_stable7600_v1/
```

### Go2X5 High-Level

状态：尚未按当前确认路线正式训练。

当前最重要的风险：

`high-level/data/cfg/go2x5_pickmulti.yaml` 中的 `low_policy_path` 仍然指向旧模型：

```yaml
low_policy_path: "../low-level/logs/go2x5-low/go2x5_b1style_20260418/model_11800.pt"
```

开始 high-level 训练前必须复制一份新配置，并把 `low_policy_path` 指向当前真正要使用的 Go2X5 low-level checkpoint。

推荐 high-level 路线：

1. 先使用 `go2x5_stable_base_v1/model_17600.pt` 或完成后的 `go2x5_ftlift_from_stable7600_v1`。
2. 做 small env smoke test，确认环境创建、low-level 加载、机器人不立即摔倒。
3. 先固定桌高训练 teacher：`--table_height 0.25`。
4. 固定桌高成功率上升后，再进入随机桌高。
5. teacher 足够好之后，再训练 student / BC 模型。

## 环境说明

### 本地环境

本地 conda 环境：

```text
vwc_go2x5
```

本地 GPU：

```text
RTX 4060
```

已处理的问题：

本地原本使用 `torch 1.10.2 + cu113`，在 RTX 40 系显卡上回放时报错：

```text
RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)
```

原因是旧版本 PyTorch / CUDA wheel 对 Ada 架构支持不足。

已升级到：

```text
torch==2.4.1+cu121
torchvision==0.19.1+cu121
torchaudio==2.4.1+cu121
```

验证命令：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

### 远端服务器环境

远端 conda 环境：

```text
b1z1
```

远端仓库路径：

```text
~/xhq_workload/VBC-gx
```

远端曾经出现 `git pull` 的 OpenSSL mismatch：

```text
OpenSSL version mismatch. Built against 30000020, you have 30500060
```

处理方式：

```bash
cd ~/xhq_workload/VBC-gx
conda deactivate
unset LD_LIBRARY_PATH
unset PYTHONPATH
GIT_PAGER=cat git pull origin main
git --no-pager log -1 --oneline
```

拉取后再激活训练环境。

### GPU 可见性约束

所有训练和回放命令都要求只看到第一张卡：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
```

设置后，程序内部仍然使用：

```bash
--sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0
```

## Low-Level 设计说明

Go2X5 low-level 不是抓取策略。它负责：

- 底盘稳定
- 腿部运动控制
- 速度命令跟踪
- 末端执行器目标跟踪
- 为 high-level 提供稳定可控的底层接口

机器人 DOF：

- 总 DOF：20
- 腿部：12
- 机械臂：6
- 夹爪：2
- policy action：18，不包含夹爪

当前 low-level 约定：

- 策略主要学习腿部行为。
- 机械臂动作分支在网络结构中存在，但环境中主要由 IK / PD 控制。
- `ManipLoco.step()` 中实际会将 arm action 清零：

```python
actions[:, 12:] = 0.
```

观测设计：

- 不带 gait command 时：`num_proprio = 66`
- privileged obs：`num_priv = 18`
- history length：`10`
- 不带 gait command 总观测：`66 * 11 + 18 = 744`
- 带 `--observe_gait_commands` 时，proprio 增加 5 维，总观测变为 799。

重要规则：

如果 checkpoint 是带 `--observe_gait_commands` 训练的，训练、恢复、回放、high-level 加载时都必须带 `--observe_gait_commands`，否则观测语义不一致。

## 当前 Go2X5 Low-Level 关键配置

文件：

```text
low-level/legged_gym/envs/manip_loco/go2x5_config.py
```

关键值：

```text
num_envs = 6144
num_proprio = 66
num_observations = 744
arm.base_offset = [0.0, 0.0, 0.08]
goal_ee.ranges.pos_p = [-0.7, pi / 3]
base_height_target = 0.28
max_contact_force = 200
feet_height_target = 0.08
max_iterations = 45000
save_interval = 200
```

当前代码已经包含的重要修复：

- `action_history` 只观察腿部 12 维动作，而不是全部 18 维。
- Go2X5 arm base offset 从 config 读取。
- X5 关节命名避免与腿部 `joint` 关键字冲突，防止 PD 增益误匹配。
- EE 四元数归一化时做了零范数保护。
- 观测组装后做了 NaN 兜底。
- EE 目标采样的 pitch 下界收紧，避免采样到地面以下导致 NaN。
- `train.py` 已经支持 `WANDB_MODE=offline`。

## 常用命令

### 远端 tmux 训练入口

创建 tmux：

```bash
tmux new -s go2x5_ftlift
```

已有 session 时进入：

```bash
tmux attach -t go2x5_ftlift
```

退出 tmux 但保持训练继续：

```text
Ctrl+b
d
```

### 远端 low-level 环境变量

```bash
cd ~/xhq_workload/VBC-gx
conda activate b1z1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PYTHONPATH"
```

确认只看到一张卡：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

### 从 7600 开始 FtLift Fine-Tune

```bash
cd low-level/legged_gym/scripts

python train.py --headless --task go2x5_ftlift \
  --proj_name go2x5-low --exptid go2x5_ftlift_from_stable7600_v1 \
  --resumeid go2x5_stable_base_v1 --checkpoint 7600 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

中断后恢复：

```bash
python train.py --headless --task go2x5_ftlift \
  --proj_name go2x5-low --exptid go2x5_ftlift_from_stable7600_v1 \
  --resumeid go2x5_ftlift_from_stable7600_v1 --checkpoint -1 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

结果路径：

```text
low-level/logs/go2x5-low/go2x5_ftlift_from_stable7600_v1/
```

### 训练 Go2X5 Stable Base

```bash
cd low-level/legged_gym/scripts

python train.py --headless --task go2x5 \
  --proj_name go2x5-low --exptid go2x5_stable_base_v1 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

恢复：

```bash
python train.py --headless --task go2x5 \
  --proj_name go2x5-low --exptid go2x5_stable_base_v1 \
  --resumeid go2x5_stable_base_v1 --checkpoint -1 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

## 本地模型回放

### 平地 0 速度 GUI 回放

注意：当前 `play.py` 的 `--headless` 默认是 `True`，因此 GUI 回放使用 Python wrapper 手动设置 `args.headless=False`。

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

GUI 注意事项：

- 按 `F` 切换自由相机。
- 不要按数字键 `1` 到 `8`，因为回放只有 1 个 env，按这些键会触发相机索引越界。
- 数字键 `0` 可以使用。

### 复杂地形 GUI 回放

去掉 `--flat_terrain`：

```bash
python -c "import isaacgym; import play as p; p.EXPORT_POLICY=False; p.SAVE_ACTOR_HIST_ENCODER=False; p.RECORD_FRAMES=False; p.MOVE_CAMERA=False; args=p.get_args(); args.headless=False; p.play(args)" \
  --task go2x5 \
  --proj_name go2x5-low --exptid go2x5_stable_base_v1 \
  --checkpoint 7600 \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --observe_gait_commands
```

## W&B 曲线

当前 low-level 使用 W&B，而不是 TensorBoard event 文件。

本地已有 W&B 离线日志：

```text
low-level/legged_gym/envs/logs/wandb/offline-run-20260531_024919-7qix21n0
low-level/legged_gym/envs/logs/wandb/offline-run-20260530_230740-ihrya8px
```

同步命令：

```bash
cd /home/lemon/research/Issac/visual-wholebody-control-go2x5
conda activate vwc_go2x5

wandb sync low-level/legged_gym/envs/logs/wandb/offline-run-20260531_024919-7qix21n0
```

如果已经部分同步：

```bash
wandb sync --include-synced --append \
  low-level/legged_gym/envs/logs/wandb/offline-run-20260531_024919-7qix21n0
```

重点曲线：

```text
Train/mean_reward
Train/mean_episode_length
Train/dones
Episode_rew/rew_collision
Episode_rew/rew_base_height
Episode_rew/rew_orientation_walking
Episode_rew/rew_tracking_lin_vel_max
Episode_rew/rew_tracking_ee_world
Episode_rew/rew_roll
Episode_rew/rew_work
Episode_rew/rew_torques
Policy/leg_mean_noise_std
Loss/value_function
Loss/surrogate
Loss/hist_latent_loss
Loss/priv_reg_loss
```

已知 W&B 问题：

- 离线日志中可能保存了指向远端路径的源码 symlink。
- 如果同步时报缺失 `files/manip_loco/b1z1_config.py` 或 `files/manip_loco/manip_loco.py`，需要用当前仓库里的真实文件替换断掉的 symlink。

## Go2X5 High-Level 训练路线

### 准备 high-level 配置

不要直接覆盖默认配置。先复制：

```bash
cd /home/lemon/research/Issac/visual-wholebody-control-go2x5/high-level

cp data/cfg/go2x5_pickmulti.yaml data/cfg/go2x5_pickmulti_stable17600.yaml
```

把 low-level checkpoint 路径改为当前有效模型：

```bash
sed -i 's#low_policy_path:.*#low_policy_path: "../low-level/logs/go2x5-low/go2x5_stable_base_v1/model_17600.pt"#' \
  data/cfg/go2x5_pickmulti_stable17600.yaml
```

如果使用 FtLift 模型，则改成：

```text
../low-level/logs/go2x5-low/go2x5_ftlift_from_stable7600_v1/<具体checkpoint>.pt
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

### High-Level 固定桌高 Teacher 训练

固定桌高是推荐的第一阶段，先不要直接随机桌高。

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

恢复：

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

## 评估标准

### Low-Level 可视化检查

平地 0 速度：

- 机身不能持续下蹲或趴地。
- roll / pitch 不能明显发散。
- 四足不能高频抖动、拖地、乱划。
- 机械臂目标变化不能明显带倒整机。

复杂地形：

- 不能一开始就摔倒。
- 脚不能长期拖地。
- 地形变化时机身不能剧烈晃动。
- 轻微扰动后应能恢复。

### High-Level 检查

Smoke test：

- 环境能正常创建。
- low-level policy 能正常加载。
- 不出现观测维度 mismatch。
- 机器人不会还没抓就倒。

Teacher 训练：

- 关注 total success rate。
- 关注不同物体 success rate。
- 区分失败原因是抓取失败，还是底盘先失稳。
- 如果底盘先失稳，应回到 low-level / FtLift。

## 踩坑与修复

### Isaac Gym 动态库缺失

现象：

```text
libpython3.8.so.1.0 not found
libmem_filesys.so not found
carb::gym::Gym acquire failed
```

修复：

- `LD_LIBRARY_PATH` 必须包含 conda lib、Isaac Gym bindings、USD plugins。
- 当前 `train.py` 和 `play.py` 已经在入口处自动处理动态库路径并重启进程。

### W&B 在线初始化超时

现象：

```text
wandb.errors.errors.CommError: Run initialization has timed out
```

修复：

```bash
export WANDB_MODE=offline
```

当前 `train.py` 已经尊重 `WANDB_MODE`。

### RTX 40 系 + 旧 PyTorch

现象：

```text
RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)
```

原因：

- `torch 1.10.2 + cu113` 对 RTX 40 系支持不足。

修复：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
rm -rf ~/.cache/torch_extensions/py38_cu113
```

### GUI 回放全局变量缺失

用 `import play as p` 方式回放时，`play.py` 中以下变量不会自动定义：

```text
EXPORT_POLICY
SAVE_ACTOR_HIST_ENCODER
RECORD_FRAMES
MOVE_CAMERA
```

因此 wrapper 命令里手动设置为 `False`。

### GUI 数字键导致相机索引越界

现象：

```text
IndexError: index 1 is out of bounds for dimension 0 with size 1
```

原因：

- 回放时只有 1 个 env。
- viewer 注册了数字键 `0` 到 `8`。
- 按 `1` 到 `8` 会尝试切换到不存在的机器人。

规避：

- 不按 `1` 到 `8`。
- 按 `F` 使用自由相机。

候选修复：

- 在 `base_task.py` 中给 `lookat(i)` 加 `i < self.num_envs` 的边界判断。

### High-Level 配置路径陈旧

问题：

默认 `go2x5_pickmulti.yaml` 的 `low_policy_path` 仍指向旧模型。

要求：

- 每次 high-level 实验都复制一份新 yaml。
- 明确写入当前使用的 low-level checkpoint。

## 阶段记录

### 2026-05-30 至 2026-05-31

- 建立 `go2x5_stable_base_v1` 作为第一版稳定 low-level。
- 下载并整理 checkpoint 到 `model_17600.pt`。
- 选择 `model_7600.pt` 作为中期参考模型，用于对比回放。
- 本地回放时发现旧 PyTorch 不支持 RTX 40 系，升级到 `torch 2.4.1+cu121`。
- W&B 离线同步时修复了断掉的源码文件 symlink。

### 2026-06-01

- 明确 `go2x5_ftlift` 是可选迁移增强步骤，不是原论文固定流程。
- 决定如需 FtLift，则从 `go2x5_stable_base_v1/model_7600.pt` 开始。
- 明确 Go2X5 high-level 路线：
  - 先修正 `low_policy_path`
  - smoke test
  - 固定桌高 teacher
  - 随机桌高
  - student / BC
- 创建英文开发文档 `docs/development_log.md`。
- 创建中文开发文档 `docs/development_log_zh.md`。

## 下一步计划

1. 在远端服务器 tmux 中启动 `go2x5_ftlift_from_stable7600_v1`。
2. 训练若干 checkpoint 后同步日志并观察曲线。
3. 本地回放 FtLift checkpoint，检查平地 0 速度和复杂地形稳定性。
4. 选择 stable base 或 FtLift checkpoint 作为 high-level low policy。
5. 复制 high-level Go2X5 yaml，修正 `low_policy_path`。
6. 运行 fixed table high-level smoke test。
7. 若 smoke test 稳定，启动 fixed table teacher 训练。
8. 如果 high-level 中底盘失稳，继续调整 low-level / FtLift，而不是直接扩大 high-level 随机化。

## 禁止事项

- 不要用默认旧路径 `go2x5_b1style_20260418/model_11800.pt` 训练 high-level。
- 不要加载带 gait command 训练的模型时省略 `--observe_gait_commands`。
- 不要把日志里的 `Dones: 0.00` 理解为完全没有 reset。
- 不要在 high-level 还没确认底盘稳定时就判断抓取策略失败。
- 不要在 RTX 40 系本地机器上使用 `torch 1.10.2 + cu113` 回放。
- 不要在远端服务器普通 terminal 里跑长训练，必须使用 tmux。
