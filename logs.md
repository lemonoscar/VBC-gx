# Go2X5 VBC 开发日志

本文档用于长期维护 Go2 + ARX-X5 复现 Visual Whole-Body Control 的开发进展。后续每次训练、配置修改、bug 修复、模型评估，都应在本文档中追加记录。

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
- 最近确认提交：`e116862 Update Go2X5 low-level URDF and training config`
- 已推送的重要提交：
  - `bda22b2 Align Go2X5 low-level stable config`
  - `a191dd7 Respect WANDB_MODE in low-level training`
  - `b4d0510 Refactor Go2X5 low-level to leg-only policy`
  - `e116862 Update Go2X5 low-level URDF and training config`
- 本地路径：`/home/lemon/research/Issac/visual-wholebody-control-go2x5`
- 远端服务器路径：`~/xhq_workload/VBC-gx`
- 当前工作区备注：存在未跟踪的项目记忆/文档文件，如 `AGENT.md`、`memory.md`、`docs/`、`alighment.md`、`logs.md` 等；训练日志和压缩包不应直接提交。

## 当前训练状态

### Go2X5 Low-Level Stable Base

#### 5.28 训练记录

本地已经验证 checkpoint 可以在 `vwc_go2x5` 环境中正常 `torch.load`。

训练趋势摘要：

- iteration 39 左右：`Mean reward` 为负，`Mean episode length` 很短，训练初期不稳定。
- iteration 2500 左右：episode length 提升到约 366，reward 变为正。
- iteration 5000 左右：episode length 约 444，reward 约 9。
- iteration 10000 左右：episode length 约 462，但 collision、base height、roll 等惩罚仍偏大。
- 日志中的 `Dones: 0.00` 是显示保留两位小数后的结果，不代表完全没有 reset。

当前判断：

- `go2x5_stable_base_v1` 已经可以用于可视化检查和 high-level test。
- 经过 7600 轮训练后，可以得到较稳定的 flat 和 rough 地形单环境回放。
- 飞书文档中的图片/视频未随文本导出；后续如果需要留档，建议把关键截图或视频文件放到 `docs/` 下并在本文档中补路径。

#### 6.5 训练内容修改

6.5 之后的 low-level 对齐重点从“继续沿用早期 Go2X5 资源”转为“对齐 Go2-X5-lab 资源和 leg-only low-level 结构”：

- low-level 机器人资源改为 `low-level/resources/robots/go2x5/go2_x5.urdf`。
- 末端执行器 body 改为 `arm_eef_link`。
- 机械臂安装偏置改为 `arm.base_offset = [0.085, 0.0, 0.094]`。
- Go2 mesh 资源完成整理，并修正 DAE up-axis，避免 Isaac Gym 中视觉姿态不一致。
- low-level policy 改为只输出腿部 12 维动作：
  - `num_actions = 12`
  - `num_torques = 12`
  - `num_leg_actions = 12`
  - `num_arm_actions = 0`
- 手臂由 IK 和位置目标驱动，不再由 low-level PPO 直接输出 6 维臂动作。
- observation 中的 action history 改为 12 维腿部动作历史。
- 当前基础观测维度：
  - `num_proprio = 66`
  - `num_priv = 18`
  - `history_len = 10`
  - `num_observations = 744`
- 初始身体高度和奖励高度目标调整：
  - `init_state.pos = [0.0, 0.0, 0.32]`
  - `rewards.base_height_target = 0.33`
- EE 目标球采样范围调整：
  - `sphere_center.x_offset = 0.22`
  - `sphere_center.z_invariant_offset = 0.37`
  - `pos_l = [0.20, 0.50]`
  - `pos_p = [-0.6, pi / 3]`
- X5 初始姿态调整为自然向前折叠，避免 `arm_joint1` 初始 yaw 和 IK target 相差约 pi 造成启动力矩冲击。
- 机械臂位置控制增益调整为：
  - `arm_pos_stiffness = 120.0`
  - `arm_pos_damping = 12.0`
  - `ik_gain = 0.5`

当前判断：

- 这些修改完成了基础形态和接口对齐。
- 但当前 domain randomization 和 reward 仍偏强，不应把当前 45000 轮配置视为最终最优训练方案。
- 最近训练显示 reward 容易被 `collision`、`hip_pos`、`roll`、`torques`、`work` 压住，应进入 staged training。

### Go2X5 FtLift Low-Level

#### 5.29 训练内容

`go2x5_ftlift` 不是原论文中的标准步骤，也不是任务书明确要求的固定步骤。它是当前仓库中为了 Go2X5 high-level 抓取迁移而设计的 low-level fine-tune。

目的：

- 让机器人更适应 high-level 抓取时的低位伸臂。
- 增强机械臂前伸、低目标跟踪和抬物扰动下的底盘支撑。
- 让 low-level 的训练分布更接近 high-level 抓取环境。

它不是用来学习抓取策略的。抓取策略仍由 high-level teacher 学习。

原计划：

- 从 `go2x5_stable_base_v1/model_7600.pt` 开始 fine-tune。
- 输出 run 名称：`go2x5_ftlift_from_stable7600_v1`

#### 6.6 训练计划调整

当前代码中 `go2x5_ftlift_config.py` 仍只是继承 `Go2X5RoughCfg` 的别名，除了 `experiment_name = 'go2x5_ftlift'` 外没有真正改变目标分布、reward 或 domain randomization。

因此，6.6 后的计划调整为：

- 不把当前 `go2x5_ftlift` 视为已经实现的低位伸臂增强阶段。
- 如果继续使用 `go2x5_ftlift` 名称，应先补齐独立配置。
- FtLift 配置应显式调整：
  - 更偏低位和前伸的 EE 目标分布。
  - 更接近 high-level 抓取时的站立/低速移动命令分布。
  - 更强的后腿支撑、站立稳定、低位目标姿态 shaping。
  - 较温和的随机化，避免 fine-tune 阶段破坏已有稳定步态。
- 在 FtLift 未真正实现前，high-level 更建议先使用已经验证可回放的 `go2x5_stable_base_v1` checkpoint 做 smoke test。

### Go2X5 High-Level

#### 6.6 训练内容和初步计划

状态：计划/训练中，优先级低于 low-level 稳定性验证。

当前 high-level 风险：

- 默认 `high-level/data/cfg/go2x5_pickmulti.yaml` 中的 `low_policy_path` 仍指向旧模型：

```yaml
low_policy_path: "../low-level/logs/go2x5-low/go2x5_b1style_20260418/model_11800.pt"
```

- high-level 当前机器人 asset 仍指向旧 URDF：

```yaml
assetFileRobot: "go2x5/urdf/go2_arx_x5.urdf"
```

- low-level 最新 URDF 使用 `arm_eef_link`，而 high-level 代码中仍有 `ee_gripper_link`、`link6/link7/link8` 等旧命名依赖。

因此，high-level 训练前必须先确认：

- low-level checkpoint 路径明确且存在。
- 训练和加载 checkpoint 时都带 `--observe_gait_commands`。
- high-level 低层 policy 的 `lowPolicyNumActions = 12`。
- small-env smoke test 不出现 observation shape mismatch、policy load error、IK NaN、机器人未抓先倒等问题。

推荐路线：

1. 复制一份新的 high-level config，不直接覆盖默认文件。
2. 将 `low_policy_path` 指向当前要评估的 Go2X5 low-level checkpoint。
3. 固定桌高 `--table_height 0.25` 做 small-env smoke test。
4. smoke test 通过后，再做固定桌高 teacher 训练。
5. 固定桌高成功率上升后，再考虑随机桌高和 student / BC。

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
- 当前 low-level policy action：12，仅腿部动作，不包含机械臂和夹爪

当前 low-level 约定：

- policy 只学习腿部行为。
- 机械臂由 IK / PD 位置控制跟踪 EE 目标。
- `ManipLoco.step()` 中根据 EE target 和实际 EE pose 计算 IK delta，并设置机械臂位置目标。
- `_compute_torques()` 只根据 12 维腿部动作计算腿部 torque，机械臂 torque 置零，由 position target 驱动。
- 当前 PPO policy 配置中 `num_arm_actions = 0`。

历史说明：

- 早期 B1Z1 / 旧 Go2X5 迁移曾使用 18 维 low-level policy，其中 12 维腿部 + 6 维机械臂。
- 当前最新 Go2X5 low-level 已不再使用 18 维动作。
- 如果旧文档中出现 `policy action: 18` 或 `actions[:, 12:] = 0`，应按历史记录理解，不代表当前最新代码。

观测设计：

- 不带 gait command 时：`num_proprio = 66`
- privileged obs：`num_priv = 18`
- history length：`10`
- 不带 gait command 总观测：`66 * 11 + 18 = 744`
- 带 `--observe_gait_commands` 时，proprio 增加 5 维，总观测变为 `799`

重要规则：

如果 checkpoint 是带 `--observe_gait_commands` 训练的，训练、恢复、回放、high-level 加载时都必须带 `--observe_gait_commands`，否则观测语义不一致。

### 6.5 最新 low-level 设置修正

当前最新 low-level 设置以 `go2x5_config.py` 为准：

- `asset.file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2x5/go2_x5.urdf'`
- `asset.gripper_name = "arm_eef_link"`
- `arm.base_offset = [0.085, 0.0, 0.094]`
- `num_actions = 12`
- `num_torques = 12`
- `num_proprio = 66`
- `num_priv = 18`
- `num_observations = 744`
- `base_height_target = 0.33`
- `init_state.pos = [0.0, 0.0, 0.32]`
- `runner.max_iterations = 45000`
- `runner.save_interval = 200`

当前仍偏强的训练设置：

- `domain_rand.friction_range = [0.3, 3.0]`
- `domain_rand.added_mass_range = [0.0, 15.0]`
- `domain_rand.added_com_range_x/y/z = [-0.15, 0.15]`
- `domain_rand.leg_motor_strength_range = [0.7, 1.3]`
- `domain_rand.push_robots = True`
- `rewards.scales.collision = -10.0`
- `rewards.scales.hip_pos = -0.3`
- `rewards.scales.roll = -2.0`
- `rewards.scales.torques = -2.5e-5`
- `rewards.scales.work = -0.003`

建议后续 Stage 1 稳定训练：

- `push_robots = False`
- `friction_range = [0.6, 1.5]`
- `added_mass_range = [0.0, 5.0]`
- COM randomization 收窄到约 `x/y [-0.05, 0.05]`，`z [-0.03, 0.03]`
- `leg_motor_strength_range = [0.85, 1.15]`
- `collision = -3.0` 或 `-5.0`
- `hip_pos = -0.1`
- `torques = -1e-5`
- `work = -0.001`
- `tracking_ee_world = 0.4 ~ 0.6`

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
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:$PWD/low-level:$PYTHONPATH"
```

确认只看到一张卡：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

注意：设置 `CUDA_VISIBLE_DEVICES=1` 后，程序内部仍使用 `cuda:0`：

```bash
--sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0
```

### 从 7600 开始 FtLift Fine-Tune

当前 `go2x5_ftlift` 仍是别名配置。以下命令仅适用于确认已补齐 FtLift 独立配置后。

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

注意：如果直接运行 `python play.py ...` 可以避免 `python -c` 路径下全局变量未定义的问题。若继续使用 wrapper，需要手动设置相关全局变量。

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
- 不要按数字键 `1` 到 `8`，因为回放只有 1 个 env，旧版本会触发相机索引越界。
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

当前 low-level 使用 W&B，重点关注：

- `Train/mean_reward`
- `Train/mean_episode_length`
- `Train/dones`
- `Episode_rew/rew_collision`
- `Episode_rew/rew_base_height`
- `Episode_rew/rew_orientation_walking`
- `Episode_rew/rew_tracking_lin_vel_max`
- `Episode_rew/rew_tracking_ee_world`
- `Episode_rew/rew_roll`
- `Episode_rew/rew_work`
- `Episode_rew/rew_torques`
- `Policy/leg_mean_noise_std`
- `Loss/value_function`
- `Loss/surrogate`
- `Loss/hist_latent_loss`
- `Loss/priv_reg_loss`

当前最近日志判断：

- `Dones` 很低，说明不是频繁倒地。
- locomotion、gait、EE tracking 都有学习。
- 但 collision、base height、roll、torques、work 仍偏大。
- 继续硬训满强度配置收益有限，应优先做 staged training。

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

因此 wrapper 命令里需要手动设置为 `False`。更稳妥的方式是直接运行 `python play.py ...`。

### GUI 数字键导致相机索引越界

现象：

```text
IndexError: index 1 is out of bounds for dimension 0 with size 1
```

原因：

- 回放时只有 1 个 env。
- viewer 注册了数字键 0 到 8。
- 按 1 到 8 会尝试切换到不存在的机器人。

规避：

- 不按 1 到 8。
- 按 `F` 使用自由相机。

候选修复：

- 在 `base_task.py` 中给 `lookat(i)` 加 `i < self.num_envs` 的边界判断。

### High-Level 配置路径陈旧

问题：

- 默认 `go2x5_pickmulti.yaml` 的 `low_policy_path` 仍指向旧模型。

要求：

- 每次 high-level 实验都复制一份新 yaml。
- 明确写入当前使用的 low-level checkpoint。
- 不要用默认 yaml 直接启动正式 high-level 训练。

### High-Level 与最新 Low-Level URDF 未完全一致

问题：

- low-level 最新使用 `go2_x5.urdf` 和 `arm_eef_link`。
- high-level 当前仍使用旧 `go2_arx_x5.urdf` 和 `ee_gripper_link`。

影响：

- EE body index、Jacobian、gripper contact、success gate 都可能不一致。
- 即使 checkpoint 能加载，high-level 结果也可能不能反映最新 low-level 的真实能力。

后续修复方向：

- 统一 high-level / low-level URDF。
- 将 high-level 的末端 link、wrist link、finger links、joint names 做成配置项。
- 为新 URDF 写 high-level smoke test，确认 body index 和 DOF order。

## 下一步计划

1. 基于当前 Go2X5 配置新建 Stage 1 stable low-level 配置，降低随机化和过强 penalty。
2. 修正 collision reward 统计阈值，避免轻微 thigh/calf 擦碰被严重惩罚。
3. 真正实现 `go2x5_ftlift` 的独立目标分布和 reward。
4. 回放并比较 `model_7600.pt`、`model_10000.pt`、`model_17600.pt`，选出 high-level smoke test 用的 low-level checkpoint。
5. 复制 high-level config 并显式设置 `low_policy_path`。
6. 固定桌高 `0.25` 做 Go2X5 high-level smoke test。
7. 如果 high-level 中底盘先失稳，回到 low-level / FtLift，而不是继续调 high-level reward。
