# memory.md

本文件记录本轮 Go2X5 VBC 复现讨论形成的项目记忆、判断和后续方向。它不是最终实验报告，而是给后续继续开发时快速恢复上下文使用。

## 项目目标

用户希望在 `visual-wholebody-control-go2x5` / `VBC-gx` 仓库中复现 Visual Whole-Body Control 框架，将原始 B1+Z1 方案迁移到 Unitree Go2 + ARX-X5。

当前优先级：

1. 先得到稳定、可信的 Go2X5 low-level model。
2. 再进行 Go2X5 high-level pick-multi teacher 训练。
3. 只有当 low-level 可稳定回放、可追踪 EE 目标且不易倒下时，high-level 成功率才有意义。

## 已讨论和处理过的问题

### 1. 环境配置

本地环境：

- 本地 repo：`/home/lemon/research/Issac/visual-wholebody-control-go2x5`
- conda env：`vwc_go2x5`
- 本地 GPU：RTX 4060
- 旧版 PyTorch 1.10.2 + CUDA 11.3 在 RTX 4060 上会触发 `nvrtc: invalid value for --gpu-architecture`。
- 建议 / 已使用 PyTorch 2.4.1 + CUDA 12.1 wheel。

远端环境：

- 服务器 repo：`/data4/duanzhibo/xhq_workload/VBC-gx`
- conda env：`b1z1`
- 所有命令都需要限制 GPU 可见性。
- 使用第二张物理卡时，应设置 `CUDA_VISIBLE_DEVICES=1`，但命令内部仍写 `cuda:0`。

Isaac Gym 相关：

- 必须处理 `LD_LIBRARY_PATH`，包括 conda lib、Isaac Gym binding 目录和 USD plugin 目录。
- 必须避免先 import torch 后 import isaacgym。
- `train.py`、`play.py`、`visualize_go2x5_config.py` 已加入动态库路径自举逻辑。

### 2. W&B 问题

训练中多次遇到 W&B SSL / timeout：

```text
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec
```

结论：

- 这会阻塞训练启动，不只是影响曲线显示。
- 远端训练默认应使用：

```bash
export WANDB_MODE=offline
```

离线同步要指定单个 run，而不是 sync 整个 `wandb/` 目录。

曾遇到 sync 报缺失 `files/manip_loco/b1z1_config.py`，原因是离线 run 记录了不存在的 saved file / symlink。后续 `train.py` 已改成按 task 保存对应配置，Go2X5 应保存 `go2x5_config.py`。

### 3. Low-Level 结构理解

原 B1Z1 low-level 训练逻辑：

- 原始代码中 low-level policy 输出 18 维：
  - 12 维腿部动作
  - 6 维机械臂动作
- 但 VBC 里手臂和腿并不是简单平行控制。手臂目标、EE tracking、IK / 动力学耦合共同影响训练。

Go2X5 迁移后，我们认为更合理的是：

- low-level policy 只控制腿：12 维。
- X5 手臂由 IK / 位置控制跟踪 EE 目标。
- 腿部 policy 通过观测 EE 目标、实际 DOF、身体状态，学习如何配合手臂移动保持稳定。

因此当前 Go2X5 已改为 leg-only low-level：

- `num_actions = 12`
- `num_torques = 12`
- `num_leg_actions = 12`
- `num_arm_actions = 0`
- `action_history` 只记录腿部 12 维动作。

这使 Go2X5 的 low-level 更像“腿部稳定器 + 手臂 IK 扰动补偿器”，不是 B1Z1 原始的 18 维直接动作策略。

### 4. 观测维度问题

早期文档中指出：

- `num_proprio` 不应是 72，而应是 66。
- 原因是 action history 不应包含 18 维全动作，而应只包含 12 维腿部动作。
- 正确观测：

```text
2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3 = 66
num_observations = 66 * 11 + 18 = 744
```

当前 Go2X5 配置已是：

- `num_proprio = 66`
- `num_observations = 744`

### 5. URDF 和模型问题

用户怀疑原 Go2X5 URDF 不正确，要求使用 `Go2-X5-lab` 仓库内的 Go2-X5 URDF。

已经完成的方向：

- 新增 / 使用 `low-level/resources/robots/go2x5/go2_x5.urdf`
- 复制 Go2-X5-lab 的 mesh 资源到 `low-level/resources/robots/go2x5/meshes/`
- 历史上曾把 Go2 `.dae` mesh 从官方 `Z_UP` 改成 `Y_UP`；该修改已证实会
  导致腿部视觉网格旋入机身平面，现已恢复 Go2-X5-lab 的 `Z_UP`。
- 保留 `arm_eef_link` 作为 EE body，避免 Isaac Gym fixed joint collapse 后找不到末端

验证过的 headless smoke 输出应包括：

- `num_dofs: 20`
- `num_actions: 12`
- `num_torques: 12`
- `num_bodies: 28`
- `EE Gripper index: 25`
- `jacobian_whole shape: [*, 28, 6, 26]`

### 6. EE 目标和可视化理解

可视化脚本：

```text
low-level/legged_gym/scripts/visualize_go2x5_config.py
```

用户问过蓝色球含义：

- 黄色球：当前 EE 目标点。
- 深蓝球：实际 EE 末端位置。
- 青色 / 蓝绿色球：目标球采样中心，不是实际末端。
- 红色轨迹点：目标轨迹历史。

用户希望：

- 末端抓取目标最远端稍微远一点。
- 世界 / 目标中心向下移动 5 cm。
- 目标机身高度改到 0.33。

当前配置反映：

- `sphere_center.z_invariant_offset = 0.37`
- `pos_l = [0.20, 0.50]`
- `base_height_target = 0.33`

### 7. 回放问题

曾遇到：

```text
NameError: EXPORT_POLICY is not defined
```

原因：

- 使用 `python -c "import play as p; ... p.play(args)"` 绕过了 `play.py` 文件底部定义 `EXPORT_POLICY = False` 的路径。

建议：

- 直接运行 `python play.py ...`。

曾遇到单环境 viewer 快捷键报：

```text
IndexError: index 1 is out of bounds for dimension 0 with size 1
```

已修复方向：

- viewer `lookat` index 对 env 数量取模。

### 8. High-Level 训练结论

Go2X5 high-level pick-multi 已尝试过，结果：

- success rate 长期为 0。
- 训练中还出现过 NaN / invalid action / PhysX CUDA illegal memory access。

当前判断：

- high-level 成功率为 0 的核心原因很可能不是 high-level PPO 本身，而是 low-level 不够稳定。
- high-level 环境中扰动更多，桌面、物体、抓取动作会放大 low-level 的站立和末端跟踪问题。
- 在 low-level 未稳定前，不建议投入大量 high-level 训练。

high-level 训练必须检查：

- `high-level/data/cfg/go2x5_pickmulti.yaml` 中的 `low_policy_path`
- 该路径必须指向当前要使用的 Go2X5 low-level checkpoint
- 当前 high-level 仍大量继承 `b1z1_*` 文件命名，不能只凭文件名判断任务是否错误

### 9. 当前训练曲线判断

最近 low-level 训练日志中，用户给出的状态：

```text
Learning iteration 13153/45000
Mean reward: 5.48
Mean episode length: 345.65
Dones: 0.00
rew_collision: -35.8780
rew_hip_pos: -5.6218
rew_roll: -3.3959
rew_torques: -6.3190
rew_work: -8.0310
rew_tracking_lin_vel_max: 13.5974
rew_tracking_contacts_shaped_vel: 22.1827
rew_walking_dof: 25.6794
rew_tracking_ee_world: 10.2802
```

我的判断：

- 这不是完全没学会。
- `Dones = 0.00` 说明它不是频繁倒地。
- 正向 locomotion / gait / EE 项已经有学习。
- 但 reward 被 collision、hip_pos、roll、torques、work 长期压住。
- `collision = -10` 且 `metric_collision` 较高，说明 thigh/calf 接触或拖腿非常严重。
- 当前配置会把策略推向“能勉强不倒、但姿态差、拖腿、高能耗”的局部最优。

因此不建议继续按当前满强度配置硬训到 45000。

### 10. B1Z1 与 Go2X5 的训练强度比较

用户问：B1Z1 是不是也这样 45000 轮同一强度？

检查结论：

- 当前仓库中 B1Z1 也是：
  - `terrain.curriculum = False`
  - `commands.curriculum = True` 但 runner 中主动调用 command curriculum 的代码被注释
  - domain randomization 从第 0 轮开始启用
  - PPO 内部只有少量 `priv_reg`、`mixing_schedule` 等 schedule

所以 B1Z1 也不是严格 staged training。

但关键区别：

- B1Z1 是原始任务配置，机体、腿长、Z1 机械臂、安装位置、reward 尺度都适配过。
- Go2X5 是迁移平台，Go2 更矮，X5 安装位置、质量分布、目标可达范围、接触行为都不同。
- B1Z1 能承受满强度，不代表 Go2X5 能直接承受。

我的观点：

- 不应再机械追求“Go2X5 和 B1Z1 配置完全一样”。
- 应保留 VBC 控制路线，但 Go2X5 训练流程需要 staged training。

## 当前推荐训练路线

### Stage 1：稳定低层

目标：

- 不倒。
- thigh/calf collision 明显下降。
- torque / work 不爆。
- EE 轻量跟踪，不压过身体稳定。

建议调整：

```python
domain_rand.push_robots = False
domain_rand.friction_range = [0.6, 1.5]
domain_rand.added_mass_range = [0.0, 5.0]
domain_rand.added_com_range_x = [-0.05, 0.05]
domain_rand.added_com_range_y = [-0.05, 0.05]
domain_rand.added_com_range_z = [-0.03, 0.03]
domain_rand.leg_motor_strength_range = [0.85, 1.15]

rewards.scales.collision = -3.0  # or -5.0
rewards.scales.hip_pos = -0.1
rewards.scales.torques = -1e-5
rewards.scales.work = -0.001
rewards.scales.roll = -1.0
rewards.scales.base_height = -0.5
rewards.arm_scales.tracking_ee_world = 0.4 ~ 0.6
```

还建议新增 thigh/calf collision contact force threshold，不要用极低阈值把轻微擦碰也计为严重 collision。

### Stage 2：增强 EE 跟踪

目标：

- 在 Stage 1 稳定基础上，提高 EE tracking。
- 逐步增大目标球范围和 EE reward。

### Stage 3：鲁棒性 fine-tune

目标：

- 加回 push。
- 加强 domain randomization。
- 加复杂 terrain。
- 让模型适应 high-level pick-multi 中更强扰动。

### Stage 4：High-Level

目标：

- 使用稳定 low-level checkpoint。
- 固定或小范围随机桌面高度开始。
- 再训练 Go2X5PickMulti teacher。

## 当前风险

- 当前 low-level 配置虽然基础修复完成，但 reward / randomization 仍偏 B1Z1 满强度风格。
- 当前 high-level 配置中的 low-level path 可能仍指向旧模型，启动前必须检查。
- 如果再次修改 action 维度、观测维度或 URDF DOF，已有 checkpoint 可能不可复用。
- 训练日志和模型很多，压缩 / scp 时要确认路径，避免拉错包。

## 对后续 agent 的要求

- 先读 `AGENT.md` 和本文件，再改代码。
- 不要默认当前 45000 轮配置是最终方案。
- 不要盲目把 Go2X5 配置改回 B1Z1 数值。
- 任何训练命令必须显式设置 `CUDA_VISIBLE_DEVICES`。
- 修改 low-level 配置后，先跑 headless smoke test，再 GUI 可视化，再训练。
- 如果用户要求“推到远端”，先检查 `git status --short`，不要把日志、模型、压缩包提交进去。
