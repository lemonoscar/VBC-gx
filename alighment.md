# Go2X5 Alignment Notes

本文说明本仓库相对于原始 Visual Whole-Body Control 仓库的主要修改，以及这些修改如何服务于 B1+Z1 到 Unitree Go2 + ARX-X5 的平台对齐。

本文不是论文结论，也不是最终实验报告。它记录的是当前复现工作的工程贡献、设计取舍和仍需闭环的问题。

## 1. 背景

原始 Visual Whole-Body Control 代码主要面向 B1 四足机器人 + Unitree Z1 机械臂。当前仓库的目标是把同一套视觉全身控制路线迁移到 Go2 四足机器人 + ARX-X5 机械臂。

这个迁移不是简单替换 URDF。Go2X5 和 B1Z1 在以下方面都有实质差异：

- 机器人几何尺寸、身体高度和腿部默认姿态不同。
- 机械臂安装位置、关节命名、末端 link 和夹爪 DOF 不同。
- Go2 更矮，低位伸臂和前伸抓取更容易影响底盘姿态。
- 原始 B1Z1 低层策略的 18 维动作设计不一定适合 Go2X5。
- high-level 抓取环境必须使用和 low-level 一致的底层策略接口。

因此，本仓库的核心工作是让 asset、action space、observation、IK、reward、训练配置和 high-level 接口逐步对齐 Go2X5 平台。

## 2. 机器人资源对齐

本仓库新增并启用了 Go2X5 机器人资源：

- low-level 当前使用 `low-level/resources/robots/go2x5/go2_x5.urdf`。
- 该 URDF 来自 Go2-X5-lab 方向的机器人模型，而不是继续沿用早期迁移时的旧 `go2_arx_x5.urdf`。
- 复制并整理了 Go2 和 X5 相关 mesh 资源到 `low-level/resources/robots/go2x5/meshes/`。
- 修正 Go2 DAE mesh 的 `up_axis`，避免 Isaac Gym 中出现视觉和碰撞姿态不一致。
- 保留 `arm_eef_link` 作为末端 link，避免 fixed joint collapse 后找不到末端刚体。

这部分贡献解决的是最基础的平台真实性问题：low-level 训练不能继续依赖和真实 Go2X5 装配不一致的旧模型。

## 3. Low-Level 动作空间对齐

原始 B1Z1 low-level policy 输出 18 维动作：

- 12 维腿部动作。
- 6 维机械臂动作。

当前 Go2X5 low-level 已重构为 leg-only policy：

- `num_actions = 12`
- `num_torques = 12`
- policy 只输出腿部动作。
- 机械臂由 IK 和位置目标驱动。
- 夹爪不属于 low-level policy 动作输出。

对应代码集中在：

- `low-level/legged_gym/envs/manip_loco/go2x5_config.py`
- `low-level/legged_gym/envs/manip_loco/manip_loco.py`
- `third_party/rsl_rl` 中 ActorCritic 的 action head 配置

这个改动的意义是把 Go2X5 low-level 明确定位为“腿部稳定器 + 手臂 IK 扰动补偿器”，而不是照搬 B1Z1 的“腿和臂都由 PPO 直接输出动作”的结构。这样更符合当前 Go2X5 迁移阶段的稳定性需求。

## 4. Observation 维度对齐

leg-only policy 改动后，原来的观测维度也必须同步修改。当前 Go2X5 low-level 的基础 proprioception 为 66 维：

```text
2 body orientation
+ 3 base angular velocity
+ 18 dof position without gripper
+ 18 dof velocity without gripper
+ 12 leg action history
+ 4 foot contacts
+ 3 velocity commands
+ 3 EE goal position
+ 3 EE goal orientation placeholder
= 66
```

历史长度为 10，privileged observation 为 18 维，因此基础 observation 为：

```text
66 * 11 + 18 = 744
```

如果训练和回放时启用 `--observe_gait_commands`，会额外加入 5 维 gait command 相关观测。这个选项必须在训练、恢复、回放和 high-level 加载时保持一致，否则 checkpoint 的输入语义不匹配。

这一部分贡献主要避免了迁移时最容易出现的隐性错误：action space 已经变成 12 维，但 observation 仍残留 18 维动作历史。

## 5. 手臂 IK 和末端目标对齐

Go2X5 low-level 中，手臂不再由 policy 直接输出动作，而是通过当前 EE target 和实际 EE pose 计算 IK delta，再设置机械臂位置目标。

相关改动包括：

- 使用 `arm_eef_link` 作为 low-level 末端刚体。
- 从配置读取 `arm.base_offset`，当前为 `[0.085, 0.0, 0.094]`。
- 将初始手臂姿态调整为向前折叠，避免启动时因为 yaw 差接近 pi 产生巨大力矩冲击。
- IK 中加入阻尼最小二乘求解。
- 对 EE 四元数归一化加入零范数保护，减少 NaN 风险。
- 将 IK 增益做成配置项，当前 `ik_gain = 0.5`，用于缓解 overshoot。

这部分贡献让 Go2X5 的手臂行为从“网络动作直接控制”转为“目标驱动的可解释 IK 控制”，降低了早期迁移训练的难度。

## 6. 身体高度和 EE 目标空间对齐

原始 B1Z1 的身体高度和目标采样范围不适合 Go2X5。当前仓库已对 Go2X5 low-level 做了以下调整：

- 初始身体高度改为约 `0.32`。
- low-level base height target 改为约 `0.33`。
- EE 目标球中心根据 X5 安装位置重新设置。
- EE 目标采样半径和 pitch 范围做了收紧，避免大量采样到不可达或地下目标。
- 可视化脚本中明确区分当前 EE target、实际 EE 位置、目标球中心和历史轨迹。

这部分贡献让训练目标从 B1Z1 的尺寸假设中脱离出来，更接近 Go2X5 的实际工作空间。

## 7. Reward 和训练强度对齐

当前仓库已经识别出一个重要问题：B1Z1 原始训练风格可以从第 0 轮承受较强 domain randomization 和 reward 约束，但 Go2X5 迁移后直接套用这套强度容易陷入局部最优。

当前 Go2X5 配置仍保留了较强训练条件：

- friction range 较宽。
- added mass 和 COM randomization 较强。
- push robots 开启。
- collision、hip position、roll、torques、work 等惩罚较重。

已有训练日志显示：

- locomotion、gait、EE tracking 已经出现学习信号。
- 但 reward 长期被 collision、hip_pos、roll、torques、work 等负项压住。
- 策略容易学成“能勉强不倒，但姿态差、拖腿、高能耗”的局部最优。

因此，本仓库的后续推荐路线已经从“直接满强度训练 45000 轮”调整为 staged training：

1. Stage 1：稳定站立和行走，降低随机化和负项强度。
2. Stage 2：增强 EE tracking，逐步扩大目标空间。
3. Stage 3：恢复强 domain randomization 和 push，提高鲁棒性。
4. Stage 4：进入 high-level pick-multi teacher 训练。

这部分贡献不是某一个单独代码 patch，而是对训练路线的重新定义：Go2X5 迁移应保留 VBC 思路，但不应机械复制 B1Z1 的训练强度。

## 8. High-Level 接口对齐

high-level 侧已经加入 Go2X5 pick-multi 任务入口和配置：

- `high-level/envs/go2x5_pickmulti.py`
- `high-level/data/cfg/go2x5_pickmulti.yaml`
- 任务名 `Go2X5PickMulti`
- 支持 `numGripperDof: 2`
- 支持 `lowPolicyNumActions: 12`
- low-level loader 按 12 维动作构建 ActorCritic。
- high-level step 中调用 low-level policy 产生腿部动作，同时手臂通过 IK 跟踪 high-level 给出的 EE goal。

这部分贡献让 high-level 不再假设低层必须是 B1Z1 的 18 维动作模型。

但是，high-level 目前仍有未完全对齐的问题：

- 默认 `go2x5_pickmulti.yaml` 的 `low_policy_path` 仍指向旧模型。
- high-level 当前 asset 仍指向旧 `go2x5/urdf/go2_arx_x5.urdf`，不是 low-level 最新的 `go2_x5.urdf`。
- high-level 代码中仍硬编码 `ee_gripper_link`，而最新 low-level URDF 使用 `arm_eef_link`。
- high-level 中 `link6/link7/link8`、`x5_joint*` 等旧命名还没有完全迁移到最新 URDF 的 `arm_link*`、`arm_joint*` 命名。

因此，当前 high-level 可以作为 Go2X5 迁移接口的基础，但还不能认为已经和最新 low-level 完全一致。

## 9. 工程可复现性改动

本仓库还做了一些工程层面的对齐和稳定性工作：

- 处理 Isaac Gym 动态库路径自举，降低手动设置 `LD_LIBRARY_PATH` 的出错概率。
- 明确 Isaac Gym 必须先于 torch 导入。
- 兼容 RTX 40 系显卡上的较新 PyTorch + CUDA wheel。
- 训练脚本尊重 `WANDB_MODE=offline`，避免 W&B 网络问题阻塞训练启动。
- 修复 `play.py` 非标准调用导致的 `EXPORT_POLICY` 问题说明。
- 修复 viewer 单环境快捷键索引越界问题。
- 增加 `visualize_go2x5_config.py` 用于检查 Go2X5 当前姿态、目标点和实际末端位置。
- 在 `AGENT.md`、`memory.md` 和 `docs/` 中记录当前工程状态、常用命令、远端 GPU 使用约束和已知坑。

这些改动的价值在于降低复现实验的环境不确定性，使后续训练问题更集中在控制和 reward 本身。

## 10. 当前贡献总结

相对于原始仓库，本仓库当前已经完成的主要对齐贡献包括：

1. 将机器人平台从 B1+Z1 迁移到 Go2+ARX-X5。
2. 引入并启用更接近 Go2-X5-lab 的机器人 URDF 和 mesh 资源。
3. 将 Go2X5 low-level policy 重构为 12 维 leg-only action space。
4. 同步修正 observation 结构、action history 和 privileged observation。
5. 将机械臂控制改为 IK/位置目标驱动，降低早期迁移训练难度。
6. 调整 Go2X5 身体高度、手臂安装偏置和 EE 目标采样空间。
7. 增加 NaN 防护、动态库路径自举、W&B offline 兼容和可视化调试工具。
8. 为 high-level 增加 Go2X5 task/config，并使其能够加载 12 维 low-level policy。
9. 明确指出 high-level success rate 为 0 很可能源于 low-level 不稳或接口不一致，而不是单纯 high-level PPO 失败。
10. 形成 staged training 的后续路线，避免继续机械复制 B1Z1 满强度训练。

## 11. 仍需完成的对齐工作

为了把 Go2X5 迁移真正闭环，后续仍建议优先完成：

- 新建 Stage 1 low-level 稳定配置，降低 domain randomization 和过强 penalty。
- 为 collision reward 增加合理 contact force threshold，避免轻微擦碰被严重惩罚。
- 真正实现 `go2x5_ftlift` 配置，而不是只作为 `go2x5` 的别名。
- 将 high-level 默认 `low_policy_path` 指向经过回放验证的 Go2X5 checkpoint。
- 统一 high-level 和 low-level 使用的 URDF。
- 将 high-level 末端 link、wrist link、finger link、joint names 做成配置项，支持最新 `go2_x5.urdf`。
- 在 high-level 训练前先做 small-env smoke test，确认 low-level 加载、EE IK、gripper contact 和 success gate 都合理。

## 12. 结论

本仓库相对于原始 Visual Whole-Body Control 仓库的主要贡献，是把 B1+Z1 的代码路径系统性迁移到 Go2+ARX-X5，并在 low-level 控制接口、机器人资源、观测维度、手臂 IK、训练流程和 high-level 任务入口上做了平台对齐。

当前最重要的认识是：Go2X5 对齐不是一次性替换模型文件，而是一个跨 URDF、控制、奖励、训练流程和 high-level 接口的系统工程。low-level 必须先成为稳定、可信、可回放的底层控制器，high-level pick-multi 的成功率才有实际意义。
