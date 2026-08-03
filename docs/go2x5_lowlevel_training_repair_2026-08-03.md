# Go2-X5 Low-Level v8 训练修复审查

日期：2026-08-03

## 结论边界

本轮修复针对“前腿长期着地、后腿乱踢、速度不跟踪”和随机 EE 目标开始后
机械臂带动机身翻滚的问题。代码与 CPU 单元门禁通过并不等于策略已经学会
走路；只有服务器 GPU readiness、from-scratch smoke 和固定命令评测通过后，
才允许启动 45000 iteration 正式训练。

## 复现出的根因

1. `mixing_schedule=[1,3000,3000]` 只延迟 arm advantage，机械臂从 iteration 0
   起仍在执行 6D IK。当前前置 X5 负载使前轴承担约 72% 的静态重量，随机
   机械臂运动会在 locomotion 尚未形成时继续扰动机身。
2. 旧 `ik_gain=0.20`、`target_max_step=0.08 rad/tick` 在 50 Hz 下允许目标命令
   以 4 rad/s 累积。32 环境、500 tick 的零腿动作探针出现 18 次 roll reset，
   最大 arm joint velocity 为 8.35 rad/s。`0.10/0.02` 的同条件探针为 0 reset，
   最大 arm joint velocity 为 1.22 rad/s。
3. 原速度核 `exp(-||c-v||²/0.05)` 在低速命令下仍高额奖励静止。例如
   `vx=0.15 m/s` 时，原地不动取得最大线速度奖励的 63.8%；再叠加零 yaw
   误差奖励和更低风险，PPO 容易收敛到静止局部最优。
4. 12D ActorCritic 的 entropy tensor 仍含 `[leg, empty_arm]` 两列，直接
   `.mean()` 会把 `entropy_coef=0.01` 的有效强度减半。
5. leg/arm advantages 原先跨两个 reward channel 一起归一化，使两类尺度和
   分布在混合前相互污染。

## v8 数学与时序

令 `K(c,v)=exp(-||c-v||²/sigma)`，`K0(c)=K(c,0)`。对非零移动命令使用：

```text
r_move = clip((K(c,v) - K0(c)) / (1 - K0(c)), -1, 1)
```

因此静止为 0，精确跟踪为 1，反向运动为负；没有被命令的轴在运动 episode
中使用 `K(0,v)-1` 作为零上界的漂移惩罚。完全停止的 episode 保留 `K(0,v)`
稳定核。最终 leg reward 仍按原逻辑求和、clip 到非负并除以 100。

训练时序：

- iteration 0--2999：机械臂 position target 与 EE trajectory timer 均冻结；
- iteration 3000：启用限速后的完整 6D IK；
- iteration 3000--6000：arm advantage 从 0 线性混入 leg policy；
- iteration 6000 以后：完整 locomotion + EE position/orientation + body 协同目标。

独立 evaluation/runtime 未设置 training iteration，默认启用完整机械臂控制，
防止 readiness 或回放只测到冻结分支。

PPO 修复：

- 12D policy entropy 只取 leg channel；18D B1-Z1 仍保留两通道原语义；
- advantage 分别沿 time/env 维度归一化，再执行 leg/arm mixing。

## 主要文件

- `go2x5_robot_spec.py`：IK gain 与每 tick target step 的唯一来源；
- `go2x5_config.py`：v8 profile、arm motion start、静止基线奖励开关；
- `manip_loco.py`：训练 iteration 接口和物理 arm freeze；
- `maniploco_rewards.py`：静止基线归一化速度/偏航奖励；
- `on_policy_runner.py`：每轮向环境传入真实 PPO iteration；
- `ppo.py`、`rollout_storage.py`：12D entropy 与逐 channel advantage；
- `check_go2x5_training_readiness.py`：冻结/启用行为、500 tick 动态检查及
  “走近桌面后可达”几何语义；
- `go2x5_pickmulti.yaml`：high-level production contract 同步到 `0.10/0.02`。

## 已执行门禁

```bash
conda run --no-capture-output -n vwc_go2x5 \
  python tests/test_go2x5_reward_semantics.py
conda run --no-capture-output -n vwc_go2x5 \
  python tests/test_go2x5_training_readiness.py
conda run --no-capture-output -n vwc_go2x5 \
  python tests/test_go2x5_alignment.py
conda run --no-capture-output -n vwc_go2x5 \
  python tests/test_low_high_runtime_parity.py
python3 low-level/legged_gym/scripts/audit_go2x5_low_level_rewards.py \
  --fail-on-mismatch
git diff --check
```

上述门禁均通过。本机 2026-08-03 无可用 NVIDIA driver；Isaac Gym 的 CPU
pipeline 仍试图分配 CUDA tensors，因此本机动态运行不构成有效结果。

## 服务器正式训练前硬门禁

1. 16 environments、500 policy ticks 的 readiness：0 nonfinite、0 early reset；
2. from-scratch smoke 至少生成可评测 checkpoint；
3. 固定 `stand/forward/backward/turn_left/turn_right` 评测方向正确；
4. translation/yaw progress ratio 至少 35%，无 early reset、无 nonfinite；
5. 通过后从随机初始化启动新的 45000 iteration run，不复用旧 checkpoint。

正式长训只能声明“已正常启动”，不能在尚未收敛时声明训练成功。
