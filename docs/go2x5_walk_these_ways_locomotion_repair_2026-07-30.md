# Go2-X5 Walk These Ways 步行修复审查（2026-07-30）

## 结论

旧 `model_45000.pt` 不是“步态不够好”，而是确定性行为门禁失败：

- `vx=+0.10 m/s` 时实际 `vx=+0.00484 m/s`，只完成命令的 4.84%；
- `vx=-0.10 m/s` 时实际 `vx=+0.00034 m/s`，没有形成反向运动；
- 左右转均没有形成正确偏航角速度；
- 前两脚在前进测试中 100% 时间保持接触，RR 几乎长期悬空；
- 无 NaN、无碰撞、无提前 reset，因此这是奖励/采样诱导出的安全退化策略，不是模拟器崩溃。

该 checkpoint 不应续训，也不能作为 high-level teacher 的合格底座。新训练必须从头开始，并先通过本文定义的固定命令行为门禁。

## Walk These Ways 对照

依据：

- 论文：[Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior](https://arxiv.org/abs/2212.03238)
- 官方基础配置：[legged_robot_config.py](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/base/legged_robot_config.py)
- 官方 Go1 配置：[go1_config.py](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/go1/go1_config.py)
- 官方奖励实现：[corl_rewards.py](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/rewards/corl_rewards.py)
- 官方策略初始化：[actor_critic.py](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym_learn/ppo_cse/actor_critic.py)

保留的基本结构：

| 项目 | Walk These Ways | Go2-X5 |
|---|---:|---:|
| policy 频率 | 50 Hz | 50 Hz |
| 物理步长 / decimation | 0.005 s / 4 | 0.005 s / 4 |
| action 维度 | 12 | 12 |
| hip action scale | 0.25 × 0.5 | 0.125 |
| thigh/calf action scale | 0.25 | 0.25 |
| 腿部 PD | 20 / 0.5 | 40 / 1（用户指定，保持同一 Kp:Kd 比） |
| 初始 action std | 1.0 | 0.25 / 0.30 / 0.30（按 X5 负载实测降额） |
| PPO entropy coefficient | 0.01 | 0.01 |
| 速度奖励核 | `exp(-||v_cmd-v||²/sigma)` | 相同 |
| 偏航奖励核 | `exp(-(w_cmd-w)²/sigma)` | 相同 |
| feet air-time | landing event | landing event |
| action-rate | -0.01 | -0.01 |
| collision | -1 | -1 |
| command hold | 10 s | 10 s |

没有照搬的部分：

- 不启用 gait clock、固定 trot 或接触相位目标；当前目标是先获得任意稳定四脚步行。
- 不启用复杂地形和 domain randomization；当前按用户要求使用原生 PhysX plane。
- 不启用固定机身水平/固定高度强惩罚；抓取任务允许机身安全俯身。
- WTW 的 `sigma=0.25` 面向约 `±0.6 m/s` 的命令。Go2-X5 只有 `±0.30 m/s`；若直接照搬，`0.10 m/s` 命令下完全静止仍可获得 96.1% 的速度奖励。这里保留平方误差形式，但将 `sigma` 缩为 `0.05`，此时静止只获得 81.9%，使低速跟踪产生有效优势。

## 最终低速命令分布

```text
vx  = [-0.30, 0.30] m/s
vy  = [-0.10, 0.10] m/s
yaw = [-0.25, 0.25] rad/s
```

每次采样互斥选择：

- 20% 精确站立；
- 35% 精确直行，`|vx| >= 0.10 m/s`，且 `vy=yaw=0`；
- 10% 原地转向，`|yaw| >= 0.10 rad/s`，且 `vx=vy=0`；
- 35% 一般 x/y/yaw 联合命令。

命令保持 10 秒，与一个 low-level episode 等长。这样既保留横移和转向，又避免在尚未形成步态时每 3 秒改变运动方向。

## 抬脚和拖脚

拖脚项只计算接触脚的水平速度：

```text
feet_drag = sum(contact * (vx_foot² + vy_foot²))
```

竖直落脚速度不计入拖脚，避免惩罚正常 touchdown。

抬脚奖励只在完成 swing 并重新落地时支付：

```text
air_bonus = clamp(air_time - 0.10, 0, 0.25)
clearance_bonus = 0.20 * normalized_clamp(
    swing_peak_height,
    foot_radius=0.022,
    target_center_height=0.05,
)
```

关键约束：

- 短步落地为 0，不再产生负奖励；
- 永久悬空不支付奖励；
- 只抬高但不落地不支付奖励；
- 四只脚采用同一公式；
- 不规定四脚的相位或先后顺序。

## Whole-body 学习顺序

当前 actor 只有 12 个腿部 action，机械臂由 6D DLS IK 控制。远端 v3
smoke 证明，即便将 EE advantage 从 0 线性渐入，iteration 1500 时的
0.5 mixing ratio 仍足以让策略先收敛到静态关节偏置，而不是步行。

另外，旧实现把 `height_adaptation` 和 `pitch_adaptation` 放在 leg reward
通道，它们从第一轮就作用于腿策略，实际绕过了 advantage mixing。

v4 将四项 whole-body 目标统一放进 arm/whole-body reward 通道：

```text
tracking_ee_world
tracking_ee_orn
height_adaptation
pitch_adaptation
```

并采用：

```text
mixing_schedule = [1.0, 3000, 3000]
```

即 iteration 0–3000 只优化 locomotion advantage，3000–6000 线性引入
whole-body advantage，6000 后使用完整最终目标。最终 reward 没有删除
EE 或俯身协同，只改变学习顺序。

## 探索强度修复

v3 错把 B1-Z1 的最低噪声量级当作随机初始化噪声：

```text
init_std = [0.15, 0.20, 0.20] × 4
min_std  = [0.08, 0.12, 0.12] × 4
```

远端确定性回放显示，actor 最终只学到每条腿不同的静态偏置，四脚始终
接触。v4 曾直接恢复 B1-Z1 的腿部探索设置：

```text
init_std = [0.80, 1.00, 1.00] × 4
min_std  = [0.15, 0.25, 0.25] × 4
```

远端 from-scratch smoke 到 iteration 43 时 roll reset 仍为 `97.9%`，
所以 v4 被明确拒绝。问题是 B1-Z1 的绝对 std 不是 Go2-X5 的正确尺度。

v5 改为对齐 Walk These Ways 的执行器扰动。WTW Go1 使用
`Kp=20`、hip/thigh/calf action scale `0.125/0.25/0.25`；Go2-X5 使用
`Kp=40`、相同 action scale。因此：

```text
std_go2 = std_wtw * Kp_wtw / Kp_go2 = 1.0 * 20 / 40 = 0.5

init_std = [0.50, 0.50, 0.50] × 4
min_std  = [0.15, 0.25, 0.25] × 4
```

两者的初始 target-error torque 标度均为每腿
`[2.5, 5.0, 5.0] Nm`。所有 action 继续经过 tanh mean、
`clip_actions=1` 和 torque limit。

但是 v5 的远端 smoke 到 iteration 39 仍有 `100%` roll reset，说明裸
Go1 执行器尺度未考虑 X5 机械臂带来的惯量和重心变化。另一方面，检查
WTW 官方 PPO 后确认其 `entropy_coef=0.01`，而 v3–v5 的 Go2 配置一直
为 `0.0`，会让 std 在没有探索激励时迅速塌到下限。Go2-X5 的 12D
wrapper 曾因空 arm entropy channel 将配置系数实际减半；修复通道平均后
显式使用等效的 `0.005`，避免前载系统的探索标准差持续增大。

v6 因此不再追求裸 Go1 的绝对扰动力矩，而使用已有稳定/失败区间的中间值：

```text
init_std = [0.25, 0.30, 0.30] × 4
min_std  = [0.08, 0.12, 0.12] × 4
entropy_coef = 0.005
```

这不修改 PPO loss 公式或网络结构；它保留非零 entropy，并把有效强度与
修复前的一通道贡献对齐，同时把初始噪声限制在带 X5 负载的实测边界内。

v6 完成了 1024 env、1000 iteration smoke，进程正常退出，且后期
episode length 接近完整 10 秒；但确定性策略仍然退化到静止：

| checkpoint | forward progress | backward progress | left yaw | right yaw | 结论 |
|---|---:|---:|---:|---:|---|
| model_200 | -0.0% | 9.3% | -0.5% | -0.7% | 失败 |
| model_400 | -15.9% | 27.5% | 7.3% | -14.6% | 失败 |
| model_600 | -8.8% | 7.5% | 2.6% | -1.2% | 失败 |
| model_800 | -0.5% | 3.1% | -6.7% | 2.3% | 失败 |
| model_1000 | 3.0% | 3.7% | 14.3% | -19.0% | 失败 |

v6 证明，仅维持 exploration 不足以摆脱旧 stability-first 奖励留下的局部
最优。当前 leg reward 仍含 `alive=1`、`termination=-100` 且
`only_positive_rewards=False`；这与 WTW 的无 survival、无 terminal
penalty、负总 reward 裁剪相反。它会对一次探索性跌倒重复惩罚，同时给
静止策略持续生存收益。

v7 因此删除这三项遗留 shaping，并减少原地抬脚的额外激励：

```text
only_positive_rewards = true
alive = 0
termination = 0
leg_action_l2_deadzone = 0
feet_air_time = 1

stand / straight / turn / general = 10% / 50% / 10% / 30%
```

环境的 roll/pitch/z/contact termination 本身仍启用；提前终止自然损失
未来 tracking reward，因此没有删除安全约束。此变更只让 locomotion
reward 回到 WTW 的简单核心，不引入 gait clock、固定步态或新网络。

## 行为门禁

训练日志中的总 reward、episode length 或 timeout 不能证明会走。每个候选 checkpoint 必须用 deterministic inference、固定命令、相同 reset 评测：

```text
stand:      vx= 0.00, yaw= 0.00
forward:    vx=+0.10, yaw= 0.00
backward:   vx=-0.10, yaw= 0.00
turn_left:  vx= 0.00, yaw=+0.15
turn_right: vx= 0.00, yaw=-0.15
```

硬门禁：

- 移动方向正确；
- 线速度/偏航至少完成命令的 35%；
- 移动速度绝对误差分别不超过 0.04 m/s / 0.05 rad/s；
- 前进和后退时每一只脚都有接触切换；
- 每只参与摆动的脚中心平均离地高度至少 0.04 m；
- 非足端碰撞 raw mean 不超过 0.10；
- 无提前 reset、无 NaN/Inf。

完整逐帧结果写 `/tmp`，不提交 checkpoint 或大数组。

## 已完成验证

```bash
python3 -m py_compile \
  low-level/legged_gym/envs/manip_loco/go2x5_config.py \
  low-level/legged_gym/envs/manip_loco/manip_loco.py \
  low-level/legged_gym/envs/rewards/maniploco_rewards.py \
  low-level/legged_gym/scripts/check_go2x5_training_readiness.py \
  low-level/legged_gym/scripts/check_go2x5_fixed_command_gait.py

conda run --no-capture-output -n vwc_go2x5 \
  python tests/test_go2x5_reward_semantics.py
conda run --no-capture-output -n vwc_go2x5 \
  python tests/test_go2x5_training_readiness.py
conda run --no-capture-output -n vwc_go2x5 \
  python tests/test_go2x5_alignment.py
conda run --no-capture-output -n vwc_go2x5 \
  python tests/test_low_high_runtime_parity.py

conda run --no-capture-output -n vwc_go2x5 \
  python low-level/legged_gym/scripts/audit_go2x5_low_level_rewards.py \
  --fail-on-mismatch \
  --output docs/06_go2x5_low_level_reward_audit.md
```

上述 CPU/静态测试通过。v3 在 `lab-server` 上的最终 GPU readiness
通过，随后完成了 1024 env、1500 iteration、36,864,000 timesteps
from-scratch smoke，训练进程退出码为 0、无 nonfinite。

但是 v3 行为门禁失败：

| checkpoint | forward progress | backward progress | left yaw | right yaw | 结论 |
|---|---:|---:|---:|---:|---|
| model_200 | -0.3% | 0.4% | 0.0% | -0.6% | 失败 |
| model_600 | -18.9% | 18.3% | -31.0% | 26.0% | 失败 |
| model_1000 | -7.5% | 2.7% | -7.6% | 6.7% | 失败 |
| model_1500 | 2.4% | -5.1% | 2.2% | -17.5% | 失败 |

最终 `model_1500` 在五种命令下四脚接触率均为 100%，没有任何接触切换。
这证明 v3 是安全静止策略，不允许进入长训。v4 与 v5 是因早期高
roll-reset 而终止的诊断运行；后来对照 v3 原始日志发现，早期高 reset
本身不能独立判失败。v6 完整 smoke 则由确定性行为明确判失败。v7 必须
重新通过 GPU readiness 和远端 smoke。

## 服务器状态与下一门禁

2026-07-30 预检与 v3 执行结果：

- SSH 主机：`lab-server`，实际 hostname：`ubuntu1`；
- canonical writable root：`/data4/duanzhibo/xhq_workload`；
- 原仓库 `/data4/duanzhibo/xhq_workload/VBC-gx` 有多项未提交修改和未跟踪训练产物，不能安全 fast-forward；
- `/data4` 使用率 99%，剩余约 23 GB；
- 使用现有干净同源副本
  `/data4/duanzhibo/xhq_workload/VBC-gx-highlevel-247b506`；
- v3 smoke 使用物理 GPU 2，确定性评测使用物理 GPU 3；
- readiness 报告：
  `/data4/duanzhibo/xhq_workload/runs/go2x5-wtw-v3-smoke-20260730-r1/readiness.json`；
- v3 最终行为报告：
  `/data4/duanzhibo/xhq_workload/runs/go2x5-wtw-v3-smoke-20260730-r1/fixed_command_model_1500.json`。

不得在脏仓库中覆盖或直接修改。后续只允许让上述干净工作副本
fast-forward 到精确提交，然后运行：

1. GPU readiness；
2. 512–1024 env、约 1000–2000 iteration smoke；
3. 固定命令五 case 行为门禁。

只有 smoke 行为门禁通过，才允许启动 4096 env、45000 iteration 正式长训。
