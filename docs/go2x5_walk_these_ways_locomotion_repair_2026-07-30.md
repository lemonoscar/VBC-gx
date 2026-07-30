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

保留的基本结构：

| 项目 | Walk These Ways | Go2-X5 |
|---|---:|---:|
| policy 频率 | 50 Hz | 50 Hz |
| 物理步长 / decimation | 0.005 s / 4 | 0.005 s / 4 |
| action 维度 | 12 | 12 |
| hip action scale | 0.25 × 0.5 | 0.125 |
| thigh/calf action scale | 0.25 | 0.25 |
| 腿部 PD | 20 / 0.5 | 40 / 1（用户指定，保持同一 Kp:Kd 比） |
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

当前 actor 只有 12 个腿部 action，机械臂由 6D DLS IK 控制。若从第一个 PPO update 就把完整 EE advantage 注入腿策略，腿在尚未学会移动时会优先用静态姿态维持 EE。

因此恢复 B1-Z1 已验证过的渐入顺序：

```text
mixing_schedule = [1.0, 0, 3000]
```

最终 reward 没有删除 EE 目标；只是在前 3000 个 PPO update 内逐渐引入 EE/机身协同 advantage。

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

上述 CPU/静态测试通过。修订前的 GPU readiness 也验证了 plane、40/1 PD、12D action、IK、finite 和 reset 基线；最终配置仍需在 lab-server 的隔离干净仓库中重新执行 readiness 和 smoke。

## 服务器状态与下一门禁

2026-07-30 预检结果：

- SSH 主机：`lab-server`，实际 hostname：`ubuntu1`；
- canonical writable root：`/data4/duanzhibo/xhq_workload`；
- 原仓库 `/data4/duanzhibo/xhq_workload/VBC-gx` 有多项未提交修改和未跟踪训练产物，不能安全 fast-forward；
- `/data4` 使用率 99%，剩余约 23 GB；
- GPU 2 和 GPU 3 当时无 compute process，各有约 24 GB 空闲显存。

不得在脏仓库中覆盖或直接修改。经用户明确允许后，应在 `~/xhq_workload` 内创建隔离干净 clone，拉取精确提交，先运行：

1. GPU readiness；
2. 512–1024 env、约 1000–2000 iteration smoke；
3. 固定命令五 case 行为门禁。

只有 smoke 行为门禁通过，才允许启动 4096 env、45000 iteration 正式长训。
