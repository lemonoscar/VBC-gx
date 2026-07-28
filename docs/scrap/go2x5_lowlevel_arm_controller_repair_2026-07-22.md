# Go2-X5 low-level 机械臂任务控制修正（2026-07-22）

## 结论

本轮机械臂任务实现修正通过 GPU smoke。原有长训已停止，未重新启动；最后保留的旧训练 checkpoint 是 `model_26400.pt`。

修正后的中心目标固定基座重力测试中，EE 位置误差从旧控制器约 `0.119 m` 的稳态偏差下降到：

- 1 秒（50 policy ticks）：`0.392 mm`；
- 2 秒（100 policy ticks）：`0.022 mm`；
- 4 秒（200 policy ticks）：`0.00048 mm`；
- 500 ticks 最终：`5.96e-8 m`。

该数值只代表固定基座、中心工作区目标的控制器隔离实验，不代表全工作区或已训练策略的最终任务成功率。

## 根因与修正

### 1. EE 目标每帧丢失重力补偿

旧实现每个 50 Hz policy tick 都使用：

```text
q_target = q_measured + ik_gain * dq
```

位置驱动器在重力下必须保留 `q_target - q_measured` 才能提供静态力矩。旧实现每帧从实测关节位置重新开始，导致控制器只能在较大的 EE 残差下维持机械臂。

新实现维护持久化 `arm_q_command`：

```text
delta = clamp(ik_gain * DLS(position_error), ±0.08 rad)
arm_q_command = clamp(arm_q_command + delta, joint_limits)
```

reset 时命令与实测关节位置同步，避免跨 episode 目标跳变；每次更新继续执行 URDF joint-limit clamp。

### 2. `ik_gain=0.25` 在持久化控制器下振荡

固定基座、重力开启、相同中心目标的 500-tick 扫描结果：

| ik_gain | 最后 100 ticks 平均误差 | 最后 100 ticks 最大误差 | 500 tick 最终误差 |
|---:|---:|---:|---:|
| 0.05 | 0.000983 mm | 0.001301 mm | 0.000962 mm |
| 0.10 | 0.000322 mm | 0.000477 mm | 0.000060 mm |
| 0.15 | 0.000574 mm | 0.001081 mm | 0.000478 mm |
| 0.25 | 6.636 mm | 11.928 mm | 8.892 mm |

因此 low/high contract 同步采用 `ik_gain=0.10`。`0.08 rad/tick` 仍作为异常大误差下的独立关节增量上限。

### 3. 空任务中夹爪持续闭合并产生手指接触

low-level 不控制抓取动作，旧位置目标默认为 0，导致两侧手指闭合并产生持续内接触。现在 low-level reset 和每帧位置目标都将两个 prismatic finger 保持在 URDF 上限 `0.044 m`。隔离实验最终两侧 finger contact force 均为 `0 N`。

high-level 仍保留自己的抓取动作语义；`gripper_hold_mode=open_upper_limit` 只描述 low-level 训练/部署 wrapper 的默认无抓取状态。

### 4. 非足碰撞不可见、近机身目标未被过滤

碰撞奖励现在按名称解析并覆盖：

- `base`、`Head_upper`、`Head_lower`；
- 四腿 hip/thigh/calf；
- `arm_link1` 至 `arm_link8`（包括 wrist/fingers）。

四个 foot body 不进入该惩罚。GPU readiness 实际解析出 23 个 penalized bodies。

任务工作区仍是机器人前方 world x `0.30–0.55 m`、y `±0.15 m`、terrain z `0.08–0.45 m`。近机身局部 x `<0.24 m` 的轨迹现在会被过滤；10 次重采样均失败时保持上一个合法目标，禁止接受最后一个碰撞目标。

## checkpoint 语义

- 旧 `model_26400.pt` 是旧 profile/arm contract，禁止 full resume；实际 production load 已 fail-closed 拒绝。
- 因 policy 是 12D leg-only 且 `num_arm_actions=0`，允许显式 `--warm_start_checkpoint` 只迁移网络权重；optimizer、exploration std、curriculum、runner/env state 均重置。
- 本轮用旧权重执行了 20-iteration smoke，生成的 `model_20.pt` 仅用于验证新训练路径，不是正式 deployment checkpoint。
- high-level production loader 使用随机 schema-v2 parity smoke checkpoint 真实通过新 contract hash；没有关闭 metadata 校验，也没有 monkey-patch production loader。

## 验证结果

| 验证 | 结果 |
|---|---|
| Python 编译、alignment、readiness、reward semantics、runtime parity 单测 | 通过 |
| S0 GPU readiness，8 env × 200 steps | 通过，0 early reset，0 nonfinite |
| S1 GPU readiness，8 env × 200 steps | 通过，0 early reset，0 nonfinite |
| 固定基座重力 EE 控制，500 ticks | 通过，最终误差 `5.96e-8 m` |
| weights-only PPO smoke，256 env × 20 iterations | 通过 |
| smoke checkpoint rollout，128 env × 500 steps | 通过，0 reset/碰撞/nonfinite |
| high-level production loader + C3 checkpoint parity | 通过，0 mismatch/oracle failure/nonfinite |

机器可读结果见 `docs/parity_reports/go2x5_arm_controller_repair_report_2026-07-22.json`。完整逐帧数据与临时 checkpoint 只保存在服务器 `/tmp` 或训练日志目录，未纳入 Git。

## 仍未证明的事项

- 本轮没有重新启动长训，也没有证明新 contract 下的最终收敛质量。
- 20-iteration checkpoint 不是正式模型。
- 固定基座中心目标结果不能替代低/高/侧向全工作区的长期闭环统计。
- 尚未恢复 domain randomization、相机、物体接触或 high-level teacher 训练。

下一次长训只能作为新 run 启动；如复用 `model_26400.pt`，必须使用 weights-only warm start，不能 `--resume`。
