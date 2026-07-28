# Go2-X5 low-level v5 奖励修复与长训就绪审查（2026-07-19）

## 审查结论

结论为：**允许使用 v5 代码从 S0 随机初始化启动新的 low-level 长训；禁止继续使用 v3/v4 checkpoint。**

此次问题已经定位到奖励语义，而不是 PPO 数值稳定性。旧策略能够稳定存活、episode 基本跑满且 loss 有限，但训练目标把四足贴地拖行塑造成了比对角小跑更优的局部最优。因此旧 run 即使继续增加 iteration，也不能作为健康 locomotion 训练继续使用。

机器可读摘要位于 `docs/parity_reports/go2x5_lowlevel_v5_training_readiness_2026-07-19.json`。

## 根因

目标 gait 是 2 Hz 对角小跑。摆动相中正常状态只有一组对角足着地，即两足支撑。旧 `_stability_safety()` 却固定要求至少三足接触，并用该结果同时门控：

- `stability_safety`；
- `tracking_ee_world_stable`。

因此出现了反向激励：

| 接触状态 | 旧稳定门控 | 结果 |
|---|---:|---|
| 正确两足对角支撑 | 0 | 丢失稳定奖励和 EE 跟踪奖励 |
| 四足全部接触 | 1 | 保留大额奖励，仅承担较小 gait contact 误差 |

这解释了观测到的行为：episode 长度和稳定性很好，但摆动腿接触率接近 100%，足端高度约 0.023 m，速度命令响应很弱，机器人表现为四脚 shuffle。

## 修复内容

### 1. gait-aware 安全门控

安全因子拆分为 body stability 与 support safety：

- 站立：至少三足支撑；
- 行走：至少两足支撑；
- roll、pitch、base height 的安全限制保持不变。

独立 runtime oracle 的加权结果为：

```text
standing, 2 contacts -> safety 0.0
standing, 3 contacts -> safety 1.0
walking, correct diagonal 2 contacts -> stability 1.0, weighted score 1.8
walking, all 4 contacts              -> stability 1.0, weighted score 1.3
```

修复后，正确对角支撑不再丢失稳定/EE 奖励，同时 gait 相位奖励仍明确区分正确两足支撑与四足拖行。

### 2. curriculum 分离 gait acquisition 与完整任务

profile 升级为：

```text
go2x5_stable_reach_curriculum_v5_gait_aware_h032
```

保留 S0–S2 的站立和 EE compensation 学习，新增两阶段 locomotion：

- S3 `S3_forward_gait_initiation`：只给正向 `vx=[0.08, 0.16]`，关闭 yaw，缩小 EE 工作域，降低与 gait acquisition 冲突的正则项；
- S4 `S4_bidirectional_locomotion_reach`：恢复双向速度、yaw 和完整 EE 工作域。

S3 必须同时满足存活、reset、速度跟踪、接触相位和摆脚高度门槛后才能进入 S4，避免单看 mean reward 自动升级。

### 3. 其他奖励实现错误

全量审查了 74 个 `_reward_*` 实现，并修复：

- `tracking_lin_vel_z_l2` 错误地使用 `commands[:, 2]`；该维实际为 yaw command，现在只惩罚 base z velocity；
- walking/standing 奖励包装器错误调用 `self.env._reward_*`，改为调用奖励容器自身实现；
- gait observation 关闭时部分函数返回 Python 标量，改为 `(num_envs,)` tensor；
- `_reward_alive` 改为逐环境 tensor；
- `num_gripper_joints=0` 时机械臂能耗切片为空的问题；
- torque reward 只计算实际受控 torque 维；
- base height、低 EE 目标和 feet height 统一相对 terrain height 计算。

站立高度合同继续保持实际 Go2-X5 的 `0.32 m`。

## 验证结果

| 验证 | 规模 | reset | nonfinite | 结果 |
|---|---:|---:|---:|---|
| 奖励静态审计 | 74/74 实现 | — | — | PASS，mismatch 0 |
| reward semantics 单元测试 | support/PD wrapper/tensor contract | — | 0 | PASS |
| readiness、alignment、runtime parity | CPU 回归 | — | 0 | PASS |
| GPU S0 readiness | 200 policy ticks | 0 | 0 | PASS |
| GPU S3 readiness | 200 policy ticks | 0 | 0 | PASS |
| GPU S4 readiness | 200 policy ticks | 0 | 0 | PASS |
| S0 PPO smoke | 512 env × 200 iter | 0 | 0 | PASS |
| model_200 production load + safety rollout | 32 env，4 cases | 0 | 0 | PASS |

200-iteration smoke 共运行 2,457,600 transitions。最终 mean reward 为 `32.91`，mean episode length 为完整的 `502.0`，contact/roll/pitch/z reset 均为 `0`。第一次使用过高探索方差的诊断 smoke 出现大量 roll reset，因此没有采用；最终配置使用较温和的 initial std `[0.15, 0.20, 0.20] × 4` 和 minimum std `[0.08, 0.12, 0.12] × 4`，完整 smoke 稳定通过。

生成的临时 `model_200.pt` 仅存于 `/tmp`，没有加入 Git。真实 runner 验证结果：

```text
schema_version = 2
action_dim = 12
num_arm_actions = 0
curriculum_profile = go2x5_stable_reach_curriculum_v5_gait_aware_h032
control_contract_sha256 = valid
model parameter nonfinite = 0
```

该 checkpoint 只完成 S0 smoke，不会行走是预期结果；安全回放使用 `--safety-only`，没有把它误报成合格 locomotion policy。

## 可复现命令

CPU gate：

```bash
python3 -m py_compile \
  low-level/legged_gym/envs/rewards/maniploco_rewards.py \
  low-level/legged_gym/envs/manip_loco/go2x5_config.py \
  low-level/legged_gym/scripts/audit_go2x5_low_level_rewards.py \
  low-level/legged_gym/scripts/check_go2x5_training_readiness.py \
  tests/test_go2x5_reward_semantics.py

python3 low-level/legged_gym/scripts/audit_go2x5_low_level_rewards.py \
  --output docs/06_go2x5_low_level_reward_audit.md \
  --fail-on-mismatch

python3 tests/test_go2x5_training_readiness.py
python3 tests/test_go2x5_alignment.py
python3 tests/test_low_high_runtime_parity.py
```

服务器 GPU0 的部署前 readiness：

```bash
cd /data4/duanzhibo/xhq_workload/VBC-gx

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
/data4/duanzhibo/miniconda3/bin/conda run --no-capture-output -n b1z1 \
  python low-level/legged_gym/scripts/check_go2x5_training_readiness.py \
  --num-envs 8 \
  --steps 500 \
  --rollout-stage 0 \
  --sim-device cuda:0 \
  --rl-device cuda:0 \
  --graphics-device-id 0 \
  --output /tmp/go2x5-v5-s0-readiness.json
```

新的长训必须随机初始化，不得添加任何 resume/checkpoint 参数：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
/data4/duanzhibo/miniconda3/bin/conda run --no-capture-output -n b1z1 \
  python low-level/legged_gym/scripts/train.py \
  --headless \
  --task go2x5 \
  --proj_name go2x5-low \
  --exptid go2x5_v5_gaitaware_h032_seed1 \
  --num_envs 4096 \
  --max_iterations 45000 \
  --seed 1 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --graphics_device_id 0
```

## 长训后的硬门禁

S3 成熟 checkpoint 必须运行 `check_go2x5_fixed_command_gait.py`。只有 stand、forward、backward 和 turn-left 全部通过速度进度、摆动接触率、摆脚高度、reset、nonfinite 和 live-foot-cache gate，才允许宣称 locomotion 训练健康。

## 尚未证明

- 尚无完成训练且通过 gait gate 的 v5 正式 checkpoint；
- 尚未证明最终策略的 1 秒或 10 秒闭环 parity；
- 尚未允许恢复额外 domain randomization；
- 尚未允许开始 high-level teacher/student 训练；
- 当前结论只授权从头启动 low-level 长训，不授权部署。

因此当前唯一正确动作是：**停止旧 v4 run，同步 v5 代码，在 GPU0 运行一次部署 checkout 的 S0 readiness，然后以新 experiment id 从随机初始化开始长训。**
