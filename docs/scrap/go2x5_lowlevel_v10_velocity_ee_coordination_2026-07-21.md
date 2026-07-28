# Go2-X5 Low-Level V10：速度与 EE 协同训练审查（2026-07-21）

## 结论

本阶段把 Go2-X5 low-level 任务化简为两个同时存在、不会互相延后的目标：

1. 精确跟踪前后向速度和 yaw 速度；
2. 通过 X5 的 position-only IK 与 Go2 的高度、俯仰协同跟踪机身前方 EE 目标。

训练不规定四拍 walk、trot 或任何足端相位。策略可以采用任意步态，但必须通过固定命令速度误差、非有限值、提前 reset、碰撞和机械臂 joint-limit hard gate。

本阶段没有修改 ActorCritic 网络结构、PPO loss 或 12D 动作拓扑。旧 `model_50000.pt` 只作为经过严格校验的网络权重初始化；旧 optimizer、history optimizer、探索标准差、runner 计数器、curriculum 和环境状态均不恢复。

## 本次发现的真实问题

### 1. 原“position-only IK”仍隐式约束姿态

旧实现把 6D pose error 的后三个旋转误差置零，却仍使用完整 `6 x 6` EE Jacobian 求解。这并不等价于 position-only：求解器仍要求 EE angular velocity 为零，占用了冗余自由度。

在旧实现的 deterministic rollout 中，`arm_joint4` 有约 `87.945%` 的 policy tick 被 clamp 到 URDF joint limit。该现象解释了 EE error 长期停滞，也说明仅看“无 NaN、机器人不倒”会产生假通过。

修复后，orientation tracking 关闭时只使用：

```text
J_position = J_ee[:3, :]
dpose_position = dpose[:3]
dq = J_position^T (J_position J_position^T + lambda^2 I_3)^-1 dpose_position
```

low-level 和 high-level 使用相同 3D 平移任务，control contract 新增：

```json
"ik_task": "position_only_translation_3d"
```

### 2. 旧 checkpoint 不能直接 full resume

新任务启用了 bounded actor output、`clip_actions=1.0`、新 reward/curriculum 和新的 IK task contract。直接把旧 `model_50000.pt` 当成当前训练 checkpoint 会把旧 optimizer、探索噪声、计数器和任务状态一并恢复，因此 production full-resume loader 会 fail-closed 拒绝它。

新增的 `--warm_start_checkpoint` 是独立路径：

- 要求 fresh runner；
- 要求新的 experiment output directory 为空，拒绝覆盖已有 checkpoint/TensorBoard；
- 校验 schema、机器人 asset hash、观测/动作维度、joint order、PD、action scale、physics、EE frame 与 IK 等权重兼容不变量；
- 要求 model state key、shape、dtype 完全一致且全部有限；
- 加载网络权重，但保留当前初始化的 `std`；
- 不恢复任何 optimizer、counter、curriculum 或 runner state；
- 在新 checkpoint metadata 中记录源文件 SHA256 和所有未恢复状态。

旧 checkpoint：

```text
model_50000.pt
source sha256 = 9d542358f62b6eec7af8a84e1309479dfb75163f0ee3020a020178d2290d83ce
```

### 3. 旧固定命令 gate 混合了速度与命名步态

旧检查把速度误差与 `desired_contact_states` 的摆动接触率、摆脚高度放进同一个结果。当前 profile 明确关闭 gait clock，因此这些相位量只能作为诊断，不能否决一种有效但不同的自然步态。

现在默认 gate 严格检查：

- 站立 vx/yaw 漂移；
- `+0.10 m/s` 前进误差；
- `-0.10 m/s` 后退误差；
- `+0.15 rad/s` 左转误差；
- 每个 case 的 early reset、nonfinite、碰撞和 foot tensor cache。

只有显式使用 `--require-gait-shape` 时才把旧摆动相位量作为 gate。

## 简化后的训练任务

### 控制合同

| 项目 | 值 |
| --- | --- |
| low-level action | 12D leg action，policy order |
| arm action head | 0D |
| leg PD | `kp=40`, `kd=1` |
| action scale | hip/thigh/calf = `0.125/0.25/0.25` |
| actor output | `tanh` |
| action clip | `[-1, 1]` |
| policy frequency | 50 Hz |
| IK | 3D translation-only damped least squares |
| nominal base height | 0.32 m |
| gait clock | disabled |

### 两阶段 curriculum

| 阶段 | iteration | vx | yaw | 站立概率 | 原地转向概率 | EE root/terrain 工作区 | EE 权重 |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: |
| S0 slow coordinated reach | 0–7999 | `[-0.12,0.12] m/s` | `[-0.15,0.15] rad/s` | 0.40 | 0.20，`abs(yaw)≥0.10` | x `0.35–0.50 m`, y `±0.10 m`, z `0.10–0.36 m` | 1.5 |
| S1 full coordinated reach | 8000+ | `[-0.30,0.30] m/s` | `[-0.40,0.40] rad/s` | 0.25 | 0.20，`abs(yaw)≥0.15` | x `0.30–0.55 m`, y `±0.15 m`, z `0.08–0.45 m` | 2.0 |

两阶段从 iteration 0 起都同时采样站立、前进、后退、转向和前方 EE 目标，不存在先纯站立、后期才学走路或 EE 的隐藏阶段。

### 奖励原则

- vx 和 yaw 使用关于 under/overspeed 对称的精确误差指数奖励；yaw 权重为 1.0，并显式采样左右原地转向；
- EE 使用未乘支撑/高度 gate 的世界坐标位置奖励，使身体动作能获得直接梯度；
- EE 目标越低，期望 base height 从 `0.32 m` 单调降到 `0.24 m`；
- EE 目标越低，期望正 pitch 从 `0` 增至 `0.12 rad`，降低前部机械臂安装点；
- 保留 roll、垂向速度、碰撞、foot drag、action rate、torque/work 和 joint-limit 约束；
- 固定 base-height、stand-still、walking posture、contact phase 和 named gait reward 均关闭。

## 文件级变更

- `low-level/legged_gym/envs/manip_loco/go2x5_config.py`：两阶段速度/EE curriculum、精确速度 reward、adaptive height/pitch、bounded action 与审查后的探索噪声。
- `low-level/legged_gym/envs/rewards/maniploco_rewards.py`：对称 vx/yaw error 与 terrain-relative adaptive body targets。
- `low-level/legged_gym/envs/manip_loco/manip_loco.py`：true 3D position-only IK、训练诊断、arm target clamp 记录、contract v2 字段和 warm-start metadata validator。
- `third_party/rsl_rl/rsl_rl/runners/on_policy_runner.py`：严格 weights-only warm start、状态恢复边界和 provenance。
- `low-level/legged_gym/scripts/train.py`、`utils/helpers.py`：新增互斥的 `--warm_start_checkpoint` CLI。
- `high-level/envs/b1z1_base.py`、`high-level/data/cfg/go2x5_pickmulti.yaml`：production loader 的 tanh/clip/contract 校验与同一 true 3D IK；未配置的新字段继续保留 B1-Z1 兼容行为。
- `check_go2x5_training_readiness.py`：3D Jacobian 独立 oracle 和 S0/S1 runtime probes。
- `check_go2x5_checkpoint_rollout.py`：deployment-history deterministic rollout、per-link collision、EE/速度/身体相关性、arm clamp 与 nonfinite hard gate。
- `check_go2x5_fixed_command_gait.py`：速度 tracking 与可选 gait-shape gate 分离。
- `audit_go2x5_low_level_rewards.py` 与 `docs/06_go2x5_low_level_reward_audit.md`：75/75 reward implementation 审计与新 profile 合同。
- `tests/test_go2x5_*.py`、`tests/test_low_high_runtime_parity.py`：warm-start、reward、IK、metadata、production parity 和 hard-gate 回归。

## 验证结果

### CPU/static

以下测试全部通过：

```bash
python3 tests/test_go2x5_reward_semantics.py
python3 tests/test_go2x5_training_readiness.py
python3 tests/test_go2x5_alignment.py
python3 tests/test_low_high_runtime_parity.py
python3 tests/test_go2x5_reachability_plot.py
python3 low-level/legged_gym/scripts/audit_go2x5_low_level_rewards.py --fail-on-mismatch
python3 -m py_compile \
  low-level/legged_gym/envs/manip_loco/go2x5_config.py \
  low-level/legged_gym/envs/manip_loco/manip_loco.py \
  low-level/legged_gym/envs/rewards/maniploco_rewards.py \
  low-level/legged_gym/scripts/train.py \
  low-level/legged_gym/scripts/check_go2x5_checkpoint_rollout.py \
  low-level/legged_gym/scripts/check_go2x5_fixed_command_gait.py \
  third_party/rsl_rl/rsl_rl/runners/on_policy_runner.py \
  high-level/envs/b1z1_base.py
git diff --check
```

### Isaac Gym runtime 与 low/high parity

| 检查 | 结果 |
| --- | --- |
| S0 readiness，8 env × 200 step | passed，early reset 0，nonfinite 0 |
| S1 readiness，8 env × 200 step | passed，early reset 0，nonfinite 0 |
| independent translation-Jacobian oracle | max error 0 |
| C4 constant-probe true IK low/high | mismatch 0，oracle failure 0，nonfinite 0 |
| schema-v2 random 12D checkpoint production loader | 真实加载通过 |
| checkpoint C2 output/action/torque/IK parity | 所有 max error 0 |

S3 deployment contract hash：

```text
38c4e29a0305da890cc0adab2e0bf7da1f6fffafc84f2b9459141a282eaf67bc
```

### smoke 选模过程

第一版 true-IK smoke `go2x5_v9_trueik_warm50000_smoke_seed1` 证明了 weights-only 路径和 3D IK 可工作，但暴露了一个真实训练分布问题：随机命令没有保证纯转向样本，且 yaw tracking 权重只有 vx 的四分之一。v9 的 model 300/400 虽然安全，但左转只有约 `0.056 rad/s`；model 600/800 又出现 roll reset，因此全部拒绝作为长训起点。

最终 profile 增加了 20% 显式原地转向样本，左右符号均衡，S0/S1 最小绝对 yaw 分别为 `0.10/0.15 rad/s`，同时把 yaw tracking 权重提高到 1.0。新 smoke：

```text
go2x5_v10_explicit_turn_from_v9_400_smoke_seed1
```

它以 v9 model 400 为网络权重来源，再次走严格 weights-only 初始化；不是跨任务 full resume。model 200 已能通过五组固定命令，但一个随机种子中出现 1 次 roll reset，故拒绝。model 300 通过两组独立随机 rollout 和全部固定命令，选为长训起点。

model 300 metadata：

| 字段 | 值 |
| --- | --- |
| schema | v2 |
| action dimension | 12 |
| num arm actions | 0 |
| profile | `go2x5_velocity_ee_coordination_v4_explicit_turn` |
| stage | `S0_slow_velocity_coordinated_reach`，iteration 300 |
| IK task | `position_only_translation_3d` |
| action scale | `0.125/0.25/0.25` × 4 legs |
| control contract SHA256 | `60a34f33eee22e6380153fe430211e7ce4700a2f17372ecf4207e27a758fc25c` |
| warm-start source | v9 model 400，SHA256 `af01bcd8f0d64f32d32d893c7cada90cfede072266088f0f2f559290854221ea` |
| restored from source | model weights only；optimizer/std/counters/curriculum/environment state 均为 false |

### model 300 随机 rollout

每组使用 deterministic deployment-history policy、128 env × 500 policy step：

| 指标 | seed 20260721 | seed 20260722 | gate |
| --- | ---: | ---: | --- |
| early reset | 0 | 0 | passed |
| nonfinite | 0 | 0 | passed |
| collision raw/tick | 0.0 | 0.0 | passed |
| arm target clamp，全部 6 joints | 0.0% | 0.0% | passed |
| action saturation | 0.0% | 0.000130% | passed |
| vx absolute error | 0.00447 m/s | 0.00417 m/s | passed |
| yaw absolute error | 0.01513 rad/s | 0.01372 rad/s | passed |
| mean EE error | 0.12687 m | 0.12709 m | 未收敛 |
| height adaptation error | 0.01167 m | 0.01160 m | passed |
| pitch adaptation error | 0.01758 rad | 0.01729 rad | passed |
| goal-z/base-height correlation | -0.0851 | -0.1636 | 未收敛 |
| goal-z/base-pitch correlation | -0.2104 | -0.2169 | passed |
| safety gate | passed | passed | passed |
| final coordination gate | failed | failed | 长训目标，不是启动阻塞项 |

这里的结论是“适合作为长训起点”，不是“已经训练完成”。EE error 和身体高度相关性必须随长训继续收敛；如果长期停滞，训练应 fail-closed 停止并重新审查 reward/target 分布，而不是放宽 gate。

### model 300 固定命令 gate

每个 case 使用 64 env、50 step warm-up、200 step measurement；未规定 gait shape：

| case | command | 实际均值 | absolute error | reset | collision | 结果 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| stand | vx 0，yaw 0 | vx 0.00189，yaw 0.00215 | vx 0.00227，yaw 0.00636 | 0 | 0.0 | passed |
| forward | vx +0.10 | vx +0.10183 | vx 0.00498 | 0 | 0.0 | passed |
| backward | vx -0.10 | vx -0.10249 | vx 0.00606 | 0 | 0.0 | passed |
| turn left | yaw +0.15 | yaw +0.12919 | yaw 0.02240 | 0 | 0.0 | passed |
| turn right | yaw -0.15 | yaw -0.14092 | yaw 0.01632 | 0 | 0.0 | passed |

因此，速度 catch、安全、动作边界、IK joint-limit 和数值有限性已经满足启动 45,000 iteration 长训的门槛。

## 可复制命令

### 从旧模型做一次且仅一次 weights-only 初始化

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
/data4/duanzhibo/miniconda3/envs/b1z1/bin/python \
  low-level/legged_gym/scripts/train.py \
  --headless \
  --task go2x5 \
  --proj_name go2x5-low \
  --exptid <fresh-v10-warm-start-run> \
  --num_envs 512 \
  --max_iterations 300 \
  --seed 1 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --graphics_device_id 0 \
  --warm_start_checkpoint /data4/duanzhibo/xhq_workload/VBC-gx/low-level/logs/go2x5-low/go2x5_v7_front_workspace_from36200_seed1/model_50000.pt
```

`--warm_start_checkpoint` 只能在新 run 的 iteration 0 使用，不能与 `--resume` 或 `--resumeid` 同时使用。

### 从选定的 model 300 启动长训

后续必须使用正常 full resume，从新 run checkpoint 恢复新 optimizer 和 curriculum：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
/data4/duanzhibo/miniconda3/envs/b1z1/bin/python \
  low-level/legged_gym/scripts/train.py \
  --headless \
  --task go2x5 \
  --proj_name go2x5-low \
  --exptid go2x5_v10_velocity_ee_coordination_seed1 \
  --num_envs 4096 \
  --max_iterations 45000 \
  --seed 1 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --graphics_device_id 0 \
  --resume \
  --resumeid go2x5_v10_explicit_turn_from_v9_400_smoke_seed1 \
  --checkpoint 300
```

这里使用正常 full resume，恢复的是同一 v10 profile 的 model 300、optimizer、探索标准差和 runner/curriculum 状态。不要再次使用旧 model 50000，也不要给这个命令同时添加 `--warm_start_checkpoint`。

### 长训监控 gate

长训启动后至少持续监控：

- nonfinite 必须始终为 0；
- 非 timeout reset 不得形成持续上升趋势；
- 固定前进、后退、左右转 gate 必须保持通过；
- `mean_ee_error_m` 应下降，goal-z/body-height correlation 应转为正并最终超过 0.30；
- arm clamp 和 action saturation 必须继续接近 0；
- S1 切换后重新执行两随机种子 rollout，不能只看总 reward。

## 仍未证明的内容

- 当前 model 300 只是通过长训启动门槛，不是训练完成的正式 12D deployment checkpoint；
- EE 误差和 body-goal correlation 需要在长训中收敛后再次通过 strict coordination gate；
- rough terrain、domain randomization、push、noise、相机和物体接触鲁棒性尚未恢复；
- high-level teacher/student 训练尚未恢复；
- sim-to-real 与真实抓取成功率尚未证明。

在 low-level final gate 通过前，不应启动 high-level teacher，也不应把当前 smoke 模型用于部署。
