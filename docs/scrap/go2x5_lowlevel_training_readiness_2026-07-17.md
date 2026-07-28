# Go2-X5 low-level 长训就绪审查（2026-07-17）

## 结论

**GO：当前版本可以从零开始 Go2-X5 low-level S0 正式长训。**

这里的“可以长训”有严格边界：它指当前 `go2x5_stable_reach_curriculum_v3_flat_step_metrics` 配置可以启动平地、确定性、12D 腿部策略的分阶段基础训练。它不表示已经得到正式训练模型，也不表示 rough terrain、domain randomization、sim-to-real 或 high-level teacher 已经就绪。

本轮没有使用任何旧模型作为训练起点。生成的 smoke checkpoint 只用于验证保存、加载、rollout 和断点续训，已列入清理范围，不提交仓库。

## 审查范围

- 分支：`agent/go2x5-runtime-parity`
- 审查开始时 HEAD：`b6ed563 Add nonzero multi-state Go2-X5 controller parity`
- task：`go2x5`
- policy action：12D，仅腿部
- observation：799D，包含 5 个 gait fields
- policy tick：50 Hz
- physics decimation：4
- 初始训练地形：零高度扰动的 flat trimesh
- DR、noise、push：关闭
- arm：position-only IK，`ik_gain=0.25`，joint-limit clamp

## 发现并修复的训练问题

| 问题 | 训练后果 | 修复 |
|---|---|---|
| S3 contact-shaped reward 的原始值是非正误差，但 scale 为负 | 把错误接触和 stance foot 滑动变成正奖励 | 两项 scale 改为 `+0.5`，并增加单调性 probe |
| contact-shaped reward 在停止命令下仍生效 | 站立时被 gait phase 伪目标干扰 | walking mask 为 false 时强制为零 |
| feet jerk 检查错误对象上的历史字段 | 历史力永远不生效，jerk 基本恒为零 | 统一使用 `env.last_contact_forces` 并在 reset 清零 |
| air-time 固定使用 0.5 s 且默认只覆盖前脚 | 与 2 Hz、50% swing 的 0.25 s 目标不符 | target 改为 0.25 s，四足全部参与 |
| feet-height 使用前脚整体 norm 和 world-z | 一只脚可掩盖另一只低脚，地形高度也会污染奖励 | 改为逐脚、只在 swing phase、terrain-relative clearance |
| `ManipLoco` 覆盖 callback 后没有更新 measured heights | base-height 和 terrain-relative shaping 使用陈旧值 | 初始化 height points，并在每个 policy tick 更新 heights |
| episode reset 没有清当前 action、torque、gait、clock、contact/history | 跨 episode 状态泄漏，首帧 observation/action 不一致 | reset 全量清理 controller 和 history 状态 |
| 12D policy 仍把空 arm-action ratio 当第二个 policy channel 平均 | 腿部 surrogate gradient 被静默减半 | `num_arm_actions=0` 时只优化一个 stochastic policy channel；arm reward 仍经 advantage mixing 影响腿策略 |
| PPO 没有明确检查 NaN/Inf | 非有限值可能污染多个 update 后才表现为崩溃 | 对输入、action、value、log-prob、rollout、return、loss、gradient norm、std 增加 hard fail |
| checkpoint 没保存 history optimizer、PPO counter、runner 统计和 `global_steps` | resume 后 schedule、history encoder、command warm-up 不连续 | 全部保存并严格恢复；缺字段直接拒绝 |
| `max_iterations` 在 resume 时被当作“再训练 N 次” | 30→32 可能错误跑到 62 | 改为总目标 iteration，实际只运行 `target-current` |
| save/curriculum 使用零基 iteration | checkpoint 编号和阶段门槛 off-by-one | 统一使用 completed iteration |
| `--resumeid` 必须隐式携带前导 `/` | 文档形式的普通 run id 会拼成错误目录 | 使用 `os.path.join`，兼容有无前导斜杠 |
| gait observation 默认关闭 | 不加 CLI flag 会训练出与部署 799D contract 不兼容的模型 | task config 默认强制启用 gait fields |
| resume metadata 校验不完整 | 18D、旧 profile、坏 contract 或错误 asset 可能被误加载 | schema/action/obs/joint order/asset/hash/curriculum/training state 全部 fail-closed |
| 原 0--2 cm rough-flat 在 Go2-X5 上产生大量 calf collision | S0 的 collision gate 在 zero policy 下也不可达 | 基础 curriculum 改为 flat；profile 升级并拒绝旧 checkpoint |
| `Episode_metric/*` 用累计值除 episode 秒数 | 原始物理量被 50 Hz 放大，EE gate `<0.50` 实际永远无法通过 | 改为每个已完成 episode 的逐 policy-step 均值；profile 升级为 v3 |

rough-flat 的隔离实验给出了直接证据：旧配置下 model-100 的 collision raw 约为 `0.539/tick`，128 个环境 10 秒出现 10 次 roll reset；zero policy 仍约为 `0.661/tick` 并出现 5 次 roll reset。相同条件改为 flat 后 collision 为 0，证明它不是单纯“策略没训练好”。

## 主要代码变更

- `low-level/legged_gym/envs/manip_loco/go2x5_config.py`
  - 固定 799D gait-aware training contract；修正 S3 reward sign；四足 gait shaping；flat v3 curriculum。
- `low-level/legged_gym/envs/manip_loco/manip_loco.py`
  - terrain height 更新；完整 reset；逐 step episode metric；严格 checkpoint metadata/training-state 恢复。
- `low-level/legged_gym/envs/rewards/maniploco_rewards.py`
  - stop mask、feet jerk、air-time、逐脚 terrain-relative clearance。
- `third_party/rsl_rl/rsl_rl/algorithms/ppo.py`
  - 12D 单 policy-channel surrogate；全路径 nonfinite hard fail。
- `third_party/rsl_rl/rsl_rl/runners/on_policy_runner.py`
  - completed-iteration 语义；完整 checkpoint；正确 ETA。
- `low-level/legged_gym/scripts/train.py`
  - resume 使用总目标 iteration。
- `low-level/legged_gym/utils/task_registry.py`
  - 修复 `--resumeid` 路径。
- `low-level/legged_gym/scripts/audit_go2x5_low_level_rewards.py`
  - 可执行、fail-closed 的 reward sign/dependency audit。
- `low-level/legged_gym/scripts/check_go2x5_training_readiness.py`
  - GPU reward/IK/reset/metadata/curriculum/gait/nonfinite gate；区分 S0 启动 gate 和后续 stage stress。
- `low-level/legged_gym/scripts/check_go2x5_checkpoint_rollout.py`
  - checkpoint deterministic/stochastic rollout；默认提前 reset 零容忍。
- `tests/test_go2x5_training_readiness.py`
  - 覆盖上述训练 contract 和 fail-closed 行为。

## 验证结果

| 验证 | 规模 | 关键结果 | 结论 |
|---|---:|---|---|
| CPU compile + parity/alignment/readiness tests | 3 个 test entrypoint | 全通过 | PASS |
| reward audit | 27 leg + 1 arm active terms，全部 curriculum override | 0 mismatch | PASS |
| S0 GPU readiness | 8 env × 500 ticks（10 s） | 47/47 checks，0 early reset，0 nonfinite | PASS |
| S3 数学/runtime probe | 8 env × 200 ticks（4 s） | 47/47 checks，0 early reset，0 nonfinite | PASS |
| PPO smoke | 256 env × 30 iterations | 184,320 transitions；89 scalar tags、2,666 points 全 finite | PASS |
| PPO 最终 episode | iteration 29 | length 502；reward 32.81；collision 0；roll/pitch/z reset 0 | PASS |
| 修复后的 EE gate metric | iteration 29 | L1 position error `0.14324 m`，门槛 `<0.50` | PASS |
| deterministic checkpoint rollout | 128 env × 500 ticks | 0 reset，0 collision，0 nonfinite | PASS |
| checkpoint resume | 30→32→33 | 只增加目标差值；optimizer/counter/global_steps/runner/contract 连续 | PASS |
| 默认容量 | 4096 env × 2 iterations | 196,608 transitions，40,767--44,420 steps/s，无 OOM | PASS |

所有 TensorBoard scalar 均有限。iteration-30 deterministic policy 的 `max_abs_action=0.2194`，`max_abs_leg_torque=14.6638 Nm`，未触发异常。

## 非阻塞压力结果及解释

为了避免把结果包装得比证据更强，本轮保留两个未通过“零 reset”严格门槛的压力结果：

1. S3 最终宽 EE 范围、未训练 fixed probe、8 env × 500 ticks：3 次 roll reset，首次在 tick 402；无 NaN/Inf。
2. 仅训练 30 iterations 的 stochastic checkpoint、128 env × 500 ticks：1 次 roll reset；无 NaN/Inf，最大 calf torque 正好被 URDF limit `45.43 Nm` clamp。

这两项不阻塞从 S0 开始长训：S0 完整 10 秒严格 gate 已通过，课程会先学习稳定/小范围 reach，并由 episode length、roll/z、collision、EE error 联合门槛控制后续升级。它们同时意味着不能宣称 S3 已训练完成；进入 S3 后仍必须用当时的 checkpoint 重新做闭环验收。

## 可复制命令

### CPU gate

```bash
python3 -m py_compile \
  low-level/legged_gym/envs/manip_loco/go2x5_config.py \
  low-level/legged_gym/envs/manip_loco/manip_loco.py \
  low-level/legged_gym/envs/rewards/maniploco_rewards.py \
  low-level/legged_gym/scripts/train.py \
  low-level/legged_gym/scripts/audit_go2x5_low_level_rewards.py \
  low-level/legged_gym/scripts/check_go2x5_training_readiness.py \
  low-level/legged_gym/scripts/check_go2x5_checkpoint_rollout.py \
  third_party/rsl_rl/rsl_rl/algorithms/ppo.py \
  third_party/rsl_rl/rsl_rl/runners/on_policy_runner.py

python3 tests/test_low_high_runtime_parity.py
python3 tests/test_go2x5_alignment.py
python3 tests/test_go2x5_training_readiness.py
python3 low-level/legged_gym/scripts/audit_go2x5_low_level_rewards.py --fail-on-mismatch
git diff --check
```

### S0 正式启动 gate

```bash
conda run -n vwc_go2x5 python \
  low-level/legged_gym/scripts/check_go2x5_training_readiness.py \
  --num-envs 8 \
  --steps 500 \
  --rollout-stage 0 \
  --sim-device cuda:0 \
  --rl-device cuda:0 \
  --graphics-device-id 0 \
  --output /tmp/go2x5_training_readiness_s0.json
```

### 从零开始正式 low-level 长训

```bash
WANDB_MODE=offline conda run -n vwc_go2x5 python \
  low-level/legged_gym/scripts/train.py \
  --headless \
  --task go2x5 \
  --proj_name go2x5-low \
  --exptid go2x5_v3_flat_seed1 \
  --num_envs 4096 \
  --max_iterations 45000 \
  --seed 1 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --graphics_device_id 0
```

这条命令不含 `--resume`/`--resumeid`，因此一定从随机初始化开始。`WANDB_MODE=offline` 可换成已配置好的 `online`。

### 断点续训到总目标 45000

```bash
WANDB_MODE=offline conda run -n vwc_go2x5 python \
  low-level/legged_gym/scripts/train.py \
  --headless \
  --task go2x5 \
  --proj_name go2x5-low \
  --exptid go2x5_v3_flat_seed1 \
  --resumeid go2x5_v3_flat_seed1 \
  --checkpoint -1 \
  --num_envs 4096 \
  --max_iterations 45000 \
  --seed 1 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --graphics_device_id 0
```

`max_iterations` 现在是总目标，不是额外 iteration 数。

## 长训监控门槛

- 任意 `FloatingPointError`、NaN/Inf：立即停止，不能用 `nan_to_num` 继续。
- S0 不应早于 iteration 1000 升级；不要手工强制切 stage。
- S0 升级前，最近窗口应满足：episode length `>450`、roll reset `<0.05`、z reset `<0.02`、collision metric `<2.0`、EE L1 error `<0.50 m`。
- 每 200 iterations 检查 checkpoint 是否包含 schema-v2 metadata、v3 profile 和完整 optimizer/runner state。
- stage 变化后重点观察 200 iterations；若 reset/collision 明显恶化，让 metric gate 保持当前 stage，不要绕过。
- 进入 S3 后，对对应 checkpoint 重跑 deterministic 10 秒 rollout；在该证据通过前，不把模型称为 deployment checkpoint。

## 尚未证明

- 尚无 45,000-iteration 正式训练收敛结果。
- 尚无可用的正式 12D trained checkpoint。
- 尚未证明 rough terrain、noise、push、payload 或 domain randomization 下的鲁棒性。
- 尚未证明实机安全性或 sim-to-real。
- 尚未允许恢复 high-level teacher/student 训练。

因此当前允许的下一步只有：**启动并监控 Go2-X5 low-level v3 flat curriculum 长训**。不能从本报告直接跳到部署或 high-level 正式训练。
