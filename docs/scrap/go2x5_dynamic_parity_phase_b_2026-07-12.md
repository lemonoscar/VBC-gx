# Go2-X5 动态 Parity Phase B 修改与验证记录

日期：2026-07-12
分支：`agent/go2x5-runtime-parity`

## 目标

本阶段把上一阶段的 XML/YAML/metadata 静态对齐扩展为两层 Isaac Gym runtime 验证：

1. runtime property snapshot：比较实际加载后的刚体、shape 和 DOF properties；
2. controller snapshot：在同一 canonical state 下比较 observation、history、gait、action、torque、EE frame 和 arm IK target。

## 新增工具

### `tools/go2x5_runtime_parity.py`

提供：

- `collect_runtime_snapshot()`；
- `collect_controller_snapshot()`；
- JSON snapshot 读写；
- 带绝对误差阈值的结构化 diff；
- mismatch path、low/high value 和 absolute error 报告。

Runtime property snapshot 包含：

- runtime body order；
- configured base body name/index；
- 每个刚体 mass、COM 和 inertia；
- total mass；
- 每个 rigid shape friction；
- DOF order、drive mode、kp/kd、position/velocity/effort limits。

Controller snapshot 包含：

- current proprio；
- observation history；
- gait phase/clock；
- policy-order action；
- URDF-order action；
- leg torque；
- arm q 和 q-target；
- EE world/local target；
- current EE position；
- arm-base world position。

### `tools/go2x5_parity_factories.py`

提供 deterministic 单环境 low/high factories：

- 1 个环境；
- domain randomization 关闭；
- command/action/gait 清零；
- canonical root/DOF state；
- 零重力、无地面接触的 kinematic refresh；
- canonical history initialization。

High-level factory 使用显式 zero-action diagnostic policy，只用于环境/controller capture，不替代生产 checkpoint loader。

### `scripts/check_go2x5_runtime_parity.py`

支持 capture 和 compare：

```bash
python3 scripts/check_go2x5_runtime_parity.py capture \
  --side low --kind runtime \
  --factory tools.go2x5_parity_factories:make_low_env \
  --output /tmp/go2x5-low-runtime.json

python3 scripts/check_go2x5_runtime_parity.py compare \
  --low /tmp/go2x5-low-runtime.json \
  --high /tmp/go2x5-high-runtime.json \
  --atol 1e-6 --report /tmp/go2x5-runtime-report.json
```

CLI 在 mismatch 时返回非零状态，并可输出机器可读 JSON report。

## 动态验证发现并修复的问题

### 1. 12 维 motor-strength 初始化崩溃

High-level 原实现分别生成 12 维 leg tensor 和 `num_actions - 12` tensor。当 Go2 policy 为 12 维时，第二个 TorchScript random tensor 的 shape 为 `(N, 0)`，在当前 PyTorch/Isaac Gym 环境触发内部 stride assert。

现改为一次生成：

```python
motor_strength.shape == (num_envs, low_policy_num_actions)
```

### 2. Gripper position-drive gains 不一致

第一次 runtime diff 得到 4 个 mismatch：

```text
dofs[18].stiffness/damping
dofs[19].stiffness/damping
```

Low-level 对 arm 和 gripper 都使用 110/7.5；high-level gripper 原先随机使用 kp 2–5、kd 2.5。

现 high-level 明确配置：

```yaml
gripperPositionDriveStiffness: 110.0
gripperPositionDriveDamping: 7.5
```

并将两项加入 checkpoint control contract。

### 3. High-level 忽略 YAML `robotStartPose`

Canonical controller diff 最初发现 EE world/local z 整体相差 0.22 m。原因是 YAML 配置为 0.33 m，但 high-level 构造函数仍采用默认参数中的 0.55 m，并用它计算 terrain-invariant EE center。

现优先读取：

```python
self.cfg["env"]["robotStartPose"]
```

### 4. Low-level arm q-target 缺少 joint-limit clamp

High-level 已 clamp arm target，low-level 没有。本阶段给 low-level 加入相同的 URDF joint-limit clamp，使 IK target 语义一致。

## Checkpoint 审计结果

配置当前指向的：

```text
low-level/logs/go2x5-low/go2x5_stable_base_v1/model_17600.pt
```

是旧 18 维 action checkpoint，包含 arm policy head和 18 维 `std`。当前 12 维 wrapper 无法加载它：

```text
Unexpected actor_arm_control_head keys
std shape checkpoint [1,18] vs current [1,12]
```

因此它不能用于 schema v2 parity 或恢复 high-level 训练。需要重新训练/保存 12 维 schema v2 checkpoint。

## 最终验证结果

### Runtime property parity

```json
{
  "atol": 1e-6,
  "mismatch_count": 0,
  "passed": true
}
```

验证覆盖 runtime body/DOF order、base index、mass、COM、inertia、friction、drive mode、kp/kd 和 limits。

### Canonical controller parity

```json
{
  "atol": 1e-5,
  "mismatch_count": 0,
  "passed": true
}
```

验证覆盖 proprio/history、gait、actions、leg torque、EE world/local frame 和 arm q-target。

### Python 测试

```text
python3 tests/test_low_high_runtime_parity.py
# runtime parity tests passed

python3 tests/test_go2x5_alignment.py
# go2x5 alignment tests passed

python3 -m py_compile ...
git diff --check
```

最终机器可读结果保存在：

- `docs/parity_reports/go2x5_runtime_report_2026-07-12.json`
- `docs/parity_reports/go2x5_controller_report_2026-07-12.json`

## 尚未覆盖

当前通过的是 runtime property parity 和单帧 canonical controller parity，尚不等于闭环 rollout parity。后续仍需：

1. 生成 12 维 schema v2 checkpoint；
2. 比较相同 observation 下的真实 policy output；
3. 执行 stop/start gait 多帧序列；
4. 执行 10 秒 zero-command trajectory；
5. 比较 roll/pitch、base height、contact、torque saturation 和 EE tracking。
