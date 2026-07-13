# Go2-X5 Runtime Parity 本轮修改审查文档

日期：2026-07-12

分支：`agent/go2x5-runtime-parity`

基线提交：`c06135a Align Go2-X5 low and high-level runtime contracts`

## 1. 审查结论摘要

本轮把上一阶段的配置、URDF 和 checkpoint metadata 静态对齐，推进为可执行的 Isaac Gym runtime parity 验证。现在可以分别从 low-level 和 high-level 环境采集统一格式的运行时快照，并对以下两层行为执行结构化比较：

1. 机器人实际加载后的刚体、碰撞 shape 和 DOF properties；
2. 同一 canonical state 下的 observation、history、gait、action、torque、EE frame 和 arm IK target。

本轮运行时验证最终结果为：

| 验证层 | 容差 | mismatch | 结果 |
| --- | ---: | ---: | --- |
| Runtime properties | `1e-6` | 0 | 通过 |
| Canonical controller | `1e-5` | 0 | 通过 |

验证过程中发现并修复了四类真实问题：

- 12 维策略下 motor-strength tensor 的零宽拼接会触发 TorchScript/PyTorch 内部错误；
- high-level gripper position-drive gains 与 low-level 不一致且包含随机值；
- high-level 忽略 YAML 中的 `robotStartPose`，导致 EE world/local target 的 z 偏差为 0.22 m；
- low-level IK 目标没有执行与 high-level 相同的 URDF joint-limit clamp。

当前配置引用的 `model_17600.pt` 经审计是旧 18 维 checkpoint，不能加载到当前 12 维 wrapper。本轮没有绕过生产加载器；仅在诊断工厂中使用显式零动作策略，以便验证 checkpoint 以外的环境与控制器语义。

## 2. 修改范围与非目标

### 2.1 本轮范围

- 建立统一 runtime/controller snapshot 数据结构；
- 建立可独立 capture 和 compare 的命令行工具；
- 建立 deterministic 单环境 low/high 诊断工厂；
- 修复由动态比较暴露的 runtime parity 问题；
- 将新增控制参数纳入 control contract；
- 增加工具单元测试、静态对齐断言和机器可读验证报告。

### 2.2 本轮非目标

- 不重新训练 low-level policy；
- 不修改奖励函数、PPO 网络或 high-level teacher；
- 不证明 10 秒闭环轨迹已经一致；
- 不证明旧 18 维 checkpoint 可用于当前 12 维控制接口；
- 不恢复 orientation tracking、物体接触或 domain randomization。

## 3. 验证架构

```text
low factory  ──> low env  ──> snapshot ──┐
                                          ├── structured diff ──> JSON report / exit code
high factory ──> high env ──> snapshot ──┘
```

工具把“环境如何构造”和“比较哪些字段”分开：

- factory 负责构造 deterministic Isaac Gym 环境并注入 canonical state；
- collector 只读取实际 runtime tensor/property，不依赖训练入口；
- comparator 递归展开字典和数组，报告具体字段路径、两侧值与绝对误差；
- CLI 在存在 mismatch 时返回非零退出码，可直接接入 CI。

## 4. 按文件说明

### 4.1 `tools/go2x5_runtime_parity.py`

新增 runtime parity 的核心采集与比较逻辑。

Runtime property snapshot 覆盖：

- runtime rigid-body 顺序；
- 配置的 base body name 与实际 index；
- 每个 rigid body 的 mass、COM、inertia；
- total mass；
- 每个 rigid shape 的 friction；
- DOF order、drive mode、stiffness、damping；
- DOF position、velocity、effort limits。

Controller snapshot 覆盖：

- current proprio observation；
- observation history；
- gait phase 与 clock inputs；
- policy-order action；
- URDF-order applied action；
- leg torque；
- arm 当前关节位置与 position target；
- EE world/local target；
- current EE position；
- arm-base world position。

比较器的成功标准是：数据结构相同，且所有数值字段的绝对误差不超过指定 `atol`。`side` 字段只用于标识快照来源，不参与数值一致性判断。

### 4.2 `tools/go2x5_parity_factories.py`

新增 `make_low_env()` 和 `make_high_env()`，用于构造可复现的单环境诊断场景：

- `num_envs = 1`；
- gravity 为零；
- domain randomization、noise、push、camera 和课程关闭；
- locomotion command、action、gait state 清零；
- action delay 为零；
- root 与 DOF state 注入相同 canonical state；
- root z 临时置为 1.0，避免单帧刷新期间引入地面或物体接触；
- 刷新 Isaac Gym runtime tensor 后，再显式恢复用于比较的 controller state 和 history。

High-level 诊断工厂会临时替换 low-policy loader，返回维度正确的零动作策略。该替换只存在于 factory 构造期间，并通过 `finally` 恢复原 loader；生产环境的 checkpoint 校验和加载逻辑没有被放宽。

### 4.3 `scripts/check_go2x5_runtime_parity.py`

新增两个子命令：

- `capture`：通过 `module:function` factory 采集 `runtime` 或 `controller` snapshot；
- `compare`：比较 low/high JSON，输出报告，并以退出码表示通过或失败。

这种接口避免把 parity 工具绑定到特定训练脚本，同时允许以后增加新的环境 factory。

### 4.4 `high-level/envs/b1z1_base.py`

#### Robot start pose

原实现直接使用构造函数默认参数 `robot_start_pose=(0, 0, 0.55)`，未读取 YAML 的 Go2-X5 起始高度 0.33 m。现改为优先读取：

```python
self.robot_start_pose = tuple(
    self.cfg["env"].get("robotStartPose", robot_start_pose)
)
```

该修复消除了 EE world/local target 的 0.22 m z 偏差。

#### Motor strength tensor

原实现分别创建固定 12 维腿 tensor 和 `num_actions - 12` tensor。12 维策略会产生 `(num_envs, 0)` 的第二个 tensor，并在当前运行环境触发内部错误。现直接一次创建：

```python
(num_envs, low_policy_num_actions)
```

数值范围仍由 YAML domain-randomization profile 控制。

#### Gripper drive gains

原 high-level 对 gripper stiffness 使用每个环境独立的 `uniform(2, 5)`，damping 固定为 2.5；low-level 实际使用 110/7.5。现改为从 asset control 配置读取，并默认继承 arm drive gains。这样 deterministic parity 下不再包含隐藏随机控制参数。

### 4.5 `high-level/data/cfg/go2x5_pickmulti.yaml`

新增：

```yaml
gripperPositionDriveStiffness: 110.0
gripperPositionDriveDamping: 7.5
```

同样的字段写入 `lowPolicyContract`，使 checkpoint contract hash 能覆盖 gripper position-drive 语义。

### 4.6 `low-level/legged_gym/envs/manip_loco/manip_loco.py`

- control contract 增加 gripper stiffness/damping；
- arm IK 更新后按 URDF `dof_pos_limits` clamp q-target；
- clamp 的关节切片与 high-level 相同，仅覆盖 6 个 arm joints，不改变 gripper target。

这防止接近奇异位形或边界目标时，low/high 对同一 IK delta 生成不同的非法 position target。

### 4.7 测试与报告

`tests/test_low_high_runtime_parity.py` 覆盖：

- 相同 snapshot 通过；
- mismatch 的字段路径和值正确；
- tolerance 边界；
- JSON write/read round trip；
- CLI report 与非零失败退出码。

`tests/test_go2x5_alignment.py` 增加：

- high-level gripper gains 与 robot spec 对齐；
- control contract 包含 gripper gains；
- high-level 读取 YAML robot start pose；
- motor-strength tensor 使用完整 action dimension。

机器可读的动态验证结果：

- `docs/parity_reports/go2x5_runtime_report_2026-07-12.json`；
- `docs/parity_reports/go2x5_controller_report_2026-07-12.json`。

简版阶段记录保留在 `docs/go2x5_dynamic_parity_phase_b_2026-07-12.md`。

## 5. 完整复现步骤

以下命令从仓库根目录执行。Isaac Gym capture 需要 `vwc_go2x5` conda 环境和可用 NVIDIA GPU。

### 5.1 采集 runtime properties

```bash
conda run -n vwc_go2x5 python scripts/check_go2x5_runtime_parity.py capture \
  --side low \
  --kind runtime \
  --factory tools.go2x5_parity_factories:make_low_env \
  --output /tmp/go2x5-low-runtime.json

conda run -n vwc_go2x5 python scripts/check_go2x5_runtime_parity.py capture \
  --side high \
  --kind runtime \
  --factory tools.go2x5_parity_factories:make_high_env \
  --output /tmp/go2x5-high-runtime.json

python3 scripts/check_go2x5_runtime_parity.py compare \
  --low /tmp/go2x5-low-runtime.json \
  --high /tmp/go2x5-high-runtime.json \
  --atol 1e-6 \
  --report /tmp/go2x5-runtime-report.json
```

### 5.2 采集 canonical controller state

```bash
conda run -n vwc_go2x5 python scripts/check_go2x5_runtime_parity.py capture \
  --side low \
  --kind controller \
  --factory tools.go2x5_parity_factories:make_low_env \
  --output /tmp/go2x5-low-controller.json

conda run -n vwc_go2x5 python scripts/check_go2x5_runtime_parity.py capture \
  --side high \
  --kind controller \
  --factory tools.go2x5_parity_factories:make_high_env \
  --output /tmp/go2x5-high-controller.json

python3 scripts/check_go2x5_runtime_parity.py compare \
  --low /tmp/go2x5-low-controller.json \
  --high /tmp/go2x5-high-controller.json \
  --atol 1e-5 \
  --report /tmp/go2x5-controller-report.json
```

### 5.3 不依赖 GPU 的测试

```bash
python3 tests/test_low_high_runtime_parity.py
python3 tests/test_go2x5_alignment.py
python3 -m py_compile \
  tools/go2x5_runtime_parity.py \
  tools/go2x5_parity_factories.py \
  scripts/check_go2x5_runtime_parity.py \
  tests/test_low_high_runtime_parity.py
git diff --check
```

## 6. Checkpoint 兼容性审计

当前 YAML 指向：

```text
low-level/logs/go2x5-low/go2x5_stable_base_v1/model_17600.pt
```

该文件包含旧的 arm policy head，并且 `std` shape 为 `[1, 18]`；当前 wrapper 要求 12 维 leg action。因此真实生产 loader 会正确拒绝该 checkpoint，典型差异包括：

```text
Unexpected actor_arm_control_head keys
std: checkpoint [1,18] vs current [1,12]
```

本轮不能声称“真实 policy output parity 已通过”。当前通过的是：环境属性、控制器输入语义、零动作 action mapping、torque 计算、EE frame 与 IK target parity。完成真实 policy parity 必须先生成带 schema v2 control-contract metadata 的 12 维 checkpoint。

## 7. 已知限制与风险

1. Controller parity 是单帧 canonical comparison，不是闭环 rollout。
2. High-level capture 使用诊断零动作 policy，因此没有覆盖真实网络推理输出。
3. 为隔离接触差异，canonical factory 使用零重力和较高 root z；这不覆盖足端接触时序。
4. 当前没有覆盖 stop/start command 下 gait state-machine 的多帧演化。
5. 当前没有覆盖 object、gripper contact、payload 或 domain randomization。
6. Gripper gains 现在与 low-level 对齐为 110/7.5；该值解决的是系统一致性，最终硬件最优值仍需独立标定。
7. Contract schema 新增字段后，旧 metadata 的 hash 不会匹配；这是预期的 fail-closed 行为。

## 8. 建议审查重点

建议审查者重点确认：

- snapshot 字段是否覆盖下一阶段闭环验证所需的最小数据集；
- diagnostic loader 是否与生产 checkpoint loader 隔离充分；
- canonical state 重注入是否会掩盖需要在多帧测试中暴露的问题；
- gripper gains 是否应继续跟随 arm gains，还是单独形成硬件标定参数；
- `robotStartPose` 的配置优先级是否符合所有 B1/Go2 调用方；
- control contract schema 增加 gripper 字段后，是否接受旧 checkpoint 全部失效；
- CLI 的 `module:function` factory 接口是否适合作为后续 CI 扩展点。

## 9. 下一阶段建议

按风险和依赖顺序建议继续：

1. 训练或转换出合法的 12 维 schema v2 checkpoint；
2. 在同一 observation 上比较真实 policy output；
3. 增加 stop/start command 的逐帧 gait parity test；
4. 增加 10 秒 zero-command rollout，比较 root state、roll/pitch、base height、contact 和 torque；
5. 再逐项引入 friction、motor strength、mass/COM、payload 和 delay；
6. 最后恢复物体接触及 high-level teacher 训练。

## 10. 提交边界

本轮提交只包含上述代码、测试、文档和 JSON 报告。工作区中的：

```text
low-level/legged_gym/envs/logs.zip
```

是未跟踪的用户文件，不属于本轮修改，不会加入提交。
