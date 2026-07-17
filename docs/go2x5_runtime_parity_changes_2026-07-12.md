# Go2-X5 Low/High Runtime Parity 修改记录

日期：2026-07-12  
范围：`visual-wholebody-control-go2x5` 第一阶段 deterministic runtime parity

## 背景与目标

Go2-X5 low-level 策略单独运行时使用的控制系统，与 high-level wrapper 内嵌运行时存在多项实现差异，包括刚体随机化目标、动力学随机化范围、PhysX 接触参数、机械臂 IK、EE 目标坐标系、动作延迟和步态状态机。

本轮修改的目标不是调整奖励或网络结构，而是先建立明确、可拒绝不匹配 checkpoint 的控制接口，使 low-level 与 high-level 在 deterministic 配置下尽可能使用相同的控制语义。

## 已完成修改

### 1. 基座刚体按名称索引

low-level 和 high-level 不再使用 `props[1]` 修改所谓的基座刚体。

- Go2-X5 配置明确使用 `baseBodyName: base` / `base_body_name = "base"`。
- 运行时通过 Isaac Gym 的 body-name dictionary 获取索引。
- 配置的 body name 不存在时立即抛出异常。
- B1 路径保留 `trunk` 作为默认名称，避免无意改变旧实验语义。

这消除了 Go2-X5 URDF 中 `props[1]` 可能指向 `Head_upper` 的高风险错误。

### 2. High-level domain randomization 配置化

以下 high-level 随机化不再固定写死在 Python 中：

- rigid-shape friction；
- leg motor strength；
- added base mass；
- base COM offset。

Go2-X5 当前使用 `go2x5_deterministic_v1`：

```yaml
friction: [1.0, 1.0]
motorStrength: [1.0, 1.0]
addedBaseMassKg: [0.0, 0.0]
baseComOffsetM: [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
```

matched domain randomization 将在 deterministic parity 通过后单独恢复。

### 3. PhysX 参数对齐

Go2-X5 high-level 主配置现与 low-level 基础配置保持一致：

| 参数 | 当前值 |
| --- | ---: |
| simulation dt | 0.005 s |
| position iterations | 4 |
| velocity iterations | 0 |
| contact offset | 0.01 m |
| rest offset | 0.0 m |
| bounce threshold | 0.5 m/s |
| max depenetration velocity | 1.0 m/s |
| buffer multiplier | 5.0 |

### 4. 机械臂 IK 语义对齐

High-level Go2-X5 arm controller 改为：

- position-only IK；
- `ik_gain = 0.25`；
- DLS damping = 0.05；
- 每个 50 Hz low-level policy step 计算一次 arm target；
- arm target 在四个 5 ms physics steps 内保持；
- arm joint target 按 URDF joint limits clamp；
- quaternion norm 使用最小值保护。

High-level orientation action 当前被冻结，直到 low-level 真正训练 orientation tracking。

### 5. EE world/local 坐标统一

Go2-X5 high-level 使用 `TERRAIN_INVARIANT_YAW`：

- EE target 的 x/y 随 base position 和 yaw；
- EE target 的 z 相对地形保持不变；
- 环境维护唯一的 `ee_goal_world`；
- 每次 low-level policy 更新前刷新 world target；
- policy observation 由当前 base pose 和实际 world target 重新计算 local goal。

High-level workspace 同时限制在当前 low-level 训练范围：

```text
x:  0.05 .. 0.60 m
y: -0.30 .. 0.30 m
z: -0.40 .. 0.42 m
```

### 6. Command、contact 和 gait 对齐

- high-level locomotion command 限制为 low-level 最终阶段范围：
  - linear x：`[-0.2, 0.2] m/s`
  - yaw：`[-0.3, 0.3] rad/s`
- linear/yaw dead zone 都设为 0.05。
- foot sensor contact threshold 统一为 1.5。
- high-level 删除未完整实现的 `gait_wait_timer` 状态机。
- 非 walking command 时，high-level 与 low-level 都立即将 gait phase 归零。

### 7. Action-delay FIFO 修复

Low-level action delay 现在按配置使用明确 FIFO：

```python
applied_action = action_history[:, -(action_delay + 1)]
```

last-action observation 使用实际 applied action，不再使用 FIFO 中最新但尚未施加的 command。

当前 deterministic parity 配置使用 `action_delay = 0`。后续应根据真机测得延迟，通过共享 contract 恢复 0–1 policy step 的随机延迟。

### 8. Checkpoint control contract schema v2

Low-level checkpoint metadata 新增完整 `control_contract` 和 canonical SHA-256，覆盖：

- effective action scale；
- leg kp/kd；
- arm position-drive kp/kd；
- sim dt、physics decimation 和 PhysX 参数；
- action delay；
- command ranges 和 dead zones；
- gait frequency；
- foot contact threshold；
- EE frame；
- IK gain、damping、orientation flag 和 target update period；
- domain-randomization ranges。

High-level 加载 checkpoint 时会：

1. 重新计算 checkpoint contract hash，检测 metadata 损坏；
2. 计算当前 high-level 期望 contract hash；
3. 两者不一致时拒绝加载。

## 兼容性说明

这是有意的 checkpoint 接口升级。

- 没有 `control_contract` 的旧 checkpoint 将被 Go2-X5 high-level 拒绝。
- 旧 checkpoint 即使 action/observation 维度匹配，也不能证明控制语义匹配。
- 需要使用修改后的 low-level runner 重新训练或重新保存 checkpoint，生成 schema v2 metadata。

不建议通过关闭 `requireLowPolicyMetadata` 绕过该检查，除非只进行受控的 legacy debugging。

## 修改文件

- `high-level/envs/b1z1_base.py`
- `high-level/data/cfg/go2x5_pickmulti.yaml`
- `low-level/legged_gym/envs/manip_loco/manip_loco.py`
- `low-level/legged_gym/envs/manip_loco/go2x5_config.py`
- `tests/test_go2x5_alignment.py`
- `docs/go2x5_runtime_parity_changes_2026-07-12.md`

## 已执行验证

```text
python3 -m py_compile high-level/envs/b1z1_base.py \
  low-level/legged_gym/envs/manip_loco/manip_loco.py \
  low-level/legged_gym/envs/manip_loco/go2x5_config.py \
  tests/test_go2x5_alignment.py

python3 tests/test_go2x5_alignment.py
# go2x5 alignment tests passed

git diff --check
```

新增静态检查覆盖：

- 不再出现 `props[1]`；
- Go2 base body name 配置；
- deterministic DR；
- PhysX contract；
- EE frame、IK 和 contact threshold；
- action-delay FIFO；
- applied-action observation；
- gait stop parity；
- control-contract hash 校验路径。

## 尚未完成的动态验收

当前修改没有宣称已经通过 Isaac Gym 闭环动态 parity。下一阶段需要使用 schema v2 checkpoint 执行：

1. runtime body-name → body-index dump；
2. total mass 和 system COM 对比；
3. DOF drive mode、kp/kd、limits 对比；
4. 同 state 下 proprio/history/gait observation 对比；
5. 同 observation 下 policy output 对比；
6. 同 action 下 URDF-order torque 对比；
7. 同 EE target 下 arm q-target 对比；
8. stop/start gait clock 逐帧对比；
9. 10 秒零命令 rollout trajectory 对比。

在这些动态验收通过前，不应恢复完整 high-level teacher 训练或扩大 domain randomization。

