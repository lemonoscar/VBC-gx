# Go2-X5 前向 EE 工作区与低桌面任务重定向审查（2026-07-21）

## 结论

原配置把机械臂默认姿态和 EE 目标集中在机身上方，high-level 又沿用 B1-Z1 的大桌面、高桌面和大范围机器人初始位姿随机化。这些设置虽然在数值上可达，却与 Go2-X5 的实际抓取任务不一致：主要工作区应位于机身前方，目标物体应在离机器人约 30 cm 的低桌面上。

本次修改把 low-level、high-level 和测试统一到同一套前向任务几何。网络结构、12D 动作接口、PD 参数、PPO 损失和奖励权重均未修改，因此现有 schema-v2 checkpoint 可以作为 warm start；它不能被视为已经学会新工作区的最终模型。

## 统一任务几何

坐标约定继续使用 `TERRAIN_INVARIANT_YAW`：XY 随机器人根节点位置和 yaw 变化，Z 相对地形保持不变。

| 项目 | 新值 |
| --- | --- |
| Go2 名义根节点高度 | 0.32 m |
| EE 中心偏移 | `[0.085, 0.0, 0.414]` |
| EE local 范围 | `x=[0.215,0.465]`, `y=[-0.15,0.15]`, `z=[-0.334,0.036]` |
| EE root/terrain 范围 | 前向 `x=[0.30,0.55]` m，侧向 `±0.15` m，高度 `z=[0.08,0.45]` m |
| X5 reset 待机姿态 | `[0.0,2.4,1.15,0.0,0.0,0.0]` |
| 待机姿态名义 EE | root 前方约 0.487 m，高度约 0.306 m |
| high-level 机器人名义起点 | `[-0.45,0.0,0.32]` |
| 桌面尺寸 | `[0.30,0.60,0.10]` m |
| 桌面表面高度 | 0.10–0.15 m |
| 桌面近边名义距离 | 机器人根节点前方 0.30 m |
| 物体桌面采样 | X `±0.05` m，Y `±0.10` m |
| Go2 high-level reset 扰动 | XY 各 `±0.03` m，yaw `±0.08` rad |

low-level curriculum 的 S0 只在桌面上方的紧凑前向范围内采样，S1 使用完整前向范围。没有引入规定四拍或小跑的步态约束；策略仍可自行学习任何满足速度、稳定性、足端拖曳和 EE 跟踪目标的走法。

## 文件级变更

- `go2x5_robot_spec.py`：集中定义 EE、桌面、起点、reset 扰动和 X5 前伸待机姿态。
- `go2x5_config.py`：low-level EE 初始轨迹及 S0/S1 采样改为前向工作区。
- `go2x5_pickmulti.yaml`：high-level 低桌面、物体、起点、目标和成功阈值与统一规范对齐。
- `b1z1_base.py`：修复 terrain-invariant EE center 在不同代码路径使用不同常量的问题；机器人 reset 扰动改为可配置，未配置时保留 B1-Z1 行为。
- `b1z1_pickmulti.py`：桌面尺寸、位置、高度和物体采样范围改为配置项；B1-Z1 默认值保留。
- `go2x5_pickmulti.py`：high-level X5 reset 姿态与 low-level 一致。
- `go2x5_runtime_parity.py`：natural-reset 报告新增桌面、物体和桌面表面高度。
- `go2x5_parity_factories.py`：canonical EE case 也改用前向目标；C4 使用工作区内最远最低角点和接近上限的 arm joint2，避免用头顶/不可达目标触发 clamp。
- `audit_go2x5_low_level_rewards.py`：静态解析器支持受限的规范常量下标，并 fail-closed 校验新 S0/S1 前向范围。
- readiness、checkpoint rollout 和可视化脚本：新增前向范围、目标采样边界和 EE 误差检查。
- `test_go2x5_alignment.py`：验证 low/high 常量、前向范围、30 cm 桌边距离、低桌面和 reset 扰动。

## 验证结果

### CPU/static

```bash
python3 -m py_compile \
  high-level/envs/b1z1_base.py \
  high-level/envs/b1z1_pickmulti.py \
  high-level/envs/go2x5_pickmulti.py \
  low-level/legged_gym/envs/manip_loco/go2x5_config.py \
  low-level/legged_gym/envs/manip_loco/go2x5_robot_spec.py \
  low-level/legged_gym/scripts/check_go2x5_checkpoint_rollout.py \
  low-level/legged_gym/scripts/check_go2x5_training_readiness.py \
  tools/go2x5_runtime_parity.py \
  tests/test_go2x5_alignment.py

python3 tests/test_go2x5_alignment.py
python3 tests/test_low_high_runtime_parity.py
git diff --check
```

结果：全部通过。

canonical controller 复验：

- C0：mismatch 0，oracle failure 0，nonfinite 0
- C3（roll/pitch/yaw + 非零命令 + 前向低位目标）：mismatch 0，oracle failure 0，nonfinite 0
- C4（arm joint2 接近上限 + 工作区内 `x=0.55,y=0.15,z=0.08 m` world target）：low/high 均将 joint2 从 `3.80038` clamp 到上限 `3.66519`，mismatch 0，oracle failure 0，nonfinite 0

### Isaac Gym readiness

8 环境、200 步、curriculum stage 1：

- observation shape：`[8, 744]`
- early reset：0
- 所有 nonfinite：0
- EE 前向范围、X5 默认关节、IK、关节限位、奖励注册和 checkpoint metadata：通过

报告：`/tmp/go2x5_front_workspace_readiness.json`（不提交）。

### high-level natural reset

真实创建 Go2-X5、桌面和物体 actor，采集 step 0/1/4/10：

- passed：true
- immediate reset：0
- nonfinite：0
- 一次代表性 reset：root `(-0.462, 0.010)` m，桌面中心 `(0,0)` m，桌面表面 `0.134` m
- 沿机器人前向到桌面近边约 0.31 m
- 物体 XY 和桌面高度均在配置范围内

报告：`/tmp/go2x5_front_table_high_natural_v2.json`（完整数组不提交）。

### checkpoint warm-start

`model_35000.pt` 能通过正式 runner 加载，action/observation shape、schema-v2 metadata 和 control contract 均未发生结构不兼容。

| 检查 | model_35000 | 20 iteration smoke 后 model_35020 |
| --- | ---: | ---: |
| 环境数 × 步数 | 128 × 500 | 128 × 500 |
| pitch/roll/z early reset | 6 / 0 / 0 | 0 / 0 / 0 |
| nonfinite | 0 | 0 |
| mean EE error | 0.1577 m | 0.1608 m |
| mean collision raw/tick | 0.4346 | 0.00275 |
| 目标采样越界 | 0 | 0 |

20 iteration smoke 使用真实 PPO optimizer/history-encoder 状态续训，所有 loss 有限。初期少量 pitch reset 随后恢复，最终 checkpoint 在同一 deterministic rollout 中无 early reset。该结果支持从既有 locomotion checkpoint 做任务几何 warm start；EE 误差仍需在正式续训中下降。

## 尚未证明的内容

- 尚未得到完成前向低桌面任务训练的正式 low-level checkpoint。
- 20 iteration smoke 只证明 warm-start 稳定性，不证明抓取成功率或 sim-to-real 表现。
- high-level teacher/student 尚未恢复训练。
- domain randomization、相机随机化和真实物体接触鲁棒性仍保持关闭或未作为当前 gate。

## 训练切换原则

在服务器上比较当前旧任务的候选 checkpoint 后，选择在新配置 deterministic rollout 中 reset 最少、EE/碰撞指标更好的 checkpoint。先运行短 smoke，确认加载路径、迭代增长、loss 有限和无 reset burst，再停止旧任务并启动独立 experiment id 的正式续训。不得覆盖旧 checkpoint 目录。
