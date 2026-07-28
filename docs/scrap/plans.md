# Go2X5 VBC 修正计划

本文档基于当前 Go2 + ARX-X5 的 VBC 复现状态，重新整理后续修改方案。方案遵守三个约束：

1. high-level 和 low-level 必须严格对齐。
2. low-level 的多 stage 训练必须自动切换，不能依赖每个阶段手动改配置。
3. 暂不考虑 FtLift 阶段，优先得到稳定、可信、可复用的 low-level 模型。

目标不是短期把 reward 推高，而是训练出一个可以稳定回放、可靠跟踪 EE 目标、能作为 high-level 抓取底座的 Go2X5 low-level policy。

## 1. 总体路线

当前项目应先收敛到一个稳定的 leg-only low-level baseline：

- policy action 固定为 12 维腿部动作。
- 手臂继续由 IK / PD 位置目标控制。
- 不引入 FtLift。
- 不急于恢复 18 维全身 action。
- 不在 low-level 稳定前投入正式 high-level teacher 训练。

训练路线分为三层：

1. Alignment Gate：先确保 high-level 和 low-level 使用同一份机器人定义、同一个 low-level 接口和可验证 checkpoint。
2. Auto Stage Low-Level：用自动 curriculum 训练稳定 low-level。
3. High-Level Readiness Test：low-level 通过固定指标后，再进入 high-level smoke test 和 teacher 训练。

## 2. High-Level / Low-Level 严格对齐

### 2.1 当前不一致点

当前 high-level 和 low-level 还没有完全一致：

- low-level 使用 `low-level/resources/robots/go2x5/go2_x5.urdf`。
- high-level 默认配置仍使用 `go2x5/urdf/go2_arx_x5.urdf`。
- low-level 最新 EE body 是 `arm_eef_link`。
- high-level 代码仍硬编码 `ee_gripper_link`。
- low-level 使用 `arm_joint*`、`arm_link*` 命名。
- high-level 旧配置仍包含 `x5_joint*`、`link6/link7/link8` 等命名。
- high-level 默认 `low_policy_path` 仍指向旧 checkpoint。

这些不一致必须在任何正式 high-level 训练前解决。否则 high-level success rate 没有解释价值。

### 2.2 建立单一机器人规格文件

新增一个单一事实来源，例如：

```text
source/go2x5_robot_spec.yaml
```

或放在低层包中：

```text
low-level/legged_gym/envs/manip_loco/go2x5_robot_spec.py
```

建议内容：

```yaml
robot:
  asset_file: "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2x5/go2_x5.urdf"
  high_level_asset_root: "../low-level/resources/robots"
  high_level_asset_file: "go2x5/go2_x5.urdf"

  num_dofs: 20
  num_leg_dofs: 12
  num_arm_dofs: 6
  num_gripper_dofs: 2

  action_dim: 12
  proprio_dim_without_gait: 66
  priv_dim: 18
  history_len: 10

  ee_body_name: "arm_eef_link"
  wrist_body_name: "arm_link6"
  finger_body_names: ["arm_link7", "arm_link8"]

  arm_base_offset: [0.085, 0.0, 0.094]
  low_action_scale: [0.4, 0.45, 0.45, 0.4, 0.45, 0.45, 0.4, 0.45, 0.45, 0.4, 0.45, 0.45]

  leg_joint_names:
    - FL_hip_joint
    - FL_thigh_joint
    - FL_calf_joint
    - FR_hip_joint
    - FR_thigh_joint
    - FR_calf_joint
    - RL_hip_joint
    - RL_thigh_joint
    - RL_calf_joint
    - RR_hip_joint
    - RR_thigh_joint
    - RR_calf_joint

  arm_joint_names:
    - arm_joint1
    - arm_joint2
    - arm_joint3
    - arm_joint4
    - arm_joint5
    - arm_joint6

  gripper_joint_names:
    - arm_joint7
    - arm_joint8
```

low-level config 和 high-level yaml 都应从这份规格同步。短期可以手动读取同一个 yaml，长期可以生成配置，避免两边再次漂移。

### 2.3 high-level 改为配置化 body names

修改 `high-level/envs/b1z1_base.py`：

- 不再硬编码 `ee_gripper_link`。
- 从 yaml 读取 `eeBodyName`。
- `wristBodyName`、`flangeBodyName`、`fingerBodyNames` 都从 yaml 读取。
- 找不到 body 时直接 raise，不能静默用错误 index。

示例 high-level yaml 字段：

```yaml
env:
  lowPolicyNumActions: 12
  lowPolicyObserveGaitCommands: true
  armBaseOffset: [0.085, 0.0, 0.094]
  eeBodyName: "arm_eef_link"
  wristBodyName: "arm_link6"
  fingerBodyNames: ["arm_link7", "arm_link8"]

  asset:
    robotAssetRoot: "../low-level/resources/robots"
    assetFileRobot: "go2x5/go2_x5.urdf"
```

### 2.4 checkpoint 必须带 metadata

low-level checkpoint 需要保存并在 high-level 加载时校验：

- git commit
- task name
- asset file
- asset hash
- action dim
- proprio dim
- privileged dim
- history len
- `observe_gait_commands`
- `num_gripper_joints`
- `ee_body_name`
- `arm_base_offset`
- low-level config hash
- curriculum stage profile name

high-level 加载 low-level checkpoint 时必须检查：

- `action_dim == lowPolicyNumActions`
- `observe_gait_commands` 和 CLI 参数一致
- `asset_hash` 或 `asset_file` 和 high-level robot asset 一致
- `ee_body_name` 和 high-level yaml 一致
- `arm_base_offset` 一致

不一致时直接报错，不允许继续训练。

### 2.5 对齐测试

新增测试：

```text
tests/test_go2x5_alignment.py
```

至少覆盖：

- low-level URDF 可加载。
- high-level URDF 路径和 low-level URDF 指向同一文件。
- DOF 数等于 20。
- 可动关节顺序和规格文件一致。
- EE body 存在且为 `arm_eef_link`。
- low-level action dim 为 12。
- low-level observation dim 在 `--observe_gait_commands` 下为 799。
- high-level low policy loader 构造的 ActorCritic 输入输出维度和 checkpoint metadata 一致。

这一步是硬门槛：测试不过，不进入 high-level 训练。

## 3. Low-Level 自动 Stage Training

### 3.1 原则

不再通过手动改 `go2x5_config.py` 来切阶段。所有 stage 写在一个配置里，由训练过程自动切换。

自动切换遵守两个条件：

- 最小训练步数或 iteration 达到阶段要求。
- 最近一段窗口的训练指标达到阈值。

如果指标没有达标，则保持当前 stage，不进入下一阶段。

### 3.2 新增 curriculum 配置

建议在 `go2x5_config.py` 增加：

```python
class auto_curriculum:
    enabled = True
    profile_name = "go2x5_stable_auto_v1"
    metric_window = 200
    log_stage = True
    save_stage_metadata = True
```

并增加 stage 表：

```python
stages = [
    {
        "name": "S0_sanity_flat",
        "start_iteration": 0,
        "min_iterations": 1000,
        "terrain": "flat",
        "push_robots": False,
        "friction_range": [0.8, 1.2],
        "added_mass_range": [0.0, 2.0],
        "added_com_range": [[-0.02, 0.02], [-0.02, 0.02], [-0.01, 0.01]],
        "leg_motor_strength_range": [0.9, 1.1],
        "ee_tracking_weight": 0.25,
        "goal_pos_l": [0.20, 0.35],
        "goal_pos_p": [-0.25, 0.50],
        "collision_scale": -2.0,
        "hip_pos_scale": -0.05,
        "roll_scale": -0.5,
        "torques_scale": -5e-6,
        "work_scale": -0.0005,
    },
    {
        "name": "S1_stable_gait_flat",
        "min_iterations": 5000,
        "terrain": "flat",
        "push_robots": False,
        "friction_range": [0.7, 1.4],
        "added_mass_range": [0.0, 4.0],
        "added_com_range": [[-0.04, 0.04], [-0.04, 0.04], [-0.02, 0.02]],
        "leg_motor_strength_range": [0.85, 1.15],
        "ee_tracking_weight": 0.4,
        "goal_pos_l": [0.20, 0.42],
        "goal_pos_p": [-0.40, 0.75],
        "collision_scale": -3.0,
        "hip_pos_scale": -0.1,
        "roll_scale": -1.0,
        "torques_scale": -1e-5,
        "work_scale": -0.001,
    },
    {
        "name": "S2_ee_tracking_flat",
        "min_iterations": 7000,
        "terrain": "flat",
        "push_robots": False,
        "friction_range": [0.6, 1.5],
        "added_mass_range": [0.0, 5.0],
        "added_com_range": [[-0.05, 0.05], [-0.05, 0.05], [-0.03, 0.03]],
        "leg_motor_strength_range": [0.85, 1.15],
        "ee_tracking_weight": 0.6,
        "goal_pos_l": [0.20, 0.50],
        "goal_pos_p": [-0.60, 1.047],
        "collision_scale": -4.0,
        "hip_pos_scale": -0.1,
        "roll_scale": -1.0,
        "torques_scale": -1e-5,
        "work_scale": -0.001,
    },
    {
        "name": "S3_rough_terrain",
        "min_iterations": 10000,
        "terrain": "rough",
        "push_robots": False,
        "friction_range": [0.5, 2.0],
        "added_mass_range": [0.0, 8.0],
        "added_com_range": [[-0.08, 0.08], [-0.08, 0.08], [-0.05, 0.05]],
        "leg_motor_strength_range": [0.8, 1.2],
        "ee_tracking_weight": 0.7,
        "collision_scale": -5.0,
        "hip_pos_scale": -0.15,
        "roll_scale": -1.5,
        "torques_scale": -1.5e-5,
        "work_scale": -0.0015,
    },
    {
        "name": "S4_robustness",
        "min_iterations": 12000,
        "terrain": "rough",
        "push_robots": True,
        "max_push_vel_xy": 0.3,
        "friction_range": [0.4, 2.5],
        "added_mass_range": [0.0, 10.0],
        "added_com_range": [[-0.10, 0.10], [-0.10, 0.10], [-0.08, 0.08]],
        "leg_motor_strength_range": [0.75, 1.25],
        "ee_tracking_weight": 0.8,
        "collision_scale": -6.0,
        "hip_pos_scale": -0.2,
        "roll_scale": -1.8,
        "torques_scale": -2e-5,
        "work_scale": -0.002,
    },
]
```

注意：这些数值是初始建议，不是最终结论。关键是自动切换机制，而不是一次性定死所有参数。

### 3.3 自动切换指标

每个 stage 都应定义进入下一阶段的最低指标。建议使用 rolling window，例如最近 200 个 PPO iteration：

S0 到 S1：

- `Train/mean_episode_length > 250`
- `Train/dones < 0.02`
- `Episode_metric/metric_roll < 1.0`
- `Episode_metric/metric_collision < 2.5`

S1 到 S2：

- `Train/mean_episode_length > 350`
- `Train/dones < 0.015`
- `Episode_metric/metric_collision < 2.0`
- `Episode_metric/metric_torques` 不继续上升
- 平地回放不持续拖腿

S2 到 S3：

- `Train/mean_episode_length > 400`
- `Train/dones < 0.01`
- `Episode_metric/metric_tracking_ee_world < 0.25` 或 EE error 明显收敛
- `Episode_metric/metric_collision < 2.0`
- arm target 切换不明显带倒底盘

S3 到 S4：

- rough terrain 上 episode length 稳定
- `Train/dones < 0.015`
- `Episode_metric/metric_collision < 2.5`
- `Episode_metric/metric_roll` 和 `metric_base_height` 无明显恶化

S4 结束：

- rough terrain + push 下稳定。
- 回放时不出现持续趴地、拖腿、高频抖动。
- EE target 跟踪能收敛。

### 3.4 实现方式

建议修改低层 runner / env 的接口：

1. 在 `ManipLoco` 中增加：

```python
def set_training_stage(self, stage_name: str, stage_cfg: dict):
    ...
```

2. 在 `OnPolicyRunner.learn()` 每个 iteration 后调用：

```python
if hasattr(env, "update_auto_curriculum"):
    env.update_auto_curriculum(iteration, episode_metrics)
```

3. `update_auto_curriculum()` 负责：

- 更新当前 stage。
- 改 reward scales。
- 改 EE goal range。
- 改 push 参数。
- 改 terrain sampling。
- 记录 stage 到 W&B / console。
- 把当前 stage 写入 checkpoint metadata。

4. 恢复训练时：

- 从 checkpoint metadata 恢复 stage。
- 重新应用 stage config。
- 不允许 resume 后回到默认强配置。

### 3.5 动态随机化的注意点

friction、base mass、COM、motor strength 当前多数是在 env 创建或 reset 时采样。自动 stage 切换时要注意：

- reward scale 和 EE target range 可以立即生效。
- push 开关可以立即生效。
- terrain sampling 可以对新 reset env 生效。
- friction、mass、COM、motor strength 如果只在创建时设置，需要在 reset 时重新采样，或增加 stage-aware resample。

因此建议把 domain randomization 的采样逻辑统一改成：

```python
current_stage_rand = self.curriculum_stage.domain_rand
```

并在 `reset_idx()` 中按当前 stage 对 env ids 重新采样。

## 4. 不使用 FtLift 的替代路线

用户明确要求不考虑 FtLift 阶段。因此：

- 不再把 `go2x5_ftlift` 作为训练主线。
- 不从 `go2x5_stable_base_v1/model_7600.pt` 单独开 FtLift fine-tune。
- 不用 FtLift 作为 high-level 前置条件。

替代方式：

- 在同一个 low-level 自动 curriculum 中逐步提高 EE tracking 和目标范围。
- 在 S2 / S3 中覆盖 high-level 需要的常见伸臂目标。
- 但不引入抓取、桌面、物体、抬物 reward。
- low-level 仍只学习“稳定 + 行走 + EE target tracking”。

这样可以保持 low-level 的任务边界清晰，避免 low-level 被高层抓取任务污染。

## 5. Low-Level 稳定能力优先级

当前最优先优化顺序：

1. 不倒。
2. 不持续趴地或过度下蹲。
3. 不拖腿，不出现 thigh/calf 大量接触。
4. torque / work 不爆。
5. 平地速度命令稳定跟踪。
6. rough terrain 下仍能保持姿态。
7. EE target tracking 在目标变化后能收敛。
8. 手臂目标变化不会明显带倒整机。

短期不要为了 EE tracking 牺牲底盘稳定。Go2 机身更低，X5 伸臂扰动更明显，先保证稳定底座比先追求末端误差更重要。

## 6. Collision Reward 修正

当前 collision reward 对 thigh/calf 接触过于敏感，建议修改为 stage-aware soft penalty。

现有逻辑类似：

```python
contact > 0.1
```

建议改为：

```python
threshold = cfg.rewards.collision_force_threshold
excess = torch.clamp(contact_force - threshold, min=0.0)
rew = torch.sum(excess / threshold, dim=1)
```

推荐初始值：

```python
collision_force_threshold = 5.0
collision_soft_clip = 50.0
```

并按 stage 调整：

- S0/S1：允许轻微擦碰，避免训练初期被打死。
- S2/S3：逐步提高碰撞约束。
- S4：接近部署/高层使用强度。

目标是减少真实拖腿和重碰撞，而不是把轻微接触全都当作失败。

## 7. High-Level 进入条件

low-level 没有通过以下检查，不进入正式 high-level teacher 训练。

### 7.1 自动评估脚本

新增：

```text
low-level/legged_gym/scripts/evaluate_go2x5_lowlevel.py
```

自动跑：

- flat terrain
- rough terrain
- zero command
- forward command
- yaw command
- EE target sweep
- arm target hold

输出 JSON：

```text
low-level/eval/go2x5/<run>/<checkpoint>/report.json
```

### 7.2 通过标准

建议 low-level checkpoint 进入 high-level 前必须满足：

- flat 回放 2 分钟不倒。
- rough 回放 2 分钟不倒。
- reset rate 低于阈值。
- 平均 base height 在合理范围内。
- roll / pitch 不持续偏大。
- thigh/calf collision 明显低于当前满强度训练结果。
- torque / work 无明显爆炸。
- EE target sweep 中误差能收敛。
- 手臂移动不会导致底盘连续失稳。

### 7.3 high-level smoke test

通过 low-level eval 后，再运行 high-level smoke test：

- 固定桌高 `0.25`。
- `num_envs = 34`。
- `timesteps = 500`。
- 禁用 camera。
- 使用 copied yaml。
- yaml 内 checkpoint 路径显式写死。

smoke test 只检查接口，不用于判断策略成功率。

## 8. High-Level 对齐后的训练原则

当 low-level 通过评估且 high/low 对齐测试通过后，high-level 才开始训练。

high-level 第一阶段：

- 固定桌高。
- 固定 low-level checkpoint。
- 不随机 low-level checkpoint。
- 不随机 URDF。
- 不启用 vision student。
- 先训练 state-based teacher。

如果 high-level 失败，要先归因：

- 如果底盘先倒或姿态崩，回到 low-level。
- 如果 EE 到不了目标，检查 high/low EE 坐标、arm offset、IK。
- 如果 gripper 接触不对，检查 finger body names 和 gripper DOF limits。
- 如果接近但抓不住，再调 high-level reward。

## 9. 实施里程碑

### Milestone A：对齐基础设施

产出：

- `go2x5_robot_spec` 单一规格文件。
- low-level 读取规格。
- high-level 读取规格。
- body names 全部配置化。
- high-level 默认 Go2X5 asset 切到 `go2_x5.urdf`。
- checkpoint metadata 保存和加载校验。
- alignment tests 通过。

完成标准：

- `python -m pytest tests/test_go2x5_alignment.py` 通过。
- high-level 和 low-level 打印出的 DOF names、EE body、action dim 完全一致。

### Milestone B：自动 Stage Curriculum

产出：

- `auto_curriculum` 配置。
- stage-aware reward / domain randomization / EE goal range。
- runner 每 iteration 自动更新 stage。
- checkpoint 保存当前 stage。
- resume 后恢复当前 stage。
- W&B / console 记录当前 stage。

完成标准：

- 从头训练不需要手动改配置。
- 训练日志中能看到 stage 自动推进或保持。
- resume 后 stage 不丢失。

### Milestone C：稳定 Low-Level 模型

产出：

- 一个主训练 run，例如 `go2x5_low_auto_stable_v1`。
- 多个 checkpoint 自动评估报告。
- 选出 stable checkpoint。

完成标准：

- flat 和 rough 单环境回放稳定。
- torque/work/collision/roll/base height 不再长期异常。
- EE target tracking 可收敛。

### Milestone D：High-Level Smoke Test

产出：

- copied high-level yaml。
- 显式 low-level checkpoint path。
- fixed table height smoke test log。

完成标准：

- 无 observation mismatch。
- low-level policy 正常加载。
- robot 不在抓取前立即失稳。
- EE body / gripper contact 指标合理。

### Milestone E：High-Level Teacher

产出：

- 固定桌高 teacher run。
- success rate 和失败原因分析。

完成标准：

- 如果失败，能明确区分是 high-level 策略问题还是 low-level 稳定性问题。
- 不再出现“路径旧、URDF 旧、body name 错”导致的无效实验。

## 10. 明确不做

当前阶段不做：

- 不使用 FtLift 作为训练阶段。
- 不手动反复改配置切 low-level stage。
- 不在 high/low 未对齐前正式训练 high-level。
- 不把默认 `go2x5_pickmulti.yaml` 直接用于正式实验。
- 不恢复 18 维 low-level action。
- 不把 high-level success rate 当作 low-level 稳定前的主要指标。

## 11. 推荐下一步

最直接的下一步不是继续训练，而是先实现 Milestone A 和 B：

1. 建立 Go2X5 单一规格文件。
2. 修 high-level body name 和 URDF 路径配置。
3. 给 low-level checkpoint 加 metadata。
4. 写 alignment test。
5. 给 low-level 加 auto curriculum stage manager。
6. 从头启动 `go2x5_low_auto_stable_v1`。

完成这些后，Go2X5 low-level 训练才会变成一个可复现、可恢复、可比较的稳定模型训练流程。
