# Go2-X5 low-level 平桌 6D 训练就绪审查

日期：2026-07-29

## 结论

本地结论为：**可以从随机初始化启动新的 low-level 长训**。

这里的“可以启动”表示：

- 机器人、地面、动作、观测、IK、PD、奖励和 checkpoint contract
  已通过 fail-closed 门禁；
- 300 iteration from-scratch smoke 能从早期 roll reset 恢复到接近完整
  10 秒 episode，优化量全为有限值；
- low/high 在 non-zero multi-state controller、真实 12D production loader
  和 stop/start command sequence 上一致。

它不表示 300 iteration smoke 已经学会最终技能，也不表示 high-level
抓取已经通过。正式 low-level checkpoint 仍需在长训中形成可靠的速度跟踪、
目标相关俯身和移动中 EE 6D 跟踪。

## 最终任务与 low-level 职责

最终任务是：Go2-X5 在平地上接近一张近边缘位于 root 前方 0.40 m、
并与最前碰撞体保留约 0.06 m 间隙、
表面高度在 0.10–0.20 m 之间的平桌，夹取桌面物体。

控制分工如下：

| 层级 | 本阶段必须学会 | 本阶段不负责 |
|---|---|---|
| low-level | `vx/yaw-rate` 跟踪、稳定行走/站立、目标相关俯身、EE 6D pose 跟踪 | 视觉感知、物体选择、夹爪闭合时机、抓取判定、物体抬升策略 |
| high-level | 产生 EE pose/底盘 command/夹爪指令并完成 reach-grasp-lift | 直接产生每个腿关节 torque |

因此，low-level 训练目标与最终任务是必要且直接吻合的执行底座，但不是完整
抓取任务本身。只有通过新的 low-level checkpoint 确定性评测后，才允许恢复
high-level teacher。

## 六项问题的处理

### 1. 机械臂末端姿态变化不足

原问题不是“Isaac Gym 调用了 Z1 的解析 IK”，而是任务仍带有 Z1 的末端姿态
约定，并且 Go2-X5 配置关闭了 orientation tracking。生产实现始终使用当前
URDF 的 Isaac Gym Jacobian。

当前实现：

- 用当前 `go2_x5.urdf` 做 FK，ready pose
  `[0, 2.4, 1.15, 0, 0, 0]` 的 EE local RPY 约为
  `[0, 1.25, 0]`；
- 删除 Z1 风格 `roll≈π/2` 的 nominal 语义；
- 目标包含 roll/pitch/yaw，范围为：
  - roll delta `[-0.35, 0.35]`
  - pitch delta `[-0.25, 0.25]`
  - yaw delta `[-0.35, 0.35]`，同时跟随目标方位角；
- observation 增加 local RPY；
- reward 使用 quaternion angular error；
- IK 使用 6D weighted damped least squares，orientation weight `0.35`；
- low/high 的目标 frame 均为 `TERRAIN_INVARIANT_YAW_LOCAL_RPY`。

公共实现依据：

- [ARX 官方 X5 Python SDK](https://github.com/ARXroboticsX/X5) 暴露
  `set_ee_pose_xyzrpy`/`get_ee_pose_xyzrpy`，说明控制接口本身是 6D；
- [ARX 官方 ROS X5 controller](https://github.com/ARXroboticsX/ARX_X5/blob/main/ROS/X5_ws/src/ARX_X5_ros_V7/arx_x5_controller/src/X5Controller.cpp)
  将 x/y/z/roll/pitch/yaw 转为 isometry 后交给 `setEndPose`；
- [Stanford 公开 X5 research SDK](https://github.com/real-stanford/arx5-sdk)
  使用 KDL 6D IK。

这些资料用于确认 X5 的 6D 控制语义。仿真中仍使用和当前 URDF 严格绑定的
Isaac Gym Jacobian，不引入另一个模型的几何参数。

### 2. 扩大机械臂训练范围

旧范围：

- root-forward x `0.30–0.55 m`
- y `-0.15–0.15 m`
- terrain z `0.08–0.45 m`

新范围：

- root-forward x `0.30–0.65 m`，最远端增加 0.10 m；
- y `-0.225–0.225 m`，全宽从 0.30 m 增至 0.45 m，扩大 50%；
- terrain z `0.05–0.45 m`，最低点离地 5 cm。

为了保留每个轴的边界，同时避免把不可达的联合角点送入 IK，增加了
`0.64 m` arm-base nominal reach envelope。轨迹的 10 个插值点同时执行：

- 近机身/头部碰撞盒检查；
- 地面下界检查；
- reach radius 检查。

10 次重采样仍不成功时保持上一个合法目标，禁止接受最后一个非法样本。

### 3. low-level 统一为平地

- `terrain.mesh_type = "plane"`；
- 直接创建原生 PhysX ground plane；
- 不创建 `Terrain`/heightfield/trimesh；
- 不使用 terrain curriculum；
- readiness 实测四个足端碰撞几何均在 plane 上方且存在支撑力。

这也修复了旧单面 trimesh 允许初始前脚位于地面下方的问题。

### 4. 简化课程和奖励

- auto curriculum 关闭，stage 列表为空；
- gait clock、四拍 walk、air-time、feet-height 和 contact phase 奖励关闭；
- 不规定 walk/trot，只要求 command tracking、稳定和低拖脚；
- 删除互相重叠的 dof acceleration、work、jerk、delta torque、hip pose
  等 active penalty；
- 保留核心项：
  - 速度：forward velocity、yaw rate；
  - 安全：alive、termination、vertical velocity、roll、collision、
    joint limit；
  - 平滑/执行：torque、action rate、action magnitude dead zone、feet drag；
  - 协同：目标相关 body height、目标相关 forward pitch；
  - 操作：EE world position、EE quaternion orientation。

12D actor 只有一组腿动作分布。EE reward 通过 value mixing 从第一个实际 PPO
更新开始加入腿策略 advantage，避免空 arm-action channel 将有效梯度减半。

### 5. 机械臂 PD

旧配置对六个 X5 关节统一使用 `kp=110, kd=7.5`，腕部过阻尼且响应拖沓。

当前 position-drive gains：

```text
kp = [120, 120, 100, 45, 35, 25]
kd = [4.0, 4.0, 3.5, 1.5, 1.2, 0.8]
```

gripper hold 单独保留 `kp=110, kd=7.5`。

公开 X5 research SDK 的 joint controller 采用近端高、腕部低的 6D gain
层级；当前值沿用该层级并针对 Isaac Gym position drive 保守调整。它们不是
ARX 对所有硬件场景发布的官方唯一参数，因此仍需在后续 sim-to-real 阶段用
实机辨识复核。

同时：

- IK gain 从 `0.10` 增至 `0.20`；
- 每 tick joint target 增量限制 `0.08 rad`；
- command 持久化并逐关节 clamp 到 URDF limit；
- 300 iteration 确定性回放的 arm target clamp fraction 为
  `3.65e-5`，无持续顶限。

### 6. 允许机身有界倾斜和俯身

固定直立高度目标被目标相关目标替代：

- body height：随 EE world z 从 `0.32 m` 降至最低 `0.22 m`；
- forward pitch：随低目标从 `0` 增至最多 `0.25 rad`；
- roll 仍受惩罚；
- hard termination：`|roll|/|pitch| > 0.8 rad` 或 root z `<0.18 m`。

这给低桌抓取留下约 4 cm 的高度安全余量，并允许腿、机身和机械臂协同，
而不是强制狗头始终抬高。

## 桌面几何覆盖

high-level 最终几何与 low-level workspace 使用同一 robot spec：

```text
robot root nominal x       = -0.45 m
robot front collision x    = -0.11 m
table center x             =  0.10 m
table length along x       =  0.30 m
table collision thickness  =  0.02 m
table near edge from root  =  0.40 m
table clearance from front =  0.06 m
table surface height       =  0.10–0.20 m
object table-local x       = -0.10–0.00 m
object table-local y       = -0.20–0.20 m
```

物体 x 映射到 root-forward `0.45–0.55 m`，y 映射到
`-0.20–0.20 m`，表面/物体/预抓取高度均包含在 low-level 的
`x=0.30–0.65, y=±0.225, z=0.05–0.45 m` 范围内。最坏桌面目标的
arm-base local radius约为 `0.596 m`，小于 `0.64 m` envelope。

## 本地验证证据

### Isaac Gym readiness

16 environments、128 policy ticks：

- native plane：通过；
- foot collision bottom above plane：通过；
- foot support force：通过；
- observation shape `744 = 66 proprio + 18 priv + 10×66 history`；
- 6D weighted DLS oracle：通过；
- orientation target 改变 arm joint target：通过；
- live orientation observation：通过；
- target interpolation：通过；
- workspace/table coverage：通过；
- early reset：0；
- nonfinite：0。

### 6D IK 边界网格

27 个 x/y/z 边界组合：

- raw IK success：24/27；
- 生产过滤器拒绝 6 个端点：
  - 3 个近机身中心线自碰撞点；
  - 3 个远、低联合超距点；
- 生产允许端点的严格 6D IK：21/21，成功率 100%；
- 无允许端点被静默当成不可达目标。

### Runtime/controller parity

| 测试 | mismatch | oracle failure | nonfinite | 结果 |
|---|---:|---:|---:|---|
| runtime properties | 0 | 0 | 0 | PASS |
| C0 zero action | 0 | 0 | 0 | PASS |
| C1 asymmetric action | 0 | 0 | 0 | PASS |
| C2 asymmetric q/qd/action | 0 | 0 | 0 | PASS |
| C3 rotated base/command/EE | 0 | 0 | 0 | PASS |
| C4 near-limit IK clamp | 0 | 0 | 0 | PASS |
| deterministic linear probe | 0 | 0 | 0 | PASS |
| stop/start command sequence | 0 | 0 | 0 | PASS |

natural reset 在 step `0/1/4/10` 两侧均：

- 可采集；
- nonfinite 0；
- immediate reset 0；
- policy observation shape 均为 726。

low 训练 buffer 额外包含 18D privileged block，因此其完整 buffer 是 744；
high production inference 只传 `66 + 10×66 = 726`。Actor 的 history encoder
固定读取末尾 `10×66`，这是训练/部署输入形式差异，不是 policy shape
不一致。

### 真实 12D production loader

固定 seed 随机初始化 schema-v2 smoke checkpoint：

- action dimension：12；
- `num_arm_actions=0`；
- purpose：`runtime_parity_smoke_only`；
- trained：false；
- metadata、control contract 和 SHA-256：真实通过；
- high-side 调用原始 `_load_low_level_model()`，无 monkey patch。

C3 共同 observation 结果：

| 字段 | max absolute error | 门槛 |
|---|---:|---:|
| observation/history | `2.235e-7` | `1e-6` |
| policy output | `7.451e-9` | `1e-7` |
| URDF-order action | `7.451e-9` | `1e-7` |
| scaled q target | `1.192e-7` | `1e-6` |
| leg torque | `4.768e-6` | `1e-5` |
| arm target | `2.384e-8` | `1e-5` |

policy output max abs 为 `0.11078`，不是零动作替代品；nonfinite 为 0。

### 300 iteration from-scratch smoke

配置：128 environments、随机初始化、无 DR/noise/push、同一生产任务合同。

最后 20 iterations：

| 指标 | 平均值 | 范围 |
|---|---:|---:|
| episode length | `490.02 / 500` | `487.14–495.71` |
| timeout fraction | `0.8556` | `0.6111–1.0` |
| value loss | `0.7358` | `0.1304–1.7923` |
| surrogate loss | `-0.01093` | `-0.01790` 至 `-0.00555` |
| EE position L1 error | `0.0372 m` | `0.0330–0.0413` |
| EE quaternion angle error | `0.0705 rad` | `0.0622–0.0760` |
| nonfinite | `0` | `0` |

同一 model 的 128 environments × 500 tick deterministic rollout：

- mean EE Euclidean error：`0.04646 m`；
- mean EE quaternion angle error：`0.12030 rad`；
- action saturation：0；
- arm target clamp fraction：`3.65e-5`；
- mean normalized non-foot collision：`0.04386/tick`；
- collision 主要来自 head/arm self-contact，不是腿穿地；
- early roll resets：13/64,000 environment-ticks；
- nonfinite：0。

300 iterations 只证明数值和学习路径健康，不作为 locomotion/coordination
性能合格 checkpoint。速度跟踪和 goal-z/body-posture correlation 尚未达到
正式 checkpoint 门槛，必须由长训继续学习。

## 长训监控标准

启动后前 10–20 个 meaningful iterations 必须满足：

- 进程未退出；
- loss、reward、action std 全部 finite；
- 没有 loader/metadata/IK/shape 错误；
- throughput 非零且 checkpoint 目录可写；
- reset 原因可解释，不出现全部环境同步立即 reset；
- arm target clamp、action saturation 和 collision 不发生单调失控。

后续 checkpoint 只有同时通过下列 deterministic gate 才能交给 high-level：

- 0 early non-timeout reset；
- mean EE position error `≤0.06 m`；
- mean EE orientation error `≤0.15 rad`；
- vx/yaw mean absolute error 各 `≤0.05`；
- height adaptation error `≤0.03 m`；
- pitch adaptation error `≤0.06 rad`；
- low-goal/body-height correlation `≥0.30`；
- low-goal/body-pitch correlation `≤-0.20`；
- mean collision `≤0.10/tick`；
- arm clamp fraction `≤0.05`；
- action saturation fraction `≤0.05`；
- nonfinite 0。

## 可复制命令

```bash
python3 -m py_compile \
  low-level/legged_gym/envs/manip_loco/manip_loco.py \
  low-level/legged_gym/envs/manip_loco/go2x5_config.py \
  low-level/legged_gym/envs/manip_loco/go2x5_robot_spec.py \
  low-level/legged_gym/scripts/check_go2x5_training_readiness.py \
  low-level/legged_gym/scripts/check_go2x5_checkpoint_rollout.py \
  low-level/legged_gym/scripts/scan_go2x5_ik_reachability.py \
  tools/go2x5_runtime_parity.py \
  tools/go2x5_parity_factories.py

python3 tests/test_low_high_runtime_parity.py
python3 tests/test_go2x5_alignment.py

conda run --no-capture-output -n vwc_go2x5 \
  python tests/test_go2x5_training_readiness.py

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n vwc_go2x5 \
  python low-level/legged_gym/scripts/check_go2x5_training_readiness.py \
  --num-envs 16 \
  --steps 128 \
  --output /tmp/go2x5_training_readiness_flat_tabletop_6d.json

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n vwc_go2x5 \
  python low-level/legged_gym/scripts/scan_go2x5_ik_reachability.py \
  --quick \
  --orientation_mode task \
  --summary_json /tmp/go2x5_ik_quick.json \
  --csv /tmp/go2x5_ik_quick.csv
```

正式长训必须从随机初始化开始，禁止传 `--resume` 或 warm-start 参数：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=<physical_gpu_index> \
WANDB_MODE=offline \
conda run --no-capture-output -n <existing_remote_env> \
  python low-level/legged_gym/scripts/train.py \
  --headless \
  --task go2x5 \
  --proj_name go2x5-low \
  --exptid go2x5_flat_tabletop_6d_seed1_20260729 \
  --num_envs 4096 \
  --max_iterations 45000 \
  --seed 1 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --graphics_device_id 0
```

## 尚未证明

- 尚无正式训练完成的 12D checkpoint；
- 尚未证明长训 checkpoint 的 1 秒或 10 秒 closed-loop task parity；
- 尚未证明物体接触、夹爪闭合和抬升成功率；
- 尚未允许恢复 domain randomization、相机或复杂地形；
- 尚未允许恢复 high-level teacher/student。

这些不是启动 low-level 长训的阻塞项，而是新 checkpoint 产生后的下一组
门禁。
