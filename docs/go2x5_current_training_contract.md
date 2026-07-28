# Go2-X5 当前训练合同

更新日期：2026-07-29

## 当前结论

- 旧 Go2-X5 low-level 和 high-level checkpoint 均已判定为不可复用并从本地清除。
- 下一次 low-level 训练必须从随机初始化开始，任务名仅为 `go2x5`。
- `go2x5_ftlift` 空别名任务已经删除，不再作为训练阶段或恢复入口。
- 当前源码已经通过静态门禁、Isaac Gym readiness、6D IK 网格、low/high
  controller parity、真实 12D production-loader parity 和 300 iteration
  from-scratch smoke，可以启动新的 low-level 长训。
- high-level 暂不可训练；必须等新的 low-level 12D schema-v2 checkpoint 通过确定性评测后，再通过 `--low_policy_path` 显式传入。

## 唯一有效的机器人与地面配置

- URDF：`low-level/resources/robots/go2x5/go2_x5.urdf`
- policy action dimension：12，仅控制四腿
- base initial height：0.32 m
- base height target：0.32 m
- leg PD：`kp=40`、`kd=1`
- policy-order action scale：每条腿 `[0.125, 0.25, 0.25]`
- X5 arm position-drive PD：
  - `kp=[120, 120, 100, 45, 35, 25]`
  - `kd=[4, 4, 3.5, 1.5, 1.2, 0.8]`
- gripper hold PD：`kp=110`、`kd=7.5`
- 地面：原生 PhysX plane
- plane 模式不创建 `Terrain` 对象，不读取 terrain tile origins
- 环境原点：3.0 m 间距的规则网格
- domain randomization、noise 和 push：当前确定性训练合同中关闭
- 自动 curriculum：关闭；使用单一静态任务分布

平面回归探针在 8 个 GPU 环境中验证：

- `env.terrain` 不存在
- 四个足端碰撞球在 5 个 policy tick 后均位于 plane 上方
- 最小碰撞球底部高度：约 1.02 mm
- 最小足端接触力：约 20.81 N
- 64 tick rollout：0 early reset、0 nonfinite
- 300 tick 零动作、移动 EE 目标：0 non-timeout reset

## EE 任务区域

EE 目标使用 `TERRAIN_INVARIANT_YAW`：

- root-forward x：0.30–0.65 m
- y：-0.225–0.225 m
- terrain z：0.05–0.45 m
- nominal arm-base reach radius：不超过 0.64 m

这相对上一合同将 y 全宽扩大 50%，x 最远端增加 0.10 m，并将最低
目标降到离地 0.05 m。近机身自碰撞区和“远、低、侧向”联合超距角点由
轨迹级 fail-closed 过滤器拒绝，不会作为训练命令进入 IK。

EE 姿态是完整 6D 目标：

- X5 nominal local RPY：`[0.0, 1.25, 0.0]`
- roll delta：`[-0.35, 0.35]`
- pitch delta：`[-0.25, 0.25]`
- yaw delta：`[-0.35, 0.35]`，并叠加目标方位角
- IK：加权 6D damped least squares，orientation weight `0.35`
- joint command：persistent、每 tick 最大更新 `0.08 rad`、URDF limit clamp

X5 机械臂不再沿用 Z1 的 `roll≈π/2` nominal 姿态。当前 ready pose 的
URDF FK 给出 local RPY 约 `[0, 1.25, 0]`；训练 observation、reward、IK
与 high-level contract 均使用同一语义。

该区域位于机器狗前方，并允许通过身体俯仰/高度与机械臂协同完成低位目标。

## 最终桌面任务覆盖

- 桌面近边缘：nominal root 前方 0.30 m
- 桌面尺寸：`0.30 × 0.60 × 0.10 m`
- 桌面表面高度：`0.10–0.20 m`
- 物体中心采样：table x `[-0.10, 0.10]`、y `[-0.20, 0.20]`
- low-level EE world range 完整覆盖桌面平面、物体高度和预抓取上方空间

low-level 的职责是：

1. 跟踪 `vx/yaw-rate`；
2. 保持稳定并允许目标相关的 `0.22–0.32 m` 高度变化与最多
   `0.25 rad` 前倾；
3. 在移动中跟踪 EE 位置和 quaternion 姿态。

low-level 不负责感知物体、闭合夹爪、判断抓取或抬升物体。这些仍属于
high-level teacher/student。因而该合同与最终抓取任务在“稳定移动和 6D
预抓取执行底座”层面吻合，但 low-level 长训完成本身不等于抓取成功。

## 训练前门禁

从仓库根目录执行：

```bash
python3 -m py_compile \
  low-level/legged_gym/envs/manip_loco/manip_loco.py \
  low-level/legged_gym/envs/manip_loco/go2x5_config.py \
  low-level/legged_gym/scripts/check_go2x5_training_readiness.py

conda run --no-capture-output -n vwc_go2x5 \
  python low-level/legged_gym/scripts/check_go2x5_training_readiness.py \
  --num-envs 16 \
  --steps 128 \
  --output /tmp/go2x5_training_readiness.json
```

只有报告 `"passed": true` 时才允许启动长训。

本轮完整证据与可复制命令见
`docs/go2x5_lowlevel_flat_tabletop_6d_readiness_2026-07-29.md`。

## 当前有效入口

- 训练：`low-level/legged_gym/scripts/train.py`
- 训练 readiness：`low-level/legged_gym/scripts/check_go2x5_training_readiness.py`
- 零动作站立：`low-level/legged_gym/scripts/check_go2x5_zero_action_stand.py`
- checkpoint rollout：`low-level/legged_gym/scripts/check_go2x5_checkpoint_rollout.py`
- 固定命令评测：`low-level/legged_gym/scripts/check_go2x5_fixed_command_gait.py`
- EE 区域可视化：`low-level/legged_gym/scripts/visualize_go2x5_arm_workspace.py`
- IK 扫描/绘图：`scan_go2x5_ik_reachability.py`、`plot_go2x5_ik_reachability.py`

历史运行名称、旧模型路径和旧阶段结论仅保存在 `docs/scrap/`，不得作为当前训练依据。
