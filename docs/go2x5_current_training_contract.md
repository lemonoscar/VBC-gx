# Go2-X5 当前训练合同

更新日期：2026-07-30

## 当前结论

- 旧 Go2-X5 low-level 和 high-level checkpoint 均已判定为不可复用并从本地清除。
- 下一次 low-level 训练必须从随机初始化开始，任务名仅为 `go2x5`。
- `go2x5_ftlift` 空别名任务已经删除，不再作为训练阶段或恢复入口。
- 当前 locomotion 修订已经通过 CPU/静态门禁；修订前的 Isaac Gym
  readiness 已验证 plane、6D IK、40/1 PD、finite 和 reset 基线。
- 当前版本仍必须在 `lab-server` 上重新执行最终 GPU readiness、短程
  from-scratch smoke 和固定命令行为门禁；在这些门禁通过前，不允许直接
  启动 45000 iteration 长训。
- high-level 暂不可训练；必须等新的 low-level 12D schema-v2 checkpoint 通过确定性评测后，再通过 `--low_policy_path` 显式传入。

## 唯一有效的机器人与地面配置

- URDF：`low-level/resources/robots/go2x5/go2_x5.urdf`
- Go2 视觉 DAE 的几何数据与 Go2-X5-lab 一致，但旧 Isaac Gym 导入器
  必须使用 `Y_UP` 元数据；Isaac Lab/Omniverse 资产中的 `Z_UP`
  不能原样照搬。服务器同状态 A/B 渲染证明，`Z_UP` 会使腿部视觉
  网格旋入水平面，而碰撞体仍按 Z-up 正常站立。
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

- 参考 B1-Z1 的落地实体支撑台，按 `0.32/0.55` 站高比缩放，不使用悬空薄板
- B1-Z1 训练时 root 到桌心为 2.0 m；按 `0.32/0.55` 缩放后，Go2-X5
  root 到桌心为 1.15 m（robot x=-0.45，table x=0.70）
- 桌面近边缘：nominal root 前方 0.95 m，与机器人初始几何完全分离
- 支撑台尺寸：`0.40 × 0.60 × 0.15 m`，底面与 plane 齐平
- 本阶段表面高度固定为 `0.15 m`；先复现 B1-Z1 的简单固定场景
- 物体中心采样：table x `[-0.18, -0.09]`、y `[-0.20, 0.20]`
- 物体初始不在机械臂工作区内；high-level 必须先走近并在约 0.45 m
  arm-base/object 距离处停下，然后才进入机械臂抓取阶段
- low-level EE world range 完整覆盖桌面平面、物体高度和预抓取上方空间

low-level 的职责是：

1. 跟踪 `vx/yaw-rate`；
2. 保持稳定并允许目标相关的 `0.22–0.32 m` 高度变化与最多
   `0.25 rad` 前倾；
3. 在移动中跟踪 EE 位置和 quaternion 姿态。

low-level 不负责感知物体、闭合夹爪、判断抓取或抬升物体。这些仍属于
high-level teacher/student。因而该合同与最终抓取任务在“稳定移动和 6D
预抓取执行底座”层面吻合，但 low-level 长训完成本身不等于抓取成功。

## 简化 locomotion 合同

- 命令范围：`vx=[-0.30, 0.30] m/s`、`vy=[-0.10, 0.10] m/s`、
  `yaw=[-0.25, 0.25] rad/s`
- 每个 10 秒 episode 只保持一个命令
- 互斥命令人口：10% 站立、50% 纯直行、10% 原地转向、30% 一般运动
- 速度奖励采用 Walk These Ways 的平方误差指数核；针对低速范围将
  `tracking_sigma` 设为 `0.05`
- `feet_air_time` 权重为 1，只奖励完成摆动并重新落地的事件；短步不罚、
  永久悬空不得分
- `feet_drag` 只惩罚接触脚的水平滑动，不惩罚正常落脚的竖直速度
- 遵循 WTW 的 `only_positive_rewards=True`；关闭旧 stability-first 的
  `alive`、额外 `termination` 和 action-magnitude shaping，避免静止保命
  成为局部最优
- 不启用 gait clock、固定 trot、四拍 walk 或接触相位目标
- 初始探索 std 为每腿 `[0.25, 0.30, 0.30]`，最低 std 为
  `[0.08, 0.12, 0.12]`；`entropy_coef=0.01` 保留 WTW 的持续探索，
  同时按带 X5 负载的远端 roll-reset 实测对裸 Go1 力矩尺度降额
- iteration 0–3000 只优化 locomotion advantage；3000–6000 再渐入
  EE、姿态追踪和机身俯身协同

完整设计、旧模型定量失败证据和固定命令门禁见
`docs/go2x5_walk_these_ways_locomotion_repair_2026-07-30.md`。

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

readiness 报告 `"passed": true` 后，还必须运行短程 smoke，并用
`check_go2x5_fixed_command_gait.py` 验证站立、前进、后退和双向转弯。
只有各方向正确、每只脚均发生接触切换且无 nonfinite/early reset 时，
才允许启动长训。

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
