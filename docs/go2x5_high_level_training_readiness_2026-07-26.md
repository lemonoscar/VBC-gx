# Go2-X5 high-level teacher 长训就绪审查

日期：2026-07-26

## 审查结论

当前 Go2-X5 代码与最终 low-level checkpoint 已通过 high-level teacher 启动前门禁，可以启动新的 high-level teacher 长训。

这个结论只表示训练路径、production loader、多环境仿真、reset、PPO 更新和日志写入已经可用，不表示尚未训练的 teacher 已经具备抓取能力，也不授权开始 student/BC 训练或部署。

## 使用的 low-level 模型

- 模型：`go2x5_v11_persistent_arm_gain010_seed1/model_45000.pt`
- SHA256：`5df0f06967963b3ceb6637282a70b193538b07eae815878f976eb8e511a92149`
- policy action dimension：12
- `num_arm_actions`：0
- production `_load_low_level_model()`：真实通过
- metadata schema、control contract 和 hash：真实通过

high-level 训练入口新增了 `--low_policy_path`，服务器可以从干净 Git checkout 显式加载外部 checkpoint，不再需要修改 YAML 或复制模型到源码目录。

## 修复的问题

### 多环境 EE controller 崩溃

`_get_low_level_ee_goal_local()` 原来把 `[num_envs, 4]` quaternion 与一维 `[3]` offset 直接传给 `quat_apply`。单环境测试未暴露问题，33 环境会报 shape error。

现在 offset 会显式展开为 `[num_envs, 3]`，33 环境 production-loader gate 已通过。

### reset 首帧使用陈旧 EE goal

原实现先计算 observation，之后才重置 EE goal。PPO 收到的 reset 首帧 observation 与随后执行的 controller target 不一致。

现在 reset 顺序为：

1. reset actor 和 tensor；
2. refresh simulation tensor；
3. 更新 robot info；
4. 初始化 EE goal 和 world transform；
5. 构造 observation。

Go2-X5 使用 `resetEEGoalToCurrent: true`，reset target 从真实 EE pose 开始并限制在 low-level workspace 内，避免 episode 开始时产生 IK 跳变。

### 起点过近导致 arm-only shortcut 和物体碰撞

机器人起点从 `x=-0.45 m` 调整为 `x=-0.65 m`。近桌边距离为 `0.50 m`，初始 arm-base 到最近物体位置的距离大于 `baseObjectDisThreshold=0.45 m`，策略必须先产生有效 base locomotion，不能仅靠机械臂在原地完成任务。

### 物体落台误判

原实现只要 object root 的 z 比桌面低一个数值 epsilon 就立即 reset。现在增加 `objectFallTolerance=0.02 m`，浅层接触求解穿透不会被误判为掉落，真正低于桌面 2 cm 才触发。

### 朝向与停止奖励错误

- `_reward_base_dir()` 原来错误地清零物体方向的 x/y 分量，使水平朝向奖励接近无效；现改为仅清零 z。
- command stop reward/penalty 原来硬编码 `0.6 m`，会在机器人尚未到达期望距离时奖励停下；现使用配置化 `commandStopDistance=0.45 m`。
- `standpick` 原来把所有负向速度视为满足停止条件；现使用绝对速度和配置 dead zone。
- position-only IK 下 orientation command 不可控，因此 Go2-X5 的 `ee_orn` reward 设为 0，不重新启用 orientation tracking。

### 训练日志与同步开销

原实现只要任一环境 reset，就计算多组 quantile/`.item()` 并打印整块诊断。33 环境、480 步 smoke 已产生接近 200 KB 日志；256 环境长训会频繁同步 GPU 并制造大量无用输出。

新增 `printResetStats`，Go2-X5 默认关闭。B1-Z1 未配置时仍保持原默认行为。关闭后 48-step PPO smoke 不再输出 reset 诊断块。

### 训练入口可靠性

- seed 现在实际传给 `set_seed(args.seed)`；
- stable launch script 改为一次性、`set -euo pipefail`；
- 删除 crash 后无限自动 resume；
- checkpoint 路径不存在时立即失败；
- 新增 fail-closed `check_go2x5_training_readiness.py`。

## 验证结果

### 本地无 GPU 测试

| 测试 | 结果 |
|---|---|
| Python compile | passed |
| `tests/test_go2x5_alignment.py` | passed |
| `tests/test_low_high_runtime_parity.py` | passed |
| `tests/test_go2x5_training_readiness.py` | passed |
| `git diff --check` | passed |

### 服务器 production runtime gate

配置：33 environments、64 high-level steps、真实 checkpoint、production loader、object features enabled。

| 指标 | 结果 |
|---|---:|
| observation shape | `[33, 1093]` |
| low observation shape | `[33, 726]` |
| resets | 0 |
| nonfinite | 0 |
| arm target limit violations | 0 |
| max EE transient error | 0.071293 m |
| max finger contact | 0 N |
| 结果 | passed |

### 服务器 PPO smoke

配置：33 environments、480 timesteps、20 个 rollout/update records。

| 指标 | 首条 | 末条 | finite |
|---|---:|---:|---|
| policy loss | -0.053790 | -0.039053 | yes |
| value loss | 1.485710 | 0.412715 | yes |
| instantaneous mean reward | 0.272574 | 0.283998 | yes |

进程以 0 退出，并写出 TensorBoard event、best checkpoint 以及 step 100/200/300/400 checkpoints。

随后使用日志抑制配置运行 48-step PPO smoke，进程正常退出，reset diagnostic block 为 0，平均约 3.89 high-level steps/s（33 environments）。

## 可复制命令

在 `high-level` 目录、只暴露一张物理 GPU 的前提下：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=<空闲物理GPU编号>
export LOW_POLICY_PATH=/absolute/path/to/model_45000.pt

python check_go2x5_training_readiness.py \
  --checkpoint "$LOW_POLICY_PATH" \
  --num-envs 33 \
  --steps 64 \
  --sim-device cuda:0 \
  --rl-device cuda:0 \
  --graphics-device-id 0
```

长训入口：

```bash
LOW_POLICY_PATH="$LOW_POLICY_PATH" \
NUM_ENVS=256 \
TIMESTEPS=60000 \
SEED=43 \
EXPERIMENT_DIR=/absolute/path/to/run/experiments \
TRAIN_NAME=go2x5_teacher_v12_cooperative_seed43 \
bash run_go2x5_train_stable.sh
```

## 仍未证明的内容

- 尚无已训练完成的 Go2-X5 high-level teacher；
- smoke 中随机初始化策略成功率为 0 属于预期，不能用于评价最终抓取质量；
- 尚未证明 teacher 收敛或最终 grasp success rate；
- 尚未开始 student/BC；
- 尚未恢复相机训练或部署验证。

## 下一阶段

启动并持续监控 `Go2-X5 high-level teacher` 长训。前期重点检查：

- process/GPU 持续存活；
- TensorBoard scalar 全部有限；
- episode length 不出现立即 reset storm；
- value/policy loss 不爆炸；
- mean reward、EE-object distance 和 success rate 随训练改善；
- checkpoint 按 interval 正常保存。

只有 teacher 评估达到稳定的 approach、grasp、lift 成功后，才允许进入 student/BC。
