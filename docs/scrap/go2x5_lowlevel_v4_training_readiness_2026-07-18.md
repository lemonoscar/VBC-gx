# Go2-X5 low-level v4 长训就绪审查（2026-07-18）

## 审查结论

结论为：**允许使用修复后的 v4 代码从 S0 随机初始化开始 low-level 长训；禁止从 `go2x5_v3_flat_seed1/model_12000.pt` 续训。**

这里的“允许长训”只表示训练入口、运行时张量、奖励信号、checkpoint 合同和 PPO 基本路径已经通过 smoke gate，不表示已经得到合格步态，也不表示可以部署。v4 是否最终学出可用步态，仍须由 S3 成熟 checkpoint 的固定命令门禁判定。

机器可读摘要位于：

- `docs/parity_reports/go2x5_lowlevel_v4_training_readiness_2026-07-18.json`

## 为什么不能续训 model_12000

`model_12000.pt` 本身是合法的 12D schema-v2 checkpoint，但它是在旧 v3 训练合同下产生的。该合同包含三个会直接改变最优策略的问题：

1. 足端位置和速度曾通过 `rigid_body_state[:, feet_indices, ...]` 的 advanced indexing 只在初始化时赋值。PyTorch advanced indexing 返回副本，而不是随 Isaac Gym tensor 自动更新的 view，因此 gait 的 stance-foot velocity reward 长期读取旧值。
2. S3 的 `walking_dof=1.0` 强烈奖励运动命令下仍保持默认姿态。旧训练在 iteration 11507 的该项约为 `+43.99/s`，足以压过摆腿接触、拖地和抬脚惩罚，形成“不迈步也能得分”的局部最优。
3. 旧合同以 0.33 m 为初始和奖励目标高度；实际 Go2-X5 站立高度应为 0.32 m。

对旧 checkpoint 的固定命令测量也证实它不是健康步态：

| Case | 命令 | 实际响应 | 摆动腿接触率 | 摆动腿平均高度 |
|---|---:|---:|---:|---:|
| forward | `vx=+0.10` | `vx=-0.000861` | `97.51%` | `0.02281 m` |
| backward | `vx=-0.10` | `vx=-0.03318` | 约 `100%` | 低于门槛 |
| turn-left | `yaw=+0.15` | `yaw=+0.03216` | `99.27%` | 低于门槛 |

旧运行中，缓存足速与实时刚体足速的平均绝对误差约为 `0.0396–0.0458 m/s`。这说明旧 checkpoint 已经围绕错误信号优化；从它续训不是普通的 warm start，而是把坏局部最优和旧 critic/history 表征一起带入新目标。

v4 的 curriculum profile 已改为：

```text
go2x5_stable_reach_curriculum_v4_live_foot_gait_h032
```

production runner 对旧 `model_12000.pt` 的真实加载结果为 fail-closed：

```text
Go2-X5 training checkpoint curriculum profile mismatch:
checkpoint=go2x5_stable_reach_curriculum_v3_flat_step_metrics,
current=go2x5_stable_reach_curriculum_v4_live_foot_gait_h032
```

因此不得修改 metadata、关闭校验或强行 resume。

## 文件级修改

| 文件 | 修改 |
|---|---|
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 新增 `_refresh_foot_kinematics()`；在刚体 tensor refresh 后、buffer 初始化和 DOF reset 后刷新足端位置/速度缓存。 |
| `low-level/legged_gym/envs/rewards/maniploco_rewards.py` | stance-foot velocity reward 直接读取当帧 `rigid_body_state`，避免任何缓存再次成为奖励真值来源。 |
| `low-level/legged_gym/envs/manip_loco/go2x5_robot_spec.py` | 初始高度和 base-height reward target 统一为 `0.32 m`。 |
| `low-level/legged_gym/envs/manip_loco/go2x5_config.py` | profile 升级到 v4；S3 关闭 `walking_dof`，提高错误接触和摆脚高度约束。 |
| `high-level/data/cfg/go2x5_pickmulti.yaml` | train/eval start pose 和 reward target 同步为 `0.32 m`。相机安装坐标中的 `0.33` 不属于站立高度，未误改。 |
| `tools/go2x5_runtime_parity.py` | runtime snapshot 的缺省 start-z 同步为 `0.32 m`。 |
| `low-level/legged_gym/scripts/check_go2x5_training_readiness.py` | 增加 0.32 合同、v4 S3 权重、旧 profile 拒绝、每 policy tick 足端缓存一致性等 GPU gate。 |
| `low-level/legged_gym/scripts/check_go2x5_fixed_command_gait.py` | 新增 stand/forward/backward/turn-left 固定命令门禁；检查进度、摆腿接触、摆腿高度、站立漂移、reset、nonfinite 和足速缓存；失败返回非零。 |
| `low-level/legged_gym/scripts/audit_go2x5_low_level_rewards.py` | 奖励审计同步实时足速依赖、0.32 高度和禁用 `walking_dof` 的设计说明。 |
| `tests/test_go2x5_alignment.py` | 增加 low/high 0.32 m 合同回归。 |
| `tests/test_go2x5_training_readiness.py` | 增加 v4 profile、实时足速和固定命令 fail-closed gate 的静态回归。 |

## 奖励合同修正

S3 当前关键权重如下：

```text
tracking_contacts_shaped_force = 1.0
tracking_contacts_shaped_vel   = 0.5
walking_dof                    = 0.0
feet_air_time                  = 0.5
feet_height                    = 1.0
```

调整原则不是简单增大奖励，而是让旧的 no-step 状态不再占优：错误摆动相接触应被明确惩罚，摆腿高度不足应有足够梯度，而运动时保持默认关节姿态不再获得独立正奖励。S0–S2 的安全站立和 EE compensation curriculum 保持原入口，生产训练仍从 S0 开始。

## 验证结果

| 验证 | 规模/容差 | reset | nonfinite | 结果 |
|---|---:|---:|---:|---|
| Python compile + unit tests + reward audit | 3 组测试；audit mismatch `0` | — | — | PASS |
| low/high runtime parity | `atol=1e-6` | — | `0` | PASS，mismatch `0` |
| v4 S0 readiness | 8 env × 500 policy ticks（10 s） | `0` | `0` | PASS，51/51 |
| 足端 cache/live invariant | 每个 S0 policy tick | — | — | PASS，position/velocity max error `0` |
| v4 S0 PPO smoke | 256 env × 30 iterations，184,320 transitions | `0`（roll/pitch/z） | 2,666 scalar points 中 `0` | PASS |
| smoke model_30 deterministic rollout | 128 env × 500 ticks | `0` | `0` | PASS |
| fixed-command gate 自检 | 未训练 model_30 | `0` | `0` | PASS：安全项通过，但 gait behavior 被正确拒绝，完整 gate exit `1` |
| 旧 v3 model_12000 production load | v4 production runner | — | — | EXPECTED REJECT：profile mismatch |
| direct-S3 random-init stress | 256 env × 200 iterations | roll reset `41.67%` | 20,196 scalar points中 `0` | DIAGNOSTIC ONLY，未通过行为门禁 |

S0 smoke 的最后一个训练点为：mean reward `33.5197`、mean episode length `502.0`。model_30 的 10 s 回放中，最大绝对 action 为 `0.13047`，最大绝对 leg torque 为 `10.9985`，collision raw mean 为 `0`。

direct-S3 stress 的作用仅是确认修复后的 gait reward 确实随实时状态变化；其最终 stance-velocity raw reward 为 `-0.01634`，已经不再是旧运行中近似固定的 `-0.0023`。随机策略直接进入 S3 出现 roll reset 是预期的反例，也再次说明不能跳过 S0–S2 或把 direct-S3 smoke 当成长训入口。

## 正式训练前的服务器操作

当前服务器上旧 v3 训练进程在审查期间没有被终止。启动新任务前应先确认 GPU0 已空闲，并确认服务器仓库已经同步包含本报告所述 v4 修改。不要让旧 v3 和新 v4 使用同一张物理卡或同一个 experiment id。

先运行 S0 readiness：

```bash
cd /data4/duanzhibo/xhq_workload/VBC-gx

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
/data4/duanzhibo/miniconda3/envs/b1z1/bin/python \
  low-level/legged_gym/scripts/check_go2x5_training_readiness.py \
  --num-envs 8 \
  --steps 500 \
  --rollout-stage 0 \
  --sim-device cuda:0 \
  --rl-device cuda:0 \
  --graphics-device-id 0 \
  --output /tmp/go2x5-v4-s0-readiness.json
```

只有报告 `passed: true` 后，才启动新的随机初始化长训：

```bash
cd /data4/duanzhibo/xhq_workload/VBC-gx

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
/data4/duanzhibo/miniconda3/envs/b1z1/bin/python \
  low-level/legged_gym/scripts/train.py \
  --headless \
  --task go2x5 \
  --proj_name go2x5-low \
  --exptid go2x5_v4_livefoot_h032_seed1 \
  --num_envs 4096 \
  --max_iterations 45000 \
  --seed 1 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --graphics_device_id 0
```

该命令不得添加 `--resume`、`--resumeid` 或旧 checkpoint 参数。

## 长训中的硬门禁

早期 S0 checkpoint 尚未学习 locomotion，固定命令行为失败是正常的；进入 S3 并训练一段时间后，必须对候选 checkpoint 执行完整门禁：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
/data4/duanzhibo/miniconda3/envs/b1z1/bin/python \
  low-level/legged_gym/scripts/check_go2x5_fixed_command_gait.py \
  --checkpoint low-level/logs/go2x5-low/go2x5_v4_livefoot_h032_seed1/model_12000.pt \
  --num-envs 64 \
  --warmup-steps 50 \
  --measure-steps 200 \
  --sim-device cuda:0 \
  --rl-device cuda:0 \
  --graphics-device-id 0 \
  --output /tmp/go2x5-v4-model12000-fixed-command.json
```

默认 hard gate 要求：

- forward/backward 平移进度比例不低于 `0.35`；
- turn-left yaw 进度比例不低于 `0.35`；
- 摆动相接触率不高于 `0.75`；
- 摆动相平均高度不低于 `0.04 m`；
- stand 平均绝对 `vx` 和 yaw-rate error 均不高于 `0.03`；
- early reset、nonfinite 和足速 cache error 均为 `0`。

如果 S3 checkpoint 不通过，不能像旧 run 一样只看 mean reward 或 episode length 继续假定训练健康；应停止扩展训练并根据该 JSON 的具体失败项调整。

## 尚未证明的事项

- 当前没有正式训练完成、通过固定命令 gate 的 v4 12D checkpoint。
- 尚未证明最终策略的 1 秒或 10 秒 low/high closed-loop parity。
- 尚未允许恢复额外 domain randomization。
- 尚未允许启动 high-level teacher/student 训练。
- direct-S3 diagnostic 未通过，不能替代按 curriculum 从 S0 开始的正式训练。

因此本阶段唯一正确动作是：**同步 v4 修复，确认 GPU0 空闲，运行 S0 readiness，然后从随机初始化启动新的 low-level run；不要续训旧 model_12000。**
