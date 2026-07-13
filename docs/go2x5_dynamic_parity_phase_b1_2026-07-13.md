# Go2-X5 Dynamic Runtime Parity Phase B.1 审查记录

日期：2026-07-13
分支：`agent/go2x5-runtime-parity`
基线：`c06135a Align Go2-X5 low and high-level runtime contracts`
审查结论：**Phase B.1 passed**

## 背景与边界

上一阶段只证明了单一 canonical state 下的零动作单帧一致性。零动作不能覆盖 policy/URDF action permutation、action scale、motor-strength 顺序和完整 PD torque path。本阶段增加非零非对称 probe、多状态、独立 oracle、真实 12D production-loader、7 秒 gait state-machine 和短 natural-reset 采样。

本阶段没有启动 low-level 训练或 high-level teacher/student 训练，没有修改 reward/PPO/network 结构，没有恢复 orientation tracking、domain randomization、相机、payload 或物体接触。随机初始化 checkpoint 只写入 `/tmp`，未纳入仓库。

## 文件级变更

- `high-level/envs/runtime_contract.py`
  - 新增无 Isaac 依赖的 start-pose resolver；优先级为显式参数 > eval pose > train pose > 代码默认值。
- `high-level/envs/b1z1_base.py`
  - 构造参数缺省改为 `None`，接入 resolver；
  - schema-v2 metadata 增加 `num_arm_actions` 校验；
  - high-level gait clock 恢复与 low-level 相同的 stance/swing phase-warp 浮点计算顺序。
- `high-level/envs/b1z1_pickmulti.py`
  - natural reset 不再要求训练入口临时注入 `env.wandb`，缺省为 `False`。
- `low-level/legged_gym/envs/manip_loco/manip_loco.py`
  - metadata 增加 `num_arm_actions`；
  - 删除 observation 上的 `torch.nan_to_num`，使 parity 对 NaN/Inf fail-closed。
- `tools/go2x5_runtime_parity.py`
  - snapshot schema 升级到 v2；
  - constant/linear diagnostic policies；
  - C0–C4 registry；
  - name-derived action permutation、独立 PD、EE frame、joint-limit oracle；
  - root/DOF/EE/Jacobian 等字段的 nonfinite 统计；
  - canonical controller、7 秒 gait 和 natural-reset step 0/1/4/10 collector；
  - schema-v2 checkpoint/hash/model-shape validator 与结构化 report。
- `tools/go2x5_parity_factories.py`
  - 支持 `canonical_injected` / `natural_reset`、C0–C4、zero/constant/linear/checkpoint；
  - canonical 使用 S3 effective action scale/command contract；
  - checkpoint high 侧不替换 `_load_low_level_model()`；diagnostic 模式仍明确使用临时 deterministic policy；
  - natural reset 不覆写 history、gait、action 或 controller buffer。
- `scripts/create_go2x5_schema_v2_smoke_checkpoint.py`
  - 固定 seed 创建当前 12D ActorCritic iteration-0 checkpoint；包含 runner-compatible model/optimizer state 和 schema-v2 metadata。
- `scripts/check_go2x5_runtime_parity.py`
  - capture 支持 `--state-mode`、`--case`、`--policy-mode`、`--checkpoint`、`--atol`，以及 runtime/controller/gait/natural-reset kind；
  - mismatch、oracle 或 nonfinite 均返回非零退出码。
- `tests/test_low_high_runtime_parity.py`、`tests/test_go2x5_alignment.py`
  - 增加 probe、oracle、tolerance、NaN/Inf、start-pose、case、JSON/CLI、hash、18D rejection 和 smoke metadata 测试。

## C0–C4 定义与结果

| Case | 状态 | Policy | 主要覆盖 | mismatch | oracle | nonfinite |
|---|---|---|---|---:|---:|---:|
| C0 | default q、qd=0、command=0 | zero | 默认 PD/history | 0 | 0 | 0 |
| C1 | default q、qd=0 | asymmetric constant | reorder/scale/nonzero torque | 0 | 0 | 0 |
| C2 | 非对称 leg q/qd 与 arm q perturbation | asymmetric constant | PD 符号、kp/kd、limit | 0 | 0 | 0 |
| C3 | rpy=(0.08,-0.06,0.25)、vx=0.10、yaw=0.15、EE=(0.32,-0.08,0.18) | deterministic linear | orientation、command、gait、EE frame | 0 | 0 | 0 |
| C4 | 稳定近 limit EE target | deterministic linear | IK clamp/invariant | 0 | 0 | 0 |

C4 的 `arm_joint3` 实际发生 clamp：unclamped `-0.1676708459854126`，lower/clamped `0.0`。所有 arm targets 满足 `lower-1e-7 <= target <= upper+1e-7`。

## Oracle 设计

Action oracle 从 policy/URDF joint name 列表独立推导 `[3,4,5,0,1,2,9,10,11,6,7,8]`，不调用 `_reindex_all()` 或 `_reindex_low_all()`。

PD oracle 独立计算：

```text
q_target = q_default + effective_action_scale * action_urdf
raw_tau = kp * (q_target - q) - kd * qd
tau = clamp(raw_tau, -limit, limit)
```

失败详情包含 joint name、q、qd、default、scale、action、kp、kd、raw/clamped torque 和 limit。Torque gate 为 `1e-5`。

EE oracle 将 `TERRAIN_INVARIANT_YAW` target 的 yaw-only world placement 与 low-policy observation 使用的 full-base inverse rotation分开验证。C3 的 world goal、arm-base world position和 reconstructed local 均为 0 mismatch。

## Smoke checkpoint 与 production loader

生成命令：

```bash
conda run -n vwc_go2x5 python scripts/create_go2x5_schema_v2_smoke_checkpoint.py \
  --contract-profile s3_deployment_smoke \
  --output /tmp/go2x5_schema_v2_smoke.pt
```

checkpoint 属性：

- current 12D `ActorCritic`，`num_arm_actions=0`；
- fixed seed `20260713`、iteration 0、未训练；
- `purpose=runtime_parity_smoke_only`、`trained=false`；
- auto curriculum disabled，contract profile `s3_deployment_smoke`；
- control contract SHA-256 `004cae99f31530d3001a01a7b01fd29d2b5ef1f33bfa4f61ab19158236d2264f`；
- 不读取、不截取、不转换旧 18D checkpoint。

High capture 使用原始 production `_load_low_level_model()`；日志出现 `Low level pretrained policy loaded!`，metadata、asset hash、control contract hash、12D state dict 均真实通过。两侧都调用 `act_inference(obs, hist_encoding=True)`。结果：policy output max abs `0.10181860625743866`，output/applied action/scaled q-target max error `0`，torque max error `0`，nonfinite `0`。

## Gait sequence

以 50 Hz、350 ticks 执行 0–7 秒指定 stop/start 序列。第一次验证发现 high 缺少 low 的 duration=0.5 phase-warp 浮点计算顺序，clock 最大误差 `5.662441253662109e-7`；修复后 walking mask、gait index、clock、dead-zone command 和 observation gait fields 全部 0 mismatch。

状态切换发生在 1.0、3.0、4.0、6.0 秒；stop tick 上 gait index 同时归零。完整逐帧数据保存在 `/tmp`，仓库只保存汇总报告。

## Natural reset

两侧均使用 normal actor creation/reset；关闭 DR、noise、push 和 camera，不注入 canonical state，不覆写 history/gait/action/controller buffer。在 step 0、1、4、10 采集 root、DOF、history、last action、gait、EE/arm target、contacts 和 reset flag。

结果：两侧四个 step 均可采集，policy observation shape 均为 781，nonfinite=0，无立即 reset、IK solve error 或 loader error。Low raw training observation 额外带 18D privileged block（799），high raw inference buffer 为 781；`hist_encoding=True` 消费的 current+history shape 一致。Natural-reset 轨迹误差本阶段仅报告，不作为闭环硬 gate。

## 执行命令

CPU/static：

```bash
python3 -m py_compile \
  tools/go2x5_runtime_parity.py \
  tools/go2x5_parity_factories.py \
  scripts/check_go2x5_runtime_parity.py \
  scripts/create_go2x5_schema_v2_smoke_checkpoint.py \
  tests/test_low_high_runtime_parity.py \
  tests/test_go2x5_alignment.py
python3 tests/test_low_high_runtime_parity.py
python3 tests/test_go2x5_alignment.py
git diff --check
```

Canonical capture（将 `SIDE/CASE/POLICY` 替换为对应值）：

```bash
conda run -n vwc_go2x5 python scripts/check_go2x5_runtime_parity.py capture \
  --side SIDE --kind controller \
  --factory tools.go2x5_parity_factories:make_SIDE_env \
  --state-mode canonical_injected --case CASE --policy-mode POLICY \
  --output /tmp/go2x5-SIDE-CASE.json
```

Checkpoint high production loader：

```bash
conda run -n vwc_go2x5 python scripts/check_go2x5_runtime_parity.py capture \
  --side high --kind controller \
  --factory tools.go2x5_parity_factories:make_high_env \
  --state-mode canonical_injected --case C3 --policy-mode checkpoint \
  --checkpoint /tmp/go2x5_schema_v2_smoke.pt \
  --output /tmp/go2x5-high-checkpoint.json
```

Gait/natural reset：

```bash
conda run -n vwc_go2x5 python scripts/check_go2x5_runtime_parity.py capture \
  --side low --kind gait --factory tools.go2x5_parity_factories:make_low_env \
  --state-mode canonical_injected --output /tmp/go2x5-low-gait.json
conda run -n vwc_go2x5 python scripts/check_go2x5_runtime_parity.py capture \
  --side high --kind natural_reset --factory tools.go2x5_parity_factories:make_high_env \
  --state-mode natural_reset --output /tmp/go2x5-high-natural.json
```

## 发现并修复的问题

1. YAML train pose 覆盖调用方显式/eval pose；
2. canonical factory 的 low/high gait tick 采样时机不同，linear policy 因 observation 不同而产生真实 output/torque mismatch；
3. high gait 缺少 low phase-warp 浮点计算顺序，clock 超过 `1e-7`；
4. high normal reset 隐式依赖训练入口注入 `env.wandb`；
5. low observation 的 `nan_to_num` 会掩盖 parity nonfinite；
6. schema-v2 metadata 未显式声明/校验 `num_arm_actions=0`。

## 未覆盖与结论

- 仍没有正式可部署的 trained 12D checkpoint；本阶段 checkpoint 明确为 untrained smoke only。
- 尚未证明 natural-reset 1 秒闭环 parity。
- 尚未证明 natural-reset 10 秒闭环 parity。
- 不允许恢复 domain randomization。
- 不允许恢复 high-level teacher/student 训练。
- 不允许开始完整 low-level 训练作为 parity 的替代。

本阶段全部硬 gate 已通过，允许进入唯一下一阶段：

**Go2-X5 Dynamic Parity Phase C: Natural-reset 1-second and 10-second closed-loop rollout parity**。
