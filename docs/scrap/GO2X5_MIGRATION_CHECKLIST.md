# Go2+X5 机器人迁移检查清单

## 当前状态分析

### ✅ 已完成部分
1. **URDF 模型文件** 
   - ✅ 已创建 `/home/hpc/visual_wholebody/low-level/resources/robots/go2x5/urdf/go2_arx_x5.urdf`
   - ✅ URDF 结构看起来完整（1742行）
   - ✅ 配置文件已指向此 URDF（`b1z1_config.py` 第166行已改为 go2x5）

### ❌ 待完成部分

#### 1. **关键问题：关节命名和配置不匹配**
   - **问题位置**: `b1z1_config.py` 中 `default_joint_angles` 和关节索引
   - **原 B1+Z1 配置**：
     - 腿部：FL/FR/RL/RR_hip/thigh/calf（12个关节）
     - 臂部：z1_waist/shoulder/elbow/wrist_angle/forearm_roll/wrist_rotate（6个关节）
     - 夹爪：z1_jointGripper（1个关节）
   - **需要验证 Go2+X5 的关节名称**：
     - Go2 的腿部关节是否也是 `*_hip/thigh/calf`？
     - X5 的臂部关节名称是什么？（不是 `z1_*`）
     - 夹爪关节名称是什么？

#### 2. **配置类需要创建** 
   - 需要创建 `Go2X5RoughCfg` 和 `Go2X5RoughCfgPPO` 配置类
   - 或在现有 `B1Z1RoughCfg` 基础上修改以支持 Go2X5

#### 3. **任务注册**
   - 需要在 `/home/hpc/visual_wholebody/low-level/legged_gym/envs/__init__.py` 中注册 Go2X5 任务

#### 4. **关键参数验证清单**

| 参数 | B1Z1 | Go2X5 | 需检查 |
|------|------|-------|--------|
| 腿部 DOF | 12 | ? | Go2 是否 12 个腿部关节？ |
| 臂部 DOF | 6 | ? | X5 是否 6 个关节？ |
| 夹爪 DOF | 1 | ? | 夹爪关节数是否为 1？ |
| total DOF | 19 | ? | 需确认总数 |
| 腿部足端名称 | foot | ? | Go2 的足端是什么名字？ |
| 臂部末端名称 | ee_gripper_link | ? | X5 的末端执行器是什么名字？ |
| 臂部 IK 关节数 | 6 | ? | X5 是否也是 6 DOF？ |

---

## 立即需要你提供的信息

### 1. Go2 URDF 的腿部关节名称
从 `/home/hpc/visual_wholebody/low-level/resources/robots/go2/urdf/go2.urdf` 中找出：
- 四条腿的所有关节名称（应该有 12 个）
- 足端执行器的链接名称

### 2. X5 URDF 的臂部关节名称
从 `/home/hpc/visual_wholebody/low-level/resources/robots/X5/X5A/urdf/X5A.urdf` 中找出：
- 所有臂部关节名称（应该有 6 个）
- 末端执行器夹爪链接的名称
- 末端执行器位置/方向信息

### 3. 合并的 URDF 验证
检查 `/home/hpc/visual_wholebody/low-level/resources/robots/go2x5/urdf/go2_arx_x5.urdf` 中：
- [ ] base link 是哪个？
- [ ] Go2 的所有关节是否正确包含？
- [ ] X5 的所有关节是否正确包含？
- [ ] 它们之间的连接（X5 如何挂到 Go2 上）是否正确？
- [ ] 所有 mesh 文件路径是否正确？

---

## 修改步骤

### Step 1: 获取关节信息（现在执行）
```bash
# 查看 Go2 的关节
grep "joint name" /home/hpc/visual_wholebody/low-level/resources/robots/go2/urdf/go2.urdf | head -20

# 查看 X5 的关节
grep "joint name" /home/hpc/visual_wholebody/low-level/resources/robots/X5/X5A/urdf/X5A.urdf | head -20

# 查看 Go2X5 合并后的关节
grep "joint name" /home/hpc/visual_wholebody/low-level/resources/robots/go2x5/urdf/go2_arx_x5.urdf | head -30
```

### Step 2: 创建 Go2X5 配置类
- 新建 `/home/hpc/visual_wholebody/low-level/legged_gym/envs/manip_loco/go2x5_config.py`
- 根据实际关节数和名称修改参数

### Step 3: 注册任务
- 修改 `/home/hpc/visual_wholebody/low-level/legged_gym/envs/__init__.py`
- 添加 `task_registry.register("go2x5", ...)`

### Step 4: 测试加载
```bash
cd /home/hpc/visual_wholebody/low-level/legged_gym/tests
python test_env.py --task go2x5 --headless --sim_device cuda:0 --rl_device cuda:0
```

---

## 需要帮助的关键点

1. **URDF 验证**: go2_arx_x5.urdf 中的关节名称是否正确？
2. **配置参数**: num_actions, num_torques, default_joint_angles 应该如何设置？
3. **IK 控制**: X5 的 IK 求解是否与 Z1 相同？
