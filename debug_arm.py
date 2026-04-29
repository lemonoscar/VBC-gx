"""Debug arm control"""
import isaacgym
import torch
import sys
import os

LOW_LEVEL_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'low-level')
if LOW_LEVEL_ROOT not in sys.path:
	sys.path.insert(0, LOW_LEVEL_ROOT)

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args

args = get_args()
args.task = "go2x5"
args.num_envs = 1

env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs = 1
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

print("\n=== Debug Info ===")
print(f"DOF names: {env.dof_names}")
print(f"num_dofs: {env.num_dofs}")
print(f"num_actions: {env.num_actions}")
print(f"num_torques: {env.num_torques}")

print(f"\ndefault_dof_pos shape: {env.default_dof_pos.shape}")
print(f"default_dof_pos: {env.default_dof_pos}")

print(f"\ndefault_dof_pos_wo_gripper shape: {env.default_dof_pos_wo_gripper.shape}")
print(f"default_dof_pos_wo_gripper: {env.default_dof_pos_wo_gripper}")

print(f"\np_gains: {env.p_gains}")
print(f"d_gains: {env.d_gains}")

print(f"\naction_scale: {env.action_scale}")

print(f"\nInitial dof_pos: {env.dof_pos}")

# Take one step with zero action
actions = torch.zeros(1, env.num_actions, device=env.device)
env.step(actions)

print(f"\nAfter step dof_pos: {env.dof_pos}")
print(f"Arm joints (12-19): {env.dof_pos[0, 12:]}")
