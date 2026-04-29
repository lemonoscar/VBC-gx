"""Debug IK and Jacobian for Go2X5"""
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

print("\n=== IK Debug Info ===")
print(f"gripper_idx: {env.gripper_idx}")
print(f"body_names: {env.body_names}")
print(f"num_bodies: {env.num_bodies}")
print(f"num_gripper_joints: {env.cfg.env.num_gripper_joints}")

print(f"\njacobian_whole shape: {env.jacobian_whole.shape}")
print(f"ee_j_eef shape: {env.ee_j_eef.shape}")
print(f"ee_j_eef:\n{env.ee_j_eef}")

print(f"\nee_pos: {env.ee_pos}")
print(f"ee_orn: {env.ee_orn}")

print(f"\ncurr_ee_goal_cart_world: {env.curr_ee_goal_cart_world}")
print(f"curr_ee_goal_sphere: {env.curr_ee_goal_sphere}")
print(f"init_start_ee_sphere: {env.init_start_ee_sphere}")

# Check if IK is working
dpos = env.curr_ee_goal_cart_world - env.ee_pos
print(f"\ndpos (goal - current): {dpos}")
