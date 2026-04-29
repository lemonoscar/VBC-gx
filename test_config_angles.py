"""Check if joint angles are correctly read from config"""
import sys
import os

LOW_LEVEL_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'low-level')
if LOW_LEVEL_ROOT not in sys.path:
    sys.path.insert(0, LOW_LEVEL_ROOT)

from legged_gym.envs.manip_loco.go2x5_config import Go2X5RoughCfg

print("=== default_joint_angles from config ===")
for name, angle in Go2X5RoughCfg.init_state.default_joint_angles.items():
    print(f"  {name}: {angle:.4f}")
