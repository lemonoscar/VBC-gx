"""Check if joint angles are correctly read from config"""
import sys
sys.path.insert(0, '/home/hpc/visual_wholebody/low-level')

from legged_gym.envs.manip_loco.go2x5_config import Go2X5RoughCfg

print("=== default_joint_angles from config ===")
for name, angle in Go2X5RoughCfg.init_state.default_joint_angles.items():
    print(f"  {name}: {angle:.4f}")
