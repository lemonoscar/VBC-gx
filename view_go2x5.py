"""Simple script to view Go2+X5 robot in IsaacGym - no control, just visualization"""
import numpy as np
import os
from isaacgym import gymapi, gymutil

# Initialize gym
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments()

# Simulation parameters
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = False
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# Load robot asset
asset_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "low-level", "resources", "robots", "go2x5", "urdf")
asset_file = "go2_arx_x5.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False  # Let robot move freely
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS  # Position control mode
robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Print robot info
num_dofs = gym.get_asset_dof_count(robot_asset)
num_bodies = gym.get_asset_rigid_body_count(robot_asset)
dof_names = gym.get_asset_dof_names(robot_asset)
body_names = gym.get_asset_rigid_body_names(robot_asset)

print(f"\n=== Go2+X5 Robot Info ===")
print(f"DOFs: {num_dofs}")
print(f"Bodies: {num_bodies}")
print(f"DOF names: {dof_names}")
print(f"Body names: {body_names}")

# Create environment
env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)

# Set initial pose - robot standing on ground
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.45)  # Start height
pose.r = gymapi.Quat(0, 0, 0, 1)

# Create actor
actor = gym.create_actor(env, robot_asset, pose, "go2x5", 0, 0)

# Set DOF properties for stable standing
dof_props = gym.get_actor_dof_properties(env, actor)
dof_props['stiffness'][:12] = 100.0  # Leg joints
dof_props['damping'][:12] = 5.0
dof_props['stiffness'][12:18] = 50.0  # Arm joints
dof_props['damping'][12:18] = 2.0
dof_props['stiffness'][18:] = 10.0  # Gripper
dof_props['damping'][18:] = 1.0
gym.set_actor_dof_properties(env, actor, dof_props)

# Set initial joint positions for stable standing
dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
# Go2 leg joints - standing pose
# FL
dof_states['pos'][0] = 0.0    # FL_hip
dof_states['pos'][1] = 0.8    # FL_thigh
dof_states['pos'][2] = -1.5   # FL_calf
# FR
dof_states['pos'][3] = 0.0    # FR_hip
dof_states['pos'][4] = 0.8    # FR_thigh
dof_states['pos'][5] = -1.5   # FR_calf
# RL
dof_states['pos'][6] = 0.0    # RL_hip
dof_states['pos'][7] = 0.8    # RL_thigh
dof_states['pos'][8] = -1.5   # RL_calf
# RR
dof_states['pos'][9] = 0.0    # RR_hip
dof_states['pos'][10] = 0.8   # RR_thigh
dof_states['pos'][11] = -1.5  # RR_calf
# Arm joints - folded position
dof_states['pos'][12] = 0.0   # joint1 - no rotation
dof_states['pos'][13] = 0.0   # joint2 - shoulder down
dof_states['pos'][14] = 0.0   # joint3 - elbow
dof_states['pos'][15] = 0.0   # joint4 - wrist
dof_states['pos'][16] = 0.0   # joint5
dof_states['pos'][17] = 0.0   # joint6
dof_states['pos'][18] = 0.02  # joint7 gripper
dof_states['pos'][19] = 0.02  # joint8 gripper

gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

# Also set as position targets
targets = np.array([s['pos'] for s in dof_states], dtype=np.float32)
gym.set_actor_dof_position_targets(env, actor, targets)

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
cam_pos = gymapi.Vec3(2.0, 2.0, 1.0)
cam_target = gymapi.Vec3(0.0, 0.0, 0.3)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print("\n=== Viewer Controls ===")
print("Mouse: Rotate view")
print("WASD: Move camera")
print("ESC: Exit")
print("\nViewing Go2+X5 robot...")

# Simulation loop
while not gym.query_viewer_has_closed(viewer):
    # Step simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # Keep applying position targets to maintain pose
    gym.set_actor_dof_position_targets(env, actor, targets)
    
    # Update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("Done")
