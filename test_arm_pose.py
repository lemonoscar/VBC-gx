"""Test different arm poses to find the folded-back position"""
import numpy as np
from isaacgym import gymapi, gymutil

gym = gymapi.acquire_gym()
args = gymutil.parse_arguments()

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = False

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True  # Fix robot to see arm pose clearly
asset = gym.load_asset(sim, "/home/hpc/visual_wholebody/low-level/resources/robots/go2x5/urdf", "go2_arx_x5.urdf", asset_options)

# Print DOF names and info
num_dofs = gym.get_asset_dof_count(asset)
dof_names = gym.get_asset_dof_names(asset)
print("\n=== DOF Info ===")
for i, name in enumerate(dof_names):
    print(f"{i}: {name}")

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0.5)
actor = gym.create_actor(env, asset, pose, "go2x5", 0, 0)

# Set arm joints to test folded position
# Arm joints are after 12 leg joints: indices 12-19
# joint1=12, joint2=13, joint3=14, joint4=15, joint5=16, joint6=17, joint7=18, joint8=19
dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)

# Try different poses - arm should fold BACK (toward tail)
# Based on joint axes:
# joint1: Z-axis rotation (yaw) - 0 or pi to point backward
# joint2: Y-axis rotation (pitch) - controls up/down
# joint3: Y-axis rotation (with pi flip) - elbow

# Test: Rotate base 180 degrees to point arm backward
test_poses = [
    {"name": "forward", "j1": 0, "j2": 0, "j3": 0, "j4": 0},
    {"name": "j1=pi (yaw back)", "j1": 3.14, "j2": 0, "j3": 0, "j4": 0},
    {"name": "j2 positive", "j1": 0, "j2": 1.5, "j3": 0, "j4": 0},
    {"name": "j2 negative", "j1": 0, "j2": -1.5, "j3": 0, "j4": 0},
    {"name": "back + fold", "j1": 3.14, "j2": -1.5, "j3": 1.5, "j4": 0},
]

# Start with first pose
current_pose = 0
poses = test_poses

# Set initial pose
dof_states['pos'][12] = poses[current_pose]["j1"]  # joint1
dof_states['pos'][13] = poses[current_pose]["j2"]  # joint2
dof_states['pos'][14] = poses[current_pose]["j3"]  # joint3
dof_states['pos'][15] = poses[current_pose]["j4"]  # joint4
gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
cam_pos = gymapi.Vec3(2, 2, 1.5)
cam_target = gymapi.Vec3(0, 0, 0.3)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print(f"\n=== Testing pose: {poses[current_pose]['name']} ===")
print("Press SPACE to cycle through poses, ESC to exit")

frame = 0
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    
    # Check for spacebar press (every 60 frames to debounce)
    frame += 1
    if frame % 60 == 0:
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "space" and evt.value > 0:
                current_pose = (current_pose + 1) % len(poses)
                dof_states['pos'][12] = poses[current_pose]["j1"]
                dof_states['pos'][13] = poses[current_pose]["j2"]
                dof_states['pos'][14] = poses[current_pose]["j3"]
                dof_states['pos'][15] = poses[current_pose]["j4"]
                gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
                print(f"\n=== Testing pose: {poses[current_pose]['name']} ===")
                print(f"j1={poses[current_pose]['j1']}, j2={poses[current_pose]['j2']}, j3={poses[current_pose]['j3']}, j4={poses[current_pose]['j4']}")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
