"""Debug DOF order"""
import isaacgym
from isaacgym import gymapi

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = False

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

asset_options = gymapi.AssetOptions()
asset = gym.load_asset(sim, "/home/hpc/visual_wholebody/low-level/resources/robots/go2x5/urdf", "go2_arx_x5.urdf", asset_options)

dof_names = gym.get_asset_dof_names(asset)
print("=== DOF Order from Asset ===")
for i, name in enumerate(dof_names):
    print(f"  {i}: {name}")

gym.destroy_sim(sim)
