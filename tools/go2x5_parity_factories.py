"""One-environment deterministic factories for Go2-X5 parity capture."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_low_env():
    low_root = REPO_ROOT / "low-level"
    for path in (low_root, REPO_ROOT / "third_party/isaacgym/python", REPO_ROOT / "third_party/rsl_rl"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import legged_gym.envs  # noqa: F401
    from legged_gym.utils import get_args, task_registry

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = get_args(test=True)
    finally:
        sys.argv = original_argv
    args.task = "go2x5"
    args.headless = True
    args.sim_device = "cuda:0"
    args.rl_device = "cuda:0"
    args.graphics_device_id = 0
    args.observe_gait_commands = True

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.observe_gait_commands = True
    env_cfg.env.action_delay = 0
    env_cfg.env.record_video = False
    env_cfg.env.teleop_mode = False
    env_cfg.sim.gravity = [0.0, 0.0, 0.0]
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.height = [0.0, 0.0]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.auto_curriculum.enabled = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _set_canonical_state(env, "low")
    import torch
    env.foot_contacts_from_sensor = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.bool)
    env.compute_observations()
    current = env.obs_buf[:, :env.cfg.env.num_proprio].clone()
    env.obs_history_buf[:] = current[:, None, :]
    env.compute_observations()
    env.torques = env._compute_torques(env.actions)
    return env


def make_high_env():
    high_root = REPO_ROOT / "high-level"
    for path in (high_root, REPO_ROOT / "third_party/isaacgym/python"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import isaacgym  # noqa: F401  # must precede modules that import torch
    import torch
    from envs import Go2X5PickMulti
    from utils.config import load_cfg

    original_cwd = Path.cwd()
    os.chdir(high_root)
    try:
        cfg = load_cfg("data/cfg/go2x5_pickmulti.yaml")
        cfg["env"]["numEnvs"] = 1
        cfg["env"]["useTanh"] = False
        cfg["env"]["near_goal_stop"] = False
        cfg["env"]["obj_move_prob"] = 0.0
        cfg["env"]["cameraMode"] = "full"
        cfg["env"]["enableDebugVis"] = False
        cfg["env"]["requireLowPolicyMetadata"] = False
        cfg["sensor"]["enableCamera"] = False
        cfg["sim"]["gravity"] = [0.0, 0.0, 0.0]
        original_loader = Go2X5PickMulti._load_low_level_model

        def diagnostic_policy_loader(instance, *args, **kwargs):
            instance.num_priv = 18
            instance.num_gripper_joints = instance.num_physical_gripper_dof
            instance.num_proprio = 71
            instance.history_len = 10

            def zero_policy(observations, hist_encoding=True):
                return torch.zeros(
                    observations.shape[0], instance.low_policy_num_actions,
                    device=observations.device, dtype=observations.dtype,
                )

            return zero_policy

        Go2X5PickMulti._load_low_level_model = diagnostic_policy_loader
        try:
            env = Go2X5PickMulti(
                cfg=cfg,
                rl_device="cuda:0",
                sim_device="cuda:0",
                graphics_device_id=0,
                headless=True,
                use_roboinfo=True,
                observe_gait_commands=True,
                no_feature=True,
                commands_curriculum=False,
            )
        finally:
            Go2X5PickMulti._load_low_level_model = original_loader
        _set_canonical_state(env, "high")
        env._compute_low_level_observations()
        current = env.low_obs_buf[:, :env.num_proprio].clone()
        env.low_obs_history_buf[:] = current[:, None, :]
        env._compute_low_level_observations()
        env.torques = env._compute_torques(env.last_low_actions)
        return env
    finally:
        os.chdir(original_cwd)


def _set_canonical_state(env, side):
    from isaacgym import gymtorch
    import torch

    root = env.root_states if side == "low" else env._robot_root_states
    root[:, :13] = 0.0
    # Keep the robot clear of terrain/object contacts during the single
    # kinematic refresh; controller inputs are injected explicitly below.
    root[:, 2] = 1.0
    root[:, 6] = 1.0
    if side == "low":
        env.dof_pos[:] = env.default_dof_pos
        env.dof_vel[:] = 0.0
        root_tensor = env._root_states
        dof_state_tensor = env.dof_state
        dof_pos = env.dof_pos
    else:
        env._dof_pos[:] = env.initial_robo_pos
        env._dof_vel[:] = 0.0
        root_tensor = env._root_states
        dof_state_tensor = env._dof_state
        dof_pos = env._dof_pos

    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(root_tensor))
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(dof_state_tensor))
    position_targets = torch.zeros_like(dof_pos)
    position_targets[:, 12:] = dof_pos[:, 12:]
    env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(position_targets))
    env.gym.set_dof_actuation_force_tensor(
        env.sim, gymtorch.unwrap_tensor(torch.zeros_like(dof_pos))
    )
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)

    if side == "low":
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.gym.refresh_jacobian_tensors(env.sim)
        env.base_lin_vel[:] = 0.0
        env.base_ang_vel[:] = 0.0
        env.projected_gravity[:] = torch.tensor([0.0, 0.0, -1.0], device=env.device)
        env.base_yaw_quat[:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
        env.root_states[:, :13] = 0.0
        env.root_states[:, 2] = 1.0
        env.root_states[:, 6] = 1.0
        env.dof_pos[:] = env.default_dof_pos
        env.dof_vel[:] = 0.0
        env.commands[:] = 0.0
        env.actions[:] = 0.0
        env.action_history_buf[:] = 0.0
        env.gait_indices[:] = 0.0
        env.clock_inputs[:] = 0.0
        env.curr_ee_goal_cart[:] = torch.tensor([0.30, 0.0, 0.20], device=env.device)
        center = torch.tensor([0.085, 0.0, 0.424], device=env.device).repeat(env.num_envs, 1)
        env.curr_ee_goal_cart_world[:] = center + env.curr_ee_goal_cart
    else:
        env._refresh_sim_tensors()
        env._robot_root_states[:, :13] = 0.0
        env._robot_root_states[:, 2] = 1.0
        env._robot_root_states[:, 6] = 1.0
        env._dof_pos[:] = env.initial_robo_pos
        env._dof_vel[:] = 0.0
        env._initial_dof_pos[:] = env.initial_robo_pos
        env._initial_dof_vel[:] = 0.0
        env._update_base_yaw_quat()
        env.update_roboinfo()
        env.commands[:] = 0.0
        env.last_low_actions[:] = 0.0
        env.foot_contacts_from_sensor[:] = False
        env.gait_indices[:] = 0.0
        env.clock_inputs[:] = 0.0
        env.curr_ee_goal_cart[:] = torch.tensor([0.30, 0.0, 0.20], device=env.device)
        env._update_ee_goal_world()
