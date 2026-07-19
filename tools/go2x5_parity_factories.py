"""One-environment deterministic factories for Go2-X5 parity capture."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_low_env(state_mode="canonical_injected", case="C0", policy_mode=None, checkpoint=""):
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
    args.observe_gait_commands = False

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.observe_gait_commands = False
    env_cfg.env.action_delay = 0
    env_cfg.env.record_video = False
    env_cfg.env.teleop_mode = False
    if state_mode == "canonical_injected":
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
    env_cfg.commands.ranges.lin_vel_x = [-0.3, 0.3]
    env_cfg.commands.ranges.ang_vel_yaw = [-0.4, 0.4]
    from legged_gym.envs.manip_loco import go2x5_robot_spec
    env_cfg.control.action_scale = list(go2x5_robot_spec.LOW_ACTION_SCALE)
    env_cfg.auto_curriculum.enabled = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    import torch
    if state_mode == "canonical_injected":
        _set_canonical_state(env, "low", case)
        env.foot_contacts_from_sensor = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.bool)
        env._step_contact_targets()
        env.compute_observations()
        current = env.obs_buf[:, :env.cfg.env.num_proprio].clone()
        env.obs_history_buf[:] = current[:, None, :]
        env.compute_observations()
    else:
        env.reset()
    policy_mode = _resolve_policy_mode(case, policy_mode)
    policy, metadata = _make_policy(policy_mode, env.cfg.env.num_proprio * (env.cfg.env.history_len + 1), env.device, checkpoint)
    with torch.no_grad():
        policy_action = policy(env.obs_buf.detach(), hist_encoding=True)
    env.parity_policy_action = policy_action
    env.parity_policy = policy
    env.parity_policy_metadata = metadata
    env.parity_policy_mode = policy_mode
    env.parity_case = case
    env.parity_state_mode = state_mode
    if state_mode == "canonical_injected":
        env.actions[:] = env._reindex_all(policy_action)
        env.torques = env._compute_torques(env.actions)
    return env


def make_high_env(state_mode="canonical_injected", case="C0", policy_mode=None, checkpoint=""):
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
        cfg["sensor"]["enableCamera"] = False
        if state_mode == "canonical_injected":
            cfg["sim"]["gravity"] = [0.0, 0.0, 0.0]
        policy_mode = _resolve_policy_mode(case, policy_mode)
        if policy_mode == "checkpoint":
            if not checkpoint:
                raise ValueError("--checkpoint is required for policy_mode=checkpoint")
            cfg["env"]["low_policy_path"] = str(Path(checkpoint).resolve())
        original_loader = Go2X5PickMulti._load_low_level_model

        def diagnostic_policy_loader(instance, *args, **kwargs):
            instance.num_priv = 18
            instance.num_gripper_joints = instance.num_physical_gripper_dof
            instance.num_proprio = 66
            instance.history_len = 10

            policy, metadata = _make_policy(
                policy_mode, instance.num_proprio * (instance.history_len + 1),
                instance.device, checkpoint,
            )
            instance.parity_policy_metadata = metadata
            return policy

        if policy_mode != "checkpoint":
            Go2X5PickMulti._load_low_level_model = diagnostic_policy_loader
        try:
            env = Go2X5PickMulti(
                cfg=cfg,
                rl_device="cuda:0",
                sim_device="cuda:0",
                graphics_device_id=0,
                headless=True,
                use_roboinfo=True,
                observe_gait_commands=False,
                no_feature=True,
                commands_curriculum=False,
            )
        finally:
            Go2X5PickMulti._load_low_level_model = original_loader
        if state_mode == "canonical_injected":
            _set_canonical_state(env, "high", case)
            env._compute_low_level_observations()
            current = env.low_obs_buf[:, :env.num_proprio].clone()
            env.low_obs_history_buf[:] = current[:, None, :]
            env.gait_indices[:] = 0.0
            env.clock_inputs[:] = 0.0
            env._compute_low_level_observations()
        else:
            env.reset()
            env._compute_low_level_observations()
        with torch.no_grad():
            policy_action = env.low_level_policy(env.low_obs_buf.detach(), hist_encoding=True)
        env.parity_policy_action = policy_action
        env.parity_policy = env.low_level_policy
        if policy_mode == "checkpoint":
            loaded = torch.load(checkpoint, map_location="cpu")
            env.parity_policy_metadata = {
                "mode": "checkpoint", "checkpoint": str(Path(checkpoint).resolve()),
                "purpose": (loaded.get("metadata") or {}).get("purpose"),
            }
        else:
            env.parity_policy_metadata = getattr(env, "parity_policy_metadata")
        env.parity_policy_mode = policy_mode
        env.parity_case = case
        env.parity_state_mode = state_mode
        if state_mode == "canonical_injected":
            env.last_low_actions[:] = env._reindex_low_all(policy_action)
            env.torques = env._compute_torques(env.last_low_actions)
        return env
    finally:
        os.chdir(original_cwd)


def _resolve_policy_mode(case, policy_mode):
    from tools.go2x5_runtime_parity import CONTROLLER_CASES
    if case not in CONTROLLER_CASES:
        raise ValueError(f"Unknown controller case: {case}")
    return policy_mode or CONTROLLER_CASES[case]["policy_mode"]


def _make_policy(policy_mode, obs_dim, device, checkpoint):
    from tools.go2x5_runtime_parity import (
        build_smoke_actor_critic, make_diagnostic_policy, validate_schema_v2_checkpoint,
    )
    if policy_mode != "checkpoint":
        return make_diagnostic_policy(policy_mode, obs_dim, device=device)
    if not checkpoint:
        raise ValueError("checkpoint path is required")
    import torch
    from rsl_rl.modules import ActorCritic
    loaded = torch.load(checkpoint, map_location=device)
    validate_schema_v2_checkpoint(loaded)
    model = build_smoke_actor_critic(ActorCritic).to(device)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()
    return model.act_inference, {
        "mode": "checkpoint", "checkpoint": str(Path(checkpoint).resolve()),
        "purpose": loaded["metadata"].get("purpose"),
    }


def _set_canonical_state(env, side, case="C0"):
    from isaacgym import gymtorch
    import torch

    root = env.root_states if side == "low" else env._robot_root_states
    root[:, :13] = 0.0
    # Keep the robot clear of terrain/object contacts during the single
    # kinematic refresh; controller inputs are injected explicitly below.
    root[:, 2] = 1.0
    root[:, 6] = 1.0
    if case == "C3":
        from isaacgym.torch_utils import quat_from_euler_xyz
        root[:, 3:7] = quat_from_euler_xyz(
            torch.full((env.num_envs,), 0.08, device=env.device),
            torch.full((env.num_envs,), -0.06, device=env.device),
            torch.full((env.num_envs,), 0.25, device=env.device),
        )
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

    if case == "C2":
        from tools.go2x5_runtime_parity import (
            LEG_QD_POLICY_ORDER, LEG_Q_OFFSET_POLICY_ORDER, policy_to_urdf_oracle,
        )
        q_offset = torch.tensor(policy_to_urdf_oracle(LEG_Q_OFFSET_POLICY_ORDER), device=env.device)
        qd = torch.tensor(policy_to_urdf_oracle(LEG_QD_POLICY_ORDER), device=env.device)
        dof_pos[:, :12] += q_offset
        velocities = env.dof_vel if side == "low" else env._dof_vel
        velocities[:, :12] = qd
        arm_delta = torch.tensor([0.015, -0.010, 0.020, -0.012, 0.008, -0.006], device=env.device)
        num_gripper = env.cfg.env.num_gripper_joints if side == "low" else env.num_gripper_joints
        dof_pos[:, -(6 + num_gripper):-num_gripper] += arm_delta

    desired_root = root.clone()
    desired_dof_pos = dof_pos.clone()
    desired_dof_vel = (env.dof_vel if side == "low" else env._dof_vel).clone()

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
        from isaacgym.torch_utils import euler_from_quat, quat_from_euler_xyz, quat_rotate_inverse
        env.root_states[:, :13] = desired_root
        env.dof_pos[:] = desired_dof_pos
        env.dof_vel[:] = desired_dof_vel
        env.base_quat[:] = desired_root[:, 3:7]
        env.base_lin_vel[:] = 0.0
        env.base_ang_vel[:] = 0.0
        base_yaw = euler_from_quat(env.base_quat)[2]
        env.base_yaw_quat[:] = quat_from_euler_xyz(
            torch.zeros_like(base_yaw), torch.zeros_like(base_yaw), base_yaw
        )
        env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
        env.commands[:] = 0.0
        env.actions[:] = 0.0
        env.action_history_buf[:] = 0.0
        env.gait_indices[:] = 0.0
        env.clock_inputs[:] = 0.0
        env.curr_ee_goal_cart[:] = torch.tensor(
            [0.32, -0.08, 0.18] if case == "C3" else
            ([0.54, 0.16, 0.34] if case == "C4" else [0.30, 0.0, 0.20]), device=env.device
        )
        from isaacgym.torch_utils import quat_apply
        env.curr_ee_goal_cart_world[:] = env._get_ee_goal_spherical_center() + quat_apply(
            env.base_yaw_quat, env.curr_ee_goal_cart
        )
    else:
        env._refresh_sim_tensors()
        env._robot_root_states[:, :13] = desired_root
        env._dof_pos[:] = desired_dof_pos
        env._dof_vel[:] = desired_dof_vel
        env._initial_dof_pos[:] = env.initial_robo_pos
        env._initial_dof_vel[:] = 0.0
        env._update_base_yaw_quat()
        env.update_roboinfo()
        env.commands[:] = 0.0
        env.last_low_actions[:] = 0.0
        env.foot_contacts_from_sensor[:] = False
        env.gait_indices[:] = 0.0
        env.clock_inputs[:] = 0.0
        env.curr_ee_goal_cart[:] = torch.tensor(
            [0.32, -0.08, 0.18] if case == "C3" else
            ([0.54, 0.16, 0.34] if case == "C4" else [0.30, 0.0, 0.20]), device=env.device
        )
        env._update_ee_goal_world()
    if case == "C3":
        env.commands[:, 0] = 0.10
        env.commands[:, 2] = 0.15
