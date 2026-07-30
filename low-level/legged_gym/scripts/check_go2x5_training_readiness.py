#!/usr/bin/env python3
"""Fail-closed Isaac Gym probes for Go2-X5 low-level training readiness."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LOW_LEVEL_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = LOW_LEVEL_ROOT.parent
ISAACGYM_BINDINGS_DIR = (
    REPO_ROOT / "third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64"
)
ISAACGYM_USD_PLUGIN_DIR = ISAACGYM_BINDINGS_DIR / "usd/plugins"

library_paths = [str(ISAACGYM_BINDINGS_DIR), str(ISAACGYM_USD_PLUGIN_DIR)]
if os.environ.get("CONDA_PREFIX"):
    library_paths.append(str(Path(os.environ["CONDA_PREFIX"]) / "lib"))
existing = os.environ.get("LD_LIBRARY_PATH", "").split(":") if os.environ.get("LD_LIBRARY_PATH") else []
os.environ["LD_LIBRARY_PATH"] = ":".join(library_paths + [path for path in existing if path])
if os.environ.get("_ISAACGYM_LIBRARY_PATH_BOOTSTRAPPED") != "1":
    os.environ["_ISAACGYM_LIBRARY_PATH_BOOTSTRAPPED"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)

for path in (LOW_LEVEL_ROOT, REPO_ROOT / "third_party/isaacgym/python", REPO_ROOT / "third_party/rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import isaacgym  # noqa: E402,F401
from isaacgym.torch_utils import (  # noqa: E402
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
)
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.envs.manip_loco import go2x5_robot_spec  # noqa: E402
from legged_gym.utils import get_args, task_registry  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/go2x5_training_readiness.json"))
    return parser.parse_args()


def make_env_args(cli):
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = get_args(test=True)
    finally:
        sys.argv = original_argv
    args.task = "go2x5"
    args.headless = True
    args.num_envs = cli.num_envs
    args.seed = cli.seed
    args.sim_device = cli.sim_device
    args.rl_device = cli.rl_device
    args.graphics_device_id = cli.graphics_device_id
    args.observe_gait_commands = False
    return args


def configure_env(cfg, cli):
    cfg.env.num_envs = cli.num_envs
    cfg.env.observe_gait_commands = False
    cfg.env.record_video = False
    cfg.env.teleop_mode = False
    cfg.terrain.num_rows = 2
    cfg.terrain.num_cols = 2
    cfg.terrain.curriculum = False
    cfg.terrain.height = [0.0, 0.0]
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_base_com = False
    cfg.domain_rand.randomize_motor = False
    cfg.domain_rand.randomize_gripper_mass = False
    cfg.domain_rand.push_robots = False
    cfg.init_state.rand_yaw_range = 0.0
    cfg.init_state.origin_perturb_range = 0.0
    cfg.init_state.init_vel_perturb_range = 0.0
    cfg.init_state.leg_reset_ratio_range = [1.0, 1.0]
    cfg.init_state.arm_reset_noise_range = [0.0, 0.0]
    return cfg


class Checks:
    def __init__(self):
        self.items = []

    def require(self, name, condition, **details):
        passed = bool(condition)
        self.items.append({"name": name, "passed": passed, **details})
        if not passed:
            raise AssertionError(f"{name} failed: {details}")


def scalar_max_abs(tensor):
    return float(torch.max(torch.abs(tensor)).item()) if tensor.numel() else 0.0


def require_finite(checks, name, tensor):
    finite = torch.isfinite(tensor)
    count = int((~finite).sum().item())
    details = {"nonfinite": count, "shape": list(tensor.shape)}
    if count:
        details["first_index"] = torch.nonzero(~finite, as_tuple=False)[0].tolist()
    checks.require(f"finite/{name}", count == 0, **details)


def probe_plane_ground(env, checks, settle_steps=5):
    checks.require(
        "terrain/native_physx_plane",
        env.cfg.terrain.mesh_type == "plane",
        mesh_type=env.cfg.terrain.mesh_type,
    )
    checks.require(
        "terrain/no_terrain_object",
        not hasattr(env, "terrain"),
        has_terrain=hasattr(env, "terrain"),
    )
    checks.require(
        "terrain/regular_grid_origins",
        env.custom_origins is False
        and bool(torch.all(env.env_origins[:, 2] == 0.0)),
        custom_origins=env.custom_origins,
        max_abs_origin_z=scalar_max_abs(env.env_origins[:, 2]),
    )

    env.reset()
    env.commands.zero_()
    zero_action = torch.zeros(
        env.num_envs,
        env.num_actions,
        device=env.device,
    )
    for _ in range(settle_steps):
        env.step(zero_action)
        env.commands.zero_()

    foot_centers_z = env.rigid_body_state[:, env.feet_indices, 2]
    foot_bottom_z = foot_centers_z - go2x5_robot_spec.FOOT_COLLISION_RADIUS
    foot_contact_norm = torch.norm(
        env.contact_forces[:, env.feet_indices, :],
        dim=-1,
    )
    min_bottom_z = float(torch.min(foot_bottom_z).item())
    min_contact = float(torch.min(foot_contact_norm).item())
    checks.require(
        "terrain/all_feet_resolved_above_plane",
        min_bottom_z >= -2.0e-3,
        settle_steps=settle_steps,
        min_foot_collision_bottom_z=min_bottom_z,
        per_foot_bottom_z=foot_bottom_z[0].detach().cpu().tolist(),
    )
    checks.require(
        "terrain/all_feet_have_contact",
        min_contact > 1.0,
        settle_steps=settle_steps,
        min_contact_force=min_contact,
        per_foot_contact_force=foot_contact_norm[0].detach().cpu().tolist(),
    )
    env.reset()


def probe_rewards(env, checks):
    reward = env.reward_container
    checks.require(
        "contract/base_height_0p32",
        abs(float(env.cfg.init_state.pos[2]) - 0.32) <= 1e-9
        and abs(float(env.cfg.rewards.base_height_target) - 0.32) <= 1e-9,
        initial=float(env.cfg.init_state.pos[2]),
        target=float(env.cfg.rewards.base_height_target),
    )
    center = [
        float(env.cfg.goal_ee.sphere_center.x_offset),
        float(env.cfg.goal_ee.sphere_center.y_offset),
        float(env.cfg.goal_ee.sphere_center.z_invariant_offset),
    ]
    local_ranges = [
        list(env.goal_ee_ranges[axis]) for axis in ("pos_x", "pos_y_cart", "pos_z")
    ]
    world_ranges = [
        [round(center[axis] + limit, 6) for limit in local_ranges[axis]]
        for axis in range(3)
    ]
    checks.require(
        "contract/front_ee_workspace",
        center == go2x5_robot_spec.EE_GOAL_CENTER_OFFSET
        and local_ranges == go2x5_robot_spec.EE_GOAL_LOCAL_RANGES
        and world_ranges == go2x5_robot_spec.EE_GOAL_WORLD_RANGES,
        center=center,
        local_ranges=local_ranges,
        world_ranges=world_ranges,
    )
    arm_default = env.default_dof_pos[-8:-2].detach().cpu().tolist()
    checks.require(
        "contract/forward_ready_arm_pose",
        all(
            abs(actual - expected) <= 1.0e-6
            for actual, expected in zip(arm_default, go2x5_robot_spec.ARM_READY_JOINT_ANGLES)
        ),
        arm_default=arm_default,
    )
    checks.require(
        "contract/persistent_arm_and_open_gripper",
        env.cfg.arm.target_mode == go2x5_robot_spec.ARM_TARGET_MODE
        and abs(float(env.cfg.arm.ik_gain) - float(go2x5_robot_spec.ARM_IK_GAIN))
        <= 1.0e-9
        and abs(
            float(env.cfg.arm.target_max_step)
            - float(go2x5_robot_spec.ARM_TARGET_MAX_STEP)
        )
        <= 1.0e-9
        and env.cfg.arm.gripper_hold_mode
        == go2x5_robot_spec.LOW_LEVEL_GRIPPER_HOLD_MODE
        and env.cfg.arm.track_ee_orientation
        and abs(
            float(env.cfg.arm.ik_orientation_weight)
            - float(go2x5_robot_spec.ARM_IK_ORIENTATION_WEIGHT)
        )
        <= 1.0e-9,
        target_mode=env.cfg.arm.target_mode,
        ik_gain=float(env.cfg.arm.ik_gain),
        target_max_step=float(env.cfg.arm.target_max_step),
        gripper_hold_mode=env.cfg.arm.gripper_hold_mode,
        track_orientation=bool(env.cfg.arm.track_ee_orientation),
        orientation_weight=float(env.cfg.arm.ik_orientation_weight),
    )
    dof_props = env.gym.get_actor_dof_properties(
        env.envs[0], env.actor_handles[0]
    )
    checks.require(
        "contract/x5_joint_specific_arm_pd",
        bool(
            torch.allclose(
                torch.as_tensor(dof_props["stiffness"][12:18]),
                torch.tensor(go2x5_robot_spec.ARM_POS_STIFFNESS),
                atol=1.0e-6,
                rtol=0.0,
            )
            and torch.allclose(
                torch.as_tensor(dof_props["damping"][12:18]),
                torch.tensor(go2x5_robot_spec.ARM_POS_DAMPING),
                atol=1.0e-6,
                rtol=0.0,
            )
            and torch.allclose(
                torch.as_tensor(dof_props["stiffness"][18:]),
                torch.full(
                    (env.cfg.env.num_gripper_joints,),
                    go2x5_robot_spec.GRIPPER_POS_STIFFNESS,
                ),
                atol=1.0e-6,
                rtol=0.0,
            )
        ),
        arm_kp=dof_props["stiffness"][12:18].tolist(),
        arm_kd=dof_props["damping"][12:18].tolist(),
        gripper_kp=dof_props["stiffness"][18:].tolist(),
    )
    penalized_names = {
        env.body_names[int(index)]
        for index in env.penalized_contact_indices.detach().cpu().tolist()
    }
    required_visible_contacts = {
        name
        for name in env.body_names
        if name == "base"
        or name.startswith("Head_")
        or name.startswith("arm_link")
    }
    checks.require(
        "contract/head_arm_finger_contacts_visible",
        required_visible_contacts.issubset(penalized_names)
        and {"arm_link7", "arm_link8"}.issubset(penalized_names),
        required=sorted(required_visible_contacts),
        penalized=sorted(penalized_names),
    )

    first_id = torch.zeros(1, device=env.device, dtype=torch.long)
    saved_start = env.ee_start_cart[first_id].clone()
    saved_goal = env.ee_goal_cart[first_id].clone()
    safe_goal = torch.tensor(
        [[0.365, 0.0, -0.064]], device=env.device, dtype=env.ee_goal_cart.dtype
    )
    near_body_goal = torch.tensor(
        [[0.215, 0.0, -0.100]], device=env.device, dtype=env.ee_goal_cart.dtype
    )
    far_low_goal = torch.tensor(
        [[0.565, 0.225, -0.364]],
        device=env.device,
        dtype=env.ee_goal_cart.dtype,
    )
    try:
        env.ee_start_cart[first_id] = safe_goal
        env.ee_goal_cart[first_id] = near_body_goal
        near_body_rejected = bool(env._collision_check(first_id).item())
        env.ee_start_cart[first_id] = far_low_goal
        env.ee_goal_cart[first_id] = far_low_goal
        far_low_rejected = bool(env._collision_check(first_id).item())
        env.ee_start_cart[first_id] = safe_goal
        env.ee_goal_cart[first_id] = safe_goal
        center_accepted = not bool(env._collision_check(first_id).item())
    finally:
        env.ee_start_cart[first_id] = saved_start
        env.ee_goal_cart[first_id] = saved_goal
    checks.require(
        "contract/workspace_filter_rejects_collision_and_overreach",
        near_body_rejected
        and far_low_rejected
        and center_accepted
        and abs(
            env.max_nominal_reach_radius
            - go2x5_robot_spec.EE_GOAL_MAX_NOMINAL_REACH_RADIUS
        )
        <= 1.0e-9,
        near_body_rejected=near_body_rejected,
        far_low_rejected=far_low_rejected,
        center_accepted=center_accepted,
        max_nominal_reach_radius=env.max_nominal_reach_radius,
    )
    expected_scale = torch.tensor(
        [0.125, 0.25, 0.25] * 4, device=env.device, dtype=env.action_scale.dtype
    )
    checks.require(
        "contract/leg_pd_and_action_scale",
        bool(
            torch.all(env.p_gains[:12] == 40.0)
            and torch.all(env.d_gains[:12] == 1.0)
            and torch.equal(env.action_scale[:12], expected_scale)
        ),
        kp=env.p_gains[:12].detach().cpu().tolist(),
        kd=env.d_gains[:12].detach().cpu().tolist(),
        action_scale=env.action_scale[:12].detach().cpu().tolist(),
    )
    checks.require(
        "contract/simple_emergent_locomotion",
        env.cfg.env.num_proprio == 66
        and not env.cfg.env.observe_gait_commands
        and env.cfg.env.policy_output_tanh
        and abs(float(env.cfg.normalization.clip_actions) - 1.0) <= 1e-9
        and not env.cfg.asset.replace_cylinder_with_capsule
        and "walking_dof" not in env.reward_scales
        and "tracking_contacts_shaped_force" not in env.reward_scales
        and "tracking_contacts_shaped_vel" not in env.reward_scales
        and "feet_height" not in env.reward_scales
        and "stability_safety" not in env.reward_scales
        and "stand_still" not in env.reward_scales
        and "base_height" not in env.reward_scales
        and "tracking_lin_vel_max" not in env.reward_scales
        and "height_adaptation" not in env.reward_scales
        and "pitch_adaptation" not in env.reward_scales
        and abs(float(env.reward_scales["leg_action_l2_deadzone"]) + 0.01) <= 1e-9
        and abs(float(env.reward_scales["tracking_lin_vel"]) - 2.0) <= 1e-9
        and abs(float(env.reward_scales["tracking_ang_vel"]) - 0.5) <= 1e-9
        and abs(float(env.reward_scales["collision"]) + 1.0) <= 1e-9
        and abs(float(env.reward_scales["action_rate"]) + 0.01) <= 1e-9
        and abs(float(env.cfg.rewards.tracking_sigma) - 0.05) <= 1e-9
        and abs(float(env.cfg.rewards.feet_air_time_target) - 0.10) <= 1e-9
        and abs(float(env.cfg.rewards.feet_clearance_target) - 0.05) <= 1e-9
        and abs(float(env.cfg.rewards.feet_clearance_landing_bonus) - 0.20)
        <= 1e-9
        and abs(float(env.reward_scales["feet_air_time"]) - 2.0) <= 1e-9
        and abs(float(env.reward_scales["feet_contact_standing"]) + 0.5) <= 1e-9
        and abs(float(env.arm_reward_scales["tracking_ee_world"]) - 2.0) <= 1e-9
        and abs(float(env.arm_reward_scales["tracking_ee_orn"]) - 0.6) <= 1e-9
        and abs(float(env.arm_reward_scales["height_adaptation"]) + 3.0) <= 1e-9
        and abs(float(env.arm_reward_scales["pitch_adaptation"]) + 1.0) <= 1e-9
        and "tracking_ee_world_stable" not in env.arm_reward_scales,
        num_proprio=int(env.cfg.env.num_proprio),
        observe_gait=bool(env.cfg.env.observe_gait_commands),
        replace_capsules=bool(env.cfg.asset.replace_cylinder_with_capsule),
        tracking_lin_vel=float(env.reward_scales["tracking_lin_vel"]),
        height_adaptation=float(env.arm_reward_scales["height_adaptation"]),
        pitch_adaptation=float(env.arm_reward_scales["pitch_adaptation"]),
        tracking_ang_vel=float(env.reward_scales["tracking_ang_vel"]),
        collision=float(env.reward_scales["collision"]),
        action_rate=float(env.reward_scales["action_rate"]),
        tracking_sigma=float(env.cfg.rewards.tracking_sigma),
        feet_air_time=float(env.reward_scales["feet_air_time"]),
        feet_contact_standing=float(env.reward_scales["feet_contact_standing"]),
        action_bound=float(env.reward_scales["leg_action_l2_deadzone"]),
        ee_tracking=float(env.arm_reward_scales["tracking_ee_world"]),
        ee_orientation_tracking=float(env.arm_reward_scales["tracking_ee_orn"]),
    )
    expected_leg_rewards = {
        "tracking_lin_vel",
        "tracking_ang_vel",
        "feet_air_time",
        "feet_contact_standing",
        "torques",
        "alive",
        "termination",
        "lin_vel_z",
        "roll",
        "collision",
        "action_rate",
        "dof_pos_limits",
        "feet_drag",
        "leg_action_l2_deadzone",
    }
    checks.require(
        "reward/minimal_active_set",
        set(env.reward_scales) == expected_leg_rewards
        and set(env.arm_reward_scales)
        == {
            "tracking_ee_world",
            "tracking_ee_orn",
            "height_adaptation",
            "pitch_adaptation",
        },
        leg_rewards=sorted(env.reward_scales),
        arm_rewards=sorted(env.arm_reward_scales),
    )

    zero = torch.zeros(1, device=env.device)
    forward_pitch = torch.full((1,), float(env.cfg.rewards.max_forward_body_pitch), device=env.device)
    pitched_quat = quat_from_euler_xyz(zero, forward_pitch, zero)
    arm_mount = torch.tensor(
        [go2x5_robot_spec.ARM_BASE_OFFSET], device=env.device, dtype=pitched_quat.dtype
    )
    pitched_mount_z = float(quat_apply(pitched_quat, arm_mount)[0, 2].item())
    checks.require(
        "contract/positive_pitch_lowers_arm_mount",
        pitched_mount_z < float(go2x5_robot_spec.ARM_BASE_OFFSET[2]),
        flat_mount_z=float(go2x5_robot_spec.ARM_BASE_OFFSET[2]),
        pitched_mount_z=pitched_mount_z,
        pitch_rad=float(forward_pitch.item()),
    )

    env.commands.zero_()
    env.desired_contact_states.fill_(0.0)
    env.contact_forces[:, env.feet_indices, 2] = 100.0
    force_disabled, _ = reward._reward_tracking_contacts_shaped_force()
    vel_disabled, _ = reward._reward_tracking_contacts_shaped_vel()
    checks.require(
        "reward/gait_schedule_disabled",
        scalar_max_abs(force_disabled) == 0.0 and scalar_max_abs(vel_disabled) == 0.0,
    )

    env.commands.zero_()
    env.commands[:, 0] = 0.10
    env.base_lin_vel[:, 0] = 0.10
    velocity_best, _ = reward._reward_tracking_lin_vel()
    env.base_lin_vel[:, 0] = 0.0
    velocity_under, velocity_error = reward._reward_tracking_lin_vel()
    env.base_lin_vel[:, 0] = 0.20
    velocity_over, _ = reward._reward_tracking_lin_vel()
    expected_velocity_error = torch.full_like(velocity_error, 0.01)
    expected_velocity_reward = torch.exp(
        -expected_velocity_error / float(env.cfg.rewards.tracking_sigma)
    )
    checks.require(
        "reward/walk_these_ways_velocity_kernel",
        bool(
            torch.all(velocity_best > velocity_under)
            and torch.allclose(velocity_under, velocity_over)
            and torch.allclose(velocity_error, expected_velocity_error)
            and torch.allclose(velocity_under, expected_velocity_reward)
        ),
        best=float(velocity_best.mean().item()),
        underspeed=float(velocity_under.mean().item()),
        overspeed=float(velocity_over.mean().item()),
        squared_error=float(velocity_error.mean().item()),
    )

    env.commands[:, 2] = 0.10
    env.base_ang_vel[:, 2] = 0.10
    yaw_best, _ = reward._reward_tracking_ang_vel()
    env.base_ang_vel[:, 2] = 0.0
    yaw_under, yaw_error = reward._reward_tracking_ang_vel()
    env.base_ang_vel[:, 2] = 0.20
    yaw_over, _ = reward._reward_tracking_ang_vel()
    checks.require(
        "reward/walk_these_ways_yaw_kernel",
        bool(
            torch.all(yaw_best > yaw_under)
            and torch.allclose(yaw_under, yaw_over)
            and torch.allclose(yaw_error, expected_velocity_error)
        ),
        best=float(yaw_best.mean().item()),
        underspeed=float(yaw_under.mean().item()),
        overspeed=float(yaw_over.mean().item()),
    )

    terrain_height = reward._terrain_height()
    env.curr_ee_goal_cart_world[:, 2] = terrain_height + 0.05
    low_height_target = reward._adaptive_body_height_target()
    env.curr_ee_goal_cart_world[:, 2] = terrain_height + 0.175
    middle_height_target = reward._adaptive_body_height_target()
    env.curr_ee_goal_cart_world[:, 2] = terrain_height + 0.30
    high_height_target = reward._adaptive_body_height_target()
    high_pitch_target = reward._adaptive_body_pitch_target()
    env.curr_ee_goal_cart_world[:, 2] = terrain_height + 0.05
    low_pitch_target = reward._adaptive_body_pitch_target()
    checks.require(
        "reward/safe_adaptive_height_mapping",
        bool(
            torch.allclose(low_height_target, torch.full_like(low_height_target, 0.22))
            and torch.allclose(middle_height_target, torch.full_like(middle_height_target, 0.27))
            and torch.allclose(high_height_target, torch.full_like(high_height_target, 0.32))
            and torch.allclose(low_pitch_target, torch.full_like(low_pitch_target, 0.25))
            and torch.allclose(high_pitch_target, torch.zeros_like(high_pitch_target))
            and float(low_height_target.min().item()) > env.cfg.termination.z_threshold
        ),
        low=float(low_height_target.mean().item()),
        middle=float(middle_height_target.mean().item()),
        high=float(high_height_target.mean().item()),
        low_pitch=float(low_pitch_target.mean().item()),
        high_pitch=float(high_pitch_target.mean().item()),
    )

    env.curr_ee_goal_cart_world[:] = env.ee_pos
    ee_exact, _ = reward._reward_tracking_ee_world()
    env.curr_ee_goal_cart_world[:, 0] += 0.10
    ee_offset, _ = reward._reward_tracking_ee_world()
    env.foot_contacts_from_sensor.zero_()
    ee_without_contacts, _ = reward._reward_tracking_ee_world()
    checks.require(
        "reward/raw_ee_tracking_allows_body_coordination",
        bool(torch.all(ee_exact > ee_offset) and torch.equal(ee_offset, ee_without_contacts)),
        exact=float(ee_exact.mean().item()),
        offset=float(ee_offset.mean().item()),
    )

    current_orn = env.ee_orn / torch.norm(
        env.ee_orn, dim=-1, keepdim=True
    ).clamp(min=1e-6)
    saved_target_orn = env.ee_goal_orn_quat.clone()
    env.ee_goal_orn_quat[:] = current_orn
    orn_exact, exact_angle = reward._reward_tracking_ee_orn()
    quarter_turn = quat_from_euler_xyz(
        torch.zeros(env.num_envs, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
        torch.full((env.num_envs,), 0.25, device=env.device),
    )
    env.ee_goal_orn_quat[:] = quat_mul(quarter_turn, current_orn)
    orn_offset, offset_angle = reward._reward_tracking_ee_orn()
    env.ee_goal_orn_quat.copy_(saved_target_orn)
    checks.require(
        "reward/quaternion_orientation_tracking",
        bool(
            torch.all(orn_exact > orn_offset)
            and torch.all(exact_angle <= 1.0e-6)
            and torch.all(offset_angle > 0.20)
        ),
        exact=float(orn_exact.mean().item()),
        offset=float(orn_offset.mean().item()),
        offset_angle=float(offset_angle.mean().item()),
    )

    env.foot_contacts_from_sensor.fill_(True)
    env.contact_filt.fill_(True)
    env.rigid_body_state[:, env.feet_indices, 7:10] = 0.0
    drag_still, _ = reward._reward_feet_drag()
    env.rigid_body_state[:, env.feet_indices[0], 9] = 1.0
    drag_vertical, _ = reward._reward_feet_drag()
    env.rigid_body_state[:, env.feet_indices[0], 9] = 0.0
    env.rigid_body_state[:, env.feet_indices[0], 7] = 0.5
    drag_moving, _ = reward._reward_feet_drag()
    env.contact_filt.zero_()
    drag_airborne, _ = reward._reward_feet_drag()
    checks.require(
        "reward/contact_horizontal_slip_penalized",
        bool(
            torch.all(drag_moving > drag_still)
            and torch.equal(drag_vertical, drag_still)
            and torch.equal(drag_airborne, drag_still)
        ),
        still=float(drag_still.mean().item()),
        vertical=float(drag_vertical.mean().item()),
        moving=float(drag_moving.mean().item()),
        airborne=float(drag_airborne.mean().item()),
    )

    env.commands.zero_()
    env.commands[:, 0] = 0.10
    env.feet_air_time.zero_()
    env.feet_swing_peak_height.zero_()
    env.feet_swing_peak_height[:, 0] = 0.055
    env.feet_air_time[:, 0] = 0.30
    env.contact_filt.zero_()
    env.contact_filt[:, 0] = True
    air_time_landing, _ = reward._reward_feet_air_time()
    air_time_reset = env.feet_air_time[:, 0].clone()
    peak_height_reset = env.feet_swing_peak_height[:, 0].clone()
    air_time_continuous, _ = reward._reward_feet_air_time()
    checks.require(
        "reward/all_foot_air_time_completed_step",
        bool(
            torch.all(air_time_landing > 0.0)
            and torch.all(air_time_reset == 0.0)
            and torch.all(peak_height_reset == 0.0)
            and torch.all(air_time_continuous == 0.0)
        ),
        landing=float(air_time_landing.mean().item()),
        continuous=float(air_time_continuous.mean().item()),
    )

    zero_commands = 0
    straight_commands = 0
    lateral_commands = 0
    turn_commands = 0
    positive_turn_commands = 0
    negative_turn_commands = 0
    sample_count = 0
    ids = torch.arange(env.num_envs, device=env.device)
    for _ in range(100):
        env._resample_commands(ids)
        zero_commands += int((torch.abs(env.commands).sum(dim=1) == 0).sum().item())
        straight_mask = (
            torch.abs(env.commands[:, 0])
            >= float(env.cfg.commands.straight_line_min_abs_vx) - 1e-7
        ) & (torch.abs(env.commands[:, 1:]).sum(dim=1) <= 1e-7)
        straight_commands += int(straight_mask.sum().item())
        lateral_commands += int(
            (torch.abs(env.commands[:, 1]) > env.cfg.commands.lin_vel_y_clip)
            .sum()
            .item()
        )
        turn_mask = (torch.abs(env.commands[:, 0]) <= 1e-7) & (
            torch.abs(env.commands[:, 2])
            >= float(env.cfg.commands.turn_in_place_min_abs_yaw) - 1e-7
        )
        turn_commands += int(turn_mask.sum().item())
        positive_turn_commands += int(
            (turn_mask & (env.commands[:, 2] > 0.0)).sum().item()
        )
        negative_turn_commands += int(
            (turn_mask & (env.commands[:, 2] < 0.0)).sum().item()
        )
        sample_count += env.num_envs
    standing_fraction = zero_commands / sample_count
    expected_standing = float(env.cfg.commands.standing_probability)
    checks.require(
        "commands/explicit_standing_population",
        expected_standing - 0.08 <= standing_fraction <= expected_standing + 0.15,
        fraction=standing_fraction,
        configured=expected_standing,
    )
    turn_fraction = turn_commands / sample_count
    expected_turn = float(env.cfg.commands.turn_in_place_probability)
    checks.require(
        "commands/explicit_turn_in_place_population",
        expected_turn - 0.08 <= turn_fraction <= expected_turn + 0.08,
        fraction=turn_fraction,
        configured=expected_turn,
    )
    positive_turn_fraction = positive_turn_commands / max(turn_commands, 1)
    checks.require(
        "commands/turn_in_place_sign_balance",
        turn_commands > 0 and 0.30 <= positive_turn_fraction <= 0.70,
        positive_fraction=positive_turn_fraction,
        positive=positive_turn_commands,
        negative=negative_turn_commands,
    )
    straight_fraction = straight_commands / sample_count
    expected_straight = float(env.cfg.commands.straight_line_probability)
    checks.require(
        "commands/explicit_straight_population",
        expected_straight - 0.08
        <= straight_fraction
        <= expected_straight + 0.08,
        fraction=straight_fraction,
        configured=expected_straight,
    )
    checks.require(
        "commands/lateral_population_present",
        lateral_commands > 0,
        samples=lateral_commands,
        fraction=lateral_commands / sample_count,
        range=list(env.command_ranges["lin_vel_y"]),
    )

    env.episode_length_buf.fill_(51)
    env.last_contact_forces = torch.zeros_like(env.force_sensor_tensor)
    env.force_sensor_tensor.fill_(1.0)
    jerk_first, _ = reward._reward_feet_jerk()
    jerk_same, _ = reward._reward_feet_jerk()
    checks.require(
        "reward/feet_jerk_tracks_previous_force",
        bool(torch.all(jerk_first > 0) and torch.all(jerk_same == 0)),
        first=float(jerk_first.mean().item()),
        unchanged=float(jerk_same.mean().item()),
    )

    env.measured_heights.zero_()
    env.root_states[:, 2] = env.cfg.rewards.base_height_target
    height_best, _ = reward._reward_base_height()
    env.root_states[:, 2] += 0.10
    height_bad, _ = reward._reward_base_height()
    checks.require(
        "reward/terrain_relative_base_height",
        bool(torch.all(height_best < height_bad)),
        best=float(height_best.mean().item()),
        bad=float(height_bad.mean().item()),
    )



def probe_all_reward_functions(env, checks):
    env.reset()
    env.commands.zero_()
    env.commands[:, 0] = 0.10
    for name in sorted(dir(env.reward_container)):
        if not name.startswith("_reward_"):
            continue
        result = getattr(env.reward_container, name)()
        checks.require(
            f"reward_contract/{name}/tuple",
            isinstance(result, tuple) and len(result) == 2,
            result_type=type(result).__name__,
        )
        for channel, tensor in zip(("raw", "metric"), result):
            checks.require(
                f"reward_contract/{name}/{channel}_tensor",
                torch.is_tensor(tensor) and tensor.shape == (env.num_envs,),
                value_type=type(tensor).__name__,
                shape=list(tensor.shape) if torch.is_tensor(tensor) else None,
            )
            require_finite(checks, f"reward_contract/{name}/{channel}", tensor)


def probe_ik(env, checks):
    num_gripper = env.cfg.env.num_gripper_joints
    arm_slice = slice(-(6 + num_gripper), -num_gripper)
    lower = env.dof_pos_limits[arm_slice, 0]
    upper = env.dof_pos_limits[arm_slice, 1]
    command_start = ((lower + upper) * 0.5).repeat(env.num_envs, 1)

    dpose = torch.tensor(
        [0.03, -0.02, 0.01, 0.20, -0.10, 0.15],
        device=env.device,
        dtype=env.ee_pos.dtype,
    ).repeat(env.num_envs, 1).unsqueeze(-1)
    orientation_weight = float(env.cfg.arm.ik_orientation_weight)
    weighted_jacobian = torch.cat(
        [
            env.ee_j_eef[:, :3, :],
            orientation_weight * env.ee_j_eef[:, 3:, :],
        ],
        dim=1,
    )
    weighted_error = torch.cat(
        [
            dpose[:, :3, :],
            orientation_weight * dpose[:, 3:, :],
        ],
        dim=1,
    )
    weighted_jacobian_t = torch.transpose(weighted_jacobian, 1, 2)
    damping = torch.eye(6, device=env.device) * (0.05 ** 2)
    weighted_oracle = torch.bmm(
        weighted_jacobian_t,
        torch.linalg.solve(
            torch.bmm(weighted_jacobian, weighted_jacobian_t)
            + damping[None, ...],
            weighted_error,
        ),
    ).squeeze(-1)
    actual_delta = env._control_ik(dpose)
    require_finite(checks, "ik_delta", actual_delta)
    checks.require(
        "ik/full_6d_uses_weighted_jacobian",
        bool(torch.allclose(actual_delta, weighted_oracle, atol=1.0e-6, rtol=0.0)),
        max_abs_error=scalar_max_abs(actual_delta - weighted_oracle),
        orientation_weight=orientation_weight,
    )

    orientation_positive = torch.zeros_like(dpose)
    orientation_positive[:, 3:, 0] = torch.tensor(
        [0.20, -0.10, 0.15], device=env.device
    )
    orientation_negative = -orientation_positive
    positive_delta = env._control_ik(orientation_positive)
    negative_delta = env._control_ik(orientation_negative)
    checks.require(
        "ik/orientation_changes_joint_target",
        scalar_max_abs(positive_delta - negative_delta) > 1.0e-4,
        max_abs_difference=scalar_max_abs(positive_delta - negative_delta),
    )

    large_dpose = torch.tensor(
        [10.0, -7.0, 5.0, 1.0, -0.8, 0.6],
        device=env.device,
        dtype=env.ee_pos.dtype,
    ).repeat(env.num_envs, 1).unsqueeze(-1)
    scaled_delta = float(env.cfg.arm.ik_gain) * env._control_ik(large_dpose)
    limited_delta = torch.clamp(
        scaled_delta,
        -float(env.cfg.arm.target_max_step),
        float(env.cfg.arm.target_max_step),
    )
    expected_first = torch.clamp(command_start + limited_delta, lower, upper)
    expected_second = torch.clamp(expected_first + limited_delta, lower, upper)
    saved_command = env.arm_q_command.clone()
    try:
        env.arm_q_command.copy_(command_start)
        actual_first = env._compute_arm_position_targets(large_dpose).clone()
        actual_second = env._compute_arm_position_targets(large_dpose).clone()
        actual_command = env.arm_q_command.clone()
    finally:
        env.arm_q_command.copy_(saved_command)
    require_finite(checks, "arm_q_target", actual_first)
    checks.require(
        "ik/joint_limits",
        bool(
            torch.all(actual_first >= lower - 1.0e-7)
            and torch.all(actual_first <= upper + 1.0e-7)
            and torch.all(actual_second >= lower - 1.0e-7)
            and torch.all(actual_second <= upper + 1.0e-7)
        ),
    )
    checks.require(
        "ik/rate_limit_is_active",
        scalar_max_abs(scaled_delta) > float(env.cfg.arm.target_max_step)
        and scalar_max_abs(actual_first - command_start)
        <= float(env.cfg.arm.target_max_step) + 1.0e-7,
        raw_scaled_max=scalar_max_abs(scaled_delta),
        applied_max=scalar_max_abs(actual_first - command_start),
        limit=float(env.cfg.arm.target_max_step),
    )
    checks.require(
        "ik/persistent_command_accumulates",
        scalar_max_abs(expected_second - expected_first) > 1.0e-7
        and bool(torch.allclose(actual_first, expected_first, atol=1.0e-7, rtol=0.0))
        and bool(torch.allclose(actual_second, expected_second, atol=1.0e-7, rtol=0.0))
        and bool(torch.allclose(actual_command, expected_second, atol=1.0e-7, rtol=0.0)),
        first_max_error=scalar_max_abs(actual_first - expected_first),
        second_max_error=scalar_max_abs(actual_second - expected_second),
        command_max_error=scalar_max_abs(actual_command - expected_second),
    )

    saved_orientation = env.curr_ee_goal_orn_rpy.clone()
    commanded_orientation = torch.tensor(
        [0.20, 1.10, -0.30], device=env.device
    ).repeat(env.num_envs, 1)
    env.curr_ee_goal_orn_rpy.copy_(commanded_orientation)
    env.compute_observations()
    observed_orientation = env.obs_buf[:, 63:66]
    checks.require(
        "observation/orientation_command_is_live",
        bool(
            torch.allclose(
                observed_orientation,
                commanded_orientation,
                atol=1.0e-7,
                rtol=0.0,
            )
            and scalar_max_abs(observed_orientation) > 1.0e-3
        ),
        max_abs_error=scalar_max_abs(
            observed_orientation - commanded_orientation
        ),
    )
    env.curr_ee_goal_orn_rpy.copy_(saved_orientation)

    target_delta = torch.tensor(
        [0.20, -0.10, 0.30], device=env.device
    ).repeat(env.num_envs, 1)
    env.ee_start_orn_delta_rpy.zero_()
    env.ee_goal_orn_delta_rpy.copy_(target_delta)
    env.goal_timer.copy_(0.5 * env.traj_timesteps)
    env._update_curr_ee_goal()
    checks.require(
        "ik/orientation_target_interpolates_without_jump",
        bool(
            torch.allclose(
                env.curr_ee_goal_orn_delta_rpy,
                0.5 * target_delta,
                atol=1.0e-6,
                rtol=0.0,
            )
        ),
        max_abs_error=scalar_max_abs(
            env.curr_ee_goal_orn_delta_rpy - 0.5 * target_delta
        ),
    )
    env.reset()


def probe_reset(env, checks):
    ids = torch.arange(env.num_envs, device=env.device)
    env.actions.fill_(0.4)
    env.torques.fill_(2.0)
    env.last_actions.fill_(0.3)
    env.last_torques.fill_(1.0)
    env.gait_indices.fill_(0.6)
    env.clock_inputs.fill_(0.7)
    env.obs_history_buf.fill_(0.8)
    env.action_history_buf.fill_(0.9)
    env.desired_contact_states.zero_()
    env.last_contact_forces = torch.ones_like(env.force_sensor_tensor)
    env.reset_idx(ids, start=True)
    zero_tensors = {
        "actions": env.actions,
        "torques": env.torques,
        "last_actions": env.last_actions,
        "last_torques": env.last_torques,
        "gait_indices": env.gait_indices,
        "clock_inputs": env.clock_inputs,
        "history": env.obs_history_buf,
        "action_history": env.action_history_buf,
        "last_contact_forces": env.last_contact_forces,
    }
    for name, tensor in zero_tensors.items():
        checks.require(f"reset/{name}_cleared", scalar_max_abs(tensor) == 0.0, max_abs=scalar_max_abs(tensor))
    checks.require(
        "reset/desired_contacts_all_stance",
        bool(torch.all(env.desired_contact_states == 1.0)),
    )
    num_gripper = env.cfg.env.num_gripper_joints
    arm_slice = slice(-(6 + num_gripper), -num_gripper)
    gripper_upper = env.dof_pos_limits[-num_gripper:, 1].repeat(env.num_envs, 1)
    checks.require(
        "reset/arm_command_matches_measured_state",
        bool(
            torch.allclose(
                env.arm_q_command, env.dof_pos[:, arm_slice], atol=1.0e-7, rtol=0.0
            )
        ),
        max_abs_error=scalar_max_abs(
            env.arm_q_command - env.dof_pos[:, arm_slice]
        ),
    )
    checks.require(
        "reset/gripper_is_open_and_commanded_open",
        bool(
            torch.allclose(
                env.dof_pos[:, -num_gripper:], gripper_upper, atol=1.0e-7, rtol=0.0
            )
            and torch.allclose(
                env.gripper_q_target, gripper_upper, atol=1.0e-7, rtol=0.0
            )
        ),
        measured=env.dof_pos[0, -num_gripper:].detach().cpu().tolist(),
        commanded=env.gripper_q_target[0].detach().cpu().tolist(),
        upper=gripper_upper[0].detach().cpu().tolist(),
    )


def probe_training_metadata(env, checks):
    env.global_steps = 123
    metadata = env.get_training_metadata()
    alignment = metadata["go2x5_alignment"]
    checks.require(
        "checkpoint/simple_runtime_contract",
        alignment["num_proprio"] == 66
        and alignment["num_observations"] == 744
        and alignment["observe_gait_commands"] is False
        and alignment["control_contract"]["replace_cylinder_with_capsule"] is False
        and alignment["control_contract"]["arm_target_mode"]
        == "persistent_joint_command"
        and alignment["control_contract"]["arm_target_max_step"] == 0.08
        and alignment["control_contract"]["gripper_hold_mode"]
        == "open_upper_limit"
        and alignment["control_contract"]["track_ee_orientation"] is True
        and alignment["control_contract"]["ik_task"]
        == "pose_6d_weighted_dls"
        and alignment["control_contract"]["ik_orientation_weight"]
        == go2x5_robot_spec.ARM_IK_ORIENTATION_WEIGHT
        and alignment["control_contract"]["arm_position_stiffness"]
        == go2x5_robot_spec.ARM_POS_STIFFNESS
        and alignment["control_contract"]["arm_position_damping"]
        == go2x5_robot_spec.ARM_POS_DAMPING
        and alignment["control_contract"]["ee_goal_ranges"]
        == go2x5_robot_spec.EE_GOAL_LOCAL_RANGES
        and alignment["control_contract"]["ee_goal_max_nominal_reach_radius"]
        == go2x5_robot_spec.EE_GOAL_MAX_NOMINAL_REACH_RADIUS
        and alignment["control_contract"]["ee_orientation_observation"]
        == "local_rpy"
        and alignment["control_contract"]["command_ranges"]["lin_vel_y"]
        == [-0.10, 0.10]
        and alignment["control_contract"]["command_dead_zone"]["lin_vel_y"]
        == 0.05
        and "gait_frequency" not in alignment["control_contract"],
    )
    env.global_steps = 0
    env.load_training_metadata(metadata)
    checks.require("checkpoint/valid_metadata_loads", env.global_steps == 123)

    wrong_action = copy.deepcopy(metadata)
    wrong_action["go2x5_alignment"]["action_dim"] = 18
    try:
        env.load_training_metadata(wrong_action)
    except RuntimeError as error:
        checks.require("checkpoint/rejects_wrong_action_dim", "action_dim" in str(error))
    else:
        checks.require("checkpoint/rejects_wrong_action_dim", False)

    corrupt_contract = copy.deepcopy(metadata)
    corrupt_contract["go2x5_alignment"]["control_contract"]["ik_gain"] = 999.0
    try:
        env.load_training_metadata(corrupt_contract)
    except RuntimeError as error:
        checks.require("checkpoint/rejects_corrupt_contract", "hash is corrupt" in str(error))
    else:
        checks.require("checkpoint/rejects_corrupt_contract", False)

    wrong_profile = copy.deepcopy(metadata)
    wrong_profile["go2x5_alignment"]["curriculum"]["profile_name"] = (
        "go2x5_stable_reach_curriculum_v3_flat_step_metrics"
    )
    try:
        env.load_training_metadata(wrong_profile)
    except RuntimeError as error:
        checks.require("checkpoint/rejects_old_curriculum_profile", "profile mismatch" in str(error))
    else:
        checks.require("checkpoint/rejects_old_curriculum_profile", False)

    missing_steps = copy.deepcopy(metadata)
    del missing_steps["go2x5_alignment"]["training_state"]
    try:
        env.load_training_metadata(missing_steps)
    except RuntimeError as error:
        checks.require("checkpoint/rejects_missing_global_steps", "global_steps" in str(error))
    else:
        checks.require("checkpoint/rejects_missing_global_steps", False)
    env.load_training_metadata(metadata)


def probe_curriculum(env, checks):
    checks.require(
        "curriculum/static_distribution",
        not env.auto_curriculum_enabled
        and len(env.curriculum_stages) == 0
        and env.curriculum_profile_name == "go2x5_flat_tabletop_6d_walk_v4"
        and env.command_ranges["lin_vel_x"] == [-0.30, 0.30]
        and env.command_ranges["lin_vel_y"] == [-0.10, 0.10]
        and env.command_ranges["ang_vel_yaw"] == [-0.25, 0.25]
        and abs(float(env.cfg.commands.straight_line_min_abs_vx) - 0.15)
        <= 1e-9
        and [
            list(env.goal_ee_ranges[axis]) for axis in ("pos_x", "pos_y_cart", "pos_z")
        ] == go2x5_robot_spec.EE_GOAL_LOCAL_RANGES
        and abs(float(env.arm_reward_scales["tracking_ee_world"]) - 2.0)
        <= 1e-9,
        enabled=bool(env.auto_curriculum_enabled),
        profile=env.curriculum_profile_name,
        command_ranges={
            axis: list(env.command_ranges[axis])
            for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
        },
        straight_line_min_abs_vx=float(
            env.cfg.commands.straight_line_min_abs_vx
        ),
        goal_ranges=[
            list(env.goal_ee_ranges[axis]) for axis in ("pos_x", "pos_y_cart", "pos_z")
        ],
    )

    robot_x = go2x5_robot_spec.HIGH_LEVEL_ROBOT_START_POSE[0]
    table_center_x = go2x5_robot_spec.HIGH_LEVEL_TABLE_POSITION_XY[0]
    table_near_edge_x = (
        table_center_x - go2x5_robot_spec.HIGH_LEVEL_TABLE_DIMS[0] / 2.0
    )
    object_root_x = [
        table_center_x + value - robot_x
        for value in go2x5_robot_spec.HIGH_LEVEL_OBJECT_POSITION_RANGE_X
    ]
    grasp_height = [
        go2x5_robot_spec.HIGH_LEVEL_TABLE_HEIGHT_RANGE[0],
        go2x5_robot_spec.HIGH_LEVEL_TABLE_HEIGHT_RANGE[1]
        + go2x5_robot_spec.HIGH_LEVEL_MAX_OBJECT_HEIGHT,
    ]
    world_ranges = go2x5_robot_spec.EE_GOAL_WORLD_RANGES
    tabletop_local_corners = torch.tensor(
        [
            [
                x - go2x5_robot_spec.EE_GOAL_CENTER_OFFSET[0],
                y,
                z - go2x5_robot_spec.EE_GOAL_CENTER_OFFSET[2],
            ]
            for x in object_root_x
            for y in go2x5_robot_spec.HIGH_LEVEL_OBJECT_POSITION_RANGE_Y
            for z in grasp_height
        ],
        dtype=torch.float64,
    )
    tabletop_max_radius = float(
        torch.linalg.vector_norm(tabletop_local_corners, dim=-1).max().item()
    )
    checks.require(
        "task_geometry/tabletop_volume_covered",
        abs((table_near_edge_x - robot_x) - 0.30) <= 1.0e-9
        and world_ranges[0][0] <= min(object_root_x)
        and max(object_root_x) <= world_ranges[0][1]
        and world_ranges[1][0]
        <= go2x5_robot_spec.HIGH_LEVEL_OBJECT_POSITION_RANGE_Y[0]
        <= go2x5_robot_spec.HIGH_LEVEL_OBJECT_POSITION_RANGE_Y[1]
        <= world_ranges[1][1]
        and world_ranges[2][0] <= grasp_height[0]
        and grasp_height[1] <= world_ranges[2][1]
        and tabletop_max_radius
        <= go2x5_robot_spec.EE_GOAL_MAX_NOMINAL_REACH_RADIUS,
        table_near_edge_distance=table_near_edge_x - robot_x,
        object_root_x=object_root_x,
        object_y=go2x5_robot_spec.HIGH_LEVEL_OBJECT_POSITION_RANGE_Y,
        grasp_height=grasp_height,
        ee_world_ranges=world_ranges,
        tabletop_max_nominal_radius=tabletop_max_radius,
        ee_max_nominal_reach_radius=(
            go2x5_robot_spec.EE_GOAL_MAX_NOMINAL_REACH_RADIUS
        ),
    )
    orientation_ranges = go2x5_robot_spec.EE_ORIENTATION_ABSOLUTE_RANGES
    nominal = go2x5_robot_spec.EE_ORIENTATION_NOMINAL_RPY
    checks.require(
        "task_geometry/x5_orientation_volume",
        all(
            bounds[0] <= value <= bounds[1]
            for bounds, value in zip(orientation_ranges, nominal)
        )
        and all(bounds[1] - bounds[0] >= 0.5 for bounds in orientation_ranges),
        nominal=nominal,
        ranges=orientation_ranges,
    )


def runtime_tensors(env):
    return {
        "observation": env.obs_buf,
        "history": env.obs_history_buf,
        "policy_action": env.actions,
        "leg_torque": env.torques[:, :12],
        "root_state": env.root_states,
        "dof_state": env.dof_state,
        "ee_pose": env.rigid_body_state[:, env.gripper_idx, :7],
        "ee_target": env.curr_ee_goal_cart_world,
        "ee_orientation_target": env.ee_goal_orn_quat,
        "ee_orientation_command": env.curr_ee_goal_orn_rpy,
        "jacobian": env.ee_j_eef,
        "arm_q_command": env.arm_q_command,
        "gripper_q_target": env.gripper_q_target,
        "contact_forces": env.contact_forces,
        "leg_reward": env.rew_buf,
        "arm_reward": env.arm_rew_buf,
        "measured_heights": env.measured_heights,
    }


def probe_rollout(env, checks, steps):
    env.reset()
    checks.require("runtime/observation_shape", env.obs_buf.shape[1] == 744, shape=list(env.obs_buf.shape))
    checks.require("runtime/action_shape", env.num_actions == 12, action_dim=int(env.num_actions))
    checks.require(
        "runtime/measured_heights_shape",
        env.measured_heights.ndim == 2 and env.measured_heights.shape[0] == env.num_envs,
        shape=list(env.measured_heights.shape),
    )

    probe = torch.tensor(
        [0.011, -0.007, 0.003, -0.005, 0.009, -0.002, 0.004, -0.008, 0.006, -0.003, 0.010, -0.001],
        device=env.device,
    ).repeat(env.num_envs, 1)
    early_resets = 0
    reset_causes = {"roll": 0, "pitch": 0, "z": 0, "contact": 0}
    first_early_reset = None
    max_abs = {name: 0.0 for name in runtime_tensors(env)}
    nonfinite = {name: 0 for name in runtime_tensors(env)}
    max_foot_velocity_cache_error = 0.0
    max_foot_position_cache_error = 0.0
    for step in range(steps):
        if step < 20:
            env.commands.zero_()
            env.commands[:, 0] = 0.10
        elif step < 30:
            env.commands.zero_()
        actions = probe if step % 2 == 0 else -probe
        env.step(actions)
        live_foot_positions = torch.index_select(
            env.rigid_body_state[:, :, 0:3], 1, env.feet_indices
        )
        live_foot_velocities = torch.index_select(
            env.rigid_body_state[:, :, 7:10], 1, env.feet_indices
        )
        max_foot_position_cache_error = max(
            max_foot_position_cache_error,
            scalar_max_abs(env.foot_positions - live_foot_positions),
        )
        max_foot_velocity_cache_error = max(
            max_foot_velocity_cache_error,
            scalar_max_abs(env.foot_velocities - live_foot_velocities),
        )
        non_timeout = env.reset_buf.bool() & ~env.time_out_buf.bool()
        if step < env.max_episode_length:
            count = int(non_timeout.sum().item())
            early_resets += count
            if count and first_early_reset is None:
                first_early_reset = {
                    "step": step,
                    "env": int(torch.nonzero(non_timeout, as_tuple=False)[0].item()),
                }
            reset_causes["roll"] += int((non_timeout & env.reset_roll_buf).sum().item())
            reset_causes["pitch"] += int((non_timeout & env.reset_pitch_buf).sum().item())
            reset_causes["z"] += int((non_timeout & env.reset_z_buf).sum().item())
            reset_causes["contact"] += int((non_timeout & env.reset_contact_buf).sum().item())
        for name, tensor in runtime_tensors(env).items():
            finite = torch.isfinite(tensor)
            count = int((~finite).sum().item())
            nonfinite[name] += count
            if count:
                checks.require(
                    f"finite/rollout/{name}",
                    False,
                    step=step,
                    nonfinite=count,
                    first_index=torch.nonzero(~finite, as_tuple=False)[0].tolist(),
                )
            max_abs[name] = max(max_abs[name], scalar_max_abs(tensor))
        if step in (19, 29):
            checks.require(
                f"gait/disabled_state_is_constant_step_{step}",
                bool(torch.all(env.gait_indices == 0.0) and torch.all(env.desired_contact_states == 1.0)),
            )
    for name, count in nonfinite.items():
        checks.require(f"finite/rollout/{name}", count == 0, nonfinite=count)
    checks.require(
        "runtime/foot_kinematics_refreshed_each_tick",
        max_foot_position_cache_error == 0.0 and max_foot_velocity_cache_error == 0.0,
        position_max_error=max_foot_position_cache_error,
        velocity_max_error=max_foot_velocity_cache_error,
    )
    checks.require(
        "runtime/no_early_reset",
        early_resets == 0,
        count=early_resets,
        causes=reset_causes,
        first=first_early_reset,
    )
    return max_abs, {
        "distribution": "static_full_task",
        "early_resets": early_resets,
        "reset_causes": reset_causes,
        "first_early_reset": first_early_reset,
        "foot_position_cache_max_error": max_foot_position_cache_error,
        "foot_velocity_cache_max_error": max_foot_velocity_cache_error,
    }


def run(cli):
    torch.manual_seed(cli.seed)
    checks = Checks()
    args = make_env_args(cli)
    env_cfg, _ = task_registry.get_cfgs(name="go2x5")
    env_cfg = configure_env(env_cfg, cli)
    env, _ = task_registry.make_env(name="go2x5", args=args, env_cfg=env_cfg)
    env.reset()

    report = {
        "schema_version": 2,
        "task": "go2x5_lowlevel_training_readiness",
        "num_envs": cli.num_envs,
        "steps": cli.steps,
        "distribution": "static_full_task",
        "seed": cli.seed,
        "checks": checks.items,
        "passed": False,
    }
    try:
        probe_plane_ground(env, checks)
        probe_rewards(env, checks)
        probe_all_reward_functions(env, checks)
        probe_ik(env, checks)
        probe_reset(env, checks)
        probe_training_metadata(env, checks)
        probe_curriculum(env, checks)
        report["max_abs"], report["rollout"] = probe_rollout(
            env,
            checks,
            cli.steps,
        )
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
