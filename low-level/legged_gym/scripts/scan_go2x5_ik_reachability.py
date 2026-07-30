#!/usr/bin/env python3
"""Headless IK reachability grid for the Go2-X5 arm.

The scan is not a training run. It fixes the base, starts from the canonical
home pose, sweeps x/y/z/rpy targets in the selected target frame, runs damped
least-squares IK, and writes per-target reachability statistics.
"""

import argparse
import csv
import json
import math
import os
import sys
from itertools import product


_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_LOW_LEVEL_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_LOW_LEVEL_ROOT, ".."))
_ISAACGYM_BINDINGS_DIR = os.path.join(
    _REPO_ROOT, "third_party", "isaacgym", "python", "isaacgym", "_bindings", "linux-x86_64"
)
_ISAACGYM_USD_PLUGIN_DIR = os.path.join(_ISAACGYM_BINDINGS_DIR, "usd", "plugins")

_ld_library_paths = [_ISAACGYM_BINDINGS_DIR, _ISAACGYM_USD_PLUGIN_DIR]
if os.environ.get("CONDA_PREFIX"):
    _ld_library_paths.append(os.path.join(os.environ["CONDA_PREFIX"], "lib"))

_existing_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
for _path in reversed(_ld_library_paths):
    if _path and _path not in _existing_ld_library_path.split(":"):
        _existing_ld_library_path = f"{_path}:{_existing_ld_library_path}" if _existing_ld_library_path else _path
os.environ["LD_LIBRARY_PATH"] = _existing_ld_library_path

_bootstrap_flag = "_ISAACGYM_LIBRARY_PATH_BOOTSTRAPPED"
if os.environ.get(_bootstrap_flag) != "1":
    os.environ[_bootstrap_flag] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)

for _path in [
    _LOW_LEVEL_ROOT,
    os.path.join(_REPO_ROOT, "third_party", "isaacgym", "python"),
    os.path.join(_REPO_ROOT, "third_party", "rsl_rl"),
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import isaacgym  # noqa: F401,E402
from isaacgym import gymtorch  # noqa: E402
from isaacgym.torch_utils import (  # noqa: E402
    euler_from_quat,
    orientation_error,
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
)
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.envs.manip_loco import go2x5_robot_spec as robot_spec  # noqa: E402
from legged_gym.utils import get_args, task_registry  # noqa: E402


def parse_float_list(text):
    return [float(item) for item in text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="go2x5")
    parser.add_argument(
        "--target_frame",
        choices=["goal_center", "arm_base"],
        default="goal_center",
        help="Frame for target x/y/z. goal_center matches the low-level terrain-invariant task sampler.",
    )
    parser.add_argument("--quick", action="store_true", help="Use a small 27-point smoke-test grid.")
    parser.add_argument("--x", default="0.215,0.33,0.45,0.565")
    parser.add_argument("--y", default="-0.225,-0.11,0.0,0.11,0.225")
    parser.add_argument("--z", default="-0.364,-0.23,-0.10,0.036")
    parser.add_argument("--roll", default="-0.35,0.0,0.35")
    parser.add_argument("--pitch", default="-0.25,0.0,0.25")
    parser.add_argument("--yaw", default="-0.35,0.0,0.35")
    parser.add_argument(
        "--orientation_mode",
        choices=["task", "absolute"],
        default="task",
        help=(
            "task interprets roll/pitch/yaw as deltas around the production "
            "X5 pose and adds target-bearing yaw; absolute uses the values directly."
        ),
    )
    parser.add_argument("--max_targets", type=int, default=0, help="Optional cap for debugging; 0 scans all.")
    parser.add_argument("--print_every", type=int, default=1, help="Print every N targets; 0 only prints summary.")
    parser.add_argument("--max_iters", type=int, default=40)
    parser.add_argument("--ik_gain", type=float, default=0.7)
    parser.add_argument("--max_delta", type=float, default=0.08, help="Maximum per-iteration arm joint update in radians.")
    parser.add_argument("--orn_weight", type=float, default=0.25, help="Orientation residual weight during IK.")
    parser.add_argument("--position_only", action="store_true", help="Optimize position only; still report orientation error.")
    parser.add_argument(
        "--base_height",
        type=float,
        default=robot_spec.BASE_INIT_HEIGHT,
        help="Fixed base-root height used by the kinematic scan.",
    )
    parser.add_argument(
        "--base_pitch",
        type=float,
        default=0.0,
        help="Fixed base pitch in radians; positive values model the configured forward lean.",
    )
    parser.add_argument("--pos_tol", type=float, default=0.03)
    parser.add_argument("--orn_tol", type=float, default=0.25)
    parser.add_argument("--limit_margin", type=float, default=0.03)
    parser.add_argument("--contact_threshold", type=float, default=1.0)
    parser.add_argument("--count_finger_contacts", action="store_true", help="Count arm_link7/8 contact forces as collisions.")
    parser.add_argument("--settle_steps", type=int, default=2)
    parser.add_argument(
        "--jacobian_body_offset",
        type=int,
        default=-1,
        help="Offset from rigid-body EE index to Isaac Gym Jacobian body index. Floating-base assets usually need -1.",
    )
    parser.add_argument("--no_auto_jacobian_body", action="store_true", help="Disable finite-difference Jacobian body index selection.")
    parser.add_argument("--csv", default="/tmp/go2x5_ik_reachability.csv")
    parser.add_argument("--summary_json", default="/tmp/go2x5_ik_reachability_summary.json")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    return parser.parse_args()


def make_legged_gym_args(cli):
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = get_args(test=True)
    finally:
        sys.argv = original_argv
    args.task = cli.task
    args.headless = True
    args.sim_device = cli.sim_device
    args.rl_device = cli.rl_device
    args.graphics_device_id = cli.graphics_device_id
    args.observe_gait_commands = False
    return args


def configure_env(env_cfg, cli):
    env_cfg.env.num_envs = 1
    env_cfg.env.record_video = False
    env_cfg.env.stand_by = True
    env_cfg.env.teleop_mode = False
    env_cfg.env.observe_gait_commands = False
    env_cfg.env.reorder_dofs = True
    env_cfg.asset.fix_base_link = True
    env_cfg.asset.disable_gravity = True

    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.height = [0.0, 0.0]

    env_cfg.noise.add_noise = False
    env_cfg.noise.noise_level = 0.0
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.init_state.rand_yaw_range = 0.0
    env_cfg.init_state.origin_perturb_range = 0.0
    env_cfg.init_state.init_vel_perturb_range = 0.0
    env_cfg.init_state.leg_reset_ratio_range = [1.0, 1.0]
    env_cfg.init_state.arm_reset_noise_range = [0.0, 0.0]
    env_cfg.init_state.pos = [0.0, 0.0, cli.base_height]
    env_cfg.init_state.rot = [
        0.0,
        math.sin(cli.base_pitch / 2.0),
        0.0,
        math.cos(cli.base_pitch / 2.0),
    ]

    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.standing_probability = 1.0
    env_cfg.commands.straight_line_probability = 0.0
    env_cfg.commands.turn_in_place_probability = 0.0
    env_cfg.commands.turn_in_place_min_abs_yaw = 0.0
    if hasattr(env_cfg, "auto_curriculum"):
        env_cfg.auto_curriculum.enabled = False
    env_cfg.arm.track_ee_orientation = not cli.position_only
    env_cfg.arm.ik_orientation_weight = cli.orn_weight
    return env_cfg


def refresh_runtime_state(env, jacobian_body_offset=-1):
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_jacobian_tensors(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.base_quat[:] = env.root_states[:, 3:7]
    _, _, yaw = euler_from_quat(env.base_quat)
    zeros = torch.zeros_like(yaw)
    env.base_yaw_quat[:] = quat_from_euler_xyz(zeros, zeros, yaw)
    arm_start = env.num_dofs - (6 + env.cfg.env.num_gripper_joints)
    arm_end = env.num_dofs - env.cfg.env.num_gripper_joints
    jacobian_body_idx = getattr(env, "_jacobian_body_idx", env.gripper_idx + jacobian_body_offset)
    env.ee_j_eef = env.jacobian_whole[:, jacobian_body_idx, :6, arm_start:arm_end]


def set_dof_positions(env, q, jacobian_body_offset=-1, simulate_steps=1):
    dof_state_view = env.dof_state.view(env.num_envs, env.num_dofs, 2)
    dof_state_view[0, :, 0] = q
    dof_state_view[0, :, 1] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(q.view(1, -1)))
    for _ in range(max(0, simulate_steps)):
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
    refresh_runtime_state(env, jacobian_body_offset=jacobian_body_offset)


def settle(env, steps, jacobian_body_offset=-1):
    for _ in range(max(0, steps)):
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
    refresh_runtime_state(env, jacobian_body_offset=jacobian_body_offset)


def target_world_from_frame(env, local_target, target_frame):
    local = torch.tensor(local_target, device=env.device, dtype=torch.float).view(1, 3)
    if target_frame == "goal_center":
        center = env._get_ee_goal_spherical_center()[0:1]
        return center + quat_apply(env.base_yaw_quat[0:1], local)

    root_quat = env.root_states[0:1, 3:7]
    root_pos = env.root_states[0:1, :3]
    arm_base_local = torch.tensor(robot_spec.ARM_BASE_OFFSET, device=env.device, dtype=torch.float).view(1, 3)
    return root_pos + quat_apply(root_quat, arm_base_local) + quat_apply(root_quat, local)


def contact_summary(env, threshold):
    forces = torch.norm(env.contact_forces[0], dim=-1).detach().cpu()
    exclude = {env.body_names_to_idx.get("base", -1)}
    exclude.update(int(idx) for idx in env.feet_indices.detach().cpu().tolist())
    if not getattr(env, "_count_finger_contacts", False):
        exclude.update(env.body_names_to_idx[name] for name in robot_spec.FINGER_BODY_NAMES if name in env.body_names_to_idx)
    contacts = []
    for idx, force in enumerate(forces.tolist()):
        if idx in exclude or force <= threshold:
            continue
        contacts.append((env.body_names[idx], force))
    max_force = max([force for _, force in contacts], default=0.0)
    return contacts, max_force


def local_ee_position(env, target_frame):
    if target_frame == "goal_center":
        center = env._get_ee_goal_spherical_center()[0:1]
        return quat_rotate_inverse(env.base_yaw_quat[0:1], env.ee_pos[0:1] - center)[0]

    root_quat = env.root_states[0:1, 3:7]
    root_pos = env.root_states[0:1, :3]
    arm_base_local = torch.tensor(robot_spec.ARM_BASE_OFFSET, device=env.device, dtype=torch.float).view(1, 3)
    arm_base_world = root_pos + quat_apply(root_quat, arm_base_local)
    return quat_rotate_inverse(root_quat, env.ee_pos[0:1] - arm_base_world)[0]


def local_ee_axes(env):
    root_quat = env.root_states[0:1, 3:7].repeat(3, 1)
    ee_quat = env.ee_orn[0:1] / torch.norm(env.ee_orn[0:1], dim=-1, keepdim=True).clamp(min=1e-6)
    basis = torch.eye(3, device=env.device, dtype=torch.float)
    axes_world = quat_apply(ee_quat.repeat(3, 1), basis)
    axes_local = quat_rotate_inverse(root_quat, axes_world)
    return {
        "x": [float(value) for value in axes_local[0].detach().cpu().tolist()],
        "y": [float(value) for value in axes_local[1].detach().cpu().tolist()],
        "z": [float(value) for value in axes_local[2].detach().cpu().tolist()],
    }


def select_jacobian_body_index(env, cli):
    arm_start = env.num_dofs - (6 + env.cfg.env.num_gripper_joints)
    arm_end = env.num_dofs - env.cfg.env.num_gripper_joints
    home_q = env.default_dof_pos.detach().clone()
    eps = 1.0e-3
    fd_columns = []
    set_dof_positions(env, home_q, jacobian_body_offset=cli.jacobian_body_offset)
    for joint_idx in range(arm_start, arm_end):
        q_plus = home_q.clone()
        q_minus = home_q.clone()
        q_plus[joint_idx] += eps
        q_minus[joint_idx] -= eps
        set_dof_positions(env, q_plus, jacobian_body_offset=cli.jacobian_body_offset)
        pos_plus = env.ee_pos[0].detach().clone()
        set_dof_positions(env, q_minus, jacobian_body_offset=cli.jacobian_body_offset)
        pos_minus = env.ee_pos[0].detach().clone()
        fd_columns.append((pos_plus - pos_minus) / (2.0 * eps))
    fd = torch.stack(fd_columns, dim=1)
    set_dof_positions(env, home_q, jacobian_body_offset=cli.jacobian_body_offset)
    refresh_runtime_state(env, jacobian_body_offset=cli.jacobian_body_offset)

    errors = []
    for body_idx in range(env.jacobian_whole.shape[1]):
        jac = env.jacobian_whole[0, body_idx, :3, arm_start:arm_end]
        errors.append((float(torch.norm(jac - fd).item()), body_idx))
    errors.sort()
    best_error, best_idx = errors[0]
    env._jacobian_body_idx = best_idx
    refresh_runtime_state(env, jacobian_body_offset=cli.jacobian_body_offset)
    return best_idx, best_error, errors[:5]


def task_target_rpy(target_xyz, orientation_command, cli):
    if cli.orientation_mode == "absolute":
        return list(orientation_command)
    target_rpy = [
        robot_spec.EE_ORIENTATION_NOMINAL_RPY[index]
        + orientation_command[index]
        for index in range(3)
    ]
    target_rpy[2] += math.atan2(target_xyz[1], target_xyz[0])
    return target_rpy


def scan_target(env, target_xyz, orientation_command, cli):
    home_q = env.default_dof_pos.detach().clone()
    arm_start = env.num_dofs - (6 + env.cfg.env.num_gripper_joints)
    arm_end = env.num_dofs - env.cfg.env.num_gripper_joints
    lower = env.dof_pos_limits[arm_start:arm_end, 0]
    upper = env.dof_pos_limits[arm_start:arm_end, 1]

    q = home_q.clone()
    set_dof_positions(env, q, jacobian_body_offset=cli.jacobian_body_offset)
    target_world = target_world_from_frame(env, target_xyz, cli.target_frame)
    target_rpy = task_target_rpy(target_xyz, orientation_command, cli)
    target_local_quat = quat_from_euler_xyz(
        torch.tensor([target_rpy[0]], device=env.device),
        torch.tensor([target_rpy[1]], device=env.device),
        torch.tensor([target_rpy[2]], device=env.device),
    )
    target_quat = quat_mul(env.base_yaw_quat[0:1], target_local_quat)

    pos_err = float("inf")
    orn_err = float("inf")
    iters = 0
    for iters in range(1, cli.max_iters + 1):
        refresh_runtime_state(env, jacobian_body_offset=cli.jacobian_body_offset)
        ee_orn_norm = torch.norm(env.ee_orn, dim=-1, keepdim=True).clamp(min=1e-6)
        dpos = target_world - env.ee_pos[0:1]
        drot = orientation_error(target_quat, env.ee_orn[0:1] / ee_orn_norm[0:1])
        ik_drot = torch.zeros_like(drot) if cli.position_only else drot
        dpose = torch.cat([dpos, ik_drot], dim=-1).unsqueeze(-1)
        pos_err = torch.norm(dpos).item()
        orn_err = torch.norm(drot).item()
        if pos_err <= cli.pos_tol and (cli.position_only or orn_err <= cli.orn_tol):
            break

        delta = torch.clamp(cli.ik_gain * env._control_ik(dpose)[0], -cli.max_delta, cli.max_delta)
        q = env.dof_pos[0].detach().clone()
        q[arm_start:arm_end] = torch.clamp(q[arm_start:arm_end] + delta, lower, upper)
        set_dof_positions(env, q, jacobian_body_offset=cli.jacobian_body_offset)

    settle(env, cli.settle_steps, jacobian_body_offset=cli.jacobian_body_offset)
    refresh_runtime_state(env, jacobian_body_offset=cli.jacobian_body_offset)

    final_local = local_ee_position(env, cli.target_frame)
    final_arm_q = env.dof_pos[0, arm_start:arm_end].detach().clone()
    margin = torch.minimum(final_arm_q - lower, upper - final_arm_q)
    limit_hits = int((margin < cli.limit_margin).sum().item())
    contacts, max_contact_force = contact_summary(env, cli.contact_threshold)

    pos_err = torch.norm(target_world - env.ee_pos[0:1]).item()
    ee_orn_norm = torch.norm(env.ee_orn, dim=-1, keepdim=True).clamp(min=1e-6)
    orn_err = torch.norm(orientation_error(target_quat, env.ee_orn[0:1] / ee_orn_norm[0:1])).item()
    raw_ik_success = pos_err <= cli.pos_tol and (cli.position_only or orn_err <= cli.orn_tol)
    success = raw_ik_success and limit_hits == 0 and not contacts
    target_local_tensor = torch.tensor(
        target_xyz, device=env.device, dtype=torch.float
    )
    endpoint_collision_rejected = bool(
        torch.all(target_local_tensor < env.collision_upper_limits)
        and torch.all(target_local_tensor > env.collision_lower_limits)
    )
    endpoint_underground_rejected = bool(
        target_local_tensor[2] < env.underground_limit
    )
    endpoint_reach_rejected = bool(
        torch.linalg.vector_norm(target_local_tensor)
        > env.max_nominal_reach_radius
    )
    task_endpoint_admissible = not (
        endpoint_collision_rejected
        or endpoint_underground_rejected
        or endpoint_reach_rejected
    )

    row = {
        "target_x": target_xyz[0],
        "target_y": target_xyz[1],
        "target_z": target_xyz[2],
        "target_roll": target_rpy[0],
        "target_pitch": target_rpy[1],
        "target_yaw": target_rpy[2],
        "orientation_command_roll": orientation_command[0],
        "orientation_command_pitch": orientation_command[1],
        "orientation_command_yaw": orientation_command[2],
        "success": int(success),
        "raw_ik_success": int(raw_ik_success),
        "pos_err": pos_err,
        "orn_err": orn_err,
        "iters": iters,
        "limit_hits": limit_hits,
        "min_limit_margin": float(margin.min().item()),
        "collision": int(bool(contacts)),
        "task_endpoint_admissible": int(task_endpoint_admissible),
        "endpoint_collision_rejected": int(endpoint_collision_rejected),
        "endpoint_underground_rejected": int(endpoint_underground_rejected),
        "endpoint_reach_rejected": int(endpoint_reach_rejected),
        "max_contact_force": max_contact_force,
        "contact_bodies": ";".join(f"{name}:{force:.3f}" for name, force in contacts),
        "final_x": float(final_local[0].item()),
        "final_y": float(final_local[1].item()),
        "final_z": float(final_local[2].item()),
    }
    for idx, value in enumerate(final_arm_q.detach().cpu().tolist(), start=1):
        row[f"arm_joint{idx}"] = value
    return row


def choose_nearest(rows, target):
    candidates = [row for row in rows if row["success"]]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            (row["target_x"] - target[0]) ** 2
            + (row["target_y"] - target[1]) ** 2
            + (row["target_z"] - target[2]) ** 2
        ),
    )


def summarize(rows, home_ee_local=None, home_ee_axes_base=None):
    total = len(rows)
    successes = [row for row in rows if row["success"]]
    raw_successes = [row for row in rows if row["raw_ik_success"]]
    admissible = [row for row in rows if row["task_endpoint_admissible"]]
    admissible_successes = [row for row in admissible if row["success"]]
    summary = {
        "total": total,
        "success_count": len(successes),
        "raw_ik_success_count": len(raw_successes),
        "success_rate": len(successes) / total if total else 0.0,
        "raw_ik_success_rate": len(raw_successes) / total if total else 0.0,
        "task_endpoint_admissible_count": len(admissible),
        "task_endpoint_admissible_success_count": len(admissible_successes),
        "task_endpoint_admissible_success_rate": (
            len(admissible_successes) / len(admissible) if admissible else 0.0
        ),
        "recommended_initial_ee_goal_cart": None,
        "recommended_mask_arm_goal_cart": None,
        "success_bounds": None,
        "home_ee_local": home_ee_local,
        "home_ee_axes_base": home_ee_axes_base,
    }
    if successes:
        summary["success_bounds"] = {
            axis: [min(row[f"target_{axis}"] for row in successes), max(row[f"target_{axis}"] for row in successes)]
            for axis in ("x", "y", "z")
        }
        initial = choose_nearest(rows, [0.28, 0.0, 0.18])
        mask = choose_nearest(rows, [0.35, 0.0, 0.25])
        if initial:
            summary["recommended_initial_ee_goal_cart"] = [initial["target_x"], initial["target_y"], initial["target_z"]]
        if mask:
            summary["recommended_mask_arm_goal_cart"] = [mask["target_x"], mask["target_y"], mask["target_z"]]
    return summary


def grid_values(cli):
    if not cli.quick:
        return (
            parse_float_list(cli.x),
            parse_float_list(cli.y),
            parse_float_list(cli.z),
            parse_float_list(cli.roll),
            parse_float_list(cli.pitch),
            parse_float_list(cli.yaw),
        )
    return (
        [0.215, 0.39, 0.565],
        [-0.225, 0.0, 0.225],
        [-0.364, -0.164, 0.036],
        [0.0],
        [0.0],
        [0.0],
    )


def main():
    cli = parse_args()
    args = make_legged_gym_args(cli)
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg = configure_env(env_cfg, cli)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env._count_finger_contacts = cli.count_finger_contacts
    env.reset()
    jacobian_candidates = None
    if not cli.no_auto_jacobian_body:
        best_idx, best_error, jacobian_candidates = select_jacobian_body_index(env, cli)
        env._jacobian_body_idx = best_idx
    set_dof_positions(env, env.default_dof_pos.detach().clone(), jacobian_body_offset=cli.jacobian_body_offset)
    settle(env, cli.settle_steps, jacobian_body_offset=cli.jacobian_body_offset)
    home_ee_local = [float(value) for value in local_ee_position(env, cli.target_frame).detach().cpu().tolist()]
    home_ee_axes_base = local_ee_axes(env)

    xs, ys, zs, rolls, pitches, yaws = grid_values(cli)
    targets = list(product(xs, ys, zs, rolls, pitches, yaws))
    if cli.max_targets > 0:
        targets = targets[: cli.max_targets]

    print("\n=== Go2-X5 IK reachability grid ===")
    print(f"target_frame: {cli.target_frame}, armBaseOffset={robot_spec.ARM_BASE_OFFSET}")
    print(f"home arm q: {[robot_spec.DEFAULT_JOINT_ANGLES[name] for name in robot_spec.ARM_JOINT_NAMES]}")
    print(f"home ee local: {[round(value, 4) for value in home_ee_local]}")
    print(f"home ee axes base: {home_ee_axes_base}")
    print(f"jacobian body index: {getattr(env, '_jacobian_body_idx', env.gripper_idx + cli.jacobian_body_offset)} (rigid body EE index {env.gripper_idx})")
    if jacobian_candidates is not None:
        print(f"jacobian fd candidates: {[(idx, round(err, 6)) for err, idx in jacobian_candidates]}")
    print(f"base height/pitch: {cli.base_height:.3f} m / {cli.base_pitch:.3f} rad")
    print(f"targets: {len(targets)}")
    print(f"position_only: {cli.position_only}")
    print(f"orientation_mode: {cli.orientation_mode}")
    print(f"csv: {cli.csv}")

    rows = []
    for index, target in enumerate(targets, start=1):
        target_xyz = target[:3]
        orientation_command = target[3:]
        row = scan_target(env, target_xyz, orientation_command, cli)
        rows.append(row)
        if cli.print_every and (index == 1 or index == len(targets) or index % cli.print_every == 0):
            print(
                "target {:04d}/{:04d} xyz={} rpy={} success={} pos_err={:.3f} orn_err={:.3f} limit_hits={} collision={}".format(
                    index,
                    len(targets),
                    [round(v, 3) for v in target_xyz],
                    [round(row[f"target_{axis}"], 3) for axis in ("roll", "pitch", "yaw")],
                    row["success"],
                    row["pos_err"],
                    row["orn_err"],
                    row["limit_hits"],
                    row["collision"],
                ),
                flush=True,
            )

    fieldnames = list(rows[0].keys()) if rows else []
    os.makedirs(os.path.dirname(os.path.abspath(cli.csv)), exist_ok=True)
    with open(cli.csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, home_ee_local=home_ee_local, home_ee_axes_base=home_ee_axes_base)
    os.makedirs(os.path.dirname(os.path.abspath(cli.summary_json)), exist_ok=True)
    with open(cli.summary_json, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
