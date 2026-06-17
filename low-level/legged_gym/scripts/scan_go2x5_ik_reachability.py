#!/usr/bin/env python3
"""Headless IK reachability grid for the Go2-X5 arm.

The scan is not a training run. It fixes the base, starts from the canonical
home pose, sweeps arm-base-frame x/y/z/rpy targets, runs damped least-squares
IK, and writes per-target reachability statistics.
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
from isaacgym.torch_utils import orientation_error, quat_apply, quat_from_euler_xyz, quat_rotate_inverse  # noqa: E402
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.envs.manip_loco import go2x5_robot_spec as robot_spec  # noqa: E402
from legged_gym.utils import get_args, task_registry  # noqa: E402


def parse_float_list(text):
    return [float(item) for item in text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="go2x5")
    parser.add_argument("--quick", action="store_true", help="Use a small 27-point smoke-test grid.")
    parser.add_argument("--x", default="0.18,0.24,0.30,0.36")
    parser.add_argument("--y", default="-0.18,-0.09,0.0,0.09,0.18")
    parser.add_argument("--z", default="0.06,0.16,0.26,0.36")
    parser.add_argument("--roll", default=f"{math.pi / 2:.12f}")
    parser.add_argument("--pitch", default="-0.25,0.0,0.25")
    parser.add_argument("--yaw", default="-0.45,0.0,0.45")
    parser.add_argument("--max_targets", type=int, default=0, help="Optional cap for debugging; 0 scans all.")
    parser.add_argument("--print_every", type=int, default=1, help="Print every N targets; 0 only prints summary.")
    parser.add_argument("--max_iters", type=int, default=40)
    parser.add_argument("--ik_gain", type=float, default=0.7)
    parser.add_argument("--max_delta", type=float, default=0.08, help="Maximum per-iteration arm joint update in radians.")
    parser.add_argument("--orn_weight", type=float, default=0.25, help="Orientation residual weight during IK.")
    parser.add_argument("--position_only", action="store_true", help="Optimize position only; still report orientation error.")
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


def configure_env(env_cfg):
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
    env_cfg.init_state.pos = [0.0, 0.0, robot_spec.BASE_INIT_HEIGHT]

    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    if hasattr(env_cfg, "auto_curriculum"):
        env_cfg.auto_curriculum.enabled = False
    return env_cfg


def refresh_runtime_state(env, jacobian_body_offset=-1):
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_jacobian_tensors(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_yaw_quat[:] = env.base_quat[:]
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


def target_world_from_arm_base(env, local_target):
    root_quat = env.root_states[0:1, 3:7]
    root_pos = env.root_states[0:1, :3]
    arm_base_local = torch.tensor(robot_spec.ARM_BASE_OFFSET, device=env.device, dtype=torch.float).view(1, 3)
    local = torch.tensor(local_target, device=env.device, dtype=torch.float).view(1, 3)
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


def local_ee_position(env):
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


def scan_target(env, target_xyz, target_rpy, cli):
    home_q = env.default_dof_pos.detach().clone()
    arm_start = env.num_dofs - (6 + env.cfg.env.num_gripper_joints)
    arm_end = env.num_dofs - env.cfg.env.num_gripper_joints
    lower = env.dof_pos_limits[arm_start:arm_end, 0]
    upper = env.dof_pos_limits[arm_start:arm_end, 1]

    q = home_q.clone()
    set_dof_positions(env, q, jacobian_body_offset=cli.jacobian_body_offset)
    target_world = target_world_from_arm_base(env, target_xyz)
    target_quat = quat_from_euler_xyz(
        torch.tensor([target_rpy[0]], device=env.device),
        torch.tensor([target_rpy[1]], device=env.device),
        torch.tensor([target_rpy[2]], device=env.device),
    )

    pos_err = float("inf")
    orn_err = float("inf")
    iters = 0
    for iters in range(1, cli.max_iters + 1):
        refresh_runtime_state(env, jacobian_body_offset=cli.jacobian_body_offset)
        ee_orn_norm = torch.norm(env.ee_orn, dim=-1, keepdim=True).clamp(min=1e-6)
        dpos = target_world - env.ee_pos[0:1]
        drot = orientation_error(target_quat, env.ee_orn[0:1] / ee_orn_norm[0:1])
        ik_drot = torch.zeros_like(drot) if cli.position_only else drot * cli.orn_weight
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

    final_local = local_ee_position(env)
    final_arm_q = env.dof_pos[0, arm_start:arm_end].detach().clone()
    margin = torch.minimum(final_arm_q - lower, upper - final_arm_q)
    limit_hits = int((margin < cli.limit_margin).sum().item())
    contacts, max_contact_force = contact_summary(env, cli.contact_threshold)

    pos_err = torch.norm(target_world - env.ee_pos[0:1]).item()
    ee_orn_norm = torch.norm(env.ee_orn, dim=-1, keepdim=True).clamp(min=1e-6)
    orn_err = torch.norm(orientation_error(target_quat, env.ee_orn[0:1] / ee_orn_norm[0:1])).item()
    raw_ik_success = pos_err <= cli.pos_tol and (cli.position_only or orn_err <= cli.orn_tol)
    success = raw_ik_success and limit_hits == 0 and not contacts

    row = {
        "target_x": target_xyz[0],
        "target_y": target_xyz[1],
        "target_z": target_xyz[2],
        "target_roll": target_rpy[0],
        "target_pitch": target_rpy[1],
        "target_yaw": target_rpy[2],
        "success": int(success),
        "raw_ik_success": int(raw_ik_success),
        "pos_err": pos_err,
        "orn_err": orn_err,
        "iters": iters,
        "limit_hits": limit_hits,
        "min_limit_margin": float(margin.min().item()),
        "collision": int(bool(contacts)),
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
    summary = {
        "total": total,
        "success_count": len(successes),
        "raw_ik_success_count": len(raw_successes),
        "success_rate": len(successes) / total if total else 0.0,
        "raw_ik_success_rate": len(raw_successes) / total if total else 0.0,
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
        [0.22, 0.30, 0.38],
        [-0.12, 0.0, 0.12],
        [0.10, 0.22, 0.34],
        [math.pi / 2],
        [0.0],
        [0.0],
    )


def main():
    cli = parse_args()
    args = make_legged_gym_args(cli)
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg = configure_env(env_cfg)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env._count_finger_contacts = cli.count_finger_contacts
    env.reset()
    jacobian_candidates = None
    if not cli.no_auto_jacobian_body:
        best_idx, best_error, jacobian_candidates = select_jacobian_body_index(env, cli)
        env._jacobian_body_idx = best_idx
    set_dof_positions(env, env.default_dof_pos.detach().clone(), jacobian_body_offset=cli.jacobian_body_offset)
    settle(env, cli.settle_steps, jacobian_body_offset=cli.jacobian_body_offset)
    home_ee_local = [float(value) for value in local_ee_position(env).detach().cpu().tolist()]
    home_ee_axes_base = local_ee_axes(env)

    xs, ys, zs, rolls, pitches, yaws = grid_values(cli)
    targets = list(product(xs, ys, zs, rolls, pitches, yaws))
    if cli.max_targets > 0:
        targets = targets[: cli.max_targets]

    print("\n=== Go2-X5 IK reachability grid ===")
    print(f"frame: arm_base, armBaseOffset={robot_spec.ARM_BASE_OFFSET}")
    print(f"home arm q: {[robot_spec.DEFAULT_JOINT_ANGLES[name] for name in robot_spec.ARM_JOINT_NAMES]}")
    print(f"home ee local: {[round(value, 4) for value in home_ee_local]}")
    print(f"home ee axes base: {home_ee_axes_base}")
    print(f"jacobian body index: {getattr(env, '_jacobian_body_idx', env.gripper_idx + cli.jacobian_body_offset)} (rigid body EE index {env.gripper_idx})")
    if jacobian_candidates is not None:
        print(f"jacobian fd candidates: {[(idx, round(err, 6)) for err, idx in jacobian_candidates]}")
    print(f"base height: {robot_spec.BASE_INIT_HEIGHT}")
    print(f"targets: {len(targets)}")
    print(f"position_only: {cli.position_only}")
    print(f"csv: {cli.csv}")

    rows = []
    for index, target in enumerate(targets, start=1):
        target_xyz = target[:3]
        target_rpy = target[3:]
        row = scan_target(env, target_xyz, target_rpy, cli)
        rows.append(row)
        if cli.print_every and (index == 1 or index == len(targets) or index % cli.print_every == 0):
            print(
                "target {:04d}/{:04d} xyz={} rpy={} success={} pos_err={:.3f} orn_err={:.3f} limit_hits={} collision={}".format(
                    index,
                    len(targets),
                    [round(v, 3) for v in target_xyz],
                    [round(v, 3) for v in target_rpy],
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
