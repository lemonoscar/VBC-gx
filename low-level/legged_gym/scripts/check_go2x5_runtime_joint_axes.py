#!/usr/bin/env python3
"""Isaac Gym runtime joint-axis sanity checks for Go2 + ARX-X5 legs.

The static URDF check verifies XML kinematics. This script verifies the same
thing after Isaac Gym has loaded the asset by directly setting DOF states and
reading the runtime rigid-body foot positions.
"""

import argparse
import os
import sys


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
from isaacgym.torch_utils import quat_rotate_inverse  # noqa: E402
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.utils import get_args, task_registry  # noqa: E402


LEGS = ("FL", "FR", "RL", "RR")
JOINT_KINDS = ("hip", "thigh", "calf")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="go2x5")
    parser.add_argument("--delta", type=float, default=0.15, help="Joint perturbation in radians.")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--min_lateral", type=float, default=0.06)
    parser.add_argument("--min_hip_dy", type=float, default=0.01)
    parser.add_argument("--settle_steps", type=int, default=1, help="Physics steps after each direct DOF-state write.")
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
    args.observe_gait_commands = True
    return args


def configure_env(env_cfg):
    env_cfg.env.num_envs = 1
    env_cfg.env.observe_gait_commands = True
    env_cfg.env.record_video = False
    env_cfg.env.stand_by = True
    env_cfg.env.teleop_mode = False
    env_cfg.asset.fix_base_link = True
    env_cfg.asset.disable_gravity = True

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

    env_cfg.init_state.rand_yaw_range = 0.0
    env_cfg.init_state.origin_perturb_range = 0.0
    env_cfg.init_state.init_vel_perturb_range = 0.0
    env_cfg.init_state.leg_reset_ratio_range = [1.0, 1.0]
    env_cfg.init_state.arm_reset_noise_range = [0.0, 0.0]

    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    if hasattr(env_cfg, "auto_curriculum"):
        env_cfg.auto_curriculum.enabled = False
    return env_cfg


def set_dof_positions(env, positions, settle_steps):
    dof_state_view = env.dof_state.view(env.num_envs, env.num_dofs, 2)
    dof_state_view[0, :, 0] = positions
    dof_state_view[0, :, 1] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    for _ in range(max(0, settle_steps)):
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
    refresh_runtime_state(env)


def refresh_runtime_state(env):
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)


def foot_positions(env):
    world_feet = env.rigid_body_state[0, env.feet_indices, :3]
    base_pos = env.root_states[0, :3].unsqueeze(0)
    base_quat = env.root_states[0, 3:7].unsqueeze(0).repeat(world_feet.shape[0], 1)
    return quat_rotate_inverse(base_quat, world_feet - base_pos).detach().clone()


def fmt_vec(vec):
    return "[" + ", ".join(f"{float(x):+0.4f}" for x in vec) + "]"


def main():
    cli = parse_args()
    args = make_legged_gym_args(cli)
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg = configure_env(env_cfg)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    default_q = env.default_dof_pos.detach().clone()
    set_dof_positions(env, default_q, cli.settle_steps)
    default_feet = foot_positions(env)
    foot_order = [env.body_names[i] for i in env.feet_indices.detach().cpu().tolist()]

    print("\n=== Go2X5 runtime joint-axis check ===")
    print(f"task: {args.task}")
    print(f"delta: {cli.delta:.3f} rad")
    print(f"settle_steps: {cli.settle_steps}")
    print(f"fix_base_link: {env.cfg.asset.fix_base_link}")
    print(f"dof_names: {env.dof_names}")
    print(f"foot_order: {foot_order}")
    print()
    print("Default foot positions, runtime base frame/world frame with fixed base:")

    failures = []
    for leg_idx, leg in enumerate(LEGS):
        pos = default_feet[leg_idx]
        side_sign = 1.0 if leg[1] == "L" else -1.0
        side_ok = float(pos[1]) * side_sign > cli.min_lateral
        print(f"  {leg}: pos={fmt_vec(pos)} side_ok={side_ok}")
        if not side_ok:
            failures.append(f"{leg} default foot y={float(pos[1]):.4f} violates side/min-lateral check")

    print()
    print("Positive +delta displacement for each leg joint:")
    for leg_idx, leg in enumerate(LEGS):
        for kind in JOINT_KINDS:
            joint_name = f"{leg}_{kind}_joint"
            joint_idx = env.dof_names.index(joint_name)
            q = default_q.clone()
            q[joint_idx] += cli.delta
            set_dof_positions(env, q, cli.settle_steps)
            displaced = foot_positions(env)[leg_idx] - default_feet[leg_idx]
            print(f"  {joint_name}: dfoot={fmt_vec(displaced)}")

    print()
    print("Hip outward command check:")
    for leg_idx, leg in enumerate(LEGS):
        joint_name = f"{leg}_hip_joint"
        joint_idx = env.dof_names.index(joint_name)
        side_sign = 1.0 if leg[1] == "L" else -1.0
        command_sign = side_sign
        q = default_q.clone()
        q[joint_idx] += command_sign * cli.delta
        set_dof_positions(env, q, cli.settle_steps)
        displaced = foot_positions(env)[leg_idx] - default_feet[leg_idx]
        outward_dy = float(displaced[1]) * side_sign
        passed = outward_dy > cli.min_hip_dy
        command_text = f"{command_sign:+.0f} * delta"
        print(
            f"  {joint_name} {command_text}: dfoot={fmt_vec(displaced)} "
            f"outward_dy={outward_dy:+0.4f} pass={passed}"
        )
        if not passed:
            failures.append(f"{joint_name} outward command failed: outward_dy={outward_dy:.4f}")

    print()
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PASS: Isaac Gym runtime joint directions match the mirrored Go2 hip convention.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
