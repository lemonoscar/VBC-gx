#!/usr/bin/env python3
"""Zero-action standing checks for Go2 + ARX-X5 in Isaac Gym.

This script removes training-time randomness and keeps leg actions at zero.
It has two modes:

- frozen_ee: keep the arm target at the current EE pose before every step.
- moving_ee: use the normal environment EE target trajectory while legs stay at
  zero action.
- moving_pos_frozen_orn: use the normal EE position target trajectory, but keep
  the EE orientation target at the current EE orientation before every step.
- frozen_pos_moving_orn: keep the EE position target at the current EE position,
  but let the normal environment orientation target evolve.

The intent is to separate base standing/contact issues from arm-target-induced
disturbances before spending time on PPO training.
"""

import argparse
import os
import sys
import time


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
from isaacgym.torch_utils import euler_from_quat, quat_rotate_inverse  # noqa: E402
import torch  # noqa: E402
from types import MethodType  # noqa: E402

from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.utils import get_args, task_registry  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="go2x5")
    parser.add_argument(
        "--mode",
        choices=("frozen_ee", "moving_ee", "moving_pos_frozen_orn", "frozen_pos_moving_orn"),
        default="frozen_ee",
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--viewer", action="store_true", help="Open Isaac Gym viewer.")
    parser.add_argument("--sleep", type=float, default=0.01, help="Viewer pacing sleep per step.")
    parser.add_argument("--flat_terrain", action="store_true", default=True)
    parser.add_argument("--observe_gait_commands", action="store_true", default=True)
    parser.add_argument("--ik_gain", type=float, default=None, help="Override cfg.arm.ik_gain for this check.")
    parser.add_argument("--arm_stiffness", type=float, default=None, help="Override arm position stiffness.")
    parser.add_argument("--arm_damping", type=float, default=None, help="Override arm position damping.")
    return parser.parse_args()


def make_legged_gym_args(cli):
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = get_args(test=True)
    finally:
        sys.argv = original_argv

    args.task = cli.task
    args.headless = not cli.viewer
    args.flat_terrain = cli.flat_terrain
    args.observe_gait_commands = cli.observe_gait_commands
    args.sim_device = cli.sim_device
    args.rl_device = cli.rl_device
    args.graphics_device_id = cli.graphics_device_id
    return args


def configure_env(env_cfg, cli):
    env_cfg.env.num_envs = 1
    env_cfg.env.record_video = False
    env_cfg.env.stand_by = True
    env_cfg.env.teleop_mode = False
    env_cfg.env.observe_gait_commands = cli.observe_gait_commands

    env_cfg.viewer.pos = [1.2, -1.2, 0.75]
    env_cfg.viewer.lookat = [0.25, 0.0, 0.30]

    # Terrain generation divides by num_rows - 1, so keep at least 2 rows.
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
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

    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    if hasattr(env_cfg, "auto_curriculum"):
        env_cfg.auto_curriculum.enabled = False
        env_cfg.auto_curriculum.stages = []
    if cli.ik_gain is not None:
        env_cfg.arm.ik_gain = cli.ik_gain
    if cli.arm_stiffness is not None:
        env_cfg.control.arm_pos_stiffness = cli.arm_stiffness
    if cli.arm_damping is not None:
        env_cfg.control.arm_pos_damping = cli.arm_damping
    return env_cfg


def freeze_ee_target(env):
    freeze_ee_position(env)
    freeze_ee_orientation(env)
    env.ee_goal_orn_delta_rpy.zero_()
    env.goal_timer.zero_()
    env.traj_timesteps[:] = 1e9
    env.traj_total_timesteps[:] = 1e9


def freeze_ee_position(env):
    ee_world = env.rigid_body_state[:, env.gripper_idx, :3].clone()
    center = env.get_ee_goal_spherical_center()
    local = quat_rotate_inverse(env.base_yaw_quat, ee_world - center)

    env.curr_ee_goal_cart[:] = local
    env.ee_start_cart[:] = local
    env.ee_goal_cart[:] = local
    env.curr_ee_goal_cart_world[:] = ee_world


def freeze_ee_orientation(env):
    ee_quat = env.ee_orn / env.ee_orn.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    env.ee_goal_orn_quat[:] = ee_quat


def contact_count(env):
    indices = getattr(env, "penalised_contact_indices", None)
    if indices is None:
        indices = getattr(env, "penalized_contact_indices", None)
    if indices is None or indices.numel() == 0:
        return 0
    contacts = torch.norm(env.contact_forces[0, indices], dim=-1) > 1.0
    return int(torch.sum(contacts).item())


def timeout_flag(env):
    time_out_buf = getattr(env, "time_out_buf", None)
    if torch.is_tensor(time_out_buf):
        return int(time_out_buf[0].item())
    return 0


def attach_termination_debug(env):
    """Store termination flags before reset_idx mutates simulator state."""

    def debug_check_termination(self):
        termination_contact_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )
        r, p, _ = euler_from_quat(self.base_quat)
        z = self.root_states[:, 2]

        r_term = torch.abs(r) > self.cfg.termination.r_threshold
        p_term = torch.abs(p) > self.cfg.termination.p_threshold
        z_term = z < self.cfg.termination.z_threshold
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = termination_contact_buf | self.time_out_buf | r_term | p_term | z_term

        self._zero_action_last_term_debug = {
            "roll": float(r[0].item()),
            "pitch": float(p[0].item()),
            "z": float(z[0].item()),
            "contact": bool(termination_contact_buf[0].item()),
            "roll_term": bool(r_term[0].item()),
            "pitch_term": bool(p_term[0].item()),
            "z_term": bool(z_term[0].item()),
            "timeout": bool(self.time_out_buf[0].item()),
            "reset": bool(self.reset_buf[0].item()),
        }

    env.check_termination = MethodType(debug_check_termination, env)
    env._zero_action_last_term_debug = {
        "roll": 0.0,
        "pitch": 0.0,
        "z": float(env.root_states[0, 2].item()),
        "contact": False,
        "roll_term": False,
        "pitch_term": False,
        "z_term": False,
        "timeout": False,
        "reset": False,
    }


def termination_reason(env):
    debug = getattr(env, "_zero_action_last_term_debug", {})
    reasons = []
    if debug.get("timeout", False):
        reasons.append("timeout")
    if debug.get("contact", False):
        reasons.append("contact")
    if debug.get("roll_term", False):
        reasons.append("roll")
    if debug.get("pitch_term", False):
        reasons.append("pitch")
    if debug.get("z_term", False):
        reasons.append("z")
    return "+".join(reasons) if reasons else "none"


def print_summary(env_cfg, env, cli):
    print("\n=== Go2X5 zero-action stand check ===")
    print(f"mode: {cli.mode}")
    print(f"viewer: {cli.viewer}")
    print(f"steps: {cli.steps}")
    print(f"asset.file: {env_cfg.asset.file}")
    print(f"init_state.pos: {env_cfg.init_state.pos}")
    print(f"base_height_target: {env_cfg.rewards.base_height_target}")
    print(f"num_actions={env.num_actions}, num_torques={env.num_torques}, num_dofs={env.num_dofs}")
    print(f"ik_gain: {getattr(env_cfg.arm, 'ik_gain', None)}")
    print(f"track_ee_orientation: {getattr(env_cfg.arm, 'track_ee_orientation', True)}")
    print(f"arm_pos_stiffness: {getattr(env_cfg.control, 'arm_pos_stiffness', None)}")
    print(f"arm_pos_damping: {getattr(env_cfg.control, 'arm_pos_damping', None)}")
    print(f"randomization: off")
    print(f"commands: zero")
    print()


def run(cli):
    args = make_legged_gym_args(cli)
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg = configure_env(env_cfg, cli)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    attach_termination_debug(env)
    print_summary(env_cfg, env, cli)

    env.reset()
    env.commands.zero_()
    env.cfg.env.teleop_mode = cli.mode == "frozen_ee"

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    failure_reset_steps = []
    timeout_steps = []

    for step in range(cli.steps):
        env.commands.zero_()
        if cli.mode == "frozen_ee":
            freeze_ee_target(env)
        elif cli.mode == "moving_pos_frozen_orn":
            freeze_ee_orientation(env)
        elif cli.mode == "frozen_pos_moving_orn":
            freeze_ee_position(env)

        env.step(actions)
        reset = int(env.reset_buf[0].item())
        timeout = timeout_flag(env)
        if reset and timeout:
            timeout_steps.append(step)
        elif reset:
            failure_reset_steps.append(step)

        if step % cli.print_every == 0 or reset:
            ee_err = torch.norm(env.ee_pos[0] - env.curr_ee_goal_cart_world[0]).item()
            print(
                "step={:05d} base_z={:.3f} ang_xy={:.3f} lin_z={:.3f} "
                "ee_err={:.3f} contacts={} reset={} timeout={} reason={} "
                "pre_roll={:.3f} pre_pitch={:.3f} pre_z={:.3f}".format(
                    step,
                    env.root_states[0, 2].item(),
                    env.base_ang_vel[0, :2].norm().item(),
                    env.base_lin_vel[0, 2].item(),
                    ee_err,
                    contact_count(env),
                    reset,
                    timeout,
                    termination_reason(env),
                    env._zero_action_last_term_debug["roll"],
                    env._zero_action_last_term_debug["pitch"],
                    env._zero_action_last_term_debug["z"],
                ),
                flush=True,
            )

        if cli.viewer and cli.sleep > 0.0:
            time.sleep(cli.sleep)

    print("\n=== Result ===")
    if failure_reset_steps:
        print(
            f"FAIL: non-timeout reset triggered {len(failure_reset_steps)} times; "
            f"first reset step={failure_reset_steps[0]}"
        )
        return 1
    if timeout_steps:
        print(f"PASS: only timeout resets observed at steps {timeout_steps[:5]}")
        return 0
    print("PASS: no reset during zero-action check")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
