# SPDX-License-Identifier: BSD-3-Clause

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

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def _make_visual_cfg(env_cfg, args):
    env_cfg.env.num_envs = 1
    env_cfg.env.record_video = False
    env_cfg.env.stand_by = True
    env_cfg.env.teleop_mode = False
    env_cfg.env.observe_gait_commands = bool(args.observe_gait_commands)

    env_cfg.viewer.pos = [1.2, -1.2, 0.75]
    env_cfg.viewer.lookat = [0.25, 0.0, 0.30]

    env_cfg.terrain.num_rows = args.rows if args.rows is not None else 2
    env_cfg.terrain.num_cols = args.cols if args.cols is not None else 2
    if args.flat_terrain:
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
    return env_cfg


def _relative_base_height(env):
    base_height = env.root_states[0, 2].item()
    measured_heights = getattr(env, "measured_heights", None)
    if torch.is_tensor(measured_heights) and measured_heights.ndim == 2:
        base_height -= torch.mean(measured_heights[0]).item()
    return base_height


def _print_static_summary(env_cfg, env):
    print("\n=== Go2X5 config visualization ===")
    print(f"asset.file: {env_cfg.asset.file}")
    print(f"asset.gripper_name: {env_cfg.asset.gripper_name}")
    print(f"arm.base_offset: {env_cfg.arm.base_offset}")
    print(f"init_state.pos: {env_cfg.init_state.pos}")
    print(f"base_height_target: {env_cfg.rewards.base_height_target}")
    print(f"goal sphere center: x={env_cfg.goal_ee.sphere_center.x_offset}, z={env_cfg.goal_ee.sphere_center.z_invariant_offset}")
    print(
        "goal local xyz: "
        f"x={env_cfg.goal_ee.ranges.pos_x}, "
        f"y={env_cfg.goal_ee.ranges.pos_y_cart}, "
        f"z={env_cfg.goal_ee.ranges.pos_z}"
    )
    print(
        "goal root/terrain xyz: "
        f"x={[round(env_cfg.goal_ee.sphere_center.x_offset + value, 3) for value in env_cfg.goal_ee.ranges.pos_x]}, "
        f"y={env_cfg.goal_ee.ranges.pos_y_cart}, "
        f"z={[round(env_cfg.goal_ee.sphere_center.z_invariant_offset + value, 3) for value in env_cfg.goal_ee.ranges.pos_z]}"
    )
    print(f"num_actions={env.num_actions}, num_torques={env.num_torques}, num_dofs={env.num_dofs}, num_bodies={env.num_bodies}")
    print(f"dof_names: {env.dof_names}")
    print(f"gripper_idx: {env.gripper_idx}")
    print("\nViewer markers:")
    print("  yellow sphere: current EE target")
    print("  blue sphere: actual EE frame")
    print("  cyan sphere: EE goal sphere center")
    print("  red dots: sampled EE target trajectory")
    print("  green sphere: world origin")
    print("\nKeyboard: ESC quit, SPACE pause, F free camera, 0 look at robot.\n")


def visualize(args):
    if args.task == "widowGo1":
        args.task = "go2x5"
    args.headless = False

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg = _make_visual_cfg(env_cfg, args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _print_static_summary(env_cfg, env)

    env.reset()
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    steps = args.max_iterations if args.max_iterations is not None else 20000

    for step in range(steps):
        start = time.time()
        env.step(actions)
        if step % 200 == 0:
            ee_error = torch.norm(env.ee_pos[0] - env.curr_ee_goal_cart_world[0]).item()
            print(
                "step={:05d} base_z={:.3f} rel_base_h={:.3f} ee_err={:.3f} ee={} target={}".format(
                    step,
                    env.root_states[0, 2].item(),
                    _relative_base_height(env),
                    ee_error,
                    [round(x, 3) for x in env.ee_pos[0].detach().cpu().tolist()],
                    [round(x, 3) for x in env.curr_ee_goal_cart_world[0].detach().cpu().tolist()],
                ),
                flush=True,
            )
        time.sleep(max(0.02 - (time.time() - start), 0.0))


if __name__ == "__main__":
    visualize(get_args(test=True))
