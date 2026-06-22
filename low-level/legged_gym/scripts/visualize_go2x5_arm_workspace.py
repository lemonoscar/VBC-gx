#!/usr/bin/env python3
"""Isaac Gym viewer for Go2-X5 arm workspace geometry.

This is a visualization-only tool. It creates one Go2-X5 environment, keeps the
robot in a quiet standing pose, and draws an arm-base-front hemisphere plus
optional IK reachability CSV points inside the Isaac Gym viewer.
"""

import argparse
import csv
import math
import os
import sys
import time
import xml.etree.ElementTree as ET


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
from isaacgym import gymapi, gymutil  # noqa: E402
from isaacgym.torch_utils import quat_apply  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.envs.manip_loco import go2x5_robot_spec as robot_spec  # noqa: E402
from legged_gym.utils import get_args, task_registry  # noqa: E402


DEFAULT_SAFE_BOX = {
    "x": [0.10, 0.38],
    "y": [-0.12, 0.12],
    "z": [0.20, 0.40],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="go2x5")
    parser.add_argument("--csv", default="", help="Optional IK reachability CSV to overlay.")
    parser.add_argument(
        "--csv_frame",
        choices=["goal_center", "arm_base"],
        default="goal_center",
        help="Frame used by CSV target_x/y/z points. goal_center matches the terrain-invariant task sampler.",
    )
    parser.add_argument("--radius", type=float, default=0.0, help="Hemisphere radius. 0 estimates from the URDF arm chain.")
    parser.add_argument("--draw_csv_points", action="store_true", help="Draw CSV points as small spheres.")
    parser.add_argument("--csv_stride", type=int, default=8, help="Draw every Nth CSV point to keep the viewer responsive.")
    parser.add_argument("--point_radius", type=float, default=0.009)
    parser.add_argument("--hemisphere_segments", type=int, default=56)
    parser.add_argument("--hemisphere_rings", type=int, default=10)
    parser.add_argument(
        "--safe_box",
        default="auto",
        help="'auto' draws the CSV success bounds when --csv is provided, otherwise use 'xmin,xmax,ymin,ymax,zmin,zmax'.",
    )
    parser.add_argument("--max_iterations", type=int, default=20000)
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--flat_terrain", action="store_true")
    parser.add_argument("--observe_gait_commands", action="store_true")
    parser.add_argument("--hide_task_box", action="store_true", help="Hide the current low-level task sampling box.")
    parser.add_argument("--hide_hemisphere", action="store_true", help="Hide the theoretical arm-base front hemisphere.")
    parser.add_argument("--hide_safe_box", action="store_true", help="Hide the IK scan/safe box.")
    parser.add_argument("--task_box_grid", type=int, default=8, help="Grid subdivisions drawn on the task sampling box faces.")
    parser.add_argument("--task_corner_radius", type=float, default=0.028)
    return parser.parse_args()


def make_legged_gym_args(cli):
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = get_args(test=True)
    finally:
        sys.argv = original_argv
    args.task = cli.task
    args.headless = False
    args.sim_device = cli.sim_device
    args.rl_device = cli.rl_device
    args.graphics_device_id = cli.graphics_device_id
    args.observe_gait_commands = cli.observe_gait_commands
    return args


def configure_env(env_cfg, cli):
    env_cfg.env.num_envs = 1
    env_cfg.env.record_video = False
    env_cfg.env.stand_by = True
    env_cfg.env.teleop_mode = False
    env_cfg.env.observe_gait_commands = bool(cli.observe_gait_commands)

    env_cfg.viewer.pos = [1.35, -1.35, 0.95]
    env_cfg.viewer.lookat = [0.35, 0.0, 0.35]

    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    if cli.flat_terrain:
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


def estimate_arm_radius_from_urdf():
    urdf_path = os.path.join(_REPO_ROOT, "low-level", "resources", "robots", "go2x5", "go2_x5.urdf")
    root = ET.parse(urdf_path).getroot()
    radius = 0.0
    for joint_name in [
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
        "arm_joint6",
        "arm_eef_joint",
    ]:
        joint = root.find(f"joint[@name='{joint_name}']")
        if joint is None:
            continue
        origin = joint.find("origin")
        xyz = [0.0, 0.0, 0.0]
        if origin is not None and origin.attrib.get("xyz"):
            xyz = [float(value) for value in origin.attrib["xyz"].split()]
        radius += math.sqrt(sum(value * value for value in xyz))
    return radius


def parse_safe_box(text, csv_success_bounds=None):
    if text == "auto":
        return csv_success_bounds if csv_success_bounds is not None else DEFAULT_SAFE_BOX
    values = [float(value) for value in text.split(",") if value.strip()]
    if len(values) != 6:
        raise ValueError("--safe_box must be 'auto' or 'xmin,xmax,ymin,ymax,zmin,zmax'")
    return {
        "x": [values[0], values[1]],
        "y": [values[2], values[3]],
        "z": [values[4], values[5]],
    }


def load_csv_points(csv_path, stride, draw_points):
    if not csv_path:
        return [], None
    points = []
    success_values = {"x": [], "y": [], "z": []}
    stride = max(1, int(stride))
    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        for index, row in enumerate(csv.DictReader(file)):
            raw_success = int(float(row.get("raw_ik_success", row.get("success", "0")) or 0))
            success = int(float(row.get("success", "0") or 0))
            local_point = np.array(
                [
                    float(row["target_x"]),
                    float(row["target_y"]),
                    float(row["target_z"]),
                ],
                dtype=np.float32,
            )
            if success:
                success_values["x"].append(float(local_point[0]))
                success_values["y"].append(float(local_point[1]))
                success_values["z"].append(float(local_point[2]))
            if not draw_points or index % stride != 0:
                continue
            if success:
                color = (0.05, 0.70, 0.25)
            elif raw_success:
                color = (1.0, 0.62, 0.18)
            else:
                color = (0.95, 0.15, 0.12)
            points.append((local_point, color))
    success_bounds = None
    if success_values["x"]:
        success_bounds = {
            axis: [min(values), max(values)]
            for axis, values in success_values.items()
        }
    return points, success_bounds


def world_from_arm_base(env, local_points):
    root_pos = env.root_states[0, :3]
    root_quat = env.root_states[0, 3:7]
    arm_base_local = torch.tensor(robot_spec.ARM_BASE_OFFSET, device=env.device, dtype=torch.float)
    points = torch.tensor(local_points, device=env.device, dtype=torch.float)
    if points.ndim == 1:
        points = points.view(1, 3)
    arm_base_world = root_pos.view(1, 3) + quat_apply(root_quat.view(1, 4), arm_base_local.view(1, 3))
    root_quat_repeated = root_quat.view(1, 4).repeat(points.shape[0], 1)
    world_points = arm_base_world + quat_apply(root_quat_repeated, points)
    return world_points.detach().cpu().numpy()


def world_from_goal_center(env, local_points):
    points = torch.tensor(local_points, device=env.device, dtype=torch.float)
    if points.ndim == 1:
        points = points.view(1, 3)
    center_world = env._get_ee_goal_spherical_center()[0:1]
    yaw_quat_repeated = env.base_yaw_quat[0:1].repeat(points.shape[0], 1)
    world_points = center_world + quat_apply(yaw_quat_repeated, points)
    return world_points.detach().cpu().numpy()


def add_lines(gym, viewer, env_handle, line_segments, color):
    if not line_segments:
        return
    vertices = np.asarray(line_segments, dtype=np.float32).reshape(-1, 3)
    colors = np.tile(np.asarray(color, dtype=np.float32), (len(line_segments), 1))
    gym.add_lines(viewer, env_handle, len(line_segments), vertices, colors)


def hemisphere_line_segments(radius, segments, rings):
    segments = max(8, int(segments))
    rings = max(2, int(rings))
    line_segments = []

    # Meridians: x = r cos(theta), radial direction sweeps the y-z plane.
    theta_values = np.linspace(0.0, math.pi / 2.0, segments + 1)
    for phi in np.linspace(0.0, 2.0 * math.pi, rings * 2, endpoint=False):
        pts = []
        for theta in theta_values:
            pts.append(
                [
                    radius * math.cos(theta),
                    radius * math.sin(theta) * math.cos(phi),
                    radius * math.sin(theta) * math.sin(phi),
                ]
            )
        line_segments.extend(zip(pts[:-1], pts[1:]))

    # Latitude rings on the hemisphere surface.
    for theta in np.linspace(math.pi / (2.0 * rings), math.pi / 2.0, rings):
        ring = []
        ring_radius = radius * math.sin(theta)
        x = radius * math.cos(theta)
        for phi in np.linspace(0.0, 2.0 * math.pi, segments + 1):
            ring.append([x, ring_radius * math.cos(phi), ring_radius * math.sin(phi)])
        line_segments.extend(zip(ring[:-1], ring[1:]))

    return [[a, b] for a, b in line_segments]


def hemisphere_emphasis_segments(radius, segments):
    segments = max(16, int(segments))
    line_segments = []

    # Front half-sphere boundary rim at x=0.
    rim = []
    for phi in np.linspace(0.0, 2.0 * math.pi, segments + 1):
        rim.append([0.0, radius * math.cos(phi), radius * math.sin(phi)])
    line_segments.extend(zip(rim[:-1], rim[1:]))

    # Three easy-to-read section curves.
    theta_values = np.linspace(0.0, math.pi / 2.0, segments + 1)
    for phi in [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]:
        pts = []
        for theta in theta_values:
            pts.append(
                [
                    radius * math.cos(theta),
                    radius * math.sin(theta) * math.cos(phi),
                    radius * math.sin(theta) * math.sin(phi),
                ]
            )
        line_segments.extend(zip(pts[:-1], pts[1:]))

    return [[a, b] for a, b in line_segments]


def box_line_segments(box):
    xs, ys, zs = box["x"], box["y"], box["z"]
    corners = [
        [x, y, z]
        for x in xs
        for y in ys
        for z in zs
    ]
    line_segments = []
    for index, a in enumerate(corners):
        for b in corners[index + 1 :]:
            if sum(abs(a[axis] - b[axis]) > 1e-9 for axis in range(3)) == 1:
                line_segments.append([a, b])
    return line_segments


def box_corners(box):
    return [
        [x, y, z]
        for x in box["x"]
        for y in box["y"]
        for z in box["z"]
    ]


def box_grid_segments(box, subdivisions):
    line_segments = box_line_segments(box)
    subdivisions = max(1, int(subdivisions))
    axes = ["x", "y", "z"]
    for fixed_axis in axes:
        sweep_axes = [axis for axis in axes if axis != fixed_axis]
        a_axis, b_axis = sweep_axes
        for fixed_value in box[fixed_axis]:
            for i in range(subdivisions + 1):
                a_value = box[a_axis][0] + (box[a_axis][1] - box[a_axis][0]) * i / subdivisions
                start = {fixed_axis: fixed_value, a_axis: a_value, b_axis: box[b_axis][0]}
                end = {fixed_axis: fixed_value, a_axis: a_value, b_axis: box[b_axis][1]}
                line_segments.append([[start["x"], start["y"], start["z"]], [end["x"], end["y"], end["z"]]])
            for i in range(subdivisions + 1):
                b_value = box[b_axis][0] + (box[b_axis][1] - box[b_axis][0]) * i / subdivisions
                start = {fixed_axis: fixed_value, a_axis: box[a_axis][0], b_axis: b_value}
                end = {fixed_axis: fixed_value, a_axis: box[a_axis][1], b_axis: b_value}
                line_segments.append([[start["x"], start["y"], start["z"]], [end["x"], end["y"], end["z"]]])
    return line_segments


def task_box_from_cfg(env_cfg):
    if getattr(env_cfg.goal_ee, "command_mode", "sphere") != "cart":
        return None
    ranges = env_cfg.goal_ee.ranges
    return {
        "x": list(ranges.pos_x),
        "y": list(ranges.pos_y_cart),
        "z": list(ranges.pos_z),
    }


def draw_local_lines(env, local_segments, color):
    if not local_segments:
        return
    flat_points = [point for segment in local_segments for point in segment]
    world = world_from_arm_base(env, flat_points)
    world_segments = [[world[i], world[i + 1]] for i in range(0, len(world), 2)]
    add_lines(env.gym, env.viewer, env.envs[0], world_segments, color)


def draw_goal_center_lines(env, local_segments, color):
    if not local_segments:
        return
    flat_points = [point for segment in local_segments for point in segment]
    world = world_from_goal_center(env, flat_points)
    world_segments = [[world[i], world[i + 1]] for i in range(0, len(world), 2)]
    add_lines(env.gym, env.viewer, env.envs[0], world_segments, color)


def draw_frame_lines(env, local_segments, color, frame):
    if frame == "goal_center":
        draw_goal_center_lines(env, local_segments, color)
    else:
        draw_local_lines(env, local_segments, color)


def draw_sphere(env, local_point, radius, color):
    world = world_from_arm_base(env, local_point)[0]
    sphere_geom = gymutil.WireframeSphereGeometry(radius, 6, 6, None, color=color)
    sphere_pose = gymapi.Transform(gymapi.Vec3(float(world[0]), float(world[1]), float(world[2])), r=None)
    gymutil.draw_lines(sphere_geom, env.gym, env.viewer, env.envs[0], sphere_pose)


def draw_goal_center_sphere(env, local_point, radius, color):
    world = world_from_goal_center(env, local_point)[0]
    sphere_geom = gymutil.WireframeSphereGeometry(radius, 6, 6, None, color=color)
    sphere_pose = gymapi.Transform(gymapi.Vec3(float(world[0]), float(world[1]), float(world[2])), r=None)
    gymutil.draw_lines(sphere_geom, env.gym, env.viewer, env.envs[0], sphere_pose)


def draw_frame_sphere(env, local_point, radius, color, frame):
    if frame == "goal_center":
        draw_goal_center_sphere(env, local_point, radius, color)
    else:
        draw_sphere(env, local_point, radius, color)


def draw_workspace(
    env,
    hemisphere_segments,
    hemisphere_emphasis,
    box_segments,
    task_box_segments,
    task_box_corners,
    csv_points,
    csv_frame,
    radius,
    point_radius,
    task_corner_radius,
):
    env.gym.clear_lines(env.viewer)
    draw_local_lines(env, hemisphere_segments, color=(0.0, 0.95, 1.0))
    draw_local_lines(env, hemisphere_emphasis, color=(1.0, 1.0, 1.0))
    draw_frame_lines(env, box_segments, color=(0.65, 0.10, 0.85), frame=csv_frame)
    draw_goal_center_lines(env, task_box_segments, color=(1.0, 0.72, 0.0))

    # Axes from arm-base: x red, y green, z blue.
    axis_len = min(0.25, radius * 0.35)
    draw_local_lines(env, [[[0, 0, 0], [axis_len, 0, 0]]], color=(1.0, 0.0, 0.0))
    draw_local_lines(env, [[[0, 0, 0], [0, axis_len, 0]]], color=(0.0, 0.9, 0.0))
    draw_local_lines(env, [[[0, 0, 0], [0, 0, axis_len]]], color=(0.0, 0.35, 1.0))
    draw_sphere(env, [0.0, 0.0, 0.0], radius=0.025, color=(0.0, 0.0, 0.0))
    if task_box_segments:
        draw_goal_center_sphere(env, [0.0, 0.0, 0.0], radius=0.022, color=(1.0, 0.72, 0.0))
        for corner in task_box_corners:
            draw_goal_center_sphere(env, corner, radius=task_corner_radius, color=(1.0, 0.48, 0.0))

    if csv_points:
        for local_point, color in csv_points:
            draw_frame_sphere(env, local_point, radius=point_radius, color=color, frame=csv_frame)

    # Current low-level EE target and measured EE, both in world state converted to local only by direct sphere draw.
    if hasattr(env, "curr_ee_goal_cart_world"):
        goal_world = env.curr_ee_goal_cart_world[0].detach().cpu().numpy()
        goal_geom = gymutil.WireframeSphereGeometry(0.022, 6, 6, None, color=(1.0, 0.95, 0.0))
        goal_pose = gymapi.Transform(gymapi.Vec3(float(goal_world[0]), float(goal_world[1]), float(goal_world[2])), r=None)
        gymutil.draw_lines(goal_geom, env.gym, env.viewer, env.envs[0], goal_pose)
    if hasattr(env, "ee_pos"):
        ee_world = env.ee_pos[0].detach().cpu().numpy()
        ee_geom = gymutil.WireframeSphereGeometry(0.020, 6, 6, None, color=(0.0, 0.1, 1.0))
        ee_pose = gymapi.Transform(gymapi.Vec3(float(ee_world[0]), float(ee_world[1]), float(ee_world[2])), r=None)
        gymutil.draw_lines(ee_geom, env.gym, env.viewer, env.envs[0], ee_pose)


def task_world_bounds(env, task_box):
    if task_box is None:
        return None
    corners = [
        [x, y, z]
        for x in task_box["x"]
        for y in task_box["y"]
        for z in task_box["z"]
    ]
    world = world_from_goal_center(env, corners)
    return {
        "x": [float(world[:, 0].min()), float(world[:, 0].max())],
        "y": [float(world[:, 1].min()), float(world[:, 1].max())],
        "z": [float(world[:, 2].min()), float(world[:, 2].max())],
    }


def print_summary(env, radius, csv_points, safe_box, task_box, csv_frame):
    print("\n=== Go2-X5 arm workspace viewer ===")
    print(f"arm-base origin: {robot_spec.ARM_BASE_OFFSET}")
    print(f"front hemisphere radius: {radius:.4f} m")
    print(f"safe box: {safe_box}")
    print(f"csv/safe-box frame: {csv_frame}")
    print(f"task sampling box local: {task_box}")
    print(f"task sampling box world: {task_world_bounds(env, task_box)}")
    print(f"csv points drawn: {len(csv_points)}")
    print("\nViewer markers:")
    print("  cyan wireframe: user-proposed front hemisphere")
    print("  purple box: safe IK scan box")
    print("  orange box: current low-level task sampling range")
    print("  orange sphere: terrain-invariant task sampling center")
    print("  black sphere: arm-base origin")
    print("  red/green/blue axes: arm-base x/y/z")
    print("  green/orange/red small spheres: IK success / rejected / failed CSV points in csv_frame")
    print("  yellow sphere: current low-level EE target")
    print("  blue sphere: actual EE")
    print("\nKeyboard: ESC quit, F free camera, 0 look at robot.\n")


def main():
    cli = parse_args()
    args = make_legged_gym_args(cli)
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg = configure_env(env_cfg, cli)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    radius = cli.radius if cli.radius > 0.0 else estimate_arm_radius_from_urdf()
    csv_points, csv_success_bounds = load_csv_points(cli.csv, cli.csv_stride, cli.draw_csv_points)
    safe_box = parse_safe_box(cli.safe_box, csv_success_bounds=csv_success_bounds)
    task_box = None if cli.hide_task_box else task_box_from_cfg(env_cfg)
    hemisphere_segments = [] if cli.hide_hemisphere else hemisphere_line_segments(radius, cli.hemisphere_segments, cli.hemisphere_rings)
    hemisphere_emphasis = [] if cli.hide_hemisphere else hemisphere_emphasis_segments(radius, cli.hemisphere_segments)
    box_segments = [] if cli.hide_safe_box else box_line_segments(safe_box)
    task_box_segments = box_grid_segments(task_box, cli.task_box_grid) if task_box is not None else []
    task_box_corners = box_corners(task_box) if task_box is not None else []
    print_summary(env, radius, csv_points, safe_box, task_box, cli.csv_frame)

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for step in range(cli.max_iterations):
        start = time.time()
        env.step(actions)
        draw_workspace(
            env,
            hemisphere_segments,
            hemisphere_emphasis,
            box_segments,
            task_box_segments,
            task_box_corners,
            csv_points,
            cli.csv_frame,
            radius,
            cli.point_radius,
            cli.task_corner_radius,
        )
        env.render(sync_frame_time=True)
        if step % 200 == 0:
            print(f"step={step:05d}", flush=True)
        time.sleep(max(0.02 - (time.time() - start), 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
