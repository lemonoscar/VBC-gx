#!/usr/bin/env python3
"""Run one honest, scripted Go2-X5 pick in the production high-level environment.

The script uses ground-truth object pose to command the existing 9D high-level
action interface. It does not load or claim a learned high-level policy, and it
never moves the object after the initial deterministic placement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path


HIGH_LEVEL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = HIGH_LEVEL_ROOT.parent
DEFAULT_CONFIG = HIGH_LEVEL_ROOT / "data/cfg/go2x5_pickmulti.yaml"
PHASE_STEPS = (
    ("settle", 18),
    ("approach", 20),
    ("descend", 15),
    ("close", 10),
    ("lift", 20),
    ("hold", 12),
)


def phase_at(step, phases=PHASE_STEPS):
    """Return ``(name, offset, duration)`` for a zero-based controller step."""
    if step < 0:
        raise ValueError("step must be non-negative")
    cursor = 0
    for name, duration in phases:
        if duration <= 0:
            raise ValueError("phase durations must be positive")
        if step < cursor + duration:
            return name, step - cursor, duration
        cursor += duration
    raise IndexError(f"step {step} is outside the {cursor}-step schedule")


def longest_true_run(values):
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def should_close_gripper(phase, ee_distance, preclose_distance):
    if phase in {"close", "lift", "hold"}:
        return True
    return (
        phase == "descend"
        and preclose_distance > 0.0
        and ee_distance <= preclose_distance
    )


def update_gripper_latch(latched, phase, ee_distance, preclose_distance):
    return bool(
        latched
        or should_close_gripper(phase, ee_distance, preclose_distance)
    )


def evaluate_pick_trace(
    lift_margins,
    ee_distances,
    finger_forces,
    *,
    minimum_lift=0.10,
    maximum_ee_distance=0.12,
    minimum_finger_force=0.5,
    required_hold_steps=6,
):
    """Apply the fail-closed physical success rule used by the runtime script."""
    if not (
        len(lift_margins) == len(ee_distances) == len(finger_forces)
        and lift_margins
    ):
        raise ValueError("trace fields must be non-empty and have equal lengths")
    if any(len(forces) != 2 for forces in finger_forces):
        raise ValueError("exactly two physical finger forces are required")

    held = [
        lift >= minimum_lift and distance <= maximum_ee_distance
        for lift, distance in zip(lift_margins, ee_distances)
    ]
    max_finger_forces = [
        max(forces[index] for forces in finger_forces) for index in range(2)
    ]
    simultaneous_contact_steps = sum(
        all(force >= minimum_finger_force for force in forces)
        for forces in finger_forces
    )
    longest_hold = longest_true_run(held)
    return {
        "longest_lift_hold_steps": longest_hold,
        "max_finger_contact_force_n": max_finger_forces,
        "simultaneous_two_finger_contact_steps": simultaneous_contact_steps,
        "lift_hold_passed": longest_hold >= required_hold_steps,
        "two_finger_contact_passed": simultaneous_contact_steps > 0,
        "passed": (
            longest_hold >= required_hold_steps
            and simultaneous_contact_steps > 0
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--video", default="")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--object", default="glue_1")
    parser.add_argument("--object-x-offset", type=float, default=0.0)
    parser.add_argument("--object-y-offset", type=float, default=0.0)
    parser.add_argument("--table-height", type=float, default=0.15)
    parser.add_argument("--object-yaw", type=float, default=math.pi / 2.0)
    parser.add_argument("--approach-distance", type=float, default=0.10)
    parser.add_argument("--grasp-standoff", type=float, default=0.035)
    parser.add_argument("--grasp-z-offset", type=float, default=0.0)
    parser.add_argument("--lift-height", type=float, default=0.18)
    parser.add_argument("--target-roll", type=float, default=0.0)
    parser.add_argument("--target-pitch", type=float, default=1.25)
    parser.add_argument("--target-yaw", type=float, default=0.0)
    parser.add_argument("--preclose-ee-distance", type=float, default=0.13)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    args = parser.parse_args(argv)

    for field in ("checkpoint", "config"):
        path = Path(getattr(args, field)).expanduser()
        if not path.is_file():
            parser.error(f"--{field.replace('_', '-')} does not exist: {path}")
    for field in ("report", "video"):
        value = getattr(args, field)
        if value and Path(value).expanduser().exists():
            parser.error(f"--{field} already exists: {value}")
    if not 0.10 <= args.table_height <= 0.20:
        parser.error("--table-height must be within the task range [0.10, 0.20]")
    for field in ("approach_distance", "grasp_standoff", "lift_height"):
        if getattr(args, field) <= 0.0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if not 1.0 <= args.target_pitch <= 1.5:
        parser.error("--target-pitch must be within the trained range [1.0, 1.5]")
    for field in ("target_roll", "target_yaw"):
        if abs(getattr(args, field)) > 0.35:
            parser.error(
                f"--{field.replace('_', '-')} must be within the trained "
                "delta range [-0.35, 0.35]"
            )
    if args.preclose_ee_distance < 0.0:
        parser.error("--preclose-ee-distance must be non-negative")
    return args


def bootstrap_runtime():
    """Make the bundled Isaac Gym libraries visible before importing torch."""
    bindings = (
        REPO_ROOT
        / "third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64"
    )
    usd_plugins = bindings / "usd/plugins"
    library_paths = [str(bindings), str(usd_plugins)]
    if os.environ.get("CONDA_PREFIX"):
        library_paths.append(str(Path(os.environ["CONDA_PREFIX"]) / "lib"))
    existing = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        library_paths + [path for path in existing if path]
    )
    if os.environ.get("_ISAACGYM_LIBRARY_PATH_BOOTSTRAPPED") != "1":
        os.environ["_ISAACGYM_LIBRARY_PATH_BOOTSTRAPPED"] = "1"
        os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)

    for path in (
        HIGH_LEVEL_ROOT,
        REPO_ROOT / "third_party/isaacgym/python",
        REPO_ROOT / "third_party/rsl_rl",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_nonfinite(fields, torch):
    for name, value in fields:
        if value is None or not torch.is_tensor(value):
            continue
        bad = torch.nonzero(~torch.isfinite(value), as_tuple=False)
        if bad.numel():
            index = bad[0].tolist()
            return {
                "field": name,
                "index": index,
                "value": repr(float(value[tuple(index)].item())),
            }
    return None


def run(args):
    bootstrap_runtime()
    from isaacgym import gymapi, gymtorch
    from isaacgym.torch_utils import (
        euler_from_quat,
        quat_apply,
        quat_from_euler_xyz,
        quat_mul,
        quat_rotate_inverse,
    )
    import cv2
    import numpy as np
    import torch

    from envs import Go2X5PickMulti
    from utils.config import load_cfg

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve() if args.video else None
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if video_path:
        video_path.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(HIGH_LEVEL_ROOT)
    cfg = load_cfg(str(config))
    objects = cfg["env"]["asset"]["asset_multi"]
    if args.object not in objects:
        raise ValueError(
            f"unknown object '{args.object}'; available: {sorted(objects)}"
        )
    for axis, value in (
        ("X", args.object_x_offset),
        ("Y", args.object_y_offset),
    ):
        limits = cfg["env"][f"objectPositionRange{axis}"]
        if not float(limits[0]) <= value <= float(limits[1]):
            raise ValueError(
                f"object {axis.lower()} offset {value} is outside configured "
                f"range {limits}"
            )
    cfg["env"]["asset"]["asset_multi"] = {args.object: objects[args.object]}
    cfg["env"]["numEnvs"] = 1
    cfg["env"]["maxEpisodeLength"] = sum(duration for _, duration in PHASE_STEPS) + 20
    cfg["env"]["holdSteps"] = cfg["env"]["maxEpisodeLength"]
    cfg["env"]["enableDebugVis"] = False
    cfg["env"]["cameraMode"] = "full"
    cfg["env"]["smallValueSetZero"] = True
    cfg["env"]["wandb"] = False
    cfg["env"]["useTanh"] = False
    cfg["env"]["near_goal_stop"] = False
    cfg["env"]["obj_move_prob"] = 0.0
    cfg["env"]["low_policy_path"] = str(checkpoint)
    cfg["env"]["tableHeightRange"] = [args.table_height, args.table_height]
    cfg["env"]["objectPositionRangeX"] = [0.0, 0.0]
    cfg["env"]["objectPositionRangeY"] = [0.0, 0.0]
    cfg["env"]["robotResetPositionRangeXY"] = [0.0, 0.0]
    cfg["env"]["robotResetYawRange"] = 0.0
    cfg["sensor"]["enableCamera"] = False
    cfg["record_video"] = bool(video_path)

    env = Go2X5PickMulti(
        cfg=cfg,
        rl_device=args.rl_device,
        sim_device=args.sim_device,
        graphics_device_id=args.graphics_device_id,
        headless=True,
        use_roboinfo=True,
        observe_gait_commands=False,
        no_feature=True,
        mask_arm=False,
        pitch_control=False,
        rand_control=False,
        arm_delay=False,
        robot_start_pose=None,
        rand_cmd_scale=False,
        rand_depth_clip=False,
        stop_pick=True,
        table_height=args.table_height,
        eval=False,
    )

    observation_dict = env.reset()
    observation = observation_dict["obs"]

    # One deterministic initial placement is allowed; the object is never
    # written again after this simulator state update.
    env._cube_root_states[:, 0] = (
        env._table_root_states[:, 0] + args.object_x_offset
    )
    env._cube_root_states[:, 1] = (
        env._table_root_states[:, 1] + args.object_y_offset
    )
    env._cube_root_states[:, 2] = env.table_heights + env.init_height
    yaw = torch.full((env.num_envs,), args.object_yaw, device=env.device)
    yaw_quat = quat_from_euler_xyz(0.0 * yaw, 0.0 * yaw, yaw)
    env._cube_root_states[:, 3:7] = quat_mul(yaw_quat, env.init_quat)
    env._cube_root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._root_states),
        gymtorch.unwrap_tensor(env._cube_actor_ids),
        env.num_envs,
    )

    initial_goal_world = env.ee_goal_world.clone()
    initial_heading = torch.stack(euler_from_quat(env.base_yaw_quat), dim=-1)[:, 2]
    target_orientation = torch.tensor(
        [args.target_roll, args.target_pitch, args.target_yaw],
        device=env.device,
        dtype=torch.float,
    ).repeat(env.num_envs, 1)
    approach_direction_local = torch.tensor(
        [
            math.cos(args.target_yaw) * math.cos(args.target_pitch),
            math.sin(args.target_yaw) * math.cos(args.target_pitch),
            -math.sin(args.target_pitch),
        ],
        device=env.device,
        dtype=torch.float,
    )

    writer = None
    trace = []
    lift_margins = []
    ee_distances = []
    finger_forces_trace = []
    first_bad = None
    reset_step = None
    gripper_latched = False
    reference_object_world = None
    reference_object_z = None
    waypoints = None

    def capture_frame(phase, lift_margin, ee_distance, finger_forces):
        nonlocal writer
        if video_path is None:
            return
        root_pos = env._robot_root_states[0, :3].detach().cpu().numpy()
        camera_position = root_pos + np.array([0.12, 1.05, 0.55])
        camera_target = root_pos + np.array([0.22, 0.0, 0.12])
        camera = env._rendering_camera_handles[0]
        env.gym.set_camera_location(
            camera,
            env.envs[0],
            gymapi.Vec3(*camera_position),
            gymapi.Vec3(*camera_target),
        )
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        image = env.gym.get_camera_image(
            env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR
        )
        height, packed_width = image.shape
        rgba = np.asarray(
            image.reshape([height, packed_width // 4, 4]), dtype=np.uint8
        )
        frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        cv2.putText(
            frame,
            f"phase: {phase}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"lift {100.0 * lift_margin:5.1f} cm  "
                f"ee {100.0 * ee_distance:4.1f} cm  "
                f"fingers {finger_forces[0]:.1f}/{finger_forces[1]:.1f} N"
            ),
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        if writer is None:
            height, width = frame.shape[:2]
            fps = 1.0 / (
                env.control_freq_low * env.control_freq_inv * env.sim_params.dt
            )
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"failed to open video writer: {video_path}")
        writer.write(frame)

    try:
        total_steps = sum(duration for _, duration in PHASE_STEPS)
        for step in range(total_steps):
            phase, offset, duration = phase_at(step)
            if phase != "settle" and waypoints is None:
                reference_object_world = env._cube_root_states[:, :3].clone()
                reference_object_z = reference_object_world[:, 2].clone()
                base_yaw = env.base_yaw_quat
                approach_direction_world = quat_apply(
                    base_yaw, approach_direction_local.repeat(env.num_envs, 1)
                )
                grasp = (
                    reference_object_world
                    - args.grasp_standoff * approach_direction_world
                )
                grasp[:, 2] += args.grasp_z_offset
                pregrasp = grasp - args.approach_distance * approach_direction_world
                lifted = grasp.clone()
                lifted[:, 2] += args.lift_height
                waypoints = {
                    "start": initial_goal_world.clone(),
                    "pregrasp": pregrasp,
                    "grasp": grasp,
                    "lift": lifted,
                }

            if phase == "settle":
                target_world = initial_goal_world
            elif phase == "approach":
                alpha = (offset + 1) / duration
                target_world = torch.lerp(
                    waypoints["start"], waypoints["pregrasp"], alpha
                )
            elif phase == "descend":
                alpha = (offset + 1) / duration
                target_world = torch.lerp(
                    waypoints["pregrasp"], waypoints["grasp"], alpha
                )
            elif phase == "close":
                target_world = waypoints["grasp"]
            elif phase == "lift":
                alpha = (offset + 1) / duration
                target_world = torch.lerp(
                    waypoints["grasp"], waypoints["lift"], alpha
                )
            else:
                target_world = waypoints["lift"]

            center = torch.cat(
                [
                    env._robot_root_states[:, :2],
                    torch.zeros(env.num_envs, 1, device=env.device),
                ],
                dim=1,
            )
            center += quat_apply(env.base_yaw_quat, env.ee_goal_center_offset)
            target_local = quat_rotate_inverse(
                env.base_yaw_quat, target_world - center
            )

            action = torch.zeros(
                env.num_envs, env.num_actions, device=env.device
            )
            position_error = target_local - env.curr_ee_goal_cart
            action[:, :3] = torch.clamp(position_error, -0.02, 0.02)
            orientation_error = target_orientation - env.curr_ee_goal_orn_rpy
            orientation_error = torch.atan2(
                torch.sin(orientation_error), torch.cos(orientation_error)
            )
            action[:, 3:6] = torch.clamp(orientation_error, -0.06, 0.06)
            ee_distance_before = float(
                torch.norm(
                    env._cube_root_states[0, :3] - env.ee_pos[0]
                ).item()
            )
            gripper_latched = update_gripper_latch(
                gripper_latched,
                phase,
                ee_distance_before,
                args.preclose_ee_distance,
            )
            gripper_closed = gripper_latched
            action[:, 6] = -1.0 if gripper_closed else 1.0
            if not gripper_closed:
                heading = torch.stack(
                    euler_from_quat(env.base_yaw_quat), dim=-1
                )[:, 2]
                heading_error = torch.atan2(
                    torch.sin(initial_heading - heading),
                    torch.cos(initial_heading - heading),
                )
                action[:, 8] = torch.clamp(heading_error, -0.10, 0.10)

            observation_dict, _, done, _ = env.step(action)
            observation = observation_dict["obs"]
            if reference_object_z is None:
                lift_margin_tensor = torch.zeros(env.num_envs, device=env.device)
            else:
                lift_margin_tensor = (
                    env._cube_root_states[:, 2] - reference_object_z
                )
            ee_distance_tensor = torch.norm(
                env._cube_root_states[:, :3] - env.ee_pos, dim=-1
            )
            if len(env.finger_indices) != 2:
                raise RuntimeError(
                    f"expected two finger bodies, got {env.finger_indices}"
                )
            finger_force_tensor = torch.norm(
                env._contact_forces[:, env.finger_indices, :], dim=-1
            )
            base_rpy = torch.stack(
                euler_from_quat(env._robot_root_states[:, 3:7]), dim=-1
            )
            ee_rpy = torch.stack(euler_from_quat(env.ee_orn), dim=-1)
            ik_height_error = torch.abs(env.ee_goal_world[:, 2] - env.ee_pos[:, 2])
            table_top = env._table_root_states[:, 2] + env.table_dimz / 2.0
            cube_fell = env._cube_root_states[:, 2] < (
                env.table_heights - env.object_fall_tolerance
            )

            first_bad = first_nonfinite(
                [
                    ("observation", observation),
                    ("action", action),
                    ("root_state", env._root_states),
                    ("dof_state", env._dof_state),
                    ("ee_pose", torch.cat([env.ee_pos, env.ee_orn], dim=-1)),
                    ("object_pose", env._cube_root_states),
                    ("jacobian", env.ee_j_eef),
                    ("arm_target", env.arm_q_command),
                    ("low_action", env.last_low_actions),
                ],
                torch,
            )

            lift_margin = float(lift_margin_tensor[0].item())
            ee_distance = float(ee_distance_tensor[0].item())
            finger_forces = finger_force_tensor[0].detach().cpu().tolist()
            lift_margins.append(lift_margin)
            ee_distances.append(ee_distance)
            finger_forces_trace.append(finger_forces)
            trace.append(
                {
                    "step": step,
                    "phase": phase,
                    "lift_margin_m": lift_margin,
                    "ee_object_distance_m": ee_distance,
                    "finger_contact_force_n": finger_forces,
                    "object_position": env._cube_root_states[0, :3]
                    .detach()
                    .cpu()
                    .tolist(),
                    "ee_position": env.ee_pos[0].detach().cpu().tolist(),
                    "ee_orientation_rpy": ee_rpy[0].detach().cpu().tolist(),
                    "ee_goal_world": env.ee_goal_world[0].detach().cpu().tolist(),
                    "base_position": env._robot_root_states[0, :3]
                    .detach()
                    .cpu()
                    .tolist(),
                    "base_orientation_rpy": base_rpy[0].detach().cpu().tolist(),
                    "gripper_closed_command": gripper_closed,
                    "gripper_dof_position": env._dof_pos[
                        0, -env.num_physical_gripper_dof:
                    ]
                    .detach()
                    .cpu()
                    .tolist(),
                    "table_top_height_m": float(table_top[0].item()),
                    "ik_height_error_m": float(ik_height_error[0].item()),
                    "termination_predicates": {
                        "roll": bool(torch.abs(base_rpy[0, 0]) > 0.8),
                        "pitch": bool(torch.abs(base_rpy[0, 1]) > 0.8),
                        "base_height": bool(
                            env._robot_root_states[0, 2] < 0.1
                        ),
                        "ik": bool(ik_height_error[0] > 0.2),
                        "object_fell": bool(cube_fell[0]),
                    },
                }
            )
            capture_frame(phase, lift_margin, ee_distance, finger_forces)

            if first_bad is not None:
                break
            if bool(done[0].item()):
                reset_step = step
                break
    finally:
        if writer is not None:
            writer.release()

    trace_result = evaluate_pick_trace(
        lift_margins,
        ee_distances,
        finger_forces_trace,
    )
    report = {
        "schema_version": 1,
        "demo_type": "scripted_ground_truth_high_level",
        "learned_high_level_policy": False,
        "production_low_level_loader": True,
        "script_object_state_writes_after_initial_placement": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config),
        "object": args.object,
        "object_position_offset_xy_m": [
            args.object_x_offset,
            args.object_y_offset,
        ],
        "table_height_m": args.table_height,
        "object_yaw_rad": args.object_yaw,
        "schedule": dict(PHASE_STEPS),
        "controller": {
            "target_orientation_local_rpy": target_orientation[0]
            .detach()
            .cpu()
            .tolist(),
            "approach_distance_m": args.approach_distance,
            "grasp_standoff_m": args.grasp_standoff,
            "grasp_z_offset_m": args.grasp_z_offset,
            "lift_height_m": args.lift_height,
            "preclose_ee_distance_m": args.preclose_ee_distance,
        },
        "thresholds": {
            "minimum_lift_m": 0.10,
            "maximum_ee_object_distance_m": 0.12,
            "minimum_finger_contact_force_n": 0.5,
            "required_hold_steps": 6,
        },
        "maximum_lift_margin_m": max(lift_margins),
        "final_lift_margin_m": lift_margins[-1],
        "minimum_ee_object_distance_m": min(ee_distances),
        "final_ee_object_distance_m": ee_distances[-1],
        **trace_result,
        "reset_step": reset_step,
        "first_nonfinite": first_bad,
        "video": str(video_path) if video_path else None,
        "trace": trace,
    }
    report["passed"] = (
        trace_result["passed"]
        and reset_step is None
        and first_bad is None
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {key: value for key, value in report.items() if key != "trace"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
