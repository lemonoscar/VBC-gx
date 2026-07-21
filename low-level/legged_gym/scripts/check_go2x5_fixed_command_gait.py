#!/usr/bin/env python3
"""Fail-closed fixed-command gait evaluation for a trained Go2-X5 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import check_go2x5_training_readiness as readiness
import torch

from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import OnPolicyRunner


CASES = (
    ("stand", 0.0, 0.0),
    ("forward", 0.10, 0.0),
    ("backward", -0.10, 0.0),
    ("turn_left", 0.0, 0.15),
    ("turn_right", 0.0, -0.15),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measure-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--min-translation-progress-ratio", type=float, default=0.35)
    parser.add_argument("--min-yaw-progress-ratio", type=float, default=0.35)
    parser.add_argument("--max-swing-contact-fraction", type=float, default=0.75)
    parser.add_argument("--min-swing-height", type=float, default=0.04)
    parser.add_argument("--max-stand-vx-error", type=float, default=0.03)
    parser.add_argument("--max-stand-yaw-error", type=float, default=0.03)
    parser.add_argument("--max-moving-vx-error", type=float, default=0.04)
    parser.add_argument("--max-moving-yaw-error", type=float, default=0.05)
    parser.add_argument("--max-collision-raw-mean", type=float, default=0.10)
    parser.add_argument(
        "--require-gait-shape",
        action="store_true",
        help=(
            "Also gate on legacy desired-contact swing metrics. The current "
            "Go2-X5 profile leaves this off because it does not prescribe a gait."
        ),
    )
    parser.add_argument(
        "--safety-only",
        action="store_true",
        help="Report gait quality without using it as an exit-code gate.",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/go2x5_fixed_command_gait.json"))
    return parser.parse_args()


def set_fixed_command(env, vx, yaw):
    # This evaluator injects an exact command after every reset/step, so normal
    # categorical command sampling must be disabled before reset_idx calls it.
    env.cfg.commands.standing_probability = 0.0
    env.cfg.commands.turn_in_place_probability = 0.0
    env.command_ranges["lin_vel_x"] = [vx, vx]
    env.command_ranges["ang_vel_yaw"] = [yaw, yaw]
    env.commands.zero_()
    env.commands[:, 0] = vx
    env.commands[:, 2] = yaw


def nonfinite_details(tensors):
    result = {}
    first = None
    for name, tensor in tensors.items():
        invalid = ~torch.isfinite(tensor)
        count = int(invalid.sum().item())
        result[name] = count
        if count and first is None:
            first = {"field": name, "index": torch.nonzero(invalid, as_tuple=False)[0].tolist()}
    return result, first


def evaluate_case(env, policy, name, vx, yaw, cli):
    set_fixed_command(env, vx, yaw)
    env.reset_idx(torch.arange(env.num_envs, device=env.device), start=True)
    set_fixed_command(env, vx, yaw)
    env.compute_observations()
    obs = env.obs_buf

    totals = {
        "base_vx": 0.0,
        "base_yaw_rate": 0.0,
        "vx_abs_error": 0.0,
        "yaw_abs_error": 0.0,
        "swing_contact": 0.0,
        "swing_weight": 0.0,
        "swing_height": 0.0,
        "stance_speed": 0.0,
        "stance_weight": 0.0,
        "collision": 0.0,
    }
    early_resets = 0
    reset_causes = {"roll": 0, "pitch": 0, "z": 0, "contact": 0}
    nonfinite = {}
    first_nonfinite = None
    foot_cache_max_error = 0.0

    total_steps = cli.warmup_steps + cli.measure_steps
    for step in range(total_steps):
        actions = policy(obs.detach(), hist_encoding=True)
        obs, _, _, _, dones, _ = env.step(actions.detach())
        set_fixed_command(env, vx, yaw)

        tensors = {
            "observation": obs,
            "policy_action": actions,
            "root_state": env.root_states,
            "dof_state": env.dof_state,
            "leg_torque": env.torques[:, :12],
            "ee_pose": env.rigid_body_state[:, env.gripper_idx, :7],
            "jacobian": env.ee_j_eef,
        }
        counts, first = nonfinite_details(tensors)
        for field, count in counts.items():
            nonfinite[field] = nonfinite.get(field, 0) + count
        if first_nonfinite is None and first is not None:
            first_nonfinite = {"step": step, **first}

        non_timeout = dones.bool() & ~env.time_out_buf.bool()
        early_resets += int(non_timeout.sum().item())
        reset_causes["roll"] += int((non_timeout & env.reset_roll_buf).sum().item())
        reset_causes["pitch"] += int((non_timeout & env.reset_pitch_buf).sum().item())
        reset_causes["z"] += int((non_timeout & env.reset_z_buf).sum().item())
        if hasattr(env, "reset_contact_buf"):
            reset_causes["contact"] += int(
                (non_timeout & env.reset_contact_buf).sum().item()
            )
        live_foot_velocity = torch.index_select(
            env.rigid_body_state[:, :, 7:10], 1, env.feet_indices
        )
        foot_cache_max_error = max(
            foot_cache_max_error,
            readiness.scalar_max_abs(env.foot_velocities - live_foot_velocity),
        )
        if step < cli.warmup_steps:
            continue

        base_vx = env.base_lin_vel[:, 0]
        yaw_rate = env.base_ang_vel[:, 2]
        totals["base_vx"] += float(base_vx.mean().item())
        totals["base_yaw_rate"] += float(yaw_rate.mean().item())
        totals["vx_abs_error"] += float(torch.abs(base_vx - vx).mean().item())
        totals["yaw_abs_error"] += float(torch.abs(yaw_rate - yaw).mean().item())
        collision, _ = env.reward_container._reward_collision()
        totals["collision"] += float(collision.mean().item())

        desired_contact = env.desired_contact_states
        swing_weight = 1.0 - desired_contact
        stance_weight = desired_contact
        contacts = env.foot_contacts_from_sensor.float()
        totals["swing_contact"] += float((swing_weight * contacts).sum().item())
        totals["swing_weight"] += float(swing_weight.sum().item())

        terrain_height = env.measured_heights.mean(dim=1, keepdim=True)
        foot_height = env.rigid_body_state[:, env.feet_indices, 2] - terrain_height
        totals["swing_height"] += float((swing_weight * foot_height).sum().item())
        foot_speed = torch.norm(live_foot_velocity, dim=2)
        totals["stance_speed"] += float((stance_weight * foot_speed).sum().item())
        totals["stance_weight"] += float(stance_weight.sum().item())

    samples = float(cli.measure_steps)
    base_vx_mean = totals["base_vx"] / samples
    yaw_rate_mean = totals["base_yaw_rate"] / samples
    translation_ratio = base_vx_mean / vx if abs(vx) > 0.0 else None
    yaw_ratio = yaw_rate_mean / yaw if abs(yaw) > 0.0 else None
    swing_contact_fraction = (
        totals["swing_contact"] / totals["swing_weight"]
        if totals["swing_weight"] > 0.0 else None
    )
    swing_height_mean = (
        totals["swing_height"] / totals["swing_weight"]
        if totals["swing_weight"] > 0.0 else None
    )
    safety_passed = (
        early_resets == 0
        and sum(nonfinite.values()) == 0
        and foot_cache_max_error <= 1.0e-7
        and totals["collision"] / samples <= cli.max_collision_raw_mean
    )
    vx_abs_error_mean = totals["vx_abs_error"] / samples
    yaw_abs_error_mean = totals["yaw_abs_error"] / samples
    tracking_checks = []
    if abs(vx) == 0.0 and abs(yaw) == 0.0:
        tracking_checks.extend(
            (
                vx_abs_error_mean <= cli.max_stand_vx_error,
                yaw_abs_error_mean <= cli.max_stand_yaw_error,
            )
        )
    if translation_ratio is not None:
        tracking_checks.append(translation_ratio >= cli.min_translation_progress_ratio)
        tracking_checks.append(vx_abs_error_mean <= cli.max_moving_vx_error)
    if yaw_ratio is not None:
        tracking_checks.append(yaw_ratio >= cli.min_yaw_progress_ratio)
        tracking_checks.append(yaw_abs_error_mean <= cli.max_moving_yaw_error)
    gait_shape_checks = []
    if swing_contact_fraction is not None:
        gait_shape_checks.append(swing_contact_fraction <= cli.max_swing_contact_fraction)
    if swing_height_mean is not None:
        gait_shape_checks.append(swing_height_mean >= cli.min_swing_height)
    tracking_passed = all(tracking_checks)
    gait_shape_passed = all(gait_shape_checks)
    behavior_passed = tracking_passed and (
        not cli.require_gait_shape or gait_shape_passed
    )

    return {
        "case": name,
        "command": {"vx": vx, "yaw": yaw},
        "base_vx_mean": base_vx_mean,
        "base_yaw_rate_mean": yaw_rate_mean,
        "vx_abs_error_mean": vx_abs_error_mean,
        "yaw_abs_error_mean": yaw_abs_error_mean,
        "translation_progress_ratio": translation_ratio,
        "yaw_progress_ratio": yaw_ratio,
        "swing_contact_fraction": swing_contact_fraction,
        "swing_height_mean": swing_height_mean,
        "stance_foot_speed_mean": (
            totals["stance_speed"] / totals["stance_weight"]
            if totals["stance_weight"] > 0.0 else None
        ),
        "collision_raw_mean": totals["collision"] / samples,
        "foot_velocity_cache_max_error": foot_cache_max_error,
        "early_resets": early_resets,
        "reset_causes": reset_causes,
        "nonfinite": nonfinite,
        "nonfinite_count": sum(nonfinite.values()),
        "first_nonfinite": first_nonfinite,
        "safety_passed": safety_passed,
        "tracking_passed": tracking_passed,
        "gait_shape_evaluated": bool(cli.require_gait_shape),
        "gait_shape_passed": gait_shape_passed,
        "behavior_passed": behavior_passed,
        "passed": safety_passed and (cli.safety_only or behavior_passed),
    }


def run(cli):
    args = readiness.make_env_args(cli)
    env_cfg, train_cfg = task_registry.get_cfgs(name="go2x5")
    env_cfg = readiness.configure_env(env_cfg, cli)
    env_cfg.commands.resampling_time = 1000.0
    env, _ = task_registry.make_env(name="go2x5", args=args, env_cfg=env_cfg)
    runner = OnPolicyRunner(env, class_to_dict(train_cfg), log_dir=None, device=cli.rl_device)
    runner.load(str(cli.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device, stochastic=False)

    cases = [evaluate_case(env, policy, *case, cli) for case in CASES]
    report = {
        "schema_version": 1,
        "checkpoint": str(cli.checkpoint),
        "checkpoint_iteration": int(runner.current_learning_iteration),
        "curriculum_stage": int(env.curriculum_stage_index),
        "num_envs": cli.num_envs,
        "warmup_steps": cli.warmup_steps,
        "measure_steps": cli.measure_steps,
        "safety_only": cli.safety_only,
        "require_gait_shape": cli.require_gait_shape,
        "thresholds": {
            "min_translation_progress_ratio": cli.min_translation_progress_ratio,
            "min_yaw_progress_ratio": cli.min_yaw_progress_ratio,
            "max_swing_contact_fraction": cli.max_swing_contact_fraction,
            "min_swing_height": cli.min_swing_height,
            "max_stand_vx_error": cli.max_stand_vx_error,
            "max_stand_yaw_error": cli.max_stand_yaw_error,
            "max_moving_vx_error": cli.max_moving_vx_error,
            "max_moving_yaw_error": cli.max_moving_yaw_error,
            "max_collision_raw_mean": cli.max_collision_raw_mean,
        },
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
