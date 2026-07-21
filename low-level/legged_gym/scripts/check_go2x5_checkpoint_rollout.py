#!/usr/bin/env python3
"""Evaluate a Go2-X5 training checkpoint with deterministic policy actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import check_go2x5_training_readiness as readiness

import torch

from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import OnPolicyRunner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--rough-terrain", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--zero-policy", action="store_true")
    parser.add_argument(
        "--privileged-latent",
        action="store_true",
        help="Diagnostic only: use privileged encoding instead of deployment history encoding.",
    )
    parser.add_argument("--require-coordination", action="store_true")
    parser.add_argument("--max-mean-ee-error-m", type=float, default=0.06)
    parser.add_argument("--max-vx-abs-error-mps", type=float, default=0.05)
    parser.add_argument("--max-yaw-abs-error-radps", type=float, default=0.05)
    parser.add_argument("--max-height-error-m", type=float, default=0.03)
    parser.add_argument("--max-pitch-error-rad", type=float, default=0.06)
    parser.add_argument("--min-height-correlation", type=float, default=0.30)
    parser.add_argument("--max-goal-z-pitch-correlation", type=float, default=-0.20)
    parser.add_argument(
        "--max-early-resets",
        type=int,
        default=0,
        help="Fail when non-timeout resets exceed this count (default: fail closed at zero).",
    )
    parser.add_argument(
        "--max-mean-collision-raw-per-tick",
        type=float,
        default=0.10,
        help="Maximum mean normalized non-foot collision force per policy tick.",
    )
    parser.add_argument(
        "--max-arm-target-clamp-fraction",
        type=float,
        default=0.05,
        help="Maximum fraction of six arm targets clamped at a URDF joint limit.",
    )
    parser.add_argument(
        "--max-action-saturation-fraction",
        type=float,
        default=0.05,
        help="Maximum fraction of policy actions with abs(action) >= 0.999.",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/go2x5_checkpoint_rollout.json"))
    return parser.parse_args()


def run(cli):
    args = readiness.make_env_args(cli)
    env_cfg, train_cfg = task_registry.get_cfgs(name="go2x5")
    env_cfg = readiness.configure_env(env_cfg, cli)
    if cli.rough_terrain:
        env_cfg.terrain.num_rows = 10
        env_cfg.terrain.num_cols = 20
        env_cfg.terrain.height = [0.0, 0.02]
    env, _ = task_registry.make_env(name="go2x5", args=args, env_cfg=env_cfg)
    runner = OnPolicyRunner(env, class_to_dict(train_cfg), log_dir=None, device=cli.rl_device)
    runner.load(str(cli.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device, stochastic=cli.stochastic)

    obs, _ = env.reset()
    nonfinite = {}
    first_nonfinite = None
    early_resets = 0
    reset_roll = 0
    reset_pitch = 0
    reset_z = 0
    collision_sum = 0.0
    collision_by_body_sum = torch.zeros(
        len(env.penalized_contact_indices), device=env.device
    )
    action_abs_sum = 0.0
    action_count = 0
    action_saturation_count = 0
    arm_target_count = 0
    arm_target_clamp_count = torch.zeros(6, device=env.device)
    max_abs_action = 0.0
    max_abs_torque = 0.0
    ee_error_sum = 0.0
    ee_error_max = 0.0
    vx_error_sum = 0.0
    yaw_error_sum = 0.0
    height_error_sum = 0.0
    pitch_error_sum = 0.0
    moment_count = 0
    goal_z_sum = 0.0
    base_height_sum = 0.0
    goal_z_sq_sum = 0.0
    base_height_sq_sum = 0.0
    goal_z_base_height_sum = 0.0
    base_pitch_sum = 0.0
    base_pitch_sq_sum = 0.0
    goal_z_base_pitch_sum = 0.0
    target_local_min = torch.full((3,), float("inf"), device=env.device)
    target_local_max = torch.full((3,), float("-inf"), device=env.device)

    for step in range(cli.steps):
        if cli.zero_policy:
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        else:
            actions = policy(obs.detach(), hist_encoding=not cli.privileged_latent)
        tensors = {"observation": obs, "policy_action": actions}
        obs, _, _, _, dones, _ = env.step(actions.detach())
        collision, _ = env.reward_container._reward_collision()
        collision_threshold = float(env.cfg.rewards.collision_force_threshold)
        collision_soft_clip = float(env.cfg.rewards.collision_soft_clip)
        penalized_forces = torch.norm(
            env.contact_forces[:, env.penalized_contact_indices, :], dim=-1
        )
        collision_by_body_sum += torch.clamp(
            penalized_forces - collision_threshold,
            min=0.0,
            max=collision_soft_clip,
        ).mean(dim=0) / max(collision_threshold, 1e-6)
        tensors.update(
            {
                "next_observation": obs,
                "collision": collision,
                "torque": env.torques[:, :12],
                "root_state": env.root_states,
                "dof_state": env.dof_state,
                "ee_pose": env.rigid_body_state[:, env.gripper_idx, :7],
                "ee_target": env.curr_ee_goal_cart_world,
                "jacobian": env.ee_j_eef,
                "arm_q_target": env.arm_q_target,
                "arm_q_target_unclamped": env.arm_q_target_unclamped,
            }
        )
        for name, tensor in tensors.items():
            count = int((~torch.isfinite(tensor)).sum().item())
            nonfinite[name] = nonfinite.get(name, 0) + count
            if count and first_nonfinite is None:
                first_nonfinite = {
                    "field": name,
                    "step": step,
                    "index": torch.nonzero(~torch.isfinite(tensor), as_tuple=False)[0].tolist(),
                }

        non_timeout = dones.bool() & ~env.time_out_buf
        early_resets += int(non_timeout.sum().item())
        reset_roll += int((non_timeout & env.reset_roll_buf).sum().item())
        reset_pitch += int((non_timeout & env.reset_pitch_buf).sum().item())
        reset_z += int((non_timeout & env.reset_z_buf).sum().item())
        collision_sum += float(collision.mean().item())
        action_abs_sum += float(actions.abs().sum().item())
        action_count += actions.numel()
        action_saturation_count += int((torch.abs(actions) >= 0.999).sum().item())
        max_abs_action = max(max_abs_action, readiness.scalar_max_abs(actions))
        max_abs_torque = max(max_abs_torque, readiness.scalar_max_abs(env.torques[:, :12]))
        arm_target_clamp_count += env.arm_q_target_clamped.sum(dim=0)
        arm_target_count += env.arm_q_target_clamped.shape[0]
        ee_error = torch.norm(env.ee_pos - env.curr_ee_goal_cart_world, dim=-1)
        ee_error_sum += float(ee_error.mean().item())
        ee_error_max = max(ee_error_max, float(ee_error.max().item()))
        terrain_height = env.reward_container._terrain_height()
        goal_z = env.curr_ee_goal_cart_world[:, 2] - terrain_height
        base_height = env.root_states[:, 2] - terrain_height
        adaptive_height = env.reward_container._adaptive_body_height_target()
        body_pitch = env._get_body_orientation()[:, 1]
        adaptive_pitch = env.reward_container._adaptive_body_pitch_target()
        vx_error_sum += float(torch.abs(env.base_lin_vel[:, 0] - env.commands[:, 0]).mean().item())
        yaw_error_sum += float(torch.abs(env.base_ang_vel[:, 2] - env.commands[:, 2]).mean().item())
        height_error_sum += float(torch.abs(base_height - adaptive_height).mean().item())
        pitch_error_sum += float(torch.abs(body_pitch - adaptive_pitch).mean().item())
        moment_count += goal_z.numel()
        goal_z_sum += float(goal_z.sum().item())
        base_height_sum += float(base_height.sum().item())
        goal_z_sq_sum += float(torch.sum(goal_z ** 2).item())
        base_height_sq_sum += float(torch.sum(base_height ** 2).item())
        goal_z_base_height_sum += float(torch.sum(goal_z * base_height).item())
        base_pitch_sum += float(body_pitch.sum().item())
        base_pitch_sq_sum += float(torch.sum(body_pitch ** 2).item())
        goal_z_base_pitch_sum += float(torch.sum(goal_z * body_pitch).item())
        target_local_min = torch.minimum(target_local_min, env.curr_ee_goal_cart.min(dim=0).values)
        target_local_max = torch.maximum(target_local_max, env.curr_ee_goal_cart.max(dim=0).values)

    expected_target_ranges = [
        list(env.goal_ee_ranges[axis]) for axis in ("pos_x", "pos_y_cart", "pos_z")
    ]
    sampled_target_ranges = [
        [float(target_local_min[axis].item()), float(target_local_max[axis].item())]
        for axis in range(3)
    ]
    target_bounds_ok = all(
        expected[0] - 1.0e-6 <= sampled[0] <= sampled[1] <= expected[1] + 1.0e-6
        for expected, sampled in zip(expected_target_ranges, sampled_target_ranges)
    )

    covariance = goal_z_base_height_sum - goal_z_sum * base_height_sum / moment_count
    goal_variance = goal_z_sq_sum - goal_z_sum ** 2 / moment_count
    height_variance = base_height_sq_sum - base_height_sum ** 2 / moment_count
    denominator = max(goal_variance * height_variance, 0.0) ** 0.5
    height_correlation = covariance / denominator if denominator > 1e-12 else 0.0
    pitch_covariance = goal_z_base_pitch_sum - goal_z_sum * base_pitch_sum / moment_count
    pitch_variance = base_pitch_sq_sum - base_pitch_sum ** 2 / moment_count
    pitch_denominator = max(goal_variance * pitch_variance, 0.0) ** 0.5
    pitch_correlation = pitch_covariance / pitch_denominator if pitch_denominator > 1e-12 else 0.0
    report = {
        "schema_version": 2,
        "checkpoint": str(cli.checkpoint),
        "checkpoint_iteration": int(runner.current_learning_iteration),
        "curriculum_stage": int(env.curriculum_stage_index),
        "num_envs": cli.num_envs,
        "steps": cli.steps,
        "rough_terrain": cli.rough_terrain,
        "stochastic": cli.stochastic,
        "zero_policy": cli.zero_policy,
        "history_encoding": not cli.privileged_latent,
        "mean_collision_raw_per_tick": collision_sum / cli.steps,
        "mean_collision_raw_per_tick_by_body": {
            env.body_names[int(body_index.item())]: float(
                collision_by_body_sum[index].item() / cli.steps
            )
            for index, body_index in enumerate(env.penalized_contact_indices)
        },
        "mean_abs_policy_action": action_abs_sum / action_count,
        "action_saturation_fraction": action_saturation_count / max(action_count, 1),
        "max_abs_policy_action": max_abs_action,
        "max_abs_leg_torque": max_abs_torque,
        "arm_target_clamp_fraction": float(
            arm_target_clamp_count.sum().item() / max(arm_target_count * 6, 1)
        ),
        "arm_target_clamp_fraction_by_joint": {
            name: float(arm_target_clamp_count[index].item() / max(arm_target_count, 1))
            for index, name in enumerate(env.dof_names[-(6 + env.cfg.env.num_gripper_joints):-env.cfg.env.num_gripper_joints])
        },
        "mean_ee_error_m": ee_error_sum / cli.steps,
        "max_ee_error_m": ee_error_max,
        "mean_vx_abs_error_mps": vx_error_sum / cli.steps,
        "mean_yaw_abs_error_radps": yaw_error_sum / cli.steps,
        "mean_height_adaptation_error_m": height_error_sum / cli.steps,
        "mean_pitch_adaptation_error_rad": pitch_error_sum / cli.steps,
        "goal_z_base_height_correlation": height_correlation,
        "goal_z_base_pitch_correlation": pitch_correlation,
        "expected_target_local_ranges": expected_target_ranges,
        "sampled_target_local_ranges": sampled_target_ranges,
        "target_bounds_ok": target_bounds_ok,
        "early_resets": early_resets,
        "max_early_resets": cli.max_early_resets,
        "max_mean_collision_raw_per_tick": cli.max_mean_collision_raw_per_tick,
        "max_arm_target_clamp_fraction": cli.max_arm_target_clamp_fraction,
        "max_action_saturation_fraction": cli.max_action_saturation_fraction,
        "reset_roll": reset_roll,
        "reset_pitch": reset_pitch,
        "reset_z": reset_z,
        "nonfinite": nonfinite,
        "nonfinite_count": sum(nonfinite.values()),
        "first_nonfinite": first_nonfinite,
        "coordination_thresholds": {
            "max_mean_ee_error_m": cli.max_mean_ee_error_m,
            "max_vx_abs_error_mps": cli.max_vx_abs_error_mps,
            "max_yaw_abs_error_radps": cli.max_yaw_abs_error_radps,
            "max_height_error_m": cli.max_height_error_m,
            "max_pitch_error_rad": cli.max_pitch_error_rad,
            "min_height_correlation": cli.min_height_correlation,
            "max_goal_z_pitch_correlation": cli.max_goal_z_pitch_correlation,
        },
    }
    report["safety_passed"] = (
        report["nonfinite_count"] == 0
        and report["early_resets"] <= report["max_early_resets"]
        and report["mean_collision_raw_per_tick"] <= report["max_mean_collision_raw_per_tick"]
        and report["arm_target_clamp_fraction"] <= report["max_arm_target_clamp_fraction"]
        and report["action_saturation_fraction"] <= report["max_action_saturation_fraction"]
        and report["target_bounds_ok"]
        and report["max_abs_policy_action"] <= 1.000001
    )
    report["coordination_passed"] = (
        report["mean_ee_error_m"] <= cli.max_mean_ee_error_m
        and report["mean_vx_abs_error_mps"] <= cli.max_vx_abs_error_mps
        and report["mean_yaw_abs_error_radps"] <= cli.max_yaw_abs_error_radps
        and report["mean_height_adaptation_error_m"] <= cli.max_height_error_m
        and report["mean_pitch_adaptation_error_rad"] <= cli.max_pitch_error_rad
        and report["goal_z_base_height_correlation"] >= cli.min_height_correlation
        and report["goal_z_base_pitch_correlation"] <= cli.max_goal_z_pitch_correlation
    )
    report["passed"] = report["safety_passed"] and (
        not cli.require_coordination or report["coordination_passed"]
    )
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
