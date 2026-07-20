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
        "--max-early-resets",
        type=int,
        default=0,
        help="Fail when non-timeout resets exceed this count (default: fail closed at zero).",
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
    action_abs_sum = 0.0
    action_count = 0
    max_abs_action = 0.0
    max_abs_torque = 0.0
    ee_error_sum = 0.0
    ee_error_max = 0.0
    target_local_min = torch.full((3,), float("inf"), device=env.device)
    target_local_max = torch.full((3,), float("-inf"), device=env.device)

    for step in range(cli.steps):
        if cli.zero_policy:
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        else:
            actions = policy(obs.detach(), hist_encoding=True)
        tensors = {"observation": obs, "policy_action": actions}
        obs, _, _, _, dones, _ = env.step(actions.detach())
        collision, _ = env.reward_container._reward_collision()
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
        max_abs_action = max(max_abs_action, readiness.scalar_max_abs(actions))
        max_abs_torque = max(max_abs_torque, readiness.scalar_max_abs(env.torques[:, :12]))
        ee_error = torch.norm(env.ee_pos - env.curr_ee_goal_cart_world, dim=-1)
        ee_error_sum += float(ee_error.mean().item())
        ee_error_max = max(ee_error_max, float(ee_error.max().item()))
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

    report = {
        "schema_version": 1,
        "checkpoint": str(cli.checkpoint),
        "checkpoint_iteration": int(runner.current_learning_iteration),
        "curriculum_stage": int(env.curriculum_stage_index),
        "num_envs": cli.num_envs,
        "steps": cli.steps,
        "rough_terrain": cli.rough_terrain,
        "stochastic": cli.stochastic,
        "zero_policy": cli.zero_policy,
        "mean_collision_raw_per_tick": collision_sum / cli.steps,
        "mean_abs_policy_action": action_abs_sum / action_count,
        "max_abs_policy_action": max_abs_action,
        "max_abs_leg_torque": max_abs_torque,
        "mean_ee_error_m": ee_error_sum / cli.steps,
        "max_ee_error_m": ee_error_max,
        "expected_target_local_ranges": expected_target_ranges,
        "sampled_target_local_ranges": sampled_target_ranges,
        "target_bounds_ok": target_bounds_ok,
        "early_resets": early_resets,
        "max_early_resets": cli.max_early_resets,
        "reset_roll": reset_roll,
        "reset_pitch": reset_pitch,
        "reset_z": reset_z,
        "nonfinite": nonfinite,
        "nonfinite_count": sum(nonfinite.values()),
        "first_nonfinite": first_nonfinite,
    }
    report["passed"] = (
        report["nonfinite_count"] == 0
        and report["early_resets"] <= report["max_early_resets"]
        and report["target_bounds_ok"]
    )
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
