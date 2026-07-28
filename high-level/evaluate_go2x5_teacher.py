#!/usr/bin/env python3
"""Deterministic, task-level evaluation for a Go2-X5 teacher checkpoint."""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--low-policy-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="data/cfg/go2x5_pickmulti.yaml"
    )
    parser.add_argument("--num-envs", type=int, default=264)
    parser.add_argument("--episodes-per-env", type=int, default=5)
    parser.add_argument("--max-episode-length", type=int, default=150)
    parser.add_argument("--policy-step", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    args = parser.parse_args(argv)

    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.episodes_per_env <= 0:
        parser.error("--episodes-per-env must be positive")
    if args.max_episode_length <= 0:
        parser.error("--max-episode-length must be positive")
    if args.policy_step < 0:
        parser.error("--policy-step must be non-negative")
    for field in ("checkpoint", "low_policy_path", "config"):
        if not Path(getattr(args, field)).is_file():
            parser.error("{} does not exist: {}".format(field, getattr(args, field)))
    if Path(args.output).exists():
        parser.error("--output already exists: {}".format(args.output))
    return args


def build_trainer_argv(args, experiment_dir):
    """Build the production trainer arguments used to instantiate the environment."""
    return [
        "--task", "Go2X5PickMulti",
        "--config", args.config,
        "--low_policy_path", args.low_policy_path,
        "--rl_device", args.rl_device,
        "--sim_device", args.sim_device,
        "--graphics_device_id", str(args.graphics_device_id),
        "--timesteps", "1",
        "--num_envs", str(args.num_envs),
        "--seed", str(args.seed),
        "--headless",
        "--experiment_dir", str(experiment_dir),
        "--wandb_name", Path(args.checkpoint).stem,
        "--roboinfo",
        "--small_value_set_zero",
        "--stop_pick",
    ]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_episode_records(records):
    """Summarize a non-empty list of JSON-compatible episode records."""
    if not records:
        return {
            "episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
        }

    outcome_counts = {
        name: sum(record["outcome"] == name for record in records)
        for name in ("success", "timeout", "object_fall", "other")
    }
    successes = outcome_counts["success"]
    min_distances = [record["min_ee_object_distance_m"] for record in records]
    max_lifts = [record["max_lift_margin_m"] for record in records]
    returns = [record["return"] for record in records]
    lengths = [record["length"] for record in records]
    contact_episodes = sum(record["finger_contact"] for record in records)
    close_episodes = sum(record["gripper_closed"] for record in records)
    reached_episodes = sum(record["reached"] for record in records)
    lifted_episodes = sum(record["lifted"] for record in records)

    return {
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "outcomes": outcome_counts,
        "reach_rate": reached_episodes / len(records),
        "lift_rate": lifted_episodes / len(records),
        "finger_contact_rate": contact_episodes / len(records),
        "gripper_close_rate": close_episodes / len(records),
        "mean_return": statistics.fmean(returns),
        "mean_episode_length": statistics.fmean(lengths),
        "mean_min_ee_object_distance_m": statistics.fmean(min_distances),
        "p50_min_ee_object_distance_m": percentile(min_distances, 0.5),
        "p90_min_ee_object_distance_m": percentile(min_distances, 0.9),
        "mean_max_lift_margin_m": statistics.fmean(max_lifts),
        "p50_max_lift_margin_m": percentile(max_lifts, 0.5),
        "p90_max_lift_margin_m": percentile(max_lifts, 0.9),
    }


def first_nonfinite(fields):
    import torch

    checks = []
    names = []
    for name, value in fields:
        if value is None or not torch.is_tensor(value):
            continue
        names.append(name)
        checks.append(torch.isfinite(value).all())
    if not checks or torch.stack(checks).all().item():
        return None

    for name, value in fields:
        if value is None or not torch.is_tensor(value):
            continue
        bad = torch.nonzero(~torch.isfinite(value), as_tuple=False)
        if bad.numel():
            index = bad[0].tolist()
            return {
                "field": name,
                "index": index,
                "value": float(value[tuple(index)].item()),
            }
    return {"field": names[0], "index": [], "value": None}


def _finger_contact_force(raw_env):
    import torch

    if not getattr(raw_env, "finger_indices", None):
        return torch.zeros(raw_env.num_envs, device=raw_env.device)
    forces = raw_env._contact_forces[:, raw_env.finger_indices, :]
    return torch.norm(forces, dim=-1).sum(dim=-1)


def evaluate(args):
    from train_multistate import get_trainer
    import torch
    from skrl.utils import set_seed
    from utils.config import get_params

    set_seed(args.seed, deterministic=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_dir = output_path.parent / "skrl"
    trainer_args = get_params(build_trainer_argv(args, experiment_dir))
    trainer = get_trainer(is_eval=True, args=trainer_args)

    env = trainer.env
    raw_env = env._env
    agent = trainer.agents
    raw_env.max_episode_length = args.max_episode_length
    raw_env.cfg["env"]["maxEpisodeLength"] = args.max_episode_length
    raw_env.global_step_counter = args.policy_step
    agent.load(args.checkpoint)
    agent.set_running_mode("eval")

    object_names = list(raw_env.obj_list)
    if args.num_envs % len(object_names):
        raise ValueError(
            "num_envs must be a multiple of the {} object types".format(
                len(object_names)
            )
        )

    states, _ = env.reset()
    device = raw_env.device
    num_envs = raw_env.num_envs
    episode_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_returns = torch.zeros(num_envs, dtype=torch.float, device=device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
    min_ee_distance = torch.full(
        (num_envs,), float("inf"), dtype=torch.float, device=device
    )
    max_lift_margin = torch.full(
        (num_envs,), -float("inf"), dtype=torch.float, device=device
    )
    max_finger_contact = torch.zeros(num_envs, dtype=torch.float, device=device)
    gripper_closed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    records = []
    maximum_action_abs = 0.0
    started = time.time()
    step_limit = args.max_episode_length * (args.episodes_per_env + 1)

    for step in range(step_limit):
        active = episode_counts < args.episodes_per_env
        with torch.no_grad():
            actions = agent.act(
                states, timestep=step, timesteps=step_limit
            )[0]
            next_states, rewards, terminated, truncated, infos = env.step(actions)

        reward_values = rewards.reshape(-1)
        done = (terminated | truncated).reshape(-1).bool()
        ee_distance = torch.norm(
            raw_env._cube_root_states[:, :3] - raw_env.ee_pos, dim=-1
        )
        lift_margin = (
            raw_env._cube_root_states[:, 2]
            - raw_env.table_heights
            - raw_env.init_height
        )
        finger_contact = _finger_contact_force(raw_env)
        object_fall = (
            raw_env._cube_root_states[:, 2]
            < raw_env.table_heights - raw_env.object_fall_tolerance
        )

        nonfinite = first_nonfinite([
            ("observation", next_states),
            ("policy_action", actions),
            ("reward", rewards),
            ("root_state", raw_env._root_states),
            ("dof_state", raw_env._dof_state),
            ("ee_pose", raw_env.ee_pos),
            ("object_pose", raw_env._cube_root_states),
            ("ee_jacobian", raw_env.ee_j_eef),
            ("low_observation", getattr(raw_env, "low_obs_buf", None)),
            ("arm_target", getattr(raw_env, "arm_q_command", None)),
        ])
        if nonfinite is not None:
            raise FloatingPointError(
                "non-finite value at evaluation step {}: {}".format(
                    step, nonfinite
                )
            )

        maximum_action_abs = max(
            maximum_action_abs, float(actions.abs().max().item())
        )
        episode_returns[active] += reward_values[active]
        episode_lengths[active] += 1
        min_ee_distance[active] = torch.minimum(
            min_ee_distance[active], ee_distance[active]
        )
        max_lift_margin[active] = torch.maximum(
            max_lift_margin[active], lift_margin[active]
        )
        max_finger_contact[active] = torch.maximum(
            max_finger_contact[active], finger_contact[active]
        )
        gripper_closed |= active & (raw_env.actions[:, 6] < 0)

        completed_ids = torch.nonzero(done & active, as_tuple=False).flatten()
        if completed_ids.numel():
            completed_cpu = completed_ids.cpu().tolist()
            success_cpu = raw_env.lifted_object[completed_ids].cpu().tolist()
            timeout_cpu = raw_env.timeout_buf[completed_ids].cpu().tolist()
            fall_cpu = object_fall[completed_ids].cpu().tolist()
            return_cpu = episode_returns[completed_ids].cpu().tolist()
            length_cpu = episode_lengths[completed_ids].cpu().tolist()
            distance_cpu = min_ee_distance[completed_ids].cpu().tolist()
            lift_cpu = max_lift_margin[completed_ids].cpu().tolist()
            contact_cpu = max_finger_contact[completed_ids].cpu().tolist()
            close_cpu = gripper_closed[completed_ids].cpu().tolist()

            for index, env_id in enumerate(completed_cpu):
                success = bool(success_cpu[index])
                timeout = bool(timeout_cpu[index])
                fell = bool(fall_cpu[index])
                if success:
                    outcome = "success"
                elif timeout:
                    outcome = "timeout"
                elif fell:
                    outcome = "object_fall"
                else:
                    outcome = "other"
                records.append({
                    "env_id": env_id,
                    "object": object_names[env_id % len(object_names)],
                    "outcome": outcome,
                    "return": float(return_cpu[index]),
                    "length": int(length_cpu[index]),
                    "min_ee_object_distance_m": float(distance_cpu[index]),
                    "max_lift_margin_m": float(lift_cpu[index]),
                    "finger_contact": bool(contact_cpu[index] > 1.0),
                    "gripper_closed": bool(close_cpu[index]),
                    "reached": bool(
                        distance_cpu[index] < raw_env.success_ee_dist_threshold
                    ),
                    "lifted": bool(
                        lift_cpu[index] > raw_env.lifted_success_threshold
                    ),
                })

            episode_counts[completed_ids] += 1
            episode_returns[completed_ids] = 0
            episode_lengths[completed_ids] = 0
            min_ee_distance[completed_ids] = float("inf")
            max_lift_margin[completed_ids] = -float("inf")
            max_finger_contact[completed_ids] = 0
            gripper_closed[completed_ids] = False

        if torch.all(episode_counts >= args.episodes_per_env).item():
            final_step = step + 1
            break

        if done.any().item():
            states, _ = env.reset()
        else:
            states = next_states
    else:
        final_step = step_limit

    completed = bool(
        torch.all(episode_counts >= args.episodes_per_env).item()
    )
    per_object = {}
    for object_name in object_names:
        per_object[object_name] = summarize_episode_records([
            record for record in records if record["object"] == object_name
        ])

    report = {
        "schema_version": 1,
        "evaluation": "go2x5_teacher_deterministic",
        "completed": completed,
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": sha256_file(args.checkpoint),
        },
        "low_policy": {
            "path": str(Path(args.low_policy_path).resolve()),
            "sha256": sha256_file(args.low_policy_path),
        },
        "seed": args.seed,
        "policy_step": args.policy_step,
        "policy_mode": "deterministic_mean_action",
        "random_control_frequency": False,
        "environment": {
            "num_envs": num_envs,
            "object_types": len(object_names),
            "replicas_per_object": num_envs // len(object_names),
            "episodes_per_env": args.episodes_per_env,
            "expected_episodes": num_envs * args.episodes_per_env,
            "completed_episodes": len(records),
            "max_episode_length": args.max_episode_length,
            "domain_randomization_profile": raw_env.cfg["env"][
                "domainRandomization"
            ]["profile"],
        },
        "success_definition": {
            "minimum_lift_margin_m": raw_env.lifted_success_threshold,
            "maximum_ee_object_distance_m": raw_env.success_ee_dist_threshold,
        },
        "contact_threshold_n": 1.0,
        "overall": summarize_episode_records(records),
        "per_object": per_object,
        "runtime": {
            "environment_steps": final_step,
            "elapsed_seconds": time.time() - started,
            "maximum_policy_action_abs": maximum_action_abs,
            "nonfinite_count": 0,
            "min_completed_episodes_per_env": int(
                episode_counts.min().item()
            ),
            "max_completed_episodes_per_env": int(
                episode_counts.max().item()
            ),
        },
    }
    return report


def main(argv=None):
    args = parse_args(argv)
    report = evaluate(args)
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "completed": report["completed"],
        "checkpoint": report["checkpoint"]["path"],
        "episodes": report["overall"]["episodes"],
        "success_rate": report["overall"]["success_rate"],
        "output": str(output_path),
    }, sort_keys=True))
    if not report["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
