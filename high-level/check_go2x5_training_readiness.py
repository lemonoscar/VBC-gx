#!/usr/bin/env python3
"""Fail-closed multi-environment smoke gate for Go2-X5 high-level training."""

from isaacgym import gymapi  # noqa: F401 - Isaac Gym must be imported before torch
from isaacgym import gymtorch  # noqa: F401

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from envs import Go2X5PickMulti
from utils.config import load_cfg


HIGH_LEVEL_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = HIGH_LEVEL_ROOT / "data/cfg/go2x5_pickmulti.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the production Go2-X5 high-level environment before training"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--num-envs", type=int, default=33)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonfinite_record(name, value):
    if value is None or not torch.is_tensor(value):
        return {"count": 0, "first_index": None}
    invalid = (~torch.isfinite(value)).nonzero(as_tuple=False)
    return {
        "count": int(invalid.shape[0]),
        "first_index": invalid[0].detach().cpu().tolist() if invalid.shape[0] else None,
    }


def collect_nonfinite(env, observation, reward):
    arm_slice = slice(
        -(6 + env.num_gripper_joints), -env.num_gripper_joints
    )
    fields = {
        "observation": observation,
        "reward": reward,
        "root_state": env._root_states,
        "dof_state": env._dof_state,
        "ee_pose": torch.cat([env.ee_pos, env.ee_orn], dim=-1),
        "ee_target": env.ee_goal_world,
        "jacobian": env.ee_j_eef,
        "low_observation": env.low_obs_buf,
        "low_history": env.low_obs_history_buf,
        "low_action": env.last_low_actions,
        "leg_torque": getattr(env, "torques", None),
        "arm_q": env._dof_pos[:, arm_slice],
        "arm_q_target": env.arm_q_command,
        "features": getattr(env, "feature_obs", None),
    }
    return {name: nonfinite_record(name, value) for name, value in fields.items()}


def merge_nonfinite(aggregate, current):
    for name, details in current.items():
        previous = aggregate.setdefault(name, {"count": 0, "first_index": None})
        previous["count"] += details["count"]
        if previous["first_index"] is None and details["first_index"] is not None:
            previous["first_index"] = details["first_index"]


def main():
    args = parse_args()
    if args.num_envs < 2:
        raise ValueError("--num-envs must be at least 2 to exercise batched runtime paths")
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Low-level checkpoint does not exist: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"High-level config does not exist: {config}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.chdir(HIGH_LEVEL_ROOT)
    cfg = load_cfg(str(config))
    cfg["env"]["numEnvs"] = args.num_envs
    cfg["env"]["enableDebugVis"] = False
    cfg["env"]["cameraMode"] = "full"
    cfg["env"]["smallValueSetZero"] = True
    cfg["env"]["wandb"] = False
    cfg["env"]["useTanh"] = False
    cfg["env"]["near_goal_stop"] = False
    cfg["env"]["obj_move_prob"] = 0.0
    cfg["env"]["low_policy_path"] = str(checkpoint)

    env = Go2X5PickMulti(
        cfg=cfg,
        rl_device=args.rl_device,
        sim_device=args.sim_device,
        graphics_device_id=args.graphics_device_id,
        headless=True,
        use_roboinfo=True,
        observe_gait_commands=False,
        no_feature=False,
        mask_arm=False,
        pitch_control=False,
        rand_control=True,
        arm_delay=False,
        robot_start_pose=None,
        rand_cmd_scale=False,
        rand_depth_clip=False,
        stop_pick=True,
        table_height=None,
        eval=False,
    )

    observation_dict = env.reset()
    observation = observation_dict["obs"]
    action = torch.zeros(
        env.num_envs, env.num_actions, device=env.device, dtype=torch.float
    )
    nonfinite = {}
    merge_nonfinite(
        nonfinite, collect_nonfinite(env, observation, torch.zeros(env.num_envs, device=env.device))
    )

    total_resets = 0
    reset_steps = {}
    maximum_ee_error = float(torch.norm(env.ee_goal_world - env.ee_pos, dim=-1).max())
    maximum_finger_contact = 0.0
    minimum_object_table_margin = float(
        (env._cube_root_states[:, 2] - env.table_heights).min()
    )

    for step in range(1, args.steps + 1):
        observation_dict, reward, done, _ = env.step(action)
        observation = observation_dict["obs"]
        merge_nonfinite(nonfinite, collect_nonfinite(env, observation, reward))

        done_count = int(done.count_nonzero().item())
        if done_count:
            total_resets += done_count
            reset_steps[str(step)] = done_count

        maximum_ee_error = max(
            maximum_ee_error,
            float(torch.norm(env.ee_goal_world - env.ee_pos, dim=-1).max()),
        )
        minimum_object_table_margin = min(
            minimum_object_table_margin,
            float((env._cube_root_states[:, 2] - env.table_heights).min()),
        )
        if env.finger_indices:
            finger_force = torch.norm(
                env._contact_forces[:, env.finger_indices, :], dim=-1
            ).sum(dim=-1)
            maximum_finger_contact = max(
                maximum_finger_contact, float(finger_force.max())
            )

        if done_count:
            env.reset_idx(done.nonzero(as_tuple=False).flatten())

    arm_slice = slice(
        -(6 + env.num_gripper_joints), -env.num_gripper_joints
    )
    arm_lower = env.dof_limits_lower[arm_slice].unsqueeze(0)
    arm_upper = env.dof_limits_upper[arm_slice].unsqueeze(0)
    arm_limit_violation = torch.logical_or(
        env.arm_q_command < arm_lower - 1.0e-7,
        env.arm_q_command > arm_upper + 1.0e-7,
    )
    nonfinite_count = sum(details["count"] for details in nonfinite.values())
    expected_observation_dim = int(env.observation_space.shape[0])
    observation_shape_ok = list(observation.shape) == [
        args.num_envs,
        expected_observation_dim,
    ]
    low_observation_shape_ok = list(env.low_obs_buf.shape) == [
        args.num_envs,
        env.num_proprio * (env.history_len + 1),
    ]

    report = {
        "schema_version": 1,
        "task": "Go2X5PickMulti",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "observation_shape": list(observation.shape),
        "expected_observation_dim": expected_observation_dim,
        "low_observation_shape": list(env.low_obs_buf.shape),
        "total_resets": total_resets,
        "reset_steps": reset_steps,
        "maximum_ee_error_m": maximum_ee_error,
        "maximum_finger_contact_force_n": maximum_finger_contact,
        "minimum_object_table_margin_m": minimum_object_table_margin,
        "arm_target_limit_violations": int(arm_limit_violation.count_nonzero().item()),
        "nonfinite": nonfinite,
        "nonfinite_count": nonfinite_count,
        "passed": (
            total_resets == 0
            and nonfinite_count == 0
            and observation_shape_ok
            and low_observation_shape_ok
            and not bool(arm_limit_violation.any())
        ),
    }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
