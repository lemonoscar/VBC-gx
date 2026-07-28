"""Runtime snapshot and structured comparison helpers for Go2-X5 parity.

The collectors deliberately use only public environment attributes already
maintained by the low/high wrappers. They do not advance simulation state.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np


SCHEMA_VERSION = 2
DEFAULT_ATOL = 1.0e-6
SMOKE_NUM_PROP = 66
SMOKE_NUM_PRIV = 18
SMOKE_HISTORY_LEN = 10

POLICY_LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
URDF_LEG_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
EXPECTED_POLICY_TO_URDF = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
PROBE_ACTION_POLICY_ORDER = [
    0.11, -0.07, 0.03, -0.05, 0.09, -0.02,
    0.04, -0.08, 0.06, -0.03, 0.10, -0.01,
]
LEG_Q_OFFSET_POLICY_ORDER = (0.01 * np.asarray([
    1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12,
])).tolist()
LEG_QD_POLICY_ORDER = (0.05 * np.asarray([
    1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6,
])).tolist()

CONTROLLER_CASES = {
    "C0": {"policy_mode": "zero", "description": "default pose, zero velocity and command"},
    "C1": {"policy_mode": "constant_probe", "description": "default pose with asymmetric action"},
    "C2": {"policy_mode": "constant_probe", "description": "asymmetric q/qd perturbation"},
    "C3": {"policy_mode": "linear_probe", "description": "rotated base, command and EE target"},
    "C4": {"policy_mode": "linear_probe", "description": "stable IK target requiring joint clamp"},
}


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def validate_schema_v2_checkpoint(checkpoint: Mapping[str, Any], expected_action_dim: int = 12) -> None:
    metadata = checkpoint.get("metadata") or {}
    alignment = metadata.get("go2x5_alignment") or {}
    if alignment.get("schema_version") != 2:
        raise ValueError("checkpoint must use go2x5 schema v2")
    if alignment.get("action_dim") != expected_action_dim:
        raise ValueError(f"checkpoint action_dim must be {expected_action_dim}")
    if alignment.get("num_arm_actions") != 0:
        raise ValueError("checkpoint num_arm_actions must be 0")
    if alignment.get("policy_output_tanh") is not True:
        raise ValueError("checkpoint policy_output_tanh must be true")
    contract = alignment.get("control_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("checkpoint has no control_contract")
    if canonical_json_sha256(contract) != alignment.get("control_contract_sha256"):
        raise ValueError("checkpoint control_contract hash is invalid")
    std = checkpoint.get("model_state_dict", {}).get("std")
    if std is None or int(_array(std).shape[-1]) != expected_action_dim:
        raise ValueError(f"checkpoint model output/std must be {expected_action_dim}D")


def smoke_actor_critic_kwargs() -> Dict[str, Any]:
    return {
        "continue_from_last_std": True, "init_std": [[0.8, 1.0, 1.0] * 4],
        "actor_hidden_dims": [128], "critic_hidden_dims": [128], "activation": "elu",
        "output_tanh": True, "leg_control_head_hidden_dims": [128, 128],
        "arm_control_head_hidden_dims": [128, 128], "priv_encoder_dims": [64, 20],
        "num_leg_actions": 12, "num_arm_actions": 0, "adaptive_arm_gains": False,
        "adaptive_arm_gains_scale": 10.0,
    }


def build_smoke_actor_critic(actor_critic_class: Any, seed: int = 20260713):
    import torch
    torch.manual_seed(seed)
    return actor_critic_class(
        SMOKE_NUM_PROP,
        SMOKE_NUM_PROP,
        12,
        **smoke_actor_critic_kwargs(),
        num_priv=SMOKE_NUM_PRIV,
        num_hist=SMOKE_HISTORY_LEN,
        num_prop=SMOKE_NUM_PROP,
    )


def policy_to_urdf_oracle(policy_action: Any) -> np.ndarray:
    """Name-derived policy -> URDF mapping; intentionally does not call runtime helpers."""
    expected = [POLICY_LEG_JOINT_NAMES.index(name) for name in URDF_LEG_JOINT_NAMES]
    if expected != EXPECTED_POLICY_TO_URDF:
        raise AssertionError(f"Unexpected Go2 joint permutation: {expected}")
    action = _array(policy_action)
    if action.shape[-1] != len(POLICY_LEG_JOINT_NAMES):
        raise ValueError(f"Expected 12D policy action, got shape {action.shape}")
    return action[..., expected]


def integrate_arm_command_oracle(
    command: Any,
    ik_delta: Any,
    gain: float,
    max_step: float,
    lower: Any,
    upper: Any,
) -> Dict[str, np.ndarray]:
    """Independent persistent arm-command update with rate and joint limits."""
    command_array = _array(command).astype(np.float64, copy=False)
    delta = float(gain) * _array(ik_delta).astype(np.float64, copy=False)
    if max_step > 0.0:
        delta = np.clip(delta, -float(max_step), float(max_step))
    unclamped = command_array + delta
    return {
        "delta": delta,
        "unclamped": unclamped,
        "target": np.clip(
            unclamped,
            _array(lower).astype(np.float64, copy=False),
            _array(upper).astype(np.float64, copy=False),
        ),
    }


def make_diagnostic_policy(mode: str, obs_dim: int, action_dim: int = 12,
                           seed: int = 20260713, scale: float = 0.05,
                           device: Any = None):
    """Return a deterministic torch policy and serializable policy metadata."""
    try:
        import torch
    except ImportError:
        torch = None

    if action_dim != 12:
        raise ValueError("Go2-X5 diagnostic policies require action_dim=12")
    metadata = {"mode": mode, "input_dim": int(obs_dim), "output_dim": action_dim}
    if mode == "zero":
        def policy(obs, hist_encoding=True):
            if torch is not None and hasattr(obs, "device"):
                return torch.zeros(obs.shape[0], action_dim, device=obs.device, dtype=obs.dtype)
            return np.zeros((obs.shape[0], action_dim), dtype=np.asarray(obs).dtype)
    elif mode == "constant_probe":
        values_np = np.asarray(PROBE_ACTION_POLICY_ORDER, dtype=np.float32)
        values_torch = torch.tensor(values_np, device=device) if torch is not None else None
        def policy(obs, hist_encoding=True):
            if torch is not None and hasattr(obs, "device"):
                return values_torch.to(device=obs.device, dtype=obs.dtype).expand(obs.shape[0], -1)
            return np.broadcast_to(
                values_np.astype(np.asarray(obs).dtype), (obs.shape[0], action_dim)
            ).copy()
    elif mode == "linear_probe":
        rng = np.random.default_rng(seed)
        weight_np = rng.standard_normal((action_dim, obs_dim), dtype=np.float32) / max(obs_dim, 1) ** 0.5
        bias_np = rng.standard_normal(action_dim, dtype=np.float32) * 0.1
        weight_torch = torch.from_numpy(weight_np).to(device=device) if torch is not None else None
        bias_torch = torch.from_numpy(bias_np).to(device=device) if torch is not None else None
        metadata.update({"seed": seed, "scale": scale})
        def policy(obs, hist_encoding=True):
            policy_obs = obs
            if obs_dim % (SMOKE_HISTORY_LEN + 1) == 0 and obs.shape[1] != obs_dim:
                num_prop = obs_dim // (SMOKE_HISTORY_LEN + 1)
                policy_obs = torch.cat([obs[:, :num_prop], obs[:, -num_prop * SMOKE_HISTORY_LEN:]], dim=-1) \
                    if torch is not None and hasattr(obs, "device") else np.concatenate(
                        [obs[:, :num_prop], obs[:, -num_prop * SMOKE_HISTORY_LEN:]], axis=-1
                    )
            if torch is not None and hasattr(obs, "device"):
                return scale * torch.tanh(
                    policy_obs @ weight_torch.to(obs).T + bias_torch.to(obs)
                )
            return scale * np.tanh(policy_obs @ weight_np.T + bias_np)
    else:
        raise ValueError(f"Unknown diagnostic policy mode: {mode}")
    return policy, metadata


def independent_pd_oracle(q: Any, qd: Any, default_q: Any, action_scale: Any,
                          action_urdf: Any, kp: Any, kd: Any, torque_limit: Any) -> Dict[str, Any]:
    arrays = [np.asarray(value, dtype=np.float64) for value in
              (q, qd, default_q, action_scale, action_urdf, kp, kd, torque_limit)]
    arrays = np.broadcast_arrays(*arrays)
    q_a, qd_a, default_a, scale_a, action_a, kp_a, kd_a, limit_a = arrays
    q_target = default_a + scale_a * action_a
    raw = kp_a * (q_target - q_a) - kd_a * qd_a
    clamped = np.clip(raw, -limit_a, limit_a)
    joints = []
    for index, name in enumerate(URDF_LEG_JOINT_NAMES):
        joints.append({
            "name": name, "q": float(q_a[index]), "qd": float(qd_a[index]),
            "default_q": float(default_a[index]), "scale": float(scale_a[index]),
            "action": float(action_a[index]), "kp": float(kp_a[index]),
            "kd": float(kd_a[index]), "raw_torque": float(raw[index]),
            "clamped_torque": float(clamped[index]), "torque_limit": float(limit_a[index]),
        })
    return {"q_target": q_target, "raw_torque": raw, "torque": clamped, "joints": joints}


def _quat_rotation(quaternion: Any) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.asarray([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


def ee_frame_oracle(base_position: Any, base_quaternion: Any, arm_base_offset: Any,
                    terrain_center: Any, ee_goal_local: Any) -> Dict[str, np.ndarray]:
    """Independent TERRAIN_INVARIANT_YAW world target and full-base local observation."""
    full_rotation = _quat_rotation(base_quaternion)
    x, y, z, w = np.asarray(base_quaternion, dtype=np.float64)
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    c, s = math.cos(yaw), math.sin(yaw)
    yaw_rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    arm_base_world = np.asarray(base_position) + yaw_rotation @ np.asarray(arm_base_offset)
    ee_goal_world = np.asarray(terrain_center) + yaw_rotation @ np.asarray(ee_goal_local)
    reconstructed_local = full_rotation.T @ (ee_goal_world - arm_base_world)
    return {
        "arm_base_world": arm_base_world,
        "ee_goal_world": ee_goal_world,
        "reconstructed_local": reconstructed_local,
    }


def nonfinite_details(fields: Mapping[str, Any]) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    counts: Dict[str, int] = {}
    failures: List[Dict[str, Any]] = []
    for name, value in fields.items():
        array = _array(value)
        if not np.issubdtype(array.dtype, np.number):
            counts[name] = 0
            continue
        mask = ~np.isfinite(array)
        count = int(mask.sum())
        counts[name] = count
        if count:
            first = tuple(int(index) for index in np.argwhere(mask)[0])
            failures.append({"path": name, "index": list(first), "value": str(array[first])})
    return counts, failures


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _numbers(value: Any) -> Any:
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return [float(value.x), float(value.y), float(value.z)]
    array = _array(value)
    if array.ndim == 0:
        item = array.item()
        return bool(item) if isinstance(item, (bool, np.bool_)) else float(item)
    return array.tolist()


def _actor_handle(env: Any) -> Any:
    handles = getattr(env, "robot_handles", None)
    if handles is None:
        handles = getattr(env, "actor_handles", None)
    if not handles:
        raise AttributeError("Environment exposes neither robot_handles nor actor_handles")
    return handles[0]


def _base_body_name(env: Any, side: str) -> str:
    if side == "low":
        return str(getattr(env.cfg.asset, "base_body_name", "trunk"))
    return str(env.cfg["env"]["asset"].get("baseBodyName", "trunk"))


def _property_record(name: str, index: int, prop: Any) -> Dict[str, Any]:
    record = {
        "name": name,
        "index": index,
        "mass": float(prop.mass),
        "com": _numbers(prop.com),
    }
    inertia = getattr(prop, "inertia", None)
    if inertia is not None:
        record["inertia"] = {
            "x": _numbers(inertia.x),
            "y": _numbers(inertia.y),
            "z": _numbers(inertia.z),
        }
    return record


def _dof_records(names: Iterable[str], props: Any) -> List[Dict[str, Any]]:
    fields = ("driveMode", "stiffness", "damping", "lower", "upper", "velocity", "effort")
    records = []
    for index, name in enumerate(names):
        record = {"name": name, "index": index}
        for field in fields:
            if field in props.dtype.names:
                value = props[field][index]
                record[field] = int(value) if field == "driveMode" else float(value)
        records.append(record)
    return records


def collect_runtime_snapshot(env: Any, side: str) -> Dict[str, Any]:
    """Collect runtime body/shape/DOF properties from env zero."""
    if side not in {"low", "high"}:
        raise ValueError("side must be 'low' or 'high'")
    env_handle = env.envs[0]
    actor_handle = _actor_handle(env)
    body_props = env.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
    shape_props = env.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
    dof_props = env.gym.get_actor_dof_properties(env_handle, actor_handle)
    body_names = list(env.body_names)
    base_name = _base_body_name(env, side)
    if base_name not in body_names:
        raise KeyError(f"Configured base body '{base_name}' not found in runtime body names")

    rigid_bodies = [_property_record(name, index, body_props[index]) for index, name in enumerate(body_names)]
    total_mass = sum(item["mass"] for item in rigid_bodies)
    shape_friction = [float(prop.friction) for prop in shape_props]

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "runtime_properties",
        "side": side,
        "base_body": {"name": base_name, "index": body_names.index(base_name)},
        "body_names": body_names,
        "rigid_bodies": rigid_bodies,
        "total_mass": total_mass,
        "shape_friction": shape_friction,
        "dofs": _dof_records(env.dof_names, dof_props),
    }


def _policy_order_action(env: Any, side: str) -> Tuple[np.ndarray, np.ndarray]:
    if side == "low":
        urdf_action = env.actions[:, :12]
        policy_action = env._reindex_all(urdf_action)[:, :12]
    else:
        urdf_action = env.last_low_actions[:, :12]
        policy_action = env._reindex_low_all(urdf_action)[:, :12]
    return _array(policy_action[0]), _array(urdf_action[0])


def _observation_parts(env: Any, side: str) -> Tuple[np.ndarray, np.ndarray]:
    if side == "low":
        full = env.obs_buf
        num_prop = int(env.cfg.env.num_proprio)
        history_len = int(env.cfg.env.history_len)
    else:
        full = env.low_obs_buf
        num_prop = int(env.num_proprio)
        history_len = int(env.history_len)
    current = full[:, :num_prop]
    history = full[:, -history_len * num_prop:]
    return _array(current[0]), _array(history[0])


def _arm_target(env: Any, side: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    num_gripper = int(env.cfg.env.num_gripper_joints) if side == "low" else int(env.num_gripper_joints)
    if side == "low":
        dpos = env.curr_ee_goal_cart_world - env.ee_pos
        import torch
        from isaacgym.torch_utils import orientation_error

        ee_orn = env.ee_orn / torch.norm(
            env.ee_orn, dim=-1, keepdim=True
        ).clamp(min=1.0e-6)
        drot = orientation_error(env.ee_goal_orn_quat, ee_orn)
        dpose = torch.cat([dpos, drot], dim=-1).unsqueeze(-1)
        ik_delta = env._control_ik(dpose)[0]
        target_mode = getattr(
            env.cfg.arm, "target_mode", "measured_joint_increment"
        )
        command = (
            env.arm_q_command[0]
            if target_mode == "persistent_joint_command"
            else env.dof_pos[0, -(6 + num_gripper):-num_gripper]
        )
        lower = env.dof_pos_limits[-(6 + num_gripper):-num_gripper, 0]
        upper = env.dof_pos_limits[-(6 + num_gripper):-num_gripper, 1]
        oracle = integrate_arm_command_oracle(
            command,
            ik_delta,
            float(env.cfg.arm.ik_gain),
            float(getattr(env.cfg.arm, "target_max_step", 0.0)),
            lower,
            upper,
        )
    else:
        dpos = env.ee_goal_world - env.ee_pos
        import torch
        from isaacgym.torch_utils import (
            orientation_error,
            quat_from_euler_xyz,
            quat_mul,
        )

        local_quat = quat_from_euler_xyz(
            env.curr_ee_goal_orn_rpy[:, 0],
            env.curr_ee_goal_orn_rpy[:, 1],
            env.curr_ee_goal_orn_rpy[:, 2],
        )
        target_quat = quat_mul(env.base_yaw_quat, local_quat)
        ee_orn = env.ee_orn / torch.norm(
            env.ee_orn, dim=-1, keepdim=True
        ).clamp(min=1.0e-6)
        drot = orientation_error(target_quat, ee_orn)
        dpose = torch.cat([dpos, drot], dim=-1).unsqueeze(-1)
        ik_delta = env.control_ik(dpose)[0]
        command = (
            env.arm_q_command[0]
            if env.arm_target_mode == "persistent_joint_command"
            else env._dof_pos[0, -(6 + num_gripper):-num_gripper]
        )
        lower = env.dof_limits_lower[-(6 + num_gripper):-num_gripper]
        upper = env.dof_limits_upper[-(6 + num_gripper):-num_gripper]
        oracle = integrate_arm_command_oracle(
            command,
            ik_delta,
            env.arm_ik_gain,
            env.arm_target_max_step,
            lower,
            upper,
        )
    unclamped = oracle["unclamped"]
    target = oracle["target"]
    names = list(getattr(env, "dof_names", []))[-(6 + num_gripper):-num_gripper]
    records = []
    for index, name in enumerate(names):
        before = float(unclamped[index])
        after = float(target[index])
        if abs(before - after) > 1.0e-8:
            records.append({
                "name": name, "unclamped": before, "clamped": after,
                "lower": float(lower[index]), "upper": float(upper[index]),
            })
    return _array(target), records


def collect_controller_snapshot(env: Any, side: str) -> Dict[str, Any]:
    """Collect named controller inputs/outputs without stepping physics."""
    current, history = _observation_parts(env, side)
    reconstructed_policy_action, urdf_action = _policy_order_action(env, side)
    policy_action = _array(getattr(env, "parity_policy_action", reconstructed_policy_action))
    if policy_action.ndim > 1:
        policy_action = policy_action[0]
    if side == "low":
        ee_world = env.curr_ee_goal_cart_world
        from isaacgym.torch_utils import quat_apply

        arm_base = env.base_pos + quat_apply(env.base_yaw_quat, env.arm_base_offset)
        ee_orientation_target = env.ee_goal_orn_quat
        ee_local = current[-11:-8] if env.cfg.env.observe_gait_commands else current[-6:-3]
        torques = env.torques[:, :12]
        arm_q = env.dof_pos[:, -(6 + env.cfg.env.num_gripper_joints):-env.cfg.env.num_gripper_joints]
        arm_q_command = env.arm_q_command
        root_state = env.root_states[0, :13]
        dof_state = env.dof_state[:env.num_dofs]
        jacobian = env.ee_j_eef[0]
        q = env.dof_pos[0, :12]
        qd = env.dof_vel[0, :12]
        default_q = env.default_dof_pos[:12]
        scale = env.action_scale[:12] * env.motor_strength[0, :12]
        kp, kd, limits = env.p_gains[:12], env.d_gains[:12], env.torque_limits[:12]
        arm_lower = env.dof_pos_limits[-(6 + env.cfg.env.num_gripper_joints):-env.cfg.env.num_gripper_joints, 0]
        arm_upper = env.dof_pos_limits[-(6 + env.cfg.env.num_gripper_joints):-env.cfg.env.num_gripper_joints, 1]
        robot_start_z = float(getattr(env.cfg.init_state, "pos", [0, 0, 0.32])[2])
    else:
        from isaacgym.torch_utils import (
            quat_apply,
            quat_from_euler_xyz,
            quat_mul,
        )

        ee_world = env.ee_goal_world
        ee_local = env._get_low_level_ee_goal_local()
        arm_base = env._robot_root_states[:, :3] + quat_apply(env.base_yaw_quat, env.arm_base_offset)
        local_ee_orientation = quat_from_euler_xyz(
            env.curr_ee_goal_orn_rpy[:, 0],
            env.curr_ee_goal_orn_rpy[:, 1],
            env.curr_ee_goal_orn_rpy[:, 2],
        )
        ee_orientation_target = quat_mul(
            env.base_yaw_quat, local_ee_orientation
        )
        torques = env.torques[:, :12]
        arm_q = env._dof_pos[:, -(6 + env.num_gripper_joints):-env.num_gripper_joints]
        arm_q_command = env.arm_q_command
        root_state = env._robot_root_states[0, :13]
        dof_state = env._dof_state[:env.num_dofs]
        jacobian = env.ee_j_eef[0]
        q = env._dof_pos[0, :12]
        qd = env._dof_vel[0, :12]
        default_q = env.default_dof_pos_wo_gripper[0, :12]
        scale = env.low_action_scale[:12] * env.motor_strength[0, :12]
        kp, kd, limits = env.p_gains[:12], env.d_gains[:12], env.torque_limits[:12]
        arm_lower = env.dof_limits_lower[-(6 + env.num_gripper_joints):-env.num_gripper_joints]
        arm_upper = env.dof_limits_upper[-(6 + env.num_gripper_joints):-env.num_gripper_joints]
        robot_start_z = float(env.robot_start_pose[2])
    arm_target, clamped_arm_joints = _arm_target(env, side)
    contract = env.get_training_metadata()["go2x5_alignment"]["control_contract"] if side == "low" else env.cfg["env"].get("lowPolicyContract", {})
    root_array, dof_array, jacobian_array = _array(root_state), _array(dof_state), _array(jacobian)
    fields = {
        "current_proprio": current, "history": history, "policy_action": policy_action,
        "applied_action": urdf_action, "leg_torque": _array(torques[0]),
        "leg_q_target": _array(default_q) + _array(scale) * urdf_action,
        "arm_q": _array(arm_q[0]), "arm_q_command": _array(arm_q_command[0]),
        "arm_q_target": arm_target,
        "root_state": root_array, "dof_state": dof_array,
        "ee_pose": np.concatenate((_array(env.ee_pos[0]), _array(env.ee_orn[0]))),
        "ee_target": _array(ee_world[0]),
        "ee_orientation_target": _array(ee_orientation_target[0]),
        "jacobian": jacobian_array,
    }
    nonfinite, nonfinite_failures = nonfinite_details(fields)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "kind": "controller_state",
        "side": side,
        "case": getattr(env, "parity_case", "C0"),
        "state_mode": getattr(env, "parity_state_mode", "canonical_injected"),
        "policy_mode": getattr(env, "parity_policy_mode", "zero"),
        "policy_metadata": getattr(env, "parity_policy_metadata", {"mode": "zero"}),
        "runtime_contract_hash": canonical_json_sha256(contract),
        "current_proprio": current.tolist(),
        "history": history.tolist(),
        "gait_phase": _numbers(env.gait_indices[0]),
        "gait_clock": _numbers(env.clock_inputs[0]),
        "policy_action": policy_action.tolist(),
        "applied_action": urdf_action.tolist(),
        "leg_torque": _numbers(torques[0]),
        "leg_q_target": fields["leg_q_target"].tolist(),
        "arm_q_target": arm_target.tolist(),
        "arm_q": _numbers(arm_q[0]),
        "arm_q_command": _numbers(arm_q_command[0]),
        "ee_position_world": _numbers(env.ee_pos[0]),
        "ee_goal_world": _numbers(ee_world[0]),
        "ee_goal_orientation": fields["ee_orientation_target"].tolist(),
        "ee_goal_orientation_rpy": _numbers(env.curr_ee_goal_orn_rpy[0]),
        "ee_goal_local": _numbers(ee_local[0] if hasattr(ee_local, "ndim") and ee_local.ndim > 1 else ee_local),
        "ee_command_local": _numbers(env.curr_ee_goal_cart[0]),
        "arm_base_world": _numbers(arm_base[0]),
        "root_state": root_array.tolist(),
        "dof_state": dof_array.tolist(),
        "ee_pose": fields["ee_pose"].tolist(),
        "jacobian": jacobian_array.tolist(),
        "pd_inputs": {
            "q": _numbers(q), "qd": _numbers(qd), "default_q": _numbers(default_q),
            "action_scale": _numbers(scale), "kp": _numbers(kp), "kd": _numbers(kd),
            "torque_limit": _numbers(limits),
        },
        "ee_oracle_inputs": {
            "base_position": _numbers(root_array[:3]), "base_quaternion": _numbers(root_array[3:7]),
            "arm_base_offset": _numbers(env.arm_base_offset[0] if getattr(env.arm_base_offset, "ndim", 1) > 1 else env.arm_base_offset),
            "terrain_center_offset_z": robot_start_z,
        },
        "arm_limits": {"lower": _numbers(arm_lower), "upper": _numbers(arm_upper)},
        "clamped_arm_joints": clamped_arm_joints,
        "nonfinite": nonfinite,
        "nonfinite_failures": nonfinite_failures,
    }
    return snapshot


def _gait_command_at(time_s: float) -> Tuple[float, float]:
    if time_s < 1.0:
        return 0.0, 0.0
    if time_s < 3.0:
        return 0.10, 0.0
    if time_s < 4.0:
        return 0.0, 0.0
    if time_s < 6.0:
        return -0.10, 0.15
    return 0.0, 0.0


def collect_gait_sequence(env: Any, side: str, tick_s: float = 0.02,
                          duration_s: float = 7.0) -> Dict[str, Any]:
    """Advance only the production gait/observation state machine at 50 Hz."""
    env.gait_indices[:] = 0.0
    env.clock_inputs[:] = 0.0
    records = []
    ticks = int(round(duration_s / tick_s))
    for tick in range(ticks):
        time_s = tick * tick_s
        vx, yaw = _gait_command_at(time_s)
        env.commands[:] = 0.0
        env.commands[:, 0] = vx
        env.commands[:, 2] = yaw
        if side == "low":
            env._step_contact_targets()
            env.compute_observations()
            current = env.obs_buf[:, :env.cfg.env.num_proprio]
            mask = env._get_walking_cmd_mask()
            lin_dead_zone = float(env.cfg.commands.lin_vel_x_clip)
            yaw_dead_zone = float(env.cfg.commands.ang_vel_yaw_clip)
        elif side == "high":
            env._compute_low_level_observations()
            current = env.low_obs_buf[:, :env.num_proprio]
            mask = env.get_walking_cmd_mask()
            lin_dead_zone = float(env.lin_vel_x_clip)
            yaw_dead_zone = float(env.ang_vel_yaw_clip)
        else:
            raise ValueError("side must be 'low' or 'high'")
        command_after_dead_zone = [
            vx if abs(vx) > lin_dead_zone else 0.0,
            yaw if abs(yaw) > yaw_dead_zone else 0.0,
        ]
        records.append({
            "tick": tick, "time_s": time_s, "walking": bool(_array(mask)[0]),
            "command_after_dead_zone": command_after_dead_zone,
            "gait_index": float(_array(env.gait_indices)[0]),
            "clock_inputs": _array(env.clock_inputs[0]).tolist(),
            "observation_gait_fields": _array(current[0, -5:]).tolist(),
        })
    counts, failures = nonfinite_details({
        "gait_index": [record["gait_index"] for record in records],
        "clock_inputs": [record["clock_inputs"] for record in records],
        "observation_gait_fields": [record["observation_gait_fields"] for record in records],
    })
    return {
        "schema_version": SCHEMA_VERSION, "kind": "gait_sequence", "side": side,
        "tick_s": tick_s, "duration_s": duration_s, "records": records,
        "nonfinite": counts, "nonfinite_failures": failures,
    }


def _natural_reset_record(env: Any, side: str, step: int) -> Dict[str, Any]:
    task_fields = {}
    if side == "low":
        root = env.root_states[0, :13]
        dof = env.dof_state[:env.num_dofs]
        history = env.obs_history_buf[0]
        action = env.actions[0]
        ee_target = env.curr_ee_goal_cart_world[0]
        arm_command = env.arm_q_command[0]
        contacts = env.foot_contacts_from_sensor[0]
        reset = env.reset_buf[0]
    else:
        root = env._robot_root_states[0, :13]
        dof = env._dof_state[:env.num_dofs]
        history = env.low_obs_history_buf[0]
        action = env.last_low_actions[0]
        ee_target = env.ee_goal_world[0]
        arm_command = env.arm_q_command[0]
        contacts = env.foot_contacts_from_sensor[0]
        reset = env.reset_buf[0]
        if hasattr(env, "_table_root_states"):
            task_fields["table_root_state"] = env._table_root_states[0, :13]
        if hasattr(env, "_cube_root_states"):
            task_fields["object_root_state"] = env._cube_root_states[0, :13]
        if hasattr(env, "table_heights"):
            task_fields["table_surface_height"] = env.table_heights[0]
    arm_target, _ = _arm_target(env, side)
    fields = {
        "root_state": root, "dof_state": dof, "history": history,
        "last_applied_action": action, "gait_index": env.gait_indices[0],
        "clock_inputs": env.clock_inputs[0], "ee_target": ee_target,
        "arm_command": arm_command, "arm_target": arm_target,
        "foot_contacts": contacts,
        **task_fields,
    }
    counts, failures = nonfinite_details(fields)
    record = {
        "step": step, "root_pose": _numbers(root[:7]), "dof_state": _numbers(dof),
        "history": _numbers(history), "last_applied_action": _numbers(action),
        "gait_index": _numbers(env.gait_indices[0]), "clock_inputs": _numbers(env.clock_inputs[0]),
        "ee_target": _numbers(ee_target), "arm_command": _numbers(arm_command),
        "arm_target": arm_target.tolist(),
        "foot_contacts": _numbers(contacts), "reset": bool(_array(reset)),
        "nonfinite": counts, "nonfinite_failures": failures,
    }
    record.update({name: _numbers(value) for name, value in task_fields.items()})
    return record


def _advance_high_policy_tick(env: Any) -> None:
    import torch
    from isaacgym import gymtorch
    env._update_ee_goal_world()
    env._compute_low_level_observations()
    with torch.no_grad():
        policy_action = env.parity_policy(env.low_obs_buf.detach(), hist_encoding=True)
    env.last_low_actions[:] = env._reindex_low_all(policy_action)
    targets = env.get_all_pos_targets(env.ee_goal_world, env.ee_orn)
    for _ in range(env.control_freq_inv):
        env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(targets))
        env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.get_torques()))
        env.gym.simulate(env.sim)
        env._refresh_sim_tensors()
    env._update_base_yaw_quat()
    env.update_roboinfo()


def collect_natural_reset_sequence(env: Any, side: str) -> Dict[str, Any]:
    """Collect normal reset state and short policy-tick evolution without buffer injection."""
    import torch
    requested = {0, 1, 4, 10}
    samples = []
    for step in range(11):
        if step in requested:
            samples.append(_natural_reset_record(env, side, step))
        if step == 10:
            break
        if side == "low":
            with torch.no_grad():
                action = env.parity_policy(env.obs_buf.detach(), hist_encoding=True)
            env.step(action)
        elif side == "high":
            _advance_high_policy_tick(env)
        else:
            raise ValueError("side must be 'low' or 'high'")
    nonfinite_count = sum(sum(sample["nonfinite"].values()) for sample in samples)
    immediate_resets = [sample["step"] for sample in samples if sample["reset"]]
    num_prop = env.cfg.env.num_proprio if side == "low" else env.num_proprio
    history_len = env.cfg.env.history_len if side == "low" else env.history_len
    return {
        "schema_version": SCHEMA_VERSION, "kind": "natural_reset", "side": side,
        "sample_steps": [0, 1, 4, 10], "samples": samples,
        "observation_shape": int(num_prop * (history_len + 1)),
        "raw_observation_shape": int(env.obs_buf.shape[1] if side == "low" else env.low_obs_buf.shape[1]),
        "nonfinite_count": nonfinite_count, "immediate_reset_steps": immediate_resets,
        "passed": nonfinite_count == 0 and not immediate_resets,
    }


def write_snapshot(snapshot: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_snapshot(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, child_prefix))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(_flatten(child, f"{prefix}[{index}]"))
        return result
    return {prefix: value}


def compare_snapshots(low: Mapping[str, Any], high: Mapping[str, Any], atol: float = DEFAULT_ATOL) -> List[Dict[str, Any]]:
    """Return structured mismatches, ignoring the expected side label."""
    low_flat = _flatten(low)
    high_flat = _flatten(high)
    ignored = {"side"}
    mismatches = []
    for path in sorted((set(low_flat) | set(high_flat)) - ignored):
        if path not in low_flat or path not in high_flat:
            mismatches.append({"path": path, "low": low_flat.get(path), "high": high_flat.get(path), "reason": "missing"})
            continue
        left, right = low_flat[path], high_flat[path]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not math.isfinite(float(left)) or not math.isfinite(float(right)):
                mismatches.append({"path": path, "low": left, "high": right, "reason": "nonfinite"})
                continue
            if not math.isclose(float(left), float(right), abs_tol=atol, rel_tol=0.0):
                mismatches.append({"path": path, "low": left, "high": right, "abs_error": abs(float(left) - float(right))})
        elif left != right:
            mismatches.append({"path": path, "low": left, "high": right, "reason": "value"})
    return mismatches


def validate_controller_oracles(snapshot: Mapping[str, Any], torque_atol: float = 1.0e-5,
                                action_atol: float = 1.0e-7, ee_atol: float = 1.0e-5) -> List[Dict[str, Any]]:
    """Validate a side independently from its production reorder/PD/frame helpers."""
    failures: List[Dict[str, Any]] = []
    policy_action = np.asarray(snapshot["policy_action"], dtype=np.float64)
    applied = np.asarray(snapshot["applied_action"], dtype=np.float64)
    expected_applied = policy_to_urdf_oracle(policy_action)
    for index, error in enumerate(np.abs(applied - expected_applied)):
        if error > action_atol:
            failures.append({
                "oracle": "action_permutation", "joint": URDF_LEG_JOINT_NAMES[index],
                "expected": float(expected_applied[index]), "actual": float(applied[index]),
                "abs_error": float(error),
            })

    inputs = snapshot["pd_inputs"]
    pd = independent_pd_oracle(
        inputs["q"], inputs["qd"], inputs["default_q"], inputs["action_scale"],
        applied, inputs["kp"], inputs["kd"], inputs["torque_limit"],
    )
    torque = np.asarray(snapshot["leg_torque"], dtype=np.float64)[:12]
    for index, error in enumerate(np.abs(torque - pd["torque"])):
        if error > torque_atol:
            detail = dict(pd["joints"][index])
            detail.update({"oracle": "pd_torque", "actual": float(torque[index]), "abs_error": float(error)})
            failures.append(detail)

    limits = snapshot["arm_limits"]
    for index, (target, lower, upper) in enumerate(zip(
        snapshot["arm_q_target"], limits["lower"], limits["upper"]
    )):
        if target < lower - 1.0e-7 or target > upper + 1.0e-7:
            failures.append({
                "oracle": "arm_joint_limit", "index": index, "target": target,
                "lower": lower, "upper": upper,
            })

    ee_inputs = snapshot["ee_oracle_inputs"]
    base_position = np.asarray(ee_inputs["base_position"], dtype=np.float64)
    base_quaternion = np.asarray(ee_inputs["base_quaternion"], dtype=np.float64)
    offset = np.asarray(ee_inputs["arm_base_offset"], dtype=np.float64)
    yaw_rotation = _quat_rotation([0.0, 0.0, math.sin(math.atan2(
        2 * (base_quaternion[3]*base_quaternion[2] + base_quaternion[0]*base_quaternion[1]),
        1 - 2*(base_quaternion[1]**2 + base_quaternion[2]**2)
    ) / 2), math.cos(math.atan2(
        2 * (base_quaternion[3]*base_quaternion[2] + base_quaternion[0]*base_quaternion[1]),
        1 - 2*(base_quaternion[1]**2 + base_quaternion[2]**2)
    ) / 2)])
    terrain_center = np.asarray([base_position[0], base_position[1], 0.0]) + yaw_rotation @ np.asarray([
        offset[0], offset[1], ee_inputs["terrain_center_offset_z"] + offset[2]
    ])
    ee = ee_frame_oracle(
        base_position, base_quaternion, offset, terrain_center, snapshot["ee_command_local"]
    )
    for field, expected in (("arm_base_world", ee["arm_base_world"]),
                            ("ee_goal_world", ee["ee_goal_world"]),
                            ("ee_goal_local", ee["reconstructed_local"])):
        actual = np.asarray(snapshot[field], dtype=np.float64)
        error = float(np.max(np.abs(actual - expected)))
        if error > ee_atol:
            failures.append({
                "oracle": "ee_frame", "field": field, "expected": expected.tolist(),
                "actual": actual.tolist(), "max_abs_error": error,
            })
    return failures


def _controller_field_atol(path: str, default_atol: float) -> float:
    """Return the acceptance tolerance for one controller snapshot field."""
    field_atols = (
        (("current_proprio", "history"), 1.0e-6),
        (("policy_action", "applied_action"), 1.0e-7),
        (("leg_q_target",), 1.0e-6),
        (("leg_torque", "arm_q_target"), 1.0e-5),
    )
    for prefixes, field_atol in field_atols:
        if path.startswith(prefixes):
            return field_atol
    return default_atol


def build_comparison_report(low: Mapping[str, Any], high: Mapping[str, Any],
                            atol: float = DEFAULT_ATOL) -> Dict[str, Any]:
    raw_mismatches = compare_snapshots(low, high, atol=0.0)
    mismatches = []
    for mismatch in raw_mismatches:
        error = mismatch.get("abs_error")
        if error is not None and error <= _controller_field_atol(
            mismatch["path"], atol
        ):
            continue
        mismatches.append(mismatch)
    has_oracle_inputs = all("pd_inputs" in snapshot and "ee_oracle_inputs" in snapshot
                            for snapshot in (low, high))
    oracle_failures = validate_controller_oracles(low) + validate_controller_oracles(high) \
        if has_oracle_inputs else []
    nonfinite_count = sum(int(value) for snapshot in (low, high)
                          for value in snapshot.get("nonfinite", {}).values())
    categories = {
        "observation": ("current_proprio", "history"),
        "policy_action": ("policy_action",),
        "applied_action": ("applied_action",),
        "scaled_q_target": ("leg_q_target",),
        "torque": ("leg_torque",),
        "arm_target": ("arm_q_target",),
    }
    low_flat = _flatten(low)
    high_flat = _flatten(high)
    max_errors = {}
    for name, prefixes in categories.items():
        values = []
        for path in set(low_flat) & set(high_flat):
            if not path.startswith(prefixes):
                continue
            left, right = low_flat[path], high_flat[path]
            if isinstance(left, (int, float)) and isinstance(
                right, (int, float)
            ) and math.isfinite(float(left)) and math.isfinite(float(right)):
                values.append(abs(float(left) - float(right)))
        max_errors[name] = max(values, default=0.0)
    passed = not mismatches and not oracle_failures and nonfinite_count == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "case": low.get("case", ""),
        "state_mode": low.get("state_mode", ""),
        "policy_mode": low.get("policy_mode", ""),
        "runtime_contract_hash": low.get("runtime_contract_hash", ""),
        "atol": atol,
        "field_atols": {
            "observation": 1.0e-6,
            "policy_action": 1.0e-7,
            "applied_action": 1.0e-7,
            "scaled_q_target": 1.0e-6,
            "torque": 1.0e-5,
            "arm_target": 1.0e-5,
        },
        "mismatch_count": len(mismatches),
        "oracle_failures": len(oracle_failures),
        "nonfinite_count": nonfinite_count,
        "max_abs_errors": max_errors,
        "passed": passed,
        "mismatches": mismatches,
        "oracle_failure_details": oracle_failures,
    }
