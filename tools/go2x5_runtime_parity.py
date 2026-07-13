"""Runtime snapshot and structured comparison helpers for Go2-X5 parity.

The collectors deliberately use only public environment attributes already
maintained by the low/high wrappers. They do not advance simulation state.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_ATOL = 1.0e-6


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


def _arm_target(env: Any, side: str) -> np.ndarray:
    num_gripper = int(env.cfg.env.num_gripper_joints) if side == "low" else int(env.num_gripper_joints)
    if side == "low":
        dpos = env.curr_ee_goal_cart_world - env.ee_pos
        drot = np.zeros((env.num_envs, 3), dtype=np.float32)
        import torch

        dpose = torch.cat([dpos, torch.as_tensor(drot, device=env.device)], dim=-1).unsqueeze(-1)
        target = float(env.cfg.arm.ik_gain) * env._control_ik(dpose) + env.dof_pos[:, -(6 + num_gripper):-num_gripper]
        lower = env.dof_pos_limits[-(6 + num_gripper):-num_gripper, 0]
        upper = env.dof_pos_limits[-(6 + num_gripper):-num_gripper, 1]
        target = torch.clamp(target, lower, upper)
    else:
        orientation = env.ee_orn
        target_all = env.get_all_pos_targets(env.ee_goal_world, orientation)
        target = target_all[:, -(6 + num_gripper):-num_gripper]
    return _array(target[0])


def collect_controller_snapshot(env: Any, side: str) -> Dict[str, Any]:
    """Collect named controller inputs/outputs without stepping physics."""
    current, history = _observation_parts(env, side)
    policy_action, urdf_action = _policy_order_action(env, side)
    if side == "low":
        ee_world = env.curr_ee_goal_cart_world
        from isaacgym.torch_utils import quat_apply

        arm_base = env.base_pos + quat_apply(env.base_yaw_quat, env.arm_base_offset)
        ee_local = current[-11:-8] if env.cfg.env.observe_gait_commands else current[-6:-3]
        torques = env.torques[:, :12]
        arm_q = env.dof_pos[:, -(6 + env.cfg.env.num_gripper_joints):-env.cfg.env.num_gripper_joints]
    else:
        from isaacgym.torch_utils import quat_apply

        ee_world = env.ee_goal_world
        ee_local = env._get_low_level_ee_goal_local()
        arm_base = env._robot_root_states[:, :3] + quat_apply(env.base_yaw_quat, env.arm_base_offset)
        torques = env.torques[:, :12]
        arm_q = env._dof_pos[:, -(6 + env.num_gripper_joints):-env.num_gripper_joints]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "controller_state",
        "side": side,
        "current_proprio": current.tolist(),
        "history": history.tolist(),
        "gait_phase": _numbers(env.gait_indices[0]),
        "gait_clock": _numbers(env.clock_inputs[0]),
        "policy_action": policy_action.tolist(),
        "urdf_action": urdf_action.tolist(),
        "leg_torque": _numbers(torques[0]),
        "arm_q_target": _arm_target(env, side).tolist(),
        "arm_q": _numbers(arm_q[0]),
        "ee_position_world": _numbers(env.ee_pos[0]),
        "ee_goal_world": _numbers(ee_world[0]),
        "ee_goal_local": _numbers(ee_local[0] if hasattr(ee_local, "ndim") and ee_local.ndim > 1 else ee_local),
        "arm_base_world": _numbers(arm_base[0]),
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
            if not math.isclose(float(left), float(right), abs_tol=atol, rel_tol=0.0):
                mismatches.append({"path": path, "low": left, "high": right, "abs_error": abs(float(left) - float(right))})
        elif left != right:
            mismatches.append({"path": path, "low": left, "high": right, "reason": "value"})
    return mismatches
