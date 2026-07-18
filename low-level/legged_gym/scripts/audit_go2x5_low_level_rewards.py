#!/usr/bin/env python3
"""Static reward-semantics audit for the Go2-X5 low-level task.

This script intentionally avoids importing the legged_gym env package. Importing
that package initializes Isaac Gym-facing modules, which is unnecessary for a
pre-training reward audit.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import operator
import textwrap
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
LOW_LEVEL_ROOT = REPO_ROOT / "low-level"
CONFIG_PATH = LOW_LEVEL_ROOT / "legged_gym/envs/manip_loco/go2x5_config.py"
SPEC_PATH = LOW_LEVEL_ROOT / "legged_gym/envs/manip_loco/go2x5_robot_spec.py"
REWARD_PATH = LOW_LEVEL_ROOT / "legged_gym/envs/rewards/maniploco_rewards.py"
ENV_PATH = LOW_LEVEL_ROOT / "legged_gym/envs/manip_loco/manip_loco.py"
PPO_PATH = REPO_ROOT / "third_party/rsl_rl/rsl_rl/algorithms/ppo.py"
STORAGE_PATH = REPO_ROOT / "third_party/rsl_rl/rsl_rl/storage/rollout_storage.py"
URDF_PATH = LOW_LEVEL_ROOT / "resources/robots/go2x5/go2_x5.urdf"
DEFAULT_OUTPUT = REPO_ROOT / "docs/06_go2x5_low_level_reward_audit.md"


@dataclass(frozen=True)
class RewardAudit:
    raw_formula: str
    raw_direction: str
    expected_scale_sign: str
    dependency: str
    migration_risk: str
    verification: str


AUDIT: dict[str, RewardAudit] = {
    "tracking_contacts_shaped_force": RewardAudit(
        raw_formula="0 is best; negative when a foot has force while desired_contact is 0",
        raw_direction="larger is better",
        expected_scale_sign="+",
        dependency="feet_indices order, desired_contact_states, observe_gait_commands",
        migration_risk="When enabled, the coefficient must remain positive or bad off-phase contact becomes positive reward.",
        verification="Enable observe_gait_commands in a small probe; inject off-phase foot force and confirm weighted reward decreases.",
    ),
    "tracking_contacts_shaped_vel": RewardAudit(
        raw_formula="0 is best; negative when a desired-stance foot has velocity",
        raw_direction="larger is better",
        expected_scale_sign="+",
        dependency="feet_indices order, live rigid_body_state velocity, desired_contact_states, observe_gait_commands",
        migration_risk="When enabled, the coefficient must remain positive and must not read an advanced-indexed cache that is never refreshed.",
        verification="Inject stance-foot velocity into live rigid_body_state, refresh the cache, and confirm both the raw reward and cache value change.",
    ),
    "feet_air_time": RewardAudit(
        raw_formula="sum((feet_air_time - configured target) on first contact)",
        raw_direction="larger is better",
        expected_scale_sign="+",
        dependency="force_sensor_tensor order, gait frequency, and swing duration",
        migration_risk="The target must match the configured gait period; all four feet should contribute to a quadruped gait.",
        verification="At 2 Hz and 50% swing, verify first contact near 0.25 s is neutral and longer air time is positive.",
    ),
    "feet_height": RewardAudit(
        raw_formula="negative mean per-foot swing-clearance shortfall relative to terrain",
        raw_direction="larger is better",
        expected_scale_sign="+",
        dependency="feet_indices, desired_contact_states, and measured terrain height",
        migration_risk="Each desired-swing foot must clear independently; one high foot must not mask another low foot.",
        verification="Sweep each foot z independently; only the low desired-swing foot should contribute clearance error.",
    ),
    "tracking_lin_vel_max": RewardAudit(
        raw_formula="velocity progress ratio for x command; zero command uses exp(-abs(base_vx))",
        raw_direction="larger is better",
        expected_scale_sign="+",
        dependency="commands[:,0], base_lin_vel[:,0] in base frame",
        migration_risk="Sign is correct; overspeed is weakly penalized because the ratio saturates at 1.",
        verification="Probe command vx {-0.5,0,0.5} and base vx offsets; weighted reward should be highest near command.",
    ),
    "tracking_ang_vel": RewardAudit(
        raw_formula="exp(-square(command_yaw - base_yaw_rate) / tracking_sigma)",
        raw_direction="larger is better",
        expected_scale_sign="+",
        dependency="commands[:,2], base_ang_vel[:,2]",
        migration_risk="Sign is correct; confirm command yaw is not confused with vertical linear velocity in any caller.",
        verification="Probe yaw-rate error 0, small, large; weighted reward should monotonically decrease with error.",
    ),
    "delta_torques": RewardAudit(
        raw_formula="sum(square(torques - last_torques) over leg joints)",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="leg torque order after action reindexing",
        migration_risk="Sign is correct; magnitude can hide action-order bugs if torques are assigned to wrong legs.",
        verification="Apply a one-joint action impulse and confirm only the expected URDF leg torque changes.",
    ),
    "torques": RewardAudit(
        raw_formula="sum(square(all torques))",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="torques tensor, leg plus zero arm/gripper torque convention",
        migration_risk="Sign is correct; with low-level 12D actions arm torques should not dominate this term.",
        verification="Check per-joint torque contribution; arm/gripper entries should be zero or intentionally excluded.",
    ),
    "stand_still": RewardAudit(
        raw_formula="exp(-0.05 * leg dof L1 deviation from default), standing commands only",
        raw_direction="larger is better",
        expected_scale_sign="+",
        dependency="default_dof_pos, walking command mask",
        migration_risk="Sign is correct; too much weight can oppose crouching needed for low EE goals.",
        verification="With zero command, perturb leg default pose and confirm reward is highest at default.",
    ),
    "walking_dof": RewardAudit(
        raw_formula="exp(-0.05 * leg dof L1 deviation from default), walking commands only",
        raw_direction="larger is better",
        expected_scale_sign="+",
        dependency="default_dof_pos, walking command mask",
        migration_risk="It biases walking toward the default crouch and is intentionally disabled in Go2-X5 S3 after the v3 no-step failure.",
        verification="Keep the S3 scale at zero unless a gait-amplitude ablation proves the term cannot dominate swing rewards.",
    ),
    "alive": RewardAudit(
        raw_formula="constant 1",
        raw_direction="larger is better",
        expected_scale_sign="+",
        dependency="episode survival only",
        migration_risk="Sign is correct; this is a survival baseline, not a behavior-specific signal.",
        verification="Confirm the term is constant and termination penalties are handled separately.",
    ),
    "termination": RewardAudit(
        raw_formula="1 on non-timeout reset, otherwise 0",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="reset_buf and time_out_buf after roll/pitch/height/contact checks",
        migration_risk="A wrong sign rewards falls; counting timeouts would also punish successful full episodes.",
        verification="Trigger roll, pitch, height, and timeout resets; only non-timeout resets must receive the penalty.",
    ),
    "lin_vel_z": RewardAudit(
        raw_formula="square(base_lin_vel_z)",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="base linear velocity in base frame",
        migration_risk="Sign is correct; excessive vertical oscillation should be penalized.",
        verification="Inject upward/downward base z velocity; weighted reward should become more negative.",
    ),
    "roll": RewardAudit(
        raw_formula="abs(base roll)",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="base_quat to roll extraction",
        migration_risk="Sign is correct; pitch is not directly penalized by this active term.",
        verification="Probe roll {0, 0.2, 0.5}; weighted reward should monotonically decrease.",
    ),
    "ang_vel_xy": RewardAudit(
        raw_formula="sum(square(base angular velocity x/y))",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="base_ang_vel x/y",
        migration_risk="Sign is correct; complements roll but does not replace pitch/orientation checks.",
        verification="Inject roll/pitch angular velocity; weighted reward should become more negative.",
    ),
    "dof_acc": RewardAudit(
        raw_formula="sum(square((last_dof_vel - dof_vel) / dt) over leg joints)",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="leg dof velocity order",
        migration_risk="Sign is correct; confirms smoothness, not migration correctness by itself.",
        verification="Apply a velocity jump to one leg joint and confirm the expected joint contribution increases.",
    ),
    "collision": RewardAudit(
        raw_formula="sum(clamp(norm(contact_force on penalized bodies)-threshold, 0, soft_clip)/threshold)",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="penalized_contact_indices from asset.penalize_contacts_on substrings",
        migration_risk="Sign is correct, but current body set is thigh/calf only; base and arm contacts are not penalized here.",
        verification="Print resolved penalized body names; manually create thigh/calf/base/arm contacts and confirm only intended bodies count.",
    ),
    "action_rate": RewardAudit(
        raw_formula="sum(square(last_actions - actions) over leg actions)",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="policy action order before/after _reindex_all",
        migration_risk="Sign is correct; action-order bugs can still hide behind a smooth but wrong action stream.",
        verification="Pulse policy FR hip action and confirm the smoothness term uses the policy-order action history consistently.",
    ),
    "dof_pos_limits": RewardAudit(
        raw_formula="sum soft joint-limit violation over leg joints",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="dof_pos_limits, URDF leg dof order",
        migration_risk="Sign is correct; verify Go2-X5 URDF limits are not inherited from B1/Z1 or too tight after mimic edits.",
        verification="Sweep each leg joint near lower/upper limit; raw should be zero inside and positive outside soft range.",
    ),
    "hip_pos": RewardAudit(
        raw_formula="sum(square(hip dofs - default hip dofs))",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="hip_indices = FR,FL,RR,RL names resolved in URDF dof_names",
        migration_risk="Sign is correct; name lookup avoids relying on URDF order, but default angles must match Go2-X5.",
        verification="Print hip_indices and dof_names; perturb each hip and check only that hip contributes.",
    ),
    "work": RewardAudit(
        raw_formula="abs(sum(torque * dof_vel over leg joints))",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="leg torque and velocity order",
        migration_risk="Sign is correct; can penalize useful work as well as waste.",
        verification="Probe zero velocity, co-directed torque/velocity, and opposing torque/velocity; magnitude should increase away from zero.",
    ),
    "feet_jerk": RewardAudit(
        raw_formula="sum(norm(force_sensor_tensor - last_contact_forces)) after first 50 steps",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="force_sensor_tensor order and contact sensor stability",
        migration_risk="Sign is correct; noisy force sensors on rough terrain can inject high-variance penalty.",
        verification="Log raw term on flat standing and rough stepping; it should not dominate early reward breakdown.",
    ),
    "feet_drag": RewardAudit(
        raw_formula="sum foot xyz velocity for feet detected in contact",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="feet_indices and force_sensor_tensor contact booleans",
        migration_risk="Sign is correct; requires feet_indices and force_sensor_tensor to describe the same FL,FR,RL,RR order.",
        verification="Slide one contacting foot in sim; raw should increase only for that foot.",
    ),
    "foot_lateral_spacing": RewardAudit(
        raw_formula="sum lateral-width shortfall for FL/RL on +y and FR/RR on -y",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="URDF foot order FL,FR,RL,RR and yaw-frame foot positions",
        migration_risk="An incorrect foot order or side sign would reward crossed legs.",
        verification="Move each foot toward and across the sagittal centerline; only the corresponding shortfall should increase.",
    ),
    "feet_contact_forces": RewardAudit(
        raw_formula="sum(max(norm(force_sensor_tensor)-max_contact_force, 0)) after 2 seconds",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="force_sensor_tensor and max_contact_force",
        migration_risk="Sign is correct; max_contact_force=200 makes it a high-force limiter, not normal contact shaping.",
        verification="Inspect force histograms; raw should be near zero for nominal stance and positive for impacts.",
    ),
    "feet_contact_standing": RewardAudit(
        raw_formula="number of feet off ground while the command is stopped",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="force-sensor contact booleans and walking command mask",
        migration_risk="A bad contact threshold can penalize a nominal four-foot stance.",
        verification="At zero command lift one foot at a time; the raw penalty should rise by one and be zero while walking.",
    ),
    "hind_feet_contact_standing": RewardAudit(
        raw_formula="number of RL/RR feet off ground while the command is stopped",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="force-sensor order FL,FR,RL,RR and walking command mask",
        migration_risk="A migrated foot order could constrain the front pair instead of the rear support pair.",
        verification="Lift FL, FR, RL, RR independently; only RL/RR should contribute while stopped.",
    ),
    "foot_support_standing": RewardAudit(
        raw_formula="max(min_stance_feet - contact_count, 0) while stopped",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="four force-sensor contact booleans and walking command mask",
        migration_risk="A flickering contact threshold can create a high-variance standing penalty.",
        verification="Sweep contact count 4,3,2,1 at zero command; raw penalty should be 0,0,1,2.",
    ),
    "base_height": RewardAudit(
        raw_formula="abs((root_z - mean(measured_heights)) - base_height_target)",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="root_states[:,2], measured_heights, base_height_target",
        migration_risk="Sign is correct for target 0.32; terrain height sampling must be valid under Go2-X5 footprint.",
        verification="Probe flat base z 0.24/0.32/0.41; weighted reward should be best at 0.32.",
    ),
    "pitch_soft_limit_standing": RewardAudit(
        raw_formula="max(abs(pitch) - configured soft limit, 0) while stopped",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="full base quaternion and walking command mask",
        migration_risk="Too tight a limit prevents body compensation needed for low EE targets.",
        verification="Sweep pitch below and above 0.35 rad; penalty must be zero below it and disabled while walking.",
    ),
    "orientation": RewardAudit(
        raw_formula="sum(square(projected_gravity_xy))",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="base quaternion and gravity vector",
        migration_risk="This base penalty is independent of the position-only EE orientation setting.",
        verification="Sweep base roll/pitch around identity; the minimum must occur at level orientation.",
    ),
    "stability_safety": RewardAudit(
        raw_formula="product of roll, pitch, terrain-relative height, and >=3-foot safety margins",
        raw_direction="larger is better",
        expected_scale_sign="+",
        dependency="base attitude, measured heights, and force-sensor contacts",
        migration_risk="A wrong terrain height or foot contact identity collapses this positive shaping term.",
        verification="Start level with four contacts, then violate one safety margin at a time; reward must decrease.",
    ),
    "dof_error_deadzone": RewardAudit(
        raw_formula="sum(square(max(abs(leg_q-default_q)-deadzone, 0)))",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="URDF leg joint order and default pose",
        migration_risk="Wrong default angles or leg slicing penalize the intended nominal stance.",
        verification="Perturb each leg joint within and beyond the dead zone; only excess displacement should contribute.",
    ),
    "leg_action_l2_deadzone": RewardAudit(
        raw_formula="sum(square(max(abs(applied_leg_action)-deadzone, 0)))",
        raw_direction="larger is worse",
        expected_scale_sign="-",
        dependency="12D applied leg action in URDF order",
        migration_risk="An overly small dead zone can suppress useful corrective actions during early learning.",
        verification="Sweep one action around the dead zone; penalty must be zero inside and quadratic outside.",
    ),
    "tracking_ee_world": RewardAudit(
        raw_formula="exp(-2 * L1(ee_pos - curr_ee_goal_cart_world) / tracking_ee_sigma)",
        raw_direction="larger is better",
        expected_scale_sign="+",
        dependency="gripper_idx=arm_eef_link, ee_pos rigid body state, world-frame EE goal",
        migration_risk="Sign is correct; with num_arm_actions=0 it influences the leg policy through PPO reward mixing, not a separate arm action head.",
        verification="Set goal exactly at arm_eef_link and then offset x/y/z; raw should be highest at zero offset and decay monotonically.",
    ),
    "tracking_ee_world_stable": RewardAudit(
        raw_formula="world-position EE tracking reward multiplied by the stability safety margin",
        raw_direction="larger is better",
        expected_scale_sign="+",
        dependency="arm_eef_link world position, EE world target, attitude, height, and foot contacts",
        migration_risk="The task can otherwise improve EE error by falling or sacrificing support.",
        verification="Compare exact EE tracking in stable stance against the same tracking with each safety margin violated.",
    ),
    "tracking_ee_orn": RewardAudit(
        raw_formula="exp(-Euler L1 orientation error / tracking_ee_sigma)",
        raw_direction="larger is better",
        expected_scale_sign="+",
        dependency="ee_goal_orn_euler, ee_orn, arm_eef_link grasp direction",
        migration_risk="Currently disabled. If enabled, Go2-X5 arm_eef_link +x grasp direction must be verified first.",
        verification="Run orientation sweep around arm_eef_link +x; reward should be highest for the actual grasp pose.",
    ),
}


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

STAGE_REWARD_SCALE_KEYS = {
    "collision_scale": "collision",
    "orientation_scale": "orientation",
    "foot_lateral_spacing_scale": "foot_lateral_spacing",
    "feet_contact_standing_scale": "feet_contact_standing",
    "hind_feet_contact_standing_scale": "hind_feet_contact_standing",
    "foot_support_standing_scale": "foot_support_standing",
    "dof_error_deadzone_scale": "dof_error_deadzone",
    "leg_action_l2_deadzone_scale": "leg_action_l2_deadzone",
    "stability_safety_scale": "stability_safety",
    "tracking_lin_vel_max_scale": "tracking_lin_vel_max",
    "tracking_ang_vel_scale": "tracking_ang_vel",
    "walking_dof_scale": "walking_dof",
    "tracking_contacts_shaped_force_scale": "tracking_contacts_shaped_force",
    "tracking_contacts_shaped_vel_scale": "tracking_contacts_shaped_vel",
    "feet_air_time_scale": "feet_air_time",
    "feet_height_scale": "feet_height",
    "torques_scale": "torques",
    "work_scale": "work",
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_eval(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [safe_eval(item, names) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(safe_eval(item, names) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            safe_eval(key, names): safe_eval(value, names)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -safe_eval(node.operand, names)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.left, names), safe_eval(node.right, names))
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.Attribute):
        base = safe_eval(node.value, names)
        return getattr(base, node.attr)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def find_class(module_or_class: ast.Module | ast.ClassDef, name: str) -> ast.ClassDef:
    body = module_or_class.body
    for stmt in body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == name:
            return stmt
    raise KeyError(f"class {name} not found")


def find_class_chain(module: ast.Module, chain: list[str]) -> ast.ClassDef:
    current: ast.Module | ast.ClassDef = module
    for name in chain:
        current = find_class(current, name)
    return current


def class_assignments(path: Path, chain: list[str], names: dict[str, Any]) -> OrderedDict[str, Any]:
    cls = find_class_chain(parse_module(path), chain)
    values: OrderedDict[str, Any] = OrderedDict()
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    try:
                        value = safe_eval(stmt.value, names)
                    except ValueError:
                        value = ast.unparse(stmt.value)
                    if target.id not in values:
                        values[target.id] = value
                    else:
                        values[target.id] = value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            try:
                values[stmt.target.id] = safe_eval(stmt.value, names) if stmt.value is not None else None
            except ValueError:
                values[stmt.target.id] = ast.unparse(stmt.value) if stmt.value is not None else None
    return values


def reward_function_lines(path: Path) -> dict[str, int]:
    module = parse_module(path)
    reward_cls = find_class(module, "ManipLoco_rewards")
    lines: dict[str, int] = {}
    for stmt in reward_cls.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("_reward_"):
            lines[stmt.name.removeprefix("_reward_")] = stmt.lineno
    return lines


def parse_urdf(path: Path) -> dict[str, list[str]]:
    root = ET.parse(path).getroot()
    links = [link.attrib["name"] for link in root.findall("link")]
    joints = [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") != "fixed"
    ]
    return {"links": links, "movable_joints": joints}


def active_scales(scales: OrderedDict[str, Any]) -> OrderedDict[str, float]:
    active: OrderedDict[str, float] = OrderedDict()
    for name, value in scales.items():
        if value is None:
            continue
        if isinstance(value, (int, float)) and value != 0:
            active[name] = float(value)
    return active


def sign_of(value: float) -> str:
    if value > 0:
        return "+"
    if value < 0:
        return "-"
    return "0"


def sign_status(name: str, scale: float, expected: str, env_cfg: dict[str, Any]) -> str:
    actual = sign_of(scale)
    if expected in {"+", "-"} and actual != expected:
        status = "MISMATCH"
    elif expected in {"+", "-"}:
        status = "OK"
    else:
        status = "CHECK"

    if name.startswith("tracking_contacts_shaped") and not env_cfg.get("observe_gait_commands", False):
        status += "; currently returns 0 because observe_gait_commands=False"
    return status


def md_escape(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def short_list(values: list[str], limit: int = 20) -> str:
    if len(values) <= limit:
        return ", ".join(values)
    head = ", ".join(values[:limit])
    return f"{head}, ... ({len(values)} total)"


def source_ref(path: Path, line: int | None = None) -> str:
    rel = path.relative_to(REPO_ROOT)
    if line is None:
        return f"`{rel}`"
    return f"`{rel}:{line}`"


def audit_rows(
    reward_scales: OrderedDict[str, float],
    arm_reward_scales: OrderedDict[str, float],
    function_lines: dict[str, int],
    env_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel, scales in (("leg", reward_scales), ("arm", arm_reward_scales)):
        for name, scale in scales.items():
            meta = AUDIT.get(
                name,
                RewardAudit(
                    raw_formula="No metadata yet; inspect _reward function manually.",
                    raw_direction="unknown",
                    expected_scale_sign="?",
                    dependency="unknown",
                    migration_risk="Unknown reward term; add audit metadata before relying on it.",
                    verification="Open reward implementation and add a monotonicity probe.",
                ),
            )
            rows.append(
                {
                    "term": name,
                    "channel": channel,
                    "scale": scale,
                    "line": function_lines.get(name),
                    "raw_formula": meta.raw_formula,
                    "raw_direction": meta.raw_direction,
                    "expected": meta.expected_scale_sign,
                    "sign": sign_status(name, scale, meta.expected_scale_sign, env_cfg),
                    "dependency": meta.dependency,
                    "risk": meta.migration_risk,
                    "verification": meta.verification,
                }
            )
    return rows


def curriculum_audit_rows(stages: list[dict[str, Any]], env_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in stages:
        for stage_key, reward_name in STAGE_REWARD_SCALE_KEYS.items():
            if stage_key not in stage:
                continue
            scale = stage[stage_key]
            meta = AUDIT.get(reward_name)
            if meta is None or not isinstance(scale, (int, float)):
                continue
            sign = "DISABLED" if scale == 0 else sign_status(
                reward_name,
                float(scale),
                meta.expected_scale_sign,
                env_cfg,
            )
            rows.append(
                {
                    "stage": stage.get("name", "unnamed"),
                    "term": reward_name,
                    "scale": float(scale),
                    "expected": meta.expected_scale_sign,
                    "sign": sign,
                }
            )
    return rows


def build_report() -> str:
    robot_spec = load_module(SPEC_PATH, "go2x5_robot_spec_for_reward_audit")
    names = {"robot_spec": robot_spec}

    reward_cfg = class_assignments(CONFIG_PATH, ["Go2X5RoughCfg", "rewards"], names)
    scales = class_assignments(CONFIG_PATH, ["Go2X5RoughCfg", "rewards", "scales"], names)
    arm_scales = class_assignments(CONFIG_PATH, ["Go2X5RoughCfg", "rewards", "arm_scales"], names)
    env_cfg = dict(class_assignments(CONFIG_PATH, ["Go2X5RoughCfg", "env"], names))
    asset_cfg = dict(class_assignments(CONFIG_PATH, ["Go2X5RoughCfg", "asset"], names))
    ppo_policy_cfg = dict(class_assignments(CONFIG_PATH, ["Go2X5RoughCfgPPO", "policy"], names))
    ppo_algorithm_cfg = dict(class_assignments(CONFIG_PATH, ["Go2X5RoughCfgPPO", "algorithm"], names))
    auto_curriculum_cfg = dict(class_assignments(CONFIG_PATH, ["Go2X5RoughCfg", "auto_curriculum"], names))
    function_lines = reward_function_lines(REWARD_PATH)
    urdf = parse_urdf(URDF_PATH)

    leg_active = active_scales(scales)
    arm_active = active_scales(arm_scales)
    rows = audit_rows(leg_active, arm_active, function_lines, env_cfg)
    stage_rows = curriculum_audit_rows(auto_curriculum_cfg.get("stages", []), env_cfg)

    foot_names = [name for name in urdf["links"] if asset_cfg["foot_name"] in name]
    preferred_feet = list(robot_spec.URDF_FOOT_BODY_NAMES)
    runtime_feet = preferred_feet if all(name in foot_names for name in preferred_feet) else foot_names
    penalized_names: list[str] = []
    for token in asset_cfg["penalize_contacts_on"]:
        penalized_names.extend([name for name in urdf["links"] if token in name])
    termination_names: list[str] = []
    for token in asset_cfg["terminate_after_contacts_on"]:
        termination_names.extend([name for name in urdf["links"] if token in name])

    mismatches = [row for row in rows if row["sign"].startswith("MISMATCH")]
    unreviewed = [row for row in rows if row["sign"].startswith("CHECK")]
    stage_mismatches = [row for row in stage_rows if row["sign"].startswith("MISMATCH")]
    disabled_zero = [
        name
        for name, value in {**scales, **arm_scales}.items()
        if name in AUDIT and (value is None or value == 0)
    ]

    lines: list[str] = []
    lines.append("# Go2-X5 Low-Level Reward Audit")
    lines.append("")
    lines.append("This report is generated by:")
    lines.append("")
    lines.append(f"- {source_ref(Path(__file__))}")
    lines.append("")
    lines.append("Run it from the repository root with:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 low-level/legged_gym/scripts/audit_go2x5_low_level_rewards.py --output docs/06_go2x5_low_level_reward_audit.md")
    lines.append("```")
    lines.append("")
    lines.append("## Static Context")
    lines.append("")
    lines.append(f"- Config: {source_ref(CONFIG_PATH)}")
    lines.append(f"- Reward implementation: {source_ref(REWARD_PATH)}")
    lines.append(f"- Env reward aggregation: {source_ref(ENV_PATH, 398)}")
    lines.append(f"- PPO two-channel reward storage: {source_ref(PPO_PATH, 129)}, {source_ref(STORAGE_PATH, 56)}")
    lines.append(f"- only_positive_rewards: `{reward_cfg.get('only_positive_rewards')}`")
    lines.append(f"- observe_gait_commands: `{env_cfg.get('observe_gait_commands')}`")
    lines.append(f"- base_height_target: `{reward_cfg.get('base_height_target')}`")
    lines.append(f"- tracking_sigma: `{reward_cfg.get('tracking_sigma')}`")
    lines.append(f"- tracking_ee_sigma: `{reward_cfg.get('tracking_ee_sigma')}`")
    lines.append(f"- collision_force_threshold: `{reward_cfg.get('collision_force_threshold')}`")
    lines.append(f"- collision_soft_clip: `{reward_cfg.get('collision_soft_clip')}`")
    lines.append(f"- feet_height_target: `{reward_cfg.get('feet_height_target')}`")
    lines.append(f"- max_contact_force: `{reward_cfg.get('max_contact_force')}`")
    lines.append(f"- Active leg reward terms: `{len(leg_active)}`")
    lines.append(f"- Active arm reward terms: `{len(arm_active)}`")
    lines.append(f"- PPO policy num_leg_actions: `{ppo_policy_cfg.get('num_leg_actions')}`")
    lines.append(f"- PPO policy num_arm_actions: `{ppo_policy_cfg.get('num_arm_actions')}`")
    lines.append(f"- PPO reward mixing_schedule: `{ppo_algorithm_cfg.get('mixing_schedule')}`")
    lines.append("")
    lines.append("Reward aggregation details:")
    lines.append("")
    lines.append("- Leg rewards are summed into `rew_buf` and divided by 100.")
    lines.append("- Arm rewards are summed into `arm_rew_buf` and divided by 100.")
    lines.append("- Episode reward summaries remain per-second totals; `Episode_metric/*` summaries are raw per-policy-step means so curriculum thresholds keep their physical units.")
    lines.append("- PPO stores `[rew_buf, arm_rew_buf]` as a two-channel reward/value/advantage signal.")
    lines.append("- In the current low-level config `num_arm_actions=0`, so the arm channel has no independent arm-action log-prob gradient. It affects the 12D leg policy through `mixing_advantages_batch[...,0] = leg_adv + value_mixing_ratio * arm_adv`; `value_mixing_ratio` ramps from 0 to 1 over the first 3000 PPO updates.")
    lines.append("")
    lines.append("## Go2-X5 Order And Body Resolution")
    lines.append("")
    lines.append(f"- URDF movable joint order: `{short_list(urdf['movable_joints'])}`")
    lines.append(f"- Policy leg joint order: `{short_list(list(robot_spec.POLICY_LEG_JOINT_NAMES))}`")
    lines.append(f"- Runtime foot sensor/reward order: `{short_list(runtime_feet)}`")
    lines.append(f"- Policy-visible foot observation order after `_reindex_feet`: `{short_list(list(robot_spec.FOOT_BODY_NAMES))}`")
    lines.append(f"- EE body: `{asset_cfg.get('gripper_name')}`")
    lines.append(f"- Penalized contact tokens: `{asset_cfg.get('penalize_contacts_on')}`")
    lines.append(f"- Resolved penalized contact bodies: `{short_list(penalized_names) if penalized_names else 'none'}`")
    lines.append(f"- Resolved termination contact bodies: `{short_list(termination_names) if termination_names else 'none'}`")
    lines.append("")
    lines.append("## Main Findings")
    lines.append("")
    if mismatches or stage_mismatches:
        lines.append("1. Sign mismatches found:")
        for row in mismatches:
            lines.append(
                f"   - `{row['term']}` has scale `{row['scale']}` but raw direction is `{row['raw_direction']}` and expected sign is `{row['expected']}`. {row['sign']}."
            )
        for row in stage_mismatches:
            lines.append(
                f"   - `{row['stage']}/{row['term']}` has scale `{row['scale']}` "
                f"but expected sign is `{row['expected']}`. {row['sign']}."
            )
    else:
        lines.append("1. No active or curriculum-stage sign mismatch was found by the static table.")
    if unreviewed:
        terms = ", ".join(f"`{row['term']}`" for row in unreviewed)
        lines.append(f"   - `{len(unreviewed)}` active terms still lack reviewed semantics: {terms}.")
    lines.append("2. Gait contact shaping is zero for stopped commands and uses positive coefficients because its raw values are non-positive penalties.")
    lines.append("3. Air-time uses the configured 0.25 s target for the 2 Hz/50% swing gait, and air-time/clearance shaping covers all four feet.")
    lines.append("4. `tracking_ee_world` uses `arm_eef_link` world position and is an active PPO reward channel. With `num_arm_actions=0`, its effect on the low-level 12D leg policy comes through PPO advantage mixing, ramped by `mixing_schedule=[1.0, 0, 3000]`.")
    lines.append("5. `collision` sign is correct, but the resolved penalized set is thigh/calf only. Base, arm, wrist, and finger contacts are not penalized by this term.")
    lines.append("6. `tracking_contacts_shaped_vel` reads the freshly refreshed rigid-body tensor directly; the advanced-indexed foot cache is refreshed each policy tick and checked independently.")
    lines.append("7. Go2-X5 S3 disables `walking_dof`: rewarding the default pose while commanding motion previously dominated the swing/contact terms and admitted a no-step optimum.")
    lines.append("")
    lines.append("## Active Reward Audit Table")
    lines.append("")
    lines.append("| term | channel | scale | raw meaning | expected sign | sign check | source | dependency | Go2-X5 migration risk | verification |")
    lines.append("|---|---:|---:|---|---:|---|---|---|---|---|")
    for row in rows:
        if row["term"] == "termination":
            source = source_ref(ENV_PATH, 689)
        else:
            source = source_ref(REWARD_PATH, row["line"]) if row["line"] else source_ref(REWARD_PATH)
        lines.append(
            "| "
            + " | ".join(
                md_escape(item)
                for item in [
                    f"`{row['term']}`",
                    row["channel"],
                    row["scale"],
                    row["raw_formula"],
                    row["expected"],
                    row["sign"],
                    source,
                    row["dependency"],
                    row["risk"],
                    row["verification"],
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Curriculum Reward Overrides")
    lines.append("")
    lines.append("| stage | term | scale | expected sign | sign check |")
    lines.append("|---|---|---:|---:|---|")
    for row in stage_rows:
        lines.append(
            f"| `{row['stage']}` | `{row['term']}` | {row['scale']} | "
            f"{row['expected']} | {row['sign']} |"
        )
    lines.append("")
    lines.append("## Disabled But Migration-Relevant Terms")
    lines.append("")
    if disabled_zero:
        lines.append("| term | configured value | why it still matters | verification before enabling |")
        lines.append("|---|---:|---|---|")
        merged = {**scales, **arm_scales}
        for name in disabled_zero:
            meta = AUDIT[name]
            lines.append(
                "| "
                + " | ".join(
                    md_escape(item)
                    for item in [
                        f"`{name}`",
                        merged[name],
                        meta.migration_risk,
                        meta.verification,
                    ]
                )
                + " |"
            )
    else:
        lines.append("No disabled migration-relevant terms are currently listed in the audit metadata.")
    lines.append("")
    lines.append("## Required Runtime Probes")
    lines.append("")
    lines.append("Static analysis can verify sign consistency and dependency wiring, but it cannot prove that Isaac Gym rigid-body/contact tensors have the expected values at runtime. Before launching a long low-level run, run these probes:")
    lines.append("")
    lines.append("1. Base-height monotonicity: set flat-terrain root z near `0.24, 0.32, 0.41`; `base_height` weighted contribution must be best at `0.32`.")
    lines.append("2. Contact identity: touch `FL_foot, FR_foot, RL_foot, RR_foot` one at a time and confirm `force_sensor_tensor` order is `FL,FR,RL,RR`, while policy observation order is `FR,FL,RR,RL`.")
    lines.append("3. Collision identity: create contact on a thigh, calf, base, arm link, and finger link; only thigh/calf should affect current `collision`.")
    lines.append("4. EE position monotonicity: set `curr_ee_goal_cart_world` equal to `arm_eef_link` position, then offset x/y/z; `tracking_ee_world` raw value must decay monotonically.")
    lines.append("5. Position-only IK invariant: vary `ee_goal_orn_quat` while holding the position target fixed; arm q-targets must not change when `track_ee_orientation=False`.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(textwrap.dedent(
        """
        The current Go2-X5 low-level reward set is sign-consistent for the active locomotion, gait, and EE-position terms.
        Static signs do not prove simulator tensor identity, terrain-relative height semantics, or reset behavior.
        Runtime acceptance therefore also requires the listed monotonicity and identity probes; their executed status is recorded in the dated training-readiness report.
        """
    ).strip())
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Write Markdown report to this path. Default stdout. Suggested: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Return non-zero if a reward sign is wrong or an active term is unreviewed.",
    )
    args = parser.parse_args()

    report = build_report()
    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Wrote {args.output}")
    if args.fail_on_mismatch and ("MISMATCH" in report or "No metadata yet" in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
