import importlib.util
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "low-level/legged_gym/envs/manip_loco/go2x5_robot_spec.py"
WORKSPACE_GEOMETRY_PATH = (
    ROOT / "low-level/legged_gym/envs/manip_loco/go2x5_workspace_geometry.py"
)
URDF_PATH = ROOT / "low-level/resources/robots/go2x5/go2_x5.urdf"
HIGH_LEVEL_CFG_PATH = ROOT / "high-level/data/cfg/go2x5_pickmulti.yaml"


def load_robot_spec():
    module_spec = importlib.util.spec_from_file_location("go2x5_robot_spec", SPEC_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_workspace_geometry():
    module_spec = importlib.util.spec_from_file_location(
        "go2x5_workspace_geometry", WORKSPACE_GEOMETRY_PATH
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_high_level_cfg():
    with open(HIGH_LEVEL_CFG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_urdf_root():
    return ET.parse(URDF_PATH).getroot()


def test_robot_spec_matches_go2x5_urdf():
    spec = load_robot_spec()
    urdf = load_urdf_root()
    link_names = {link.attrib["name"] for link in urdf.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in urdf.findall("joint")}
    movable_joint_names = [
        joint.attrib["name"]
        for joint in urdf.findall("joint")
        if joint.attrib.get("type") != "fixed"
    ]

    assert URDF_PATH.exists()
    assert len(movable_joint_names) == spec.NUM_DOFS
    assert movable_joint_names == spec.MOVABLE_JOINT_NAMES
    assert spec.URDF_LEG_JOINT_NAMES == [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]
    assert spec.POLICY_LEG_JOINT_NAMES == [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    assert spec.URDF_FOOT_BODY_NAMES == ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    assert spec.FOOT_BODY_NAMES == ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
    for foot_name in spec.URDF_FOOT_BODY_NAMES:
        collision_sphere = urdf.find(
            f"./link[@name='{foot_name}']/collision/geometry/sphere"
        )
        assert collision_sphere is not None
        assert float(collision_sphere.attrib["radius"]) == spec.FOOT_COLLISION_RADIUS
    assert spec.EE_BODY_NAME in link_names
    assert spec.WRIST_BODY_NAME in link_names
    assert spec.FLANGE_BODY_NAME in link_names
    assert set(spec.FINGER_BODY_NAMES).issubset(link_names)
    assert spec.GRIPPER_JOINT_NAMES == ["arm_joint7", "arm_joint8"]
    assert spec.NUM_GRIPPER_DOFS == 1
    assert spec.NUM_PHYSICAL_GRIPPER_DOFS == 2
    assert joints["arm_joint7"].find("axis").attrib["xyz"] == "0 1 0"
    assert joints["arm_joint8"].find("axis").attrib["xyz"] == "0 -1 0"
    mimic = joints["arm_joint8"].find("mimic")
    assert mimic is not None
    assert mimic.attrib["joint"] == "arm_joint7"
    assert mimic.attrib["multiplier"] == "1"


def test_high_level_yaml_uses_same_robot_interface():
    spec = load_robot_spec()
    cfg = load_high_level_cfg()
    env_cfg = cfg["env"]
    asset_cfg = env_cfg["asset"]

    assert env_cfg["lowPolicyNumActions"] == spec.ACTION_DIM
    assert env_cfg["lowPolicyObserveGaitCommands"] is False
    assert env_cfg["lowPolicyReorderDofs"] is True
    assert env_cfg["lowPolicyOutputTanh"] is True
    assert env_cfg["lowPolicyActionClip"] == 1.0
    assert env_cfg["requireLowPolicyMetadata"] is True
    assert env_cfg["low_policy_path"] == ""
    assert env_cfg["numGripperDof"] == spec.NUM_GRIPPER_DOFS
    assert env_cfg["numPhysicalGripperDof"] == spec.NUM_PHYSICAL_GRIPPER_DOFS
    assert env_cfg["gripperOpenAtUpper"] is True
    assert env_cfg["lowActionScale"] == spec.LOW_ACTION_SCALE
    assert spec.LEG_STIFFNESS == 40.0
    assert spec.LEG_DAMPING == 1.0
    assert len(env_cfg["lowActionScale"]) == env_cfg["lowPolicyNumActions"]
    assert env_cfg["robotStartPose"][2] == spec.BASE_INIT_HEIGHT
    assert env_cfg["evalRobotStartPose"][2] == spec.BASE_INIT_HEIGHT
    assert cfg["reward"]["base_height_target"] == spec.BASE_HEIGHT_TARGET
    assert env_cfg["lowEeGoalRanges"] == spec.EE_GOAL_LOCAL_RANGES
    assert env_cfg["initialEEGoalCart"] == spec.EE_GOAL_INIT_END_LOCAL
    assert env_cfg["maskArmGoalCart"] == spec.EE_GOAL_MASK_LOCAL
    assert env_cfg["eeGoalCenterOffset"] == spec.EE_GOAL_CENTER_OFFSET
    assert env_cfg["robotStartPose"] == spec.HIGH_LEVEL_ROBOT_START_POSE
    assert env_cfg["evalRobotStartPose"] == spec.HIGH_LEVEL_ROBOT_START_POSE
    assert env_cfg["tableDims"] == spec.HIGH_LEVEL_TABLE_DIMS
    assert env_cfg["tablePositionXY"] == spec.HIGH_LEVEL_TABLE_POSITION_XY
    assert env_cfg["tableHeightRange"] == spec.HIGH_LEVEL_TABLE_HEIGHT_RANGE
    assert env_cfg["objectPositionRangeX"] == spec.HIGH_LEVEL_OBJECT_POSITION_RANGE_X
    assert env_cfg["objectPositionRangeY"] == spec.HIGH_LEVEL_OBJECT_POSITION_RANGE_Y
    assert env_cfg["robotResetPositionRangeXY"] == spec.HIGH_LEVEL_ROBOT_RESET_POSITION_RANGE_XY
    assert env_cfg["robotResetYawRange"] == spec.HIGH_LEVEL_ROBOT_RESET_YAW_RANGE
    assert env_cfg["resetEEGoalToCurrent"] is spec.HIGH_LEVEL_RESET_EE_GOAL_TO_CURRENT
    assert env_cfg["objectFallTolerance"] == spec.HIGH_LEVEL_OBJECT_FALL_TOLERANCE
    assert env_cfg["liftedSuccessThreshold"] == spec.HIGH_LEVEL_LIFT_SUCCESS_HEIGHT
    assert env_cfg["successEeDistThreshold"] == spec.HIGH_LEVEL_EE_SUCCESS_DISTANCE
    assert env_cfg["baseObjectDisThreshold"] == spec.HIGH_LEVEL_BASE_OBJECT_DISTANCE
    assert env_cfg["commandStopDistance"] == spec.HIGH_LEVEL_COMMAND_STOP_DISTANCE
    assert env_cfg["printResetStats"] is False
    assert cfg["reward"]["scales"]["ee_orn"] == 0.25
    assert asset_cfg["control"]["armPositionDriveStiffness"] == spec.ARM_POS_STIFFNESS
    assert asset_cfg["control"]["armPositionDriveDamping"] == spec.ARM_POS_DAMPING
    assert (
        asset_cfg["control"]["gripperPositionDriveStiffness"]
        == spec.GRIPPER_POS_STIFFNESS
    )
    assert (
        asset_cfg["control"]["gripperPositionDriveDamping"]
        == spec.GRIPPER_POS_DAMPING
    )
    assert env_cfg["lowEeGoalOrnRanges"] == spec.EE_ORIENTATION_ABSOLUTE_RANGES
    assert env_cfg["initialEEGoalOrnRPY"] == spec.EE_ORIENTATION_NOMINAL_RPY
    assert env_cfg["armBaseOffset"] == spec.ARM_BASE_OFFSET
    assert env_cfg["eeBodyName"] == spec.EE_BODY_NAME
    assert env_cfg["wristBodyName"] == spec.WRIST_BODY_NAME
    assert env_cfg["flangeBodyName"] == spec.FLANGE_BODY_NAME
    assert env_cfg["fingerBodyNames"] == spec.FINGER_BODY_NAMES
    assert asset_cfg["robotAssetRoot"] == spec.HIGH_LEVEL_ASSET_ROOT
    assert asset_cfg["assetFileRobot"] == spec.HIGH_LEVEL_ASSET_FILE

    high_level_root = ROOT / "high-level"
    high_level_asset = (high_level_root / asset_cfg["robotAssetRoot"] / asset_cfg["assetFileRobot"]).resolve()
    assert high_level_asset.samefile(URDF_PATH)


def test_low_level_observation_dimensions_are_explicit():
    spec = load_robot_spec()

    assert spec.ACTION_DIM == 12
    assert spec.NUM_GRIPPER_DOFS == 1
    assert spec.NUM_PHYSICAL_GRIPPER_DOFS == 2
    assert spec.PROPRIO_DIM_WITHOUT_GAIT == 66
    assert spec.PRIV_DIM == 18
    assert spec.HISTORY_LEN == 10
    assert spec.observation_dim(False) == 744
    assert spec.observation_dim(True) == 799
    assert spec.BASE_HEIGHT_TARGET == 0.32
    assert spec.BASE_INIT_HEIGHT == 0.32
    assert spec.ARM_TARGET_MODE == "persistent_joint_command"
    assert spec.ARM_IK_GAIN == 0.20
    assert spec.ARM_IK_ORIENTATION_WEIGHT == 0.35
    assert spec.ARM_TARGET_MAX_STEP == 0.08
    assert spec.LOW_LEVEL_GRIPPER_HOLD_MODE == "open_upper_limit"
    assert spec.DEFAULT_JOINT_ANGLES["arm_joint2"] == spec.ARM_READY_JOINT_ANGLES[1] == 2.4
    assert spec.DEFAULT_JOINT_ANGLES["arm_joint3"] == spec.ARM_READY_JOINT_ANGLES[2] == 1.15
    assert spec.DEFAULT_JOINT_ANGLES["RR_thigh_joint"] == 1.0


def test_go2x5_ee_workspace_and_table_are_in_front_of_robot():
    spec = load_robot_spec()

    world_ranges = []
    for center, local_range in zip(spec.EE_GOAL_CENTER_OFFSET, spec.EE_GOAL_LOCAL_RANGES):
        world_ranges.append([round(center + value, 6) for value in local_range])
    assert world_ranges == spec.EE_GOAL_WORLD_RANGES
    assert spec.EE_GOAL_WORLD_RANGES[0] == [0.30, 0.65]
    assert spec.EE_GOAL_WORLD_RANGES[1] == [-0.225, 0.225]
    assert spec.EE_GOAL_WORLD_RANGES[2] == [0.05, 0.45]
    assert spec.EE_GOAL_MAX_NOMINAL_REACH_RADIUS == 0.64

    init_start_world = [
        round(center + value, 6)
        for center, value in zip(spec.EE_GOAL_CENTER_OFFSET, spec.EE_GOAL_INIT_START_LOCAL)
    ]
    init_end_world = [
        round(center + value, 6)
        for center, value in zip(spec.EE_GOAL_CENTER_OFFSET, spec.EE_GOAL_INIT_END_LOCAL)
    ]
    assert init_start_world == [0.487, 0.0, 0.306]
    assert init_end_world == [0.45, 0.0, 0.25]

    robot_x = spec.HIGH_LEVEL_ROBOT_START_POSE[0]
    table_center_x = spec.HIGH_LEVEL_TABLE_POSITION_XY[0]
    table_near_edge_x = table_center_x - spec.HIGH_LEVEL_TABLE_DIMS[0] / 2.0
    assert round(table_near_edge_x - robot_x, 6) == 0.30
    object_root_forward_x = [
        spec.HIGH_LEVEL_TABLE_POSITION_XY[0] + bound - robot_x
        for bound in spec.HIGH_LEVEL_OBJECT_POSITION_RANGE_X
    ]
    assert (
        spec.EE_GOAL_WORLD_RANGES[0][0]
        <= min(object_root_forward_x)
        <= max(object_root_forward_x)
        <= spec.EE_GOAL_WORLD_RANGES[0][1]
    )
    assert (
        spec.EE_GOAL_WORLD_RANGES[1][0]
        <= spec.HIGH_LEVEL_OBJECT_POSITION_RANGE_Y[0]
        <= spec.HIGH_LEVEL_OBJECT_POSITION_RANGE_Y[1]
        <= spec.EE_GOAL_WORLD_RANGES[1][1]
    )
    assert spec.HIGH_LEVEL_TABLE_HEIGHT_RANGE == [0.10, 0.20]
    assert spec.HIGH_LEVEL_TABLE_HEIGHT_RANGE[0] >= spec.HIGH_LEVEL_TABLE_DIMS[2]
    assert spec.HIGH_LEVEL_ROBOT_RESET_POSITION_RANGE_XY == [0.03, 0.03]
    assert spec.HIGH_LEVEL_ROBOT_RESET_YAW_RANGE == 0.08


def test_go2x5_colored_workspace_uses_production_collision_predicates():
    spec = load_robot_spec()
    geometry = load_workspace_geometry()
    box = {
        axis: bounds
        for axis, bounds in zip(("x", "y", "z"), spec.EE_GOAL_LOCAL_RANGES)
    }
    points = geometry.cartesian_grid(box, geometry.parse_grid_resolution("11,9,11"))
    accepted, collision_rejected, underground_rejected = geometry.classify_cartesian_goals(
        points,
        collision_lower_limits=[-0.8, -0.2, -0.7],
        collision_upper_limits=[0.24, 0.2, 0.05],
        underground_limit=-0.6,
    )

    assert points.shape == (1089, 3)
    assert accepted.sum() == 1012
    assert collision_rejected.sum() == 77
    assert underground_rejected.sum() == 0
    assert np.all(points[collision_rejected, 0] < 0.24)
    assert not np.any(
        np.all(points[accepted] < np.array([0.24, 0.2, 0.05]), axis=1)
        & np.all(points[accepted] > np.array([-0.8, -0.2, -0.7]), axis=1)
    )
    reach_rejected = geometry.nominal_reach_rejected(
        points, spec.EE_GOAL_MAX_NOMINAL_REACH_RADIUS
    )
    assert reach_rejected.any()
    assert geometry.nominal_reach_rejected(
        np.array([[0.565, 0.225, -0.364]]),
        spec.EE_GOAL_MAX_NOMINAL_REACH_RADIUS,
    ).item()
    assert not geometry.nominal_reach_rejected(
        np.array([[0.365, 0.0, -0.064]]),
        spec.EE_GOAL_MAX_NOMINAL_REACH_RADIUS,
    ).item()

    segments = geometry.cross_marker_segments(points[:2], half_extent=0.005)
    assert segments.shape == (6, 2, 3)
    assert np.isfinite(segments).all()


def test_low_level_action_interfaces_keep_b1z1_full_dim():
    go2x5_config = (ROOT / "low-level/legged_gym/envs/manip_loco/go2x5_config.py").read_text(encoding="utf-8")
    b1z1_config = (ROOT / "low-level/legged_gym/envs/manip_loco/b1z1_config.py").read_text(encoding="utf-8")
    manip_loco = (ROOT / "low-level/legged_gym/envs/manip_loco/manip_loco.py").read_text(encoding="utf-8")

    assert "num_actions = robot_spec.ACTION_DIM" in go2x5_config
    assert "num_torques = robot_spec.NUM_TORQUES" in go2x5_config
    assert "num_actions = 12 + 6" in b1z1_config
    assert "num_torques = 12 + 6" in b1z1_config
    assert "if actions.shape[1] == 12 and self.num_torques == 12:" in manip_loco
    assert "elif actions.shape[1] == self.num_torques:" in manip_loco
    assert "default_torques[:, 12:] = 0." in manip_loco
    assert "self._reindex_all(self.actions)[:, :12]" in manip_loco


def test_go2x5_joint_and_foot_reindexing_is_self_inverse():
    spec = load_robot_spec()
    manip_loco = (ROOT / "low-level/legged_gym/envs/manip_loco/manip_loco.py").read_text(encoding="utf-8")
    b1z1_base = (ROOT / "high-level/envs/b1z1_base.py").read_text(encoding="utf-8")

    leg_perm = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    foot_perm = [1, 0, 3, 2]

    assert [spec.URDF_LEG_JOINT_NAMES[i] for i in leg_perm] == spec.POLICY_LEG_JOINT_NAMES
    assert [spec.POLICY_LEG_JOINT_NAMES[i] for i in leg_perm] == spec.URDF_LEG_JOINT_NAMES
    assert [spec.URDF_FOOT_BODY_NAMES[i] for i in foot_perm] == spec.FOOT_BODY_NAMES
    assert [spec.FOOT_BODY_NAMES[i] for i in foot_perm] == spec.URDF_FOOT_BODY_NAMES

    assert "actions = self._reindex_all(actions)" in manip_loco
    assert "low_actions = self._reindex_low_all(low_actions)" in b1z1_base
    assert "low_action_obs = self._reindex_low_all(self.last_low_actions)[:, :12]" in b1z1_base


def test_go2x5_default_joint_angles_cover_all_dofs_and_respect_limits():
    spec = load_robot_spec()
    urdf = load_urdf_root()
    joints = {
        joint.attrib["name"]: joint
        for joint in urdf.findall("joint")
        if joint.attrib.get("type") != "fixed"
    }

    assert set(spec.DEFAULT_JOINT_ANGLES) == set(spec.MOVABLE_JOINT_NAMES)
    assert len(spec.LOW_ACTION_SCALE) == spec.ACTION_DIM

    for joint_name, default_angle in spec.DEFAULT_JOINT_ANGLES.items():
        limit = joints[joint_name].find("limit")
        assert limit is not None
        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])
        assert lower <= default_angle <= upper, (joint_name, lower, default_angle, upper)


def test_configs_do_not_fall_back_to_old_go2x5_names():
    go2x5_pickmulti = (ROOT / "high-level/envs/go2x5_pickmulti.py").read_text(encoding="utf-8")
    b1z1_base = (ROOT / "high-level/envs/b1z1_base.py").read_text(encoding="utf-8")
    go2x5_config = (ROOT / "low-level/legged_gym/envs/manip_loco/go2x5_config.py").read_text(encoding="utf-8")
    manip_loco = (ROOT / "low-level/legged_gym/envs/manip_loco/manip_loco.py").read_text(encoding="utf-8")

    assert "x5_joint" not in go2x5_pickmulti
    assert 'find_actor_rigid_body_index(self.envs[0], self.robot_handles[0], "ee_gripper_link"' not in b1z1_base
    assert 'self.cfg["env"].get("eeBodyName"' in b1z1_base
    assert "self.ee_jacobian_idx = self.gripper_idx - 1" in b1z1_base
    assert 'self.num_physical_gripper_dof = self.cfg["env"].get("numPhysicalGripperDof", self.num_gripper_dof)' in b1z1_base
    assert "lowActionScale length must match lowPolicyNumActions" in b1z1_base
    assert 'self.low_policy_output_tanh_configured = "lowPolicyOutputTanh" in self.cfg["env"]' in b1z1_base
    assert 'if self.low_policy_output_tanh_configured or "policy_output_tanh" in alignment:' in b1z1_base
    assert "open_target = upper if self.gripper_open_at_upper else lower" in b1z1_base
    assert "self.gripper_dof_pos[:] = torch.where(u_gripper >= 0, open_target, close_target)" in b1z1_base
    assert 'control_cfg.get("armPositionDriveStiffness", 400.0)' in b1z1_base
    assert 'control_cfg.get("armPositionDriveDamping", 40.0)' in b1z1_base
    assert "robot_spec.ACTION_DIM" in go2x5_config
    assert "reorder_dofs = True" in go2x5_config
    assert "command_mode = 'cart'" in go2x5_config
    assert "center_mode = 'terrain_invariant'" in go2x5_config
    assert "x_offset = robot_spec.EE_GOAL_CENTER_OFFSET[0]" in go2x5_config
    assert "z_invariant_offset = robot_spec.EE_GOAL_CENTER_OFFSET[2]" in go2x5_config
    assert "pos_x = robot_spec.EE_GOAL_LOCAL_RANGES[0]" in go2x5_config
    assert "pos_y_cart = robot_spec.EE_GOAL_LOCAL_RANGES[1]" in go2x5_config
    assert "pos_z = robot_spec.EE_GOAL_LOCAL_RANGES[2]" in go2x5_config
    assert "pos_l = [0.20, 0.56]" in go2x5_config
    assert "pos_p = [0.15, 1.05]" in go2x5_config
    assert "pos_y = [-0.65, 0.65]" in go2x5_config
    assert "enabled = False" in go2x5_config
    assert 'profile_name = "go2x5_flat_tabletop_6d_walk_v7"' in go2x5_config
    assert "class auto_curriculum" in go2x5_config
    assert "stages = []" in go2x5_config
    assert '"name": "S0_' not in go2x5_config
    assert '"name": "S1_' not in go2x5_config
    assert "safety_min_feet_contacts_standing = 3.0" in go2x5_config
    assert "safety_min_feet_contacts_walking = 2.0" in go2x5_config
    assert "feet_height_target = 0.12" in go2x5_config
    assert "standing_probability = 0.10" in go2x5_config
    assert "straight_line_probability = 0.50" in go2x5_config
    assert "turn_in_place_probability = 0.10" in go2x5_config
    assert "straight_line_min_abs_vx = 0.15" in go2x5_config
    assert "lin_vel_x = [-0.30, 0.30]" in go2x5_config
    assert "lin_vel_y = [-0.10, 0.10]" in go2x5_config
    assert "ang_vel_yaw = [-0.25, 0.25]" in go2x5_config
    assert "base_height = 0.0" in go2x5_config
    assert "height_adaptation = -3.0" in go2x5_config
    assert "pitch_adaptation = -1.0" in go2x5_config
    assert "stand_still = 0.0" in go2x5_config
    assert "termination = 0.0" in go2x5_config
    assert "tracking_contacts_shaped_force = 0.0" in go2x5_config
    assert "tracking_lin_vel_max = 0.0" in go2x5_config
    assert "tracking_lin_vel_x_exp = 0.0" in go2x5_config
    assert "tracking_lin_vel = 2.0" in go2x5_config
    assert "tracking_ang_vel_yaw_exp = 0.0" in go2x5_config
    assert "tracking_ang_vel = 0.5" in go2x5_config
    assert "tracking_ee_world = 2.0" in go2x5_config
    assert "tracking_ee_orn = 0.6" in go2x5_config
    assert "tracking_ee_world_stable = 0.0" in go2x5_config
    assert "observe_gait_commands = False" in go2x5_config
    assert "replace_cylinder_with_capsule = False" in go2x5_config
    assert "collision_force_threshold = 5.0" in go2x5_config
    assert "randomize_friction = False" in go2x5_config
    assert "friction_range = [1.0, 1.0]" in go2x5_config
    assert "added_mass_range = [0.0, 0.0]" in go2x5_config
    assert "added_com_range_x = [0.0, 0.0]" in go2x5_config
    assert "leg_motor_strength_range = [1.0, 1.0]" in go2x5_config
    assert "push_robots = False" in go2x5_config
    assert "max_push_vel_xy = 0.0" in go2x5_config
    assert "tracking_ee_sigma = 0.15" in go2x5_config
    assert "ik_gain = robot_spec.ARM_IK_GAIN" in go2x5_config
    assert "track_ee_orientation = True" in go2x5_config
    assert "orientation_in_observation = True" in go2x5_config
    assert "ik_orientation_weight = robot_spec.ARM_IK_ORIENTATION_WEIGHT" in go2x5_config
    assert "target_mode = robot_spec.ARM_TARGET_MODE" in go2x5_config
    assert "target_max_step = robot_spec.ARM_TARGET_MAX_STEP" in go2x5_config
    assert "gripper_hold_mode = robot_spec.LOW_LEVEL_GRIPPER_HOLD_MODE" in go2x5_config
    assert 'penalize_contacts_on = ["base", "Head", "hip", "thigh", "calf", "arm_link"]' in go2x5_config
    assert "collision_upper_limits = [0.24, 0.2, 0.05]" in go2x5_config
    assert "mesh_type = 'plane'" in go2x5_config
    assert "env_spacing = 3.0" in go2x5_config
    assert 'if mesh_type in ["heightfield", "trimesh"]:' in manip_loco
    assert 'if mesh_type == "plane":' in manip_loco
    assert "self._create_ground_plane()" in manip_loco
    assert "self.custom_origins = False" in manip_loco
    assert "A PhysX plane has no Terrain object" in manip_loco
    assert "self.terrain = Terrain(self.cfg.terrain, )" not in manip_loco
    reward_file = (ROOT / "low-level/legged_gym/envs/rewards/maniploco_rewards.py").read_text(encoding="utf-8")
    runner = (ROOT / "third_party/rsl_rl/rsl_rl/runners/on_policy_runner.py").read_text(encoding="utf-8")

    assert "def _reward_foot_lateral_spacing" in reward_file
    assert "def _reward_tracking_ee_world_stable" in reward_file
    assert "def _reward_dof_error_deadzone" in reward_file
    assert "def _reward_leg_action_l2_deadzone" in reward_file
    assert "def _reward_foot_support_standing" in reward_file
    assert "if not self.cfg.env.reorder_dofs:" in manip_loco
    assert "self.ee_jacobian_idx = self.gripper_idx - 1" in manip_loco
    assert 'getattr(self.cfg.arm, "track_ee_orientation", True)' in manip_loco
    assert "task_jacobian = self.ee_j_eef[:, :3, :]" in manip_loco
    assert "task_error = dpose[:, :3, :]" in manip_loco
    assert "orientation_weight * self.ee_j_eef[:, 3:, :]" in manip_loco
    assert "self.curr_ee_goal_orn_rpy" in manip_loco
    assert "target_base = self.arm_q_command" in manip_loco
    assert "self.arm_q_command.copy_(arm_pos_targets)" in manip_loco
    assert "self.arm_q_command[env_ids] = self.dof_pos[env_ids, arm_slice]" in manip_loco
    assert "all_pos_targets[:, -self.cfg.env.num_gripper_joints:] = self.gripper_q_target" in manip_loco
    assert "def _reward_termination(self):" in manip_loco
    assert '"reset_roll_buf"' in manip_loco
    assert 'self.extras["episode"]["reset_" + name]' in manip_loco
    assert 'wandb_dict[\'Episode/\' + key] = value' in runner
    assert 'if "action_scale" in stage_cfg:' in manip_loco
    assert "self.action_scale = torch.tensor(self.cfg.control.action_scale, device=self.device)" in manip_loco
    assert "def _sync_reward_functions_and_sums" in manip_loco
    assert '"reorder_dofs": self.cfg.env.reorder_dofs' in manip_loco
    assert 'if self.cfg.goal_ee.command_mode == "cart":' in manip_loco
    assert "def _resample_ee_goal_cart_once" in manip_loco
    assert 'getattr(self.cfg.goal_ee, "center_mode", "terrain_invariant") == "arm_base"' in manip_loco


def test_go2x5_simple_training_design_matches_current_plan():
    go2x5_config = (ROOT / "low-level/legged_gym/envs/manip_loco/go2x5_config.py").read_text(encoding="utf-8")

    assert "actor_hidden_dims = [128]" in go2x5_config
    assert "critic_hidden_dims = [128]" in go2x5_config
    assert "leg_control_head_hidden_dims = [128, 128]" in go2x5_config
    assert "arm_control_head_hidden_dims = [128, 128]" in go2x5_config
    assert "priv_encoder_dims = [64, 20]" in go2x5_config
    assert "num_leg_actions = 12" in go2x5_config
    assert "num_arm_actions = 0" in go2x5_config

    assert "tracking_contacts_shaped_force = 0.0" in go2x5_config
    assert "tracking_contacts_shaped_vel = 0.0" in go2x5_config
    assert "feet_height = 0.0" in go2x5_config
    assert "feet_air_time = 1.0" in go2x5_config
    assert "feet_contact_standing = -0.5" in go2x5_config
    assert "walking_dof = 0.0" in go2x5_config
    assert "stability_safety = 0.0" in go2x5_config
    assert "leg_action_l2_deadzone = 0.0" in go2x5_config
    assert "tracking_ee_world_stable = 0.0" in go2x5_config

    assert "base_height = 0.0" in go2x5_config
    assert "height_adaptation = -3.0" in go2x5_config
    assert "pitch_adaptation = -1.0" in go2x5_config
    assert "tracking_lin_vel = 2.0" in go2x5_config
    assert "tracking_ang_vel = 0.5" in go2x5_config
    assert "termination = 0.0" in go2x5_config
    assert "lin_vel_z = -1.0" in go2x5_config
    assert "roll = -2.0" in go2x5_config
    assert "ang_vel_xy = 0.0" in go2x5_config
    assert "collision = -1.0" in go2x5_config
    assert "action_rate = -0.01" in go2x5_config
    assert "feet_drag = -0.20" in go2x5_config
    assert "action_scale = robot_spec.LOW_ACTION_SCALE" in go2x5_config
    assert "enabled = False" in go2x5_config
    assert "output_tanh = True" in go2x5_config
    assert "clip_actions = 1.0" in go2x5_config
    assert "init_std = [[0.25, 0.30, 0.30] * 4]" in go2x5_config
    assert "entropy_coef = 0.01" in go2x5_config
    assert "min_policy_std = [[0.08, 0.12, 0.12] * 4]" in go2x5_config


def test_go2x5_runtime_contract_is_deterministic_and_name_based():
    spec = load_robot_spec()
    cfg = load_high_level_cfg()
    env_cfg = cfg["env"]
    asset_cfg = env_cfg["asset"]
    high_level = (ROOT / "high-level/envs/b1z1_base.py").read_text(encoding="utf-8")
    low_level = (ROOT / "low-level/legged_gym/envs/manip_loco/manip_loco.py").read_text(encoding="utf-8")
    go2x5_config = (ROOT / "low-level/legged_gym/envs/manip_loco/go2x5_config.py").read_text(encoding="utf-8")

    assert env_cfg["asset"]["baseBodyName"] == "base"
    assert "props[1]" not in high_level
    assert "props[1]" not in low_level
    assert 'base_body_name = "base"' in go2x5_config

    dr = env_cfg["domainRandomization"]
    assert dr["friction"] == [1.0, 1.0]
    assert dr["motorStrength"] == [1.0, 1.0]
    assert dr["addedBaseMassKg"] == [0.0, 0.0]
    assert dr["baseComOffsetM"] == [[0.0, 0.0]] * 3

    physx = cfg["sim"]["physx"]
    contract = env_cfg["lowPolicyContract"]
    assert physx["num_position_iterations"] == contract["physx"]["num_position_iterations"] == 4
    assert physx["contact_offset"] == contract["physx"]["contact_offset"] == 0.01
    assert physx["bounce_threshold_velocity"] == contract["physx"]["bounce_threshold_velocity"] == 0.5
    assert physx["max_depenetration_velocity"] == contract["physx"]["max_depenetration_velocity"] == 1.0
    assert physx["default_buffer_size_multiplier"] == contract["physx"]["default_buffer_size_multiplier"] == 5.0

    assert env_cfg["eeFrame"] == contract["ee_frame"] == "TERRAIN_INVARIANT_YAW"
    assert env_cfg["armIkGain"] == contract["ik_gain"] == spec.ARM_IK_GAIN
    assert env_cfg["trackEeOrientation"] is contract["track_ee_orientation"] is True
    assert (
        env_cfg["armIkOrientationWeight"]
        == contract["ik_orientation_weight"]
        == spec.ARM_IK_ORIENTATION_WEIGHT
    )
    assert contract["ik_task"] == "pose_6d_weighted_dls"
    assert contract["ee_goal_ranges"] == spec.EE_GOAL_LOCAL_RANGES
    assert (
        env_cfg["lowEeGoalMaxNominalReachRadius"]
        == contract["ee_goal_max_nominal_reach_radius"]
        == spec.EE_GOAL_MAX_NOMINAL_REACH_RADIUS
    )
    assert (
        contract["ee_orientation_delta_ranges"]
        == spec.EE_ORIENTATION_DELTA_RANGES
    )
    assert (
        contract["ee_orientation_nominal_rpy"]
        == spec.EE_ORIENTATION_NOMINAL_RPY
    )
    assert contract["ee_orientation_observation"] == "local_rpy"
    assert env_cfg["armTargetMode"] == contract["arm_target_mode"] == spec.ARM_TARGET_MODE
    assert env_cfg["armTargetMaxStep"] == contract["arm_target_max_step"] == spec.ARM_TARGET_MAX_STEP
    assert contract["gripper_hold_mode"] == spec.LOW_LEVEL_GRIPPER_HOLD_MODE
    assert "self.arm_ik_orientation_weight * self.ee_j_eef[:, 3:, :]" in high_level
    assert "target_base = self.arm_q_command" in high_level
    assert "self.arm_q_command.copy_(arm_pos_targets)" in high_level
    assert "self.arm_q_command[env_ids] = self._dof_pos[env_ids, arm_slice]" in high_level
    assert env_cfg["armTargetUpdatePeriod"] == contract["arm_target_update_period"] == 4
    assert env_cfg["lowFootContactThreshold"] == contract["foot_contact_threshold"] == 1.5
    assert contract["arm_position_stiffness"] == spec.ARM_POS_STIFFNESS
    assert contract["arm_position_damping"] == spec.ARM_POS_DAMPING
    assert contract["gripper_position_stiffness"] == spec.GRIPPER_POS_STIFFNESS
    assert contract["gripper_position_damping"] == spec.GRIPPER_POS_DAMPING
    assert contract["leg_stiffness"] == 40.0
    assert contract["leg_damping"] == 1.0
    assert asset_cfg["control"]["stiffness"]["hip"] == 40
    assert asset_cfg["control"]["damping"]["hip"] == 1.0

    assert 'actions = self.action_history_buf[:, -(self.action_delay + 1)]' in low_level
    assert "self._reindex_all(self.actions)[:, :12]" in low_level
    assert env_cfg["lowPolicyObserveGaitCommands"] is False
    assert "gait_frequency" not in contract
    assert contract["replace_cylinder_with_capsule"] is False
    assert contract["policy_action_clip"] == env_cfg["lowPolicyActionClip"] == 1.0
    assert '"control_contract_sha256": control_contract_hash' in low_level
    assert "Low-level checkpoint control contract mismatch" in high_level
    assert "resolve_robot_start_pose(" in high_level
    assert "robot_start_pose=None" in high_level
    assert "self.arm_base_offset.unsqueeze(0).expand(self.num_envs, -1)" in high_level
    assert "ee_goal_global = self.ee_goal_world" in high_level
    assert '"num_arm_actions": max(int(self.cfg.env.num_actions) - 12, 0)' in low_level
    assert '"policy_output_tanh": bool(self.cfg.env.policy_output_tanh)' in low_level
    assert '"output_tanh": self.low_policy_output_tanh' in high_level
    assert "low_actions = torch.clamp(" in high_level
    assert "torch.nan_to_num" not in low_level
    assert "(self.num_envs, self.low_policy_num_actions)" in high_level


def test_deprecated_go2x5_training_paths_are_removed():
    env_registry = (ROOT / "low-level/legged_gym/envs/__init__.py").read_text(
        encoding="utf-8"
    )
    train_entrypoint = (
        ROOT / "low-level/legged_gym/scripts/train.py"
    ).read_text(encoding="utf-8")

    assert "go2x5_ftlift" not in env_registry
    assert "go2x5_ftlift" not in train_entrypoint
    assert not (
        ROOT
        / "low-level/legged_gym/envs/manip_loco/go2x5_ftlift_config.py"
    ).exists()
    assert not (
        ROOT / "low-level/resources/robots/go2x5/urdf/go2_arx_x5.urdf"
    ).exists()
    assert not (
        ROOT / "low-level/resources/robots/go2x5/urdf/go2_arx_x5_clean.urdf"
    ).exists()
    assert not (ROOT / "high-level/go2x5-pick-multi-teacher").exists()


def test_high_level_training_entrypoint_is_fail_closed_and_one_shot():
    runtime_path = ROOT / "high-level/envs/runtime_contract.py"
    runtime_spec = importlib.util.spec_from_file_location(
        "runtime_contract_training", runtime_path
    )
    runtime = importlib.util.module_from_spec(runtime_spec)
    runtime_spec.loader.exec_module(runtime)

    assert runtime.object_fell_below_table(0.079, 0.10, 0.02)
    assert not runtime.object_fell_below_table(0.080, 0.10, 0.02)
    assert not runtime.object_fell_below_table(0.099, 0.10, 0.02)
    try:
        runtime.object_fell_below_table(0.0, 0.1, -0.01)
    except ValueError:
        pass
    else:
        raise AssertionError("negative object-fall tolerance was accepted")

    base = (ROOT / "high-level/envs/b1z1_base.py").read_text(encoding="utf-8")
    reset_start = base.index("    def _reset_envs(self, env_ids):")
    reset_end = base.index("    def _reset_ee_goal(self, env_ids):")
    reset_body = base[reset_start:reset_end]
    assert reset_body.index("self._reset_ee_goal(env_ids)") < reset_body.index(
        "self._compute_observations(env_ids)"
    )

    config_source = (ROOT / "high-level/utils/config.py").read_text(encoding="utf-8")
    trainer_source = (ROOT / "high-level/train_multistate.py").read_text(encoding="utf-8")
    reward_source = (ROOT / "high-level/envs/reward_vec_task.py").read_text(
        encoding="utf-8"
    )
    pickmulti_source = (ROOT / "high-level/envs/b1z1_pickmulti.py").read_text(
        encoding="utf-8"
    )
    launch_source = (ROOT / "high-level/run_go2x5_train_stable.sh").read_text(
        encoding="utf-8"
    )
    readiness_source = (
        ROOT / "high-level/check_go2x5_training_readiness.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--low_policy_path"' in config_source
    assert 'cfg["env"]["low_policy_path"] = low_policy_path' in trainer_source
    assert "set_seed(args.seed)" in trainer_source
    assert "base_obj_dis < self.command_stop_distance" in reward_source
    assert "obj_dir[:, 2] = 0." in reward_source
    assert "torch.abs(self.commands[:, 0]) <= self.lin_vel_x_clip" in pickmulti_source
    assert "if not report_to_wandb and not self.print_reset_stats:" in pickmulti_source
    assert "while true" not in launch_source
    assert "set -euo pipefail" in launch_source
    assert "LOW_POLICY_PATH" in launch_source
    assert "total_resets == 0" in readiness_source
    assert "nonfinite_count == 0" in readiness_source


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("go2x5 alignment tests passed")
