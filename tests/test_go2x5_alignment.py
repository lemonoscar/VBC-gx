import importlib.util
import pathlib
import xml.etree.ElementTree as ET

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "low-level/legged_gym/envs/manip_loco/go2x5_robot_spec.py"
URDF_PATH = ROOT / "low-level/resources/robots/go2x5/go2_x5.urdf"
HIGH_LEVEL_CFG_PATH = ROOT / "high-level/data/cfg/go2x5_pickmulti.yaml"


def load_robot_spec():
    module_spec = importlib.util.spec_from_file_location("go2x5_robot_spec", SPEC_PATH)
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
    assert env_cfg["requireLowPolicyMetadata"] is True
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
    assert env_cfg["liftedSuccessThreshold"] == spec.HIGH_LEVEL_LIFT_SUCCESS_HEIGHT
    assert env_cfg["successEeDistThreshold"] == spec.HIGH_LEVEL_EE_SUCCESS_DISTANCE
    assert env_cfg["baseObjectDisThreshold"] == spec.HIGH_LEVEL_BASE_OBJECT_DISTANCE
    assert asset_cfg["control"]["armPositionDriveStiffness"] == spec.ARM_POS_STIFFNESS
    assert asset_cfg["control"]["armPositionDriveDamping"] == spec.ARM_POS_DAMPING
    assert asset_cfg["control"]["gripperPositionDriveStiffness"] == spec.ARM_POS_STIFFNESS
    assert asset_cfg["control"]["gripperPositionDriveDamping"] == spec.ARM_POS_DAMPING
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
    assert spec.DEFAULT_JOINT_ANGLES["arm_joint2"] == spec.ARM_READY_JOINT_ANGLES[1] == 2.4
    assert spec.DEFAULT_JOINT_ANGLES["arm_joint3"] == spec.ARM_READY_JOINT_ANGLES[2] == 1.15
    assert spec.DEFAULT_JOINT_ANGLES["RR_thigh_joint"] == 1.0


def test_go2x5_ee_workspace_and_table_are_in_front_of_robot():
    spec = load_robot_spec()

    world_ranges = []
    for center, local_range in zip(spec.EE_GOAL_CENTER_OFFSET, spec.EE_GOAL_LOCAL_RANGES):
        world_ranges.append([round(center + value, 6) for value in local_range])
    assert world_ranges == spec.EE_GOAL_WORLD_RANGES
    assert spec.EE_GOAL_WORLD_RANGES[0] == [0.30, 0.55]
    assert spec.EE_GOAL_WORLD_RANGES[2] == [0.08, 0.45]

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
    assert spec.HIGH_LEVEL_TABLE_HEIGHT_RANGE == [0.10, 0.15]
    assert spec.HIGH_LEVEL_TABLE_HEIGHT_RANGE[0] >= spec.HIGH_LEVEL_TABLE_DIMS[2]
    assert spec.HIGH_LEVEL_ROBOT_RESET_POSITION_RANGE_XY == [0.03, 0.03]
    assert spec.HIGH_LEVEL_ROBOT_RESET_YAW_RANGE == 0.08


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
    assert "enabled = True" in go2x5_config
    assert 'profile_name = "go2x5_simple_locomotion_reach_v2_bounded_action"' in go2x5_config
    assert "class auto_curriculum" in go2x5_config
    assert go2x5_config.count('"name": "S0_locomotion_center_reach"') == 1
    assert go2x5_config.count('"name": "S1_bidirectional_coordinated_reach"') == 1
    assert '"name": "S2_' not in go2x5_config
    assert "safety_min_feet_contacts_standing = 3.0" in go2x5_config
    assert "safety_min_feet_contacts_walking = 2.0" in go2x5_config
    assert "feet_height_target = 0.12" in go2x5_config
    assert "standing_probability = 0.25" in go2x5_config
    assert "lin_vel_x = [0.0, 0.25]" in go2x5_config
    assert "ang_vel_yaw = [-0.15, 0.15]" in go2x5_config
    assert "base_height = -1.0" in go2x5_config
    assert "termination = -100.0" in go2x5_config
    assert "tracking_contacts_shaped_force = 0.0" in go2x5_config
    assert "tracking_lin_vel_max = 2.0" in go2x5_config
    assert "tracking_ee_world = 0.4" in go2x5_config
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
    assert "tracking_ee_sigma = 1.0" in go2x5_config
    assert "ik_gain = 0.25" in go2x5_config
    assert "track_ee_orientation = False" in go2x5_config
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
    assert "walking_dof = 0.0" in go2x5_config
    assert "stability_safety = 0.0" in go2x5_config
    assert "leg_action_l2_deadzone = -0.02" in go2x5_config
    assert "tracking_ee_world_stable = 0.0" in go2x5_config

    assert "base_height = -1.0" in go2x5_config
    assert "termination = -100.0" in go2x5_config
    assert "lin_vel_z = -1.5" in go2x5_config
    assert "roll = -2.0" in go2x5_config
    assert "ang_vel_xy = -0.2" in go2x5_config
    assert "collision = -10.0" in go2x5_config
    assert "feet_drag = -0.15" in go2x5_config
    assert "action_scale = robot_spec.LOW_ACTION_SCALE" in go2x5_config
    assert '"action_scale": robot_spec.LOW_ACTION_SCALE' in go2x5_config
    assert "init_std = [[0.8, 1.0, 1.0] * 4]" in go2x5_config
    assert "min_policy_std = [[0.15, 0.25, 0.25] * 4]" in go2x5_config


def test_go2x5_runtime_contract_is_deterministic_and_name_based():
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
    assert env_cfg["armIkGain"] == contract["ik_gain"] == 0.25
    assert env_cfg["trackEeOrientation"] is contract["track_ee_orientation"] is False
    assert env_cfg["armTargetUpdatePeriod"] == contract["arm_target_update_period"] == 4
    assert env_cfg["lowFootContactThreshold"] == contract["foot_contact_threshold"] == 1.5
    assert contract["gripper_position_stiffness"] == 110.0
    assert contract["gripper_position_damping"] == 7.5
    assert contract["leg_stiffness"] == 40.0
    assert contract["leg_damping"] == 1.0
    assert asset_cfg["control"]["stiffness"]["hip"] == 40
    assert asset_cfg["control"]["damping"]["hip"] == 1.0

    assert 'actions = self.action_history_buf[:, -(self.action_delay + 1)]' in low_level
    assert "self._reindex_all(self.actions)[:, :12]" in low_level
    assert env_cfg["lowPolicyObserveGaitCommands"] is False
    assert "gait_frequency" not in contract
    assert contract["replace_cylinder_with_capsule"] is False
    assert '"control_contract_sha256": control_contract_hash' in low_level
    assert "Low-level checkpoint control contract mismatch" in high_level
    assert "resolve_robot_start_pose(" in high_level
    assert "robot_start_pose=None" in high_level
    assert '"num_arm_actions": max(int(self.cfg.env.num_actions) - 12, 0)' in low_level
    assert "torch.nan_to_num" not in low_level
    assert "(self.num_envs, self.low_policy_num_actions)" in high_level


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("go2x5 alignment tests passed")
