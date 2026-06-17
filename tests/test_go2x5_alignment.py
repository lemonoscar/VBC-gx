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
    assert env_cfg["lowPolicyObserveGaitCommands"] is True
    assert env_cfg["lowPolicyReorderDofs"] is True
    assert env_cfg["requireLowPolicyMetadata"] is True
    assert env_cfg["numGripperDof"] == spec.NUM_GRIPPER_DOFS
    assert env_cfg["numPhysicalGripperDof"] == spec.NUM_PHYSICAL_GRIPPER_DOFS
    assert env_cfg["gripperOpenAtUpper"] is True
    assert env_cfg["lowActionScale"] == spec.LOW_ACTION_SCALE
    assert len(env_cfg["lowActionScale"]) == env_cfg["lowPolicyNumActions"]
    assert env_cfg["initialEEGoalCart"] == [0.30, 0.0, 0.20]
    assert env_cfg["maskArmGoalCart"] == [0.34, 0.0, 0.24]
    assert asset_cfg["control"]["armPositionDriveStiffness"] == spec.ARM_POS_STIFFNESS
    assert asset_cfg["control"]["armPositionDriveDamping"] == spec.ARM_POS_DAMPING
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
    assert spec.BASE_HEIGHT_TARGET == 0.33
    assert spec.DEFAULT_JOINT_ANGLES["arm_joint2"] == 0.3
    assert spec.DEFAULT_JOINT_ANGLES["arm_joint3"] == 0.5
    assert spec.DEFAULT_JOINT_ANGLES["RR_thigh_joint"] == 1.0


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
    assert "self._reindex_all(self.action_history_buf[:, -1])[:, :12]" in manip_loco


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
    assert "x_offset = 0.0" in go2x5_config
    assert "z_invariant_offset = robot_spec.BASE_INIT_HEIGHT + 0.20" in go2x5_config
    assert "pos_l = [0.20, 0.56]" in go2x5_config
    assert "pos_p = [0.15, 1.05]" in go2x5_config
    assert "pos_y = [-0.65, 0.65]" in go2x5_config
    assert "enabled = False" in go2x5_config
    assert 'profile_name = "go2x5_b1z1_unified_v1"' in go2x5_config
    assert "class auto_curriculum" in go2x5_config
    assert "stages = []" in go2x5_config
    assert "S0_stand_sanity" not in go2x5_config
    assert "S4_robustness" not in go2x5_config
    assert "feet_height_target = 0.10" in go2x5_config
    assert "base_height = -1.5" in go2x5_config
    assert "tracking_contacts_shaped_force = -2.0" in go2x5_config
    assert "tracking_lin_vel_max = 2.0" in go2x5_config
    assert "collision_force_threshold = 5.0" in go2x5_config
    assert "def _reward_foot_lateral_spacing" in (ROOT / "low-level/legged_gym/envs/rewards/maniploco_rewards.py").read_text(encoding="utf-8")
    assert "if not self.cfg.env.reorder_dofs:" in manip_loco
    assert "self.ee_jacobian_idx = self.gripper_idx - 1" in manip_loco
    assert '"reorder_dofs": self.cfg.env.reorder_dofs' in manip_loco


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("go2x5 alignment tests passed")
