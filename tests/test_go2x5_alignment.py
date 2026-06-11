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
    movable_joint_names = [
        joint.attrib["name"]
        for joint in urdf.findall("joint")
        if joint.attrib.get("type") != "fixed"
    ]

    assert URDF_PATH.exists()
    assert len(movable_joint_names) == spec.NUM_DOFS
    assert movable_joint_names == spec.MOVABLE_JOINT_NAMES
    assert spec.POLICY_LEG_JOINT_NAMES == spec.LEG_JOINT_NAMES
    assert spec.FOOT_BODY_NAMES == ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    assert spec.EE_BODY_NAME in link_names
    assert spec.WRIST_BODY_NAME in link_names
    assert spec.FLANGE_BODY_NAME in link_names
    assert set(spec.FINGER_BODY_NAMES).issubset(link_names)


def test_high_level_yaml_uses_same_robot_interface():
    spec = load_robot_spec()
    cfg = load_high_level_cfg()
    env_cfg = cfg["env"]
    asset_cfg = env_cfg["asset"]

    assert env_cfg["lowPolicyNumActions"] == spec.ACTION_DIM
    assert env_cfg["lowPolicyObserveGaitCommands"] is True
    assert env_cfg["lowPolicyReorderDofs"] is False
    assert env_cfg["requireLowPolicyMetadata"] is True
    assert env_cfg["lowActionScale"] == spec.LOW_ACTION_SCALE
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
    assert spec.PROPRIO_DIM_WITHOUT_GAIT == 66
    assert spec.PRIV_DIM == 18
    assert spec.HISTORY_LEN == 10
    assert spec.observation_dim(False) == 744
    assert spec.observation_dim(True) == 799


def test_configs_do_not_fall_back_to_old_go2x5_names():
    go2x5_pickmulti = (ROOT / "high-level/envs/go2x5_pickmulti.py").read_text(encoding="utf-8")
    b1z1_base = (ROOT / "high-level/envs/b1z1_base.py").read_text(encoding="utf-8")
    go2x5_config = (ROOT / "low-level/legged_gym/envs/manip_loco/go2x5_config.py").read_text(encoding="utf-8")
    manip_loco = (ROOT / "low-level/legged_gym/envs/manip_loco/manip_loco.py").read_text(encoding="utf-8")

    assert "x5_joint" not in go2x5_pickmulti
    assert 'find_actor_rigid_body_index(self.envs[0], self.robot_handles[0], "ee_gripper_link"' not in b1z1_base
    assert 'self.cfg["env"].get("eeBodyName"' in b1z1_base
    assert "robot_spec.ACTION_DIM" in go2x5_config
    assert "reorder_dofs = False" in go2x5_config
    assert 'profile_name = "go2x5_stable_auto_v2"' in go2x5_config
    assert "class auto_curriculum" in go2x5_config
    assert "S0_sanity_flat" in go2x5_config
    assert "S4_robustness" in go2x5_config
    assert '"max_terrain_level": 1' in go2x5_config
    assert '"max_terrain_level": 10' in go2x5_config
    assert "collision_force_threshold = 5.0" in go2x5_config
    assert "def _reward_foot_lateral_spacing" in (ROOT / "low-level/legged_gym/envs/rewards/maniploco_rewards.py").read_text(encoding="utf-8")
    assert "if not self.cfg.env.reorder_dofs:" in manip_loco
    assert '"reorder_dofs": self.cfg.env.reorder_dofs' in manip_loco


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("go2x5 alignment tests passed")
