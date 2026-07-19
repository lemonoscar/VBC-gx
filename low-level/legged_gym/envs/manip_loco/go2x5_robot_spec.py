"""Shared Go2 + ARX-X5 interface constants for low-level and high-level configs."""

LOW_LEVEL_ASSET_FILE = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2x5/go2_x5.urdf"
HIGH_LEVEL_ASSET_ROOT = "../low-level/resources/robots"
HIGH_LEVEL_ASSET_FILE = "go2x5/go2_x5.urdf"

NUM_DOFS = 20
NUM_LEG_DOFS = 12
NUM_ARM_DOFS = 6
NUM_PHYSICAL_GRIPPER_DOFS = 2
NUM_GRIPPER_DOFS = 1
ACTION_DIM = 12
NUM_TORQUES = 12

PROPRIO_DIM_WITHOUT_GAIT = 66
GAIT_COMMAND_DIM = 5
PRIV_DIM = 18
HISTORY_LEN = 10

EE_BODY_NAME = "arm_eef_link"
WRIST_BODY_NAME = "arm_link6"
FLANGE_BODY_NAME = "arm_link6"
FINGER_BODY_NAMES = ["arm_link7", "arm_link8"]
ARM_BASE_OFFSET = [0.085, 0.0, 0.094]

BASE_HEIGHT_TARGET = 0.32
BASE_INIT_HEIGHT = 0.32

GO2X5_LAB_LEG_ACTION_SCALE = [0.125, 0.25, 0.25] * 4
LEGACY_LEG_ACTION_SCALE = [0.4, 0.45, 0.45] * 4
# Keep the VBC low-level action contract identical to the Go2-X5-lab asset.
LOW_ACTION_SCALE = GO2X5_LAB_LEG_ACTION_SCALE

LEG_STIFFNESS = 60.0
LEG_DAMPING = 1.5
ARM_POS_STIFFNESS = 110.0
ARM_POS_DAMPING = 7.5

URDF_LEG_JOINT_NAMES = [
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

POLICY_LEG_JOINT_NAMES = [
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

ARM_JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
]

GRIPPER_JOINT_NAMES = ["arm_joint7", "arm_joint8"]
MOVABLE_JOINT_NAMES = URDF_LEG_JOINT_NAMES + ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES
URDF_FOOT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
FOOT_BODY_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

LEG_JOINT_NAMES = POLICY_LEG_JOINT_NAMES

DEFAULT_JOINT_ANGLES = {
    "FR_hip_joint": 0.1,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "FL_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "RR_hip_joint": 0.1,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.5,
    "RL_hip_joint": -0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,
    "arm_joint1": 0.0,
    "arm_joint2": 0.3,
    "arm_joint3": 0.5,
    "arm_joint4": 0.0,
    "arm_joint5": 0.0,
    "arm_joint6": 0.0,
    "arm_joint7": 0.022,
    "arm_joint8": 0.022,
}


def proprio_dim(observe_gait_commands=False):
    if observe_gait_commands:
        return PROPRIO_DIM_WITHOUT_GAIT + GAIT_COMMAND_DIM
    return PROPRIO_DIM_WITHOUT_GAIT


def observation_dim(observe_gait_commands=False):
    return proprio_dim(observe_gait_commands) * (HISTORY_LEN + 1) + PRIV_DIM
