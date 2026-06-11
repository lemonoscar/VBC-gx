"""Shared Go2 + ARX-X5 interface constants for low-level and high-level configs."""

LOW_LEVEL_ASSET_FILE = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2x5/go2_x5.urdf"
HIGH_LEVEL_ASSET_ROOT = "../low-level/resources/robots"
HIGH_LEVEL_ASSET_FILE = "go2x5/go2_x5.urdf"

NUM_DOFS = 20
NUM_LEG_DOFS = 12
NUM_ARM_DOFS = 6
NUM_GRIPPER_DOFS = 2
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

LOW_ACTION_SCALE = [0.4, 0.45, 0.45] * 4

LEG_JOINT_NAMES = [
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

ARM_JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
]

GRIPPER_JOINT_NAMES = ["arm_joint7", "arm_joint8"]
MOVABLE_JOINT_NAMES = LEG_JOINT_NAMES + ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES
POLICY_LEG_JOINT_NAMES = LEG_JOINT_NAMES
FOOT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

DEFAULT_JOINT_ANGLES = {
    "FL_hip_joint": 0.2,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "RL_hip_joint": 0.2,
    "RL_thigh_joint": 0.8,
    "RL_calf_joint": -1.5,
    "FR_hip_joint": -0.2,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RR_hip_joint": -0.2,
    "RR_thigh_joint": 0.8,
    "RR_calf_joint": -1.5,
    "arm_joint1": 0.0,
    "arm_joint2": 0.5,
    "arm_joint3": 1.5,
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
