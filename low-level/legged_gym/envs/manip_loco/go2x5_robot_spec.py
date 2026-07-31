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

# Terrain-invariant EE task coordinates. The local ranges are expressed from
# the nominal arm-base center and cover the complete tabletop task volume:
# root-forward x=[0.30, 0.65] m, y=[-0.225, 0.225] m, and terrain
# z=[0.05, 0.45] m.
EE_GOAL_CENTER_OFFSET = [0.085, 0.0, 0.414]
EE_GOAL_LOCAL_RANGES = [[0.215, 0.565], [-0.225, 0.225], [-0.364, 0.036]]
EE_GOAL_WORLD_RANGES = [[0.30, 0.65], [-0.225, 0.225], [0.05, 0.45]]
# Keep all requested marginal extrema while excluding only the combined
# far/low/lateral corners that exceed the X5's nominal arm-base reach.  The
# final tabletop volume is inside 0.60 m; 0.64 m leaves useful coordination
# margin without feeding persistent joint-limit targets to the IK controller.
EE_GOAL_MAX_NOMINAL_REACH_RADIUS = 0.64
EE_GOAL_INIT_START_LOCAL = [0.402, 0.0, -0.108]
EE_GOAL_INIT_END_LOCAL = [0.365, 0.0, -0.164]
EE_GOAL_MASK_LOCAL = [0.365, 0.0, -0.064]

# The X5 end-effector x-axis is the gripper approach direction. At the
# canonical ready pose its local RPY is approximately [0, 1.25, 0], not the
# [pi/2, ..., ...] convention inherited from Z1. The target yaw follows the
# target bearing and these bounded deltas provide genuine 6-D supervision.
EE_ORIENTATION_NOMINAL_RPY = [0.0, 1.25, 0.0]
EE_ORIENTATION_DELTA_RANGES = [
    [-0.35, 0.35],
    [-0.25, 0.25],
    [-0.35, 0.35],
]
EE_ORIENTATION_ABSOLUTE_RANGES = [
    [-0.35, 0.35],
    [1.00, 1.50],
    [-1.00, 1.00],
]

# A forward-ready X5 pose.  Its nominal EE position is approximately
# root-forward [0.487, 0.0, 0.306] m at the 0.32 m Go2 stance height.
ARM_READY_JOINT_ANGLES = [0.0, 2.4, 1.15, 0.0, 0.0, 0.0]

# Go2-X5 tabletop task geometry. Distances must account for the robot's front
# collision geometry rather than only its root: the head reaches about 0.34 m
# forward, so a table edge 0.30 m from the root visually overlaps the robot.
# The edge below is 0.40 m from the nominal root and leaves 0.06 m in front of
# the head. The collision box is a thin tabletop, not a solid platform.
HIGH_LEVEL_ROBOT_START_POSE = [-0.45, 0.0, BASE_INIT_HEIGHT]
HIGH_LEVEL_ROBOT_FRONT_COLLISION_EXTENT = 0.34
HIGH_LEVEL_TABLE_MIN_FRONT_CLEARANCE = 0.05
HIGH_LEVEL_TABLE_DIMS = [0.30, 0.60, 0.02]
HIGH_LEVEL_TABLE_POSITION_XY = [0.10, 0.0]
HIGH_LEVEL_TABLE_COLOR = [0.36, 0.20, 0.08]
HIGH_LEVEL_TABLE_HEIGHT_RANGE = [0.10, 0.20]
HIGH_LEVEL_OBJECT_POSITION_RANGE_X = [-0.10, 0.0]
HIGH_LEVEL_OBJECT_POSITION_RANGE_Y = [-0.20, 0.20]
HIGH_LEVEL_MAX_OBJECT_HEIGHT = 0.127
HIGH_LEVEL_ROBOT_RESET_POSITION_RANGE_XY = [0.03, 0.03]
HIGH_LEVEL_ROBOT_RESET_YAW_RANGE = 0.08
HIGH_LEVEL_RESET_EE_GOAL_TO_CURRENT = True
HIGH_LEVEL_OBJECT_FALL_TOLERANCE = 0.02
HIGH_LEVEL_LIFT_SUCCESS_HEIGHT = 0.15
HIGH_LEVEL_EE_SUCCESS_DISTANCE = 0.12
HIGH_LEVEL_BASE_OBJECT_DISTANCE = 0.45
HIGH_LEVEL_COMMAND_STOP_DISTANCE = 0.45

GO2X5_LAB_LEG_ACTION_SCALE = [0.125, 0.25, 0.25] * 4
LEGACY_LEG_ACTION_SCALE = [0.4, 0.45, 0.45] * 4
# Keep the VBC low-level action contract identical to the Go2-X5-lab asset.
LOW_ACTION_SCALE = GO2X5_LAB_LEG_ACTION_SCALE

LEG_STIFFNESS = 40.0
LEG_DAMPING = 1.0
# Per-joint X5 position-drive gains. The proximal joints retain the stiffness
# needed to carry the arm, while the wrist uses the lower damping/stiffness
# hierarchy used by public X5 controllers. This avoids the old uniform,
# over-damped 110/7.5 response without making the wrist unrealistically rigid.
ARM_POS_STIFFNESS = [120.0, 120.0, 100.0, 45.0, 35.0, 25.0]
ARM_POS_DAMPING = [4.0, 4.0, 3.5, 1.5, 1.2, 0.8]
GRIPPER_POS_STIFFNESS = 110.0
GRIPPER_POS_DAMPING = 7.5
ARM_IK_GAIN = 0.20
ARM_IK_ORIENTATION_WEIGHT = 0.35
ARM_TARGET_MODE = "persistent_joint_command"
ARM_TARGET_MAX_STEP = 0.08
LOW_LEVEL_GRIPPER_HOLD_MODE = "open_upper_limit"

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
FOOT_COLLISION_RADIUS = 0.022

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
    "arm_joint1": ARM_READY_JOINT_ANGLES[0],
    "arm_joint2": ARM_READY_JOINT_ANGLES[1],
    "arm_joint3": ARM_READY_JOINT_ANGLES[2],
    "arm_joint4": ARM_READY_JOINT_ANGLES[3],
    "arm_joint5": ARM_READY_JOINT_ANGLES[4],
    "arm_joint6": ARM_READY_JOINT_ANGLES[5],
    "arm_joint7": 0.022,
    "arm_joint8": 0.022,
}


def proprio_dim(observe_gait_commands=False):
    if observe_gait_commands:
        return PROPRIO_DIM_WITHOUT_GAIT + GAIT_COMMAND_DIM
    return PROPRIO_DIM_WITHOUT_GAIT


def observation_dim(observe_gait_commands=False):
    return proprio_dim(observe_gait_commands) * (HISTORY_LEN + 1) + PRIV_DIM
