from .b1z1_base import B1Z1Base
from .b1z1_pickmulti import B1Z1PickMulti


class Go2X5PickMulti(B1Z1PickMulti):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("num_gripper_dof", 1)
        super().__init__(*args, **kwargs)

    def _set_default_joint_angles_dict(self):
        if self.floating_base:
            default_joint_angles = {
                'arm_joint1': 0.0,
                'arm_joint2': 2.4,
                'arm_joint3': 1.15,
                'arm_joint4': 0.0,
                'arm_joint5': 0.0,
                'arm_joint6': 0.0,
                'arm_joint7': 0.022,
                'arm_joint8': 0.022,
            }
        else:
            default_joint_angles = {
                'FR_hip_joint': 0.1,
                'FR_thigh_joint': 0.8,
                'FR_calf_joint': -1.5,

                'FL_hip_joint': -0.1,
                'FL_thigh_joint': 0.8,
                'FL_calf_joint': -1.5,

                'RR_hip_joint': 0.1,
                'RR_thigh_joint': 1.0,
                'RR_calf_joint': -1.5,

                'RL_hip_joint': -0.1,
                'RL_thigh_joint': 1.0,
                'RL_calf_joint': -1.5,

                'arm_joint1': 0.0,
                'arm_joint2': 2.4,
                'arm_joint3': 1.15,
                'arm_joint4': 0.0,
                'arm_joint5': 0.0,
                'arm_joint6': 0.0,
                'arm_joint7': 0.022,
                'arm_joint8': 0.022,
            }
        return default_joint_angles

    def _setup_obs_and_action_info(self):
        # Dimension-aligned setup for Go2X5 robot observations
        B1Z1Base._setup_obs_and_action_info(self, removed_dim=9, num_action=9, num_obs=37 + self.num_features - 1)


class Go2X5Float(Go2X5PickMulti):
    pass
