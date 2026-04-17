from .b1z1_base import B1Z1Base
from .b1z1_pickmulti import B1Z1PickMulti


class Go2X5PickMulti(B1Z1PickMulti):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("num_gripper_dof", 2)
        super().__init__(*args, **kwargs)

    def _set_default_joint_angles_dict(self):
        if self.floating_base:
            default_joint_angles = {
                'x5_joint1': 0.0,
                'x5_joint2': 0.5,
                'x5_joint3': 1.5,
                'x5_joint4': 0.0,
                'x5_joint5': 0.0,
                'x5_joint6': 0.0,
                'x5_joint7': 0.022,
                'x5_joint8': 0.022,
            }
        else:
            default_joint_angles = {
                'FL_hip_joint': 0.0,
                'FL_thigh_joint': 0.9,
                'FL_calf_joint': -1.8,

                'RL_hip_joint': 0.0,
                'RL_thigh_joint': 1.0,
                'RL_calf_joint': -1.8,

                'FR_hip_joint': 0.0,
                'FR_thigh_joint': 0.9,
                'FR_calf_joint': -1.8,

                'RR_hip_joint': 0.0,
                'RR_thigh_joint': 1.0,
                'RR_calf_joint': -1.8,

                'x5_joint1': 0.0,
                'x5_joint2': 0.5,
                'x5_joint3': 1.5,
                'x5_joint4': 0.0,
                'x5_joint5': 0.0,
                'x5_joint6': 0.0,
                'x5_joint7': 0.022,
                'x5_joint8': 0.022,
            }
        return default_joint_angles

    def _setup_obs_and_action_info(self):
        # Dimension-aligned setup for Go2X5 robot observations
        B1Z1Base._setup_obs_and_action_info(self, removed_dim=9, num_action=9, num_obs=37 + self.num_features - 1)


class Go2X5Float(Go2X5PickMulti):
    pass
