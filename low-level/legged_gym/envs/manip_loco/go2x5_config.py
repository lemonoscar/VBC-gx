# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class Go2X5RoughCfg( LeggedRobotCfg ):
    """Configuration for Go2 quadruped with ARX-X5 6-DOF manipulator arm"""
    
    class goal_ee:
        num_commands = 3
        traj_time = [1, 3]
        hold_time = [0.5, 2]
        collision_upper_limits = [0.1, 0.2, -0.05]
        collision_lower_limits = [-0.8, -0.2, -0.7]
        underground_limit = -0.7  # local cartesian z (world_z = z_invariant_offset + local_z = 0.45 + (-0.7) = -0.25m, well below ground but limits extreme downward goals)
        num_collision_check_samples = 10
        command_mode = 'sphere'
        arm_induced_pitch = 0.38

        class sphere_center:
            x_offset = 0.3  # Relative to base
            y_offset = 0    # Relative to base
            z_invariant_offset = 0.45  # Relative to terrain
            # Lower the goal sphere so low targets are reachable while moving.

        class ranges:
            init_pos_start = [0.36, np.pi/10, 0]
            init_pos_end = [0.52, 0, 0]
            pos_l = [0.25, 0.50]  # Slightly wider reach window while enabling lower goals.
            pos_p = [-0.85, 1 * np.pi / 3]  # Lower low-goal floor to increase low-target exposure.
            pos_y = [-1.2, 1.2]
            
            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]
            final_tracking_ee_reward = 0.55

        sphere_error_scale = [1, 1, 1]
        orn_error_scale = [1, 1, 1]

    class noise:
        add_noise = False
        noise_level = 1.0  # 增大观测/动作噪声
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class commands:
        curriculum = True
        num_commands = 3
        resampling_time = 3.

        lin_vel_x_schedule = [0, 0.5]
        ang_vel_yaw_schedule = [0, 1]
        tracking_ang_vel_yaw_schedule = [0, 1]

        ang_vel_yaw_clip = 0.35
        lin_vel_x_clip = 0.05

        class ranges:
            lin_vel_x = [-0.8, 0.8]
            ang_vel_yaw = [-0.8, 0.8]

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class env:
        num_envs = 6144
        num_actions = 18  # policy/PD only controls non-gripper joints (12 leg + 6 arm)
        num_torques = 18
        action_delay = 3
        num_gripper_joints = 2  # ARX-X5 gripper has 2 DOF (sliding mechanism with 2 claws)
        # Observation breakdown:
        # - body_orientation: 2
        # - base_ang_vel: 3
        # - dof_pos (without gripper): 18 (20 total - 2 gripper)
        # - dof_vel (without gripper): 18
        # - action_history: 12 (leg actions only, VBC low-level convention)
        # - foot_contacts: 4
        # - commands: 3
        # - ee_goal_cart: 3
        # - ee_goal_orient: 3
        num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3  # = 66
        # Privileged observation breakdown:
        # - mass_params: 5
        # - friction: 1
        # - motor_strength (legs only, hardcoded in code): 12
        num_priv = 5 + 1 + 12  # = 18 (code uses motor_strength[:, :12])
        history_len = 10
        num_observations = num_proprio * (history_len+1) + num_priv  # = 66*11 + 18 = 744
        num_privileged_obs = None
        send_timeouts = True
        episode_length_s = 10
        reorder_dofs = True
        teleop_mode = False
        record_video = False
        stand_by = False
        observe_gait_commands = False
        frequencies = 2

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.40]  # Start slightly higher to avoid immediate ground penetration on reset.
        leg_reset_ratio_range = [0.98, 1.02]
        arm_reset_noise_range = [-0.05, 0.05]
        # Go2 leg joints and X5 arm joints
        default_joint_angles = {
            # Go2 leg joints - adjusted for balance with arm
            # Hip: positive = inward, negative = outward
            'FL_hip_joint': 0.15,
            'FL_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,

            'RL_hip_joint': 0.15,
            'RL_thigh_joint': 0.8,
            'RL_calf_joint': -1.5,

            'FR_hip_joint': -0.15,
            'FR_thigh_joint': 0.8,
            'FR_calf_joint': -1.5,

            'RR_hip_joint': -0.15,
            'RR_thigh_joint': 0.8,
            'RR_calf_joint': -1.5,

            # X5 arm joints - natural forward-pointing folded position
            # x5_joint1: Z-axis rotation - 0.0 = arm points FORWARD, matching init IK target (yaw=0)
            # Using π (backward) caused 800 N*m startup torque shock due to ~π mismatch with forward IK target
            'x5_joint1': 0.0,      # Arm points forward - aligned with init_pos_start yaw=0
            'x5_joint2': 0.5,      # Lift arm slightly
            'x5_joint3': 1.5,      # Bend elbow to fold
            'x5_joint4': 0.0,      # Wrist - neutral
            'x5_joint5': 0.0,      # Wrist 2
            'x5_joint6': 0.0,      # Wrist 3
            'x5_joint7': 0.022,    # Gripper
            'x5_joint8': 0.022,    # Gripper
        }
        rand_yaw_range = np.pi/2
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1

    class control:
        stiffness = {'hip': 80, 'thigh': 80, 'calf': 80, 'x5': 5}  # [N*m/rad]
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0, 'x5': 0.5}  # [N*m*s/rad]
        arm_pos_stiffness = 80.0
        arm_pos_damping = 8.0

        adaptive_arm_gains = False
        # action_scale: [leg_scales (12)] + [arm_scales (6)] 
        # Total 18 scales for 18 actions (gripper joints 7,8 handled separately via force control)
        action_scale = [0.4, 0.45, 0.45] * 4 + [2.1, 0.6, 0.6, 0.4, 0.4, 0.4]
        decimation = 4
        torque_supervision = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2x5/urdf/go2_arx_x5.urdf'
        foot_name = "foot"  # Go2 foot links are named *_foot
        gripper_name = "ee_gripper_link"  # ARX-X5 gripper end-effector link
        # Note: Go2 has no "trunk" like B1, use "thigh" instead for penalization
        penalize_contacts_on = ["thigh", "calf"]  # Changed from ["thigh", "trunk", "calf"] for Go2
        terminate_after_contacts_on = []
        self_collisions = 0
        flip_visual_attachments = False
        collapse_fixed_joints = True
        fix_base_link = False

    class box:
        box_size = 0.1
        randomize_base_mass = True
        added_mass_range = [-0.001, 0.050]
        box_env_origins_x = 0
        box_env_origins_y_range = [0.1, 0.3]
        box_env_origins_z = box_size / 2 + 0.16

    class arm:
        base_offset = [0.0, 0.0, 0.08]  # x5_mount in URDF is at base frame z=0.08
        init_target_ee_base = [0.2, 0.0, 0.2]
        grasp_offset = 0.08
        ik_gain = 0.3
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)

    class domain_rand:
        observe_priv = True
        randomize_friction = True
        friction_range = [0.3, 3.0]
        randomize_base_mass = True
        added_mass_range = [0., 15.]
        randomize_base_com = True
        added_com_range_x = [-0.15, 0.15]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.15, 0.15]
        randomize_motor = True
        leg_motor_strength_range = [0.7, 1.3]
        arm_motor_strength_range = [0.7, 1.3]
        randomize_gripper_mass = True
        gripper_added_mass_range = [0, 0.1]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.3

    class rewards:
        reward_container_name = "maniploco_rewards"
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_ee_sigma = 1
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.4
        # Keep the locomotion height target consistent with Go2's nominal stance.
        # 0.55 is the B1 target and strongly biases Go2 toward an unrealistically tall/stiff posture.
        base_height_target = 0.35
        max_contact_force = 40.
        gait_vel_sigma = 0.5
        gait_force_sigma = 0.5
        kappa_gait_probs = 0.07
        feet_height_target = 0.08

        feet_aritime_allfeet = False
        feet_height_allfeet = False
        min_body_height = 0.15        # Minimum crouching height (~full knee bend)
        low_goal_height_thresh = 0.30 # Only trigger posture shaping when EE goal is low
        low_goal_hind_force_ratio_target = 0.30  # For low goals, hind legs should still carry a meaningful fraction of load

        class scales:
            # -------Gait control rewards ---------
            tracking_contacts_shaped_force = -2.0
            tracking_contacts_shaped_vel = -2.0
            feet_air_time = 2.0
            feet_height = 1.0

            # -------Tracking rewards ----------
            tracking_lin_vel_max = 3.0
            tracking_lin_vel_x_l1 = 0.
            tracking_lin_vel_x_exp = 0
            tracking_ang_vel = 0.5

            delta_torques = -1.0e-7/4.0
            work = 0
            energy_square = 0.0
            torques = -2.5e-5 
            stand_still = 0.0
            walking_dof = 1.8
            dof_default_pos = 0.0
            dof_error = 0.0 
            alive = 1.0
            lin_vel_z = -1.5
            roll = -2

            # common rewards
            ang_vel_xy = -0.2 
            dof_acc = -7.5e-7 
            collision = -10.
            action_rate = -0.015
            dof_pos_limits = -10.0
            delta_torques = -1.0e-7
            hip_pos = -0.3
            work = -0.003
            feet_jerk = -0.0002
            feet_drag = -0.08
            feet_contact_forces = -0.001
            height_adaptation = -2.0
            low_goal_front_leg_bend = 0.3
            low_goal_posture_asymmetry = 0.2
            low_goal_hind_leg_extension = 0.2
            low_goal_hind_support_force = 0.5
            feet_contact_standing = 0.0
            hind_feet_contact_standing = 0.0
            pitch_soft_limit_standing = -0.25
            orientation = 0.0
            orientation_walking = -5.0   # Heavily penalize pitching while walking to prevent diving
            orientation_standing = 0.0
            base_height = -2.5
            torques_walking = 0.0
            torques_standing = 0.0
            energy_square = 0.0
            energy_square_walking = 0.0
            energy_square_standing = 0.0
            base_height_walking = 0.0
            base_height_standing = 0.0
            penalty_lin_vel_y = 0.

        class arm_scales:
            arm_termination = None
            tracking_ee_sphere = 0.
            tracking_ee_world = 0.6
            tracking_ee_sphere_walking = 0.0
            tracking_ee_sphere_standing = 0.0
            tracking_ee_cart = None
            arm_orientation = None
            arm_energy_abs_sum = None
            tracking_ee_orn = 0.
            tracking_ee_orn_ry = None

    class viewer:
        pos = [-20, 0, 20]
        lookat = [0, 0, -2]

    class termination:
        r_threshold = 0.6
        p_threshold = 0.5
        z_threshold = 0.15

    class terrain:
        mesh_type = 'trimesh'
        hf2mesh_method = "fast"
        max_error = 0.1
        horizontal_scale = 0.05
        vertical_scale = 0.005
        border_size = 25
        height = [0.00, 0.1]
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = False

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20

        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.,
                        "gaps": 0., 
                        "rough flat": 1.0,
                        "pit": 0.0,
                        "wall": 0.0}
        terrain_proportions = list(terrain_dict.values())
        slope_treshold = None
        origin_zero_z = False


class Go2X5RoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        continue_from_last_std = True
        init_std = [[0.8, 1.0, 1.0] * 4 + [1.0] * 6]
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = 'elu'
        output_tanh = False

        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]

        priv_encoder_dims = [64, 20]

        num_leg_actions = 12
        num_arm_actions = 6

        adaptive_arm_gains = Go2X5RoughCfg.control.adaptive_arm_gains
        adaptive_arm_gains_scale = 10.0
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.002
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2e-4
        schedule = 'fixed'
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.
        min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3]

        mixing_schedule = [1.0, 0, 3000]
        torque_supervision = Go2X5RoughCfg.control.torque_supervision
        torque_supervision_schedule = [0.0, 1000, 1000]
        adaptive_arm_gains = Go2X5RoughCfg.control.adaptive_arm_gains
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000]

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 45000
        save_interval = 200
        experiment_name = 'go2x5'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
