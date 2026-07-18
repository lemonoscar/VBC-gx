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
from . import go2x5_robot_spec as robot_spec

class Go2X5RoughCfg( LeggedRobotCfg ):
    """Configuration for Go2 quadruped with ARX-X5 6-DOF manipulator arm"""
    
    class goal_ee:
        num_commands = 3
        traj_time = [2.5, 4.0]
        hold_time = [1.0, 2.5]
        collision_upper_limits = [0.1, 0.2, -0.05]
        collision_lower_limits = [-0.8, -0.2, -0.7]
        underground_limit = -0.6  # local cartesian z; keeps sampled EE goals from spending too much time below the terrain.
        num_collision_check_samples = 10
        command_mode = 'cart'
        center_mode = 'terrain_invariant'
        arm_induced_pitch = 0.38

        class sphere_center:
            x_offset = robot_spec.ARM_BASE_OFFSET[0]  # Relative to base root xy/yaw; z remains terrain-invariant.
            y_offset = 0    # Relative to base
            z_invariant_offset = robot_spec.BASE_INIT_HEIGHT + robot_spec.ARM_BASE_OFFSET[2]  # Relative to terrain.

        class ranges:
            # Cartesian targets relative to the terrain-invariant nominal arm-base center. This wide
            # box is a task target region, not an IK-feasible-only set; some targets intentionally
            # require body/arm coordination.
            init_pos_start = [0.174, 0.0, 0.318]  # Default arm home from the IK reachability scan.
            init_pos_end = [0.30, 0.0, 0.16]
            pos_x = [0.05, 0.60]
            pos_y_cart = [-0.30, 0.30]
            pos_z = [-0.40, 0.42]

            # Legacy spherical ranges remain defined for B1Z1-compatible helpers and diagnostics.
            pos_l = [0.20, 0.56]
            pos_p = [0.15, 1.05]
            pos_y = [-0.65, 0.65]
            
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
        curriculum = False
        num_commands = 3
        resampling_time = 3.

        lin_vel_x_schedule = [0, 0]
        ang_vel_yaw_schedule = [0, 0]
        tracking_ang_vel_yaw_schedule = [0, 0]

        ang_vel_yaw_clip = 0.05
        lin_vel_x_clip = 0.05

        class ranges:
            lin_vel_x = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

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
        num_envs = 4096
        num_actions = robot_spec.ACTION_DIM  # low-level policy controls legs only; arm is driven by IK position targets
        num_torques = robot_spec.NUM_TORQUES
        action_delay = 0  # deterministic parity first; reintroduce measured delay through a shared contract
        num_gripper_joints = robot_spec.NUM_PHYSICAL_GRIPPER_DOFS  # Isaac Gym loads the mirrored finger sliders as physical DOFs.
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
        num_proprio = robot_spec.PROPRIO_DIM_WITHOUT_GAIT
        # Privileged observation breakdown:
        # - mass_params: 5
        # - friction: 1
        # - motor_strength (legs only, hardcoded in code): 12
        num_priv = robot_spec.PRIV_DIM
        history_len = robot_spec.HISTORY_LEN
        num_observations = robot_spec.observation_dim(True)
        num_privileged_obs = None
        send_timeouts = True
        episode_length_s = 10
        reorder_dofs = True
        teleop_mode = False
        record_video = False
        stand_by = False
        # Go2-X5 deployment always consumes the five gait fields.  Keep this
        # enabled in the task itself so a missing CLI flag cannot silently
        # produce an incompatible checkpoint.
        observe_gait_commands = True
        require_training_metadata = True
        frequencies = 2

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, robot_spec.BASE_INIT_HEIGHT]  # Go2-X5 canonical screening stance height.
        leg_reset_ratio_range = [1.0, 1.0]
        arm_reset_noise_range = [0.0, 0.0]
        # Go2 leg joints and X5 arm joints
        default_joint_angles = robot_spec.DEFAULT_JOINT_ANGLES.copy()
        rand_yaw_range = 0.0
        origin_perturb_range = 0.0
        init_vel_perturb_range = 0.0

    class control:
        stiffness = {'hip': robot_spec.LEG_STIFFNESS, 'thigh': robot_spec.LEG_STIFFNESS, 'calf': robot_spec.LEG_STIFFNESS, 'arm': 5}  # [N*m/rad]
        damping = {'hip': robot_spec.LEG_DAMPING, 'thigh': robot_spec.LEG_DAMPING, 'calf': robot_spec.LEG_DAMPING, 'arm': 0.5}  # [N*m*s/rad]
        arm_pos_stiffness = robot_spec.ARM_POS_STIFFNESS
        arm_pos_damping = robot_spec.ARM_POS_DAMPING

        adaptive_arm_gains = False
        # action_scale: leg scales only; arm joints are controlled by IK position targets.
        action_scale = [0.10, 0.16, 0.16] * 4
        decimation = 4
        torque_supervision = False

    class asset( LeggedRobotCfg.asset ):
        file = robot_spec.LOW_LEVEL_ASSET_FILE
        base_body_name = "base"
        foot_name = "foot"  # Go2 foot links are named *_foot
        gripper_name = robot_spec.EE_BODY_NAME  # End-effector frame from the Go2-X5-lab URDF
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
        base_offset = robot_spec.ARM_BASE_OFFSET  # arm_base_joint origin in the Go2-X5-lab URDF
        init_target_ee_base = [0.30, 0.0, 0.20]
        grasp_offset = 0.08
        # Go2-X5 zero-action checks show that simultaneous position+orientation
        # IK can roll the base over. Train the low-level controller with
        # position-only EE IK first, then reintroduce orientation later.
        ik_gain = 0.25
        track_ee_orientation = False
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)

    class domain_rand:
        observe_priv = True
        # Stability-first low-level training: first learn to survive and track
        # small commands before reintroducing B1Z1-style randomization.
        randomize_friction = False
        friction_range = [1.0, 1.0]
        randomize_base_mass = False
        added_mass_range = [0.0, 0.0]
        randomize_base_com = False
        added_com_range_x = [0.0, 0.0]
        added_com_range_y = [0.0, 0.0]
        added_com_range_z = [0.0, 0.0]
        randomize_motor = False
        leg_motor_strength_range = [1.0, 1.0]
        arm_motor_strength_range = [1.0, 1.0]
        randomize_gripper_mass = False
        gripper_added_mass_range = [0.0, 0.0]
        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 0.0

    class rewards:
        reward_container_name = "maniploco_rewards"
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_ee_sigma = 1.0
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.4
        # Keep the locomotion height target consistent with Go2's nominal stance.
        # 0.55 is the B1 target and strongly biases Go2 toward an unrealistically tall/stiff posture.
        base_height_target = robot_spec.BASE_HEIGHT_TARGET
        max_contact_force = 200.
        collision_force_threshold = 5.0
        collision_soft_clip = 50.0
        gait_vel_sigma = 0.5
        gait_force_sigma = 0.5
        kappa_gait_probs = 0.07
        feet_height_target = 0.12

        feet_aritime_allfeet = True
        feet_height_allfeet = True
        feet_air_time_target = 0.25  # 2 Hz gait with a 50% swing phase.
        foot_lateral_min = 0.06
        min_stance_feet = 3.0
        safety_min_feet_contacts = 3.0
        safety_roll_soft = 0.25
        safety_roll_hard = 0.55
        safety_pitch_soft = 0.25
        safety_pitch_hard = 0.55
        safety_base_height_min = 0.26
        safety_base_height_floor = 0.20
        dof_error_deadzone = 0.12
        leg_action_deadzone = 0.20
        min_body_height = 0.15        # Minimum crouching height (~full knee bend)
        low_goal_height_thresh = 0.35 # Only trigger posture shaping when EE goal is low
        low_goal_hind_force_ratio_target = 0.30  # For low goals, hind legs should still carry a meaningful fraction of load
        pitch_soft_limit = 0.35

        class scales:
            # -------Gait control rewards ---------
            tracking_contacts_shaped_force = 0.0
            tracking_contacts_shaped_vel = 0.0
            feet_air_time = 0.0
            feet_height = 0.0

            # -------Tracking rewards ----------
            tracking_lin_vel_max = 0.0
            tracking_lin_vel_x_l1 = 0.
            tracking_lin_vel_x_exp = 0
            tracking_ang_vel = 0.0

            torques = -2.5e-5
            stand_still = 3.0
            walking_dof = 0.0
            dof_default_pos = 0.0
            dof_error = 0.0
            alive = 3.0
            termination = -100.0
            lin_vel_z = -5.0
            roll = -8.0

            # common rewards
            ang_vel_xy = -2.0
            dof_acc = -1.0e-6
            collision = -8.0
            action_rate = -0.03
            dof_pos_limits = -10.0
            delta_torques = -1.0e-7
            hip_pos = -0.5
            work = -0.003
            feet_jerk = -0.0002
            feet_drag = -0.25
            foot_lateral_spacing = -2.0
            feet_contact_forces = -0.001
            height_adaptation = 0.0
            low_goal_front_leg_bend = 0.0
            low_goal_posture_asymmetry = 0.0
            low_goal_hind_leg_extension = 0.0
            low_goal_hind_support_force = 0.0
            feet_contact_standing = -2.0
            hind_feet_contact_standing = -2.5
            foot_support_standing = -2.0
            pitch_soft_limit_standing = -2.0
            orientation = -3.0
            orientation_walking = 0.0
            orientation_standing = 0.0
            base_height = -6.0
            stability_safety = 1.0
            dof_error_deadzone = -1.0
            leg_action_l2_deadzone = -0.1
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
            tracking_ee_world = 0.0
            tracking_ee_world_stable = 0.2
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
        r_threshold = 0.8
        p_threshold = 0.8
        z_threshold = 0.18

    class terrain:
        mesh_type = 'trimesh'
        hf2mesh_method = "fast"
        max_error = 0.1
        horizontal_scale = 0.05
        vertical_scale = 0.005
        border_size = 25
        # The stability/reach curriculum is intentionally flat. Runtime probes
        # showed that even the zero policy makes calf contact on the previous
        # 0--2 cm rough-flat mesh, making the S0 collision gate impossible to
        # satisfy. Rough terrain belongs in a later, explicitly versioned
        # robustness curriculum after flat closed-loop parity is established.
        height = [0.00, 0.00]
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = False

        all_vertical = False
        no_flat = False
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 0
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

    class auto_curriculum:
        enabled = True
        # v4 invalidates checkpoints trained with stale foot velocity copies,
        # the S3 default-pose exploit, and the old 0.33 m body-height contract.
        profile_name = "go2x5_stable_reach_curriculum_v4_live_foot_gait_h032"
        metric_window = 200
        log_stage = True
        save_stage_metadata = True
        current_stage_index = 0
        current_stage_name = "S0_safe_small_reach"
        stages = [
            {
                "name": "S0_safe_small_reach",
                "min_iterations": 1000,
                "action_scale": [0.10, 0.16, 0.16] * 4,
                "lin_vel_x_range": [0.0, 0.0],
                "ang_vel_yaw_range": [0.0, 0.0],
                "goal_pos_x": [0.18, 0.35],
                "goal_pos_y_cart": [-0.10, 0.10],
                "goal_pos_z": [0.05, 0.25],
                "traj_time": [3.0, 5.0],
                "hold_time": [1.0, 3.0],
                "collision_scale": -8.0,
                "orientation_scale": -3.0,
                "foot_lateral_spacing_scale": -2.0,
                "feet_contact_standing_scale": -2.0,
                "hind_feet_contact_standing_scale": -2.5,
                "foot_support_standing_scale": -2.0,
                "dof_error_deadzone_scale": -1.0,
                "leg_action_l2_deadzone_scale": -0.1,
                "stability_safety_scale": 1.0,
                "ee_tracking_stable_weight": 0.2,
                "tracking_lin_vel_max_scale": 0.0,
                "tracking_contacts_shaped_force_scale": 0.0,
                "tracking_contacts_shaped_vel_scale": 0.0,
                "walking_dof_scale": 0.0,
                "advance": {
                    "Train/mean_episode_length": [">", 450.0],
                    "Episode/reset_roll": ["<", 0.05],
                    "Episode/reset_z": ["<", 0.02],
                    "Episode_metric/metric_collision": ["<", 2.0],
                    "Episode_metric/metric_tracking_ee_world_stable": ["<", 0.50],
                },
            },
            {
                "name": "S1_mid_reach_compensation",
                "min_iterations": 3000,
                "action_scale": [0.14, 0.22, 0.22] * 4,
                "lin_vel_x_range": [0.0, 0.0],
                "ang_vel_yaw_range": [0.0, 0.0],
                "goal_pos_x": [0.12, 0.45],
                "goal_pos_y_cart": [-0.18, 0.18],
                "goal_pos_z": [-0.15, 0.32],
                "traj_time": [2.5, 4.0],
                "hold_time": [1.0, 2.5],
                "collision_scale": -10.0,
                "orientation_scale": -3.0,
                "foot_lateral_spacing_scale": -2.0,
                "feet_contact_standing_scale": -2.0,
                "hind_feet_contact_standing_scale": -2.5,
                "foot_support_standing_scale": -2.0,
                "dof_error_deadzone_scale": -0.8,
                "leg_action_l2_deadzone_scale": -0.08,
                "stability_safety_scale": 1.0,
                "ee_tracking_stable_weight": 0.4,
                "advance": {
                    "Train/mean_episode_length": [">", 430.0],
                    "Episode/reset_roll": ["<", 0.08],
                    "Episode/reset_z": ["<", 0.02],
                    "Episode_metric/metric_collision": ["<", 2.5],
                    "Episode_metric/metric_tracking_ee_world_stable": ["<", 0.60],
                },
            },
            {
                "name": "S2_full_reach_compensation",
                "min_iterations": 7000,
                "action_scale": [0.20, 0.30, 0.30] * 4,
                "lin_vel_x_range": [0.0, 0.0],
                "ang_vel_yaw_range": [0.0, 0.0],
                "goal_pos_x": [0.05, 0.60],
                "goal_pos_y_cart": [-0.30, 0.30],
                "goal_pos_z": [-0.40, 0.42],
                "traj_time": [2.0, 4.0],
                "hold_time": [1.0, 2.5],
                "collision_scale": -12.0,
                "orientation_scale": -3.0,
                "foot_lateral_spacing_scale": -1.5,
                "feet_contact_standing_scale": -1.5,
                "hind_feet_contact_standing_scale": -2.0,
                "foot_support_standing_scale": -1.5,
                "dof_error_deadzone_scale": -0.5,
                "leg_action_l2_deadzone_scale": -0.05,
                "stability_safety_scale": 1.0,
                "ee_tracking_stable_weight": 0.8,
                "advance": {
                    "Train/mean_episode_length": [">", 430.0],
                    "Episode/reset_roll": ["<", 0.08],
                    "Episode/reset_z": ["<", 0.02],
                    "Episode_metric/metric_collision": ["<", 3.0],
                    "Episode_metric/metric_tracking_ee_world_stable": ["<", 0.70],
                },
            },
            {
                "name": "S3_small_locomotion_reach",
                "min_iterations": 11000,
                "action_scale": robot_spec.LOW_ACTION_SCALE,
                "lin_vel_x_range": [-0.2, 0.2],
                "ang_vel_yaw_range": [-0.3, 0.3],
                "goal_pos_x": [0.05, 0.60],
                "goal_pos_y_cart": [-0.30, 0.30],
                "goal_pos_z": [-0.40, 0.42],
                "traj_time": [2.0, 4.0],
                "hold_time": [1.0, 2.5],
                "collision_scale": -12.0,
                "orientation_scale": -3.0,
                "foot_lateral_spacing_scale": -1.5,
                "feet_contact_standing_scale": -1.0,
                "hind_feet_contact_standing_scale": -1.5,
                "foot_support_standing_scale": -1.0,
                "dof_error_deadzone_scale": -0.3,
                "leg_action_l2_deadzone_scale": -0.03,
                "stability_safety_scale": 1.0,
                "ee_tracking_stable_weight": 0.8,
                "tracking_lin_vel_max_scale": 1.0,
                "tracking_ang_vel_scale": 0.25,
                # The raw shaped-contact rewards are zero at the target and
                # negative on error, so their coefficients must be positive.
                # In the observed v3 no-step state, the roughly -0.47 raw
                # contact error must cancel its roughly +0.46 velocity score;
                # clearance and drag then make that behavior net-negative.
                "tracking_contacts_shaped_force_scale": 1.0,
                "tracking_contacts_shaped_vel_scale": 0.5,
                # Rewarding the default pose while walking produced +44/s in
                # the v3 run and suppressed the gait amplitude. Keep the term
                # for B1 compatibility, but disable it for Go2-X5 S3.
                "walking_dof_scale": 0.0,
                "feet_air_time_scale": 0.5,
                "feet_height_scale": 1.0,
                "advance": {},
            },
        ]


class Go2X5RoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        continue_from_last_std = True
        init_std = [[0.08, 0.10, 0.10] * 4]
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = 'elu'
        output_tanh = False

        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]

        priv_encoder_dims = [64, 20]

        num_leg_actions = 12
        num_arm_actions = 0

        adaptive_arm_gains = Go2X5RoughCfg.control.adaptive_arm_gains
        adaptive_arm_gains_scale = 10.0
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2e-4
        schedule = 'fixed'
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.
        min_policy_std = [[0.04, 0.05, 0.05] * 4]

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
