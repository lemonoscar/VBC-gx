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
        traj_time = [3.0, 5.0]
        hold_time = [1.5, 3.0]
        # Reject trajectories that enter the near-body/head region.  The old
        # x upper bound (0.1) predated the front-workspace retarget and could
        # never reject the current local x range, whose minimum is 0.215.
        collision_upper_limits = [0.24, 0.2, 0.05]
        collision_lower_limits = [-0.8, -0.2, -0.7]
        underground_limit = -0.6  # local cartesian z; keeps sampled EE goals from spending too much time below the terrain.
        max_nominal_reach_radius = robot_spec.EE_GOAL_MAX_NOMINAL_REACH_RADIUS
        num_collision_check_samples = 10
        command_mode = 'cart'
        center_mode = 'terrain_invariant'
        arm_induced_pitch = 0.38

        class sphere_center:
            x_offset = robot_spec.EE_GOAL_CENTER_OFFSET[0]  # Relative to base root xy/yaw.
            y_offset = robot_spec.EE_GOAL_CENTER_OFFSET[1]
            z_invariant_offset = robot_spec.EE_GOAL_CENTER_OFFSET[2]  # Relative to terrain.

        class ranges:
            # Cartesian targets relative to the terrain-invariant nominal arm-base center.
            # These map to root-forward x=[0.30, 0.65] m, y=[-0.225, 0.225] m,
            # and terrain z=[0.05, 0.45] m.
            init_pos_start = robot_spec.EE_GOAL_INIT_START_LOCAL
            init_pos_end = robot_spec.EE_GOAL_INIT_END_LOCAL
            pos_x = robot_spec.EE_GOAL_LOCAL_RANGES[0]
            pos_y_cart = robot_spec.EE_GOAL_LOCAL_RANGES[1]
            pos_z = robot_spec.EE_GOAL_LOCAL_RANGES[2]

            # Legacy spherical ranges remain defined for B1Z1-compatible helpers and diagnostics.
            pos_l = [0.20, 0.56]
            pos_p = [0.15, 1.05]
            pos_y = [-0.65, 0.65]
            
            delta_orn_r = robot_spec.EE_ORIENTATION_DELTA_RANGES[0]
            delta_orn_p = robot_spec.EE_ORIENTATION_DELTA_RANGES[1]
            delta_orn_y = robot_spec.EE_ORIENTATION_DELTA_RANGES[2]
            final_tracking_ee_reward = 0.55

        sphere_error_scale = [1, 1, 1]
        orn_error_scale = [1, 1, 1]
        orientation_nominal_rpy = robot_spec.EE_ORIENTATION_NOMINAL_RPY
        orientation_follow_target_yaw = True
        orientation_in_observation = True

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
        # Walk These Ways holds velocity commands for 10 s.  Go2-X5 episodes
        # are also 10 s, so this gives the policy one coherent locomotion goal
        # per episode instead of changing direction three times while it is
        # still acquiring a gait.
        resampling_time = 10.
        # Keep explicit, easily learned command populations.  The previous
        # sampler devoted only a small fraction of episodes to exact straight
        # walking because almost every non-standing sample also contained yaw.
        standing_probability = 0.20
        straight_line_probability = 0.35
        turn_in_place_probability = 0.10
        # The v3 remote smoke converged to an all-feet-contact policy because
        # 0.10 m/s commands were still well rewarded at zero velocity.  Keep
        # 0.10 m/s in the general distribution, but make the dedicated
        # locomotion population large enough to require a real step.
        straight_line_min_abs_vx = 0.15
        turn_in_place_min_abs_yaw = 0.10

        lin_vel_x_schedule = [0, 0]
        ang_vel_yaw_schedule = [0, 0]
        tracking_ang_vel_yaw_schedule = [0, 0]

        ang_vel_yaw_clip = 0.05
        lin_vel_x_clip = 0.05
        lin_vel_y_clip = 0.05

        class ranges:
            lin_vel_x = [-0.30, 0.30]
            # A small lateral range preserves useful body mobility without
            # making the tabletop approach task primarily sideways.
            lin_vel_y = [-0.10, 0.10]
            ang_vel_yaw = [-0.25, 0.25]

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        # Keep stochastic exploration inside the deployed tanh action contract.
        clip_actions = 1.0

    class env:
        num_envs = 4096
        env_spacing = 3.0
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
        num_observations = robot_spec.observation_dim(False)
        num_privileged_obs = None
        send_timeouts = True
        episode_length_s = 10
        reorder_dofs = True
        teleop_mode = False
        record_video = False
        stand_by = False
        # No named gait is prescribed. The policy learns any stable locomotion
        # pattern that satisfies velocity, contact, and drag objectives.
        observe_gait_commands = False
        policy_output_tanh = True
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
        gripper_pos_stiffness = robot_spec.GRIPPER_POS_STIFFNESS
        gripper_pos_damping = robot_spec.GRIPPER_POS_DAMPING

        adaptive_arm_gains = False
        # action_scale: leg scales only; arm joints are controlled by IK position targets.
        action_scale = robot_spec.LOW_ACTION_SCALE
        decimation = 4
        torque_supervision = False

    class asset( LeggedRobotCfg.asset ):
        file = robot_spec.LOW_LEVEL_ASSET_FILE
        base_body_name = "base"
        foot_name = "foot"  # Go2 foot links are named *_foot
        gripper_name = robot_spec.EE_BODY_NAME  # End-effector frame from the Go2-X5-lab URDF
        # Note: Go2 has no "trunk" like B1, use "thigh" instead for penalization
        # Cover every non-foot collision body.  In particular, arm/finger and
        # head contacts must not be hidden behind a zero collision metric.
        penalize_contacts_on = ["base", "Head", "hip", "thigh", "calf", "arm_link"]
        terminate_after_contacts_on = []
        self_collisions = 0
        flip_visual_attachments = False
        collapse_fixed_joints = True
        replace_cylinder_with_capsule = False
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
        ik_gain = robot_spec.ARM_IK_GAIN
        ik_orientation_weight = robot_spec.ARM_IK_ORIENTATION_WEIGHT
        track_ee_orientation = True
        target_mode = robot_spec.ARM_TARGET_MODE
        target_max_step = robot_spec.ARM_TARGET_MAX_STEP
        gripper_hold_mode = robot_spec.LOW_LEVEL_GRIPPER_HOLD_MODE
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)

    class domain_rand:
        observe_priv = True
        # Establish the deterministic flat-terrain baseline before adding a
        # separately versioned robustness phase.
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
        # Keep Walk These Ways' squared-error exponential kernel, but scale its
        # width to this task's much smaller (+/-0.30 m/s) command envelope.
        # With the paper's 0.25 width, standing still at a 0.10 m/s command
        # already receives 96.1% of the maximum reward; 0.05 lowers that to
        # 81.9% and gives the policy a meaningful low-speed tracking signal.
        tracking_sigma = 0.05
        # Metres in exp(-2 * L1_error / sigma).
        tracking_ee_sigma = 0.15
        # Radians in exp(-quaternion_angle / sigma).
        tracking_ee_orientation_sigma = 0.35
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

        feet_air_time_all_feet = True
        feet_height_allfeet = True
        # Phase-free, completed-step shaping.  A short landing is neutral
        # rather than negative, while a real swing that clears the foot radius
        # and lands receives a small bounded bonus.
        feet_air_time_target = 0.10
        feet_air_time_max_bonus = 0.25
        feet_clearance_floor = robot_spec.FOOT_COLLISION_RADIUS
        feet_clearance_target = 0.05
        feet_clearance_landing_bonus = 0.20
        foot_lateral_min = 0.06
        min_stance_feet = 3.0
        # Retained for diagnostics; the simplified active reward does not gate
        # EE tracking on a prescribed contact pattern.
        safety_min_feet_contacts_standing = 3.0
        safety_min_feet_contacts_walking = 2.0
        safety_roll_soft = 0.25
        safety_roll_hard = 0.55
        safety_pitch_soft = 0.25
        safety_pitch_hard = 0.55
        safety_base_height_min = 0.26
        safety_base_height_floor = 0.20
        dof_error_deadzone = 0.12
        leg_action_deadzone = 0.20
        min_body_height = 0.22        # Still 4 cm above the 0.18 m termination threshold.
        height_adaptation_goal_z_low = 0.05
        height_adaptation_goal_z_high = 0.30
        max_forward_body_pitch = 0.25 # Positive pitch lowers the Go2 front/arm mount.
        low_goal_height_thresh = 0.35 # Only trigger posture shaping when EE goal is low
        low_goal_hind_force_ratio_target = 0.30  # For low goals, hind legs should still carry a meaningful fraction of load
        pitch_soft_limit = 0.35

        class scales:
            # No gait clock/contact schedule is active.
            tracking_contacts_shaped_force = 0.0
            tracking_contacts_shaped_vel = 0.0
            # Phase-free Walk These Ways/legged-gym stepping incentive.  It
            # rewards completed swing-and-land events for all four feet but
            # does not prescribe trot, walk, or a fixed phase relationship.
            feet_air_time = 2.0
            feet_height = 0.0

            # -------Tracking rewards ----------
            tracking_lin_vel_max = 0.0
            tracking_lin_vel_x_l1 = 0.
            tracking_lin_vel_x_exp = 0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            tracking_ang_vel_yaw_exp = 0.0

            torques = -2.5e-5
            # Standing targets may still require coordinated crouching.
            stand_still = 0.0
            walking_dof = 0.0
            dof_default_pos = 0.0
            dof_error = 0.0
            alive = 1.0
            termination = -100.0
            lin_vel_z = -1.0
            roll = -2.0

            # Minimal task/stability set: no prescribed gait and no overlapping
            # acceleration/work/jerk curricula.
            ang_vel_xy = 0.0
            dof_acc = 0.0
            # Use the Walk These Ways order of magnitude.  The local collision
            # metric is already force-normalized and can reach O(10) for one
            # bad body contact; -10 suppressed early stepping exploration.
            collision = -1.0
            action_rate = -0.01
            dof_pos_limits = -10.0
            delta_torques = 0.0
            hip_pos = 0.0
            work = 0.0
            feet_jerk = 0.0
            feet_drag = -0.20
            foot_lateral_spacing = 0.0
            feet_contact_forces = 0.0
            # EE-conditioned body coordination belongs to the whole-body
            # advantage channel below. Keeping it here bypasses the PPO mixing
            # delay and produced a static crouch before locomotion was learned.
            height_adaptation = 0.0
            pitch_adaptation = 0.0
            low_goal_front_leg_bend = 0.0
            low_goal_posture_asymmetry = 0.0
            low_goal_hind_leg_extension = 0.0
            low_goal_hind_support_force = 0.0
            # A stopped robot should not learn the old three-leg stance with
            # one rear foot permanently airborne.
            feet_contact_standing = -0.5
            hind_feet_contact_standing = 0.0
            foot_support_standing = 0.0
            pitch_soft_limit_standing = 0.0
            orientation = 0.0
            orientation_walking = 0.0
            orientation_standing = 0.0
            # The target-dependent height term replaces a conflicting fixed height.
            base_height = 0.0
            stability_safety = 0.0
            dof_error_deadzone = 0.0
            # Do not prescribe a gait, but prevent an unbounded actor mean from
            # buying velocity with saturated PD targets. Actions inside +/-0.2
            # remain free; the quadratic tail only constrains large offsets.
            leg_action_l2_deadzone = -0.01
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
            tracking_ee_world = 2.0
            tracking_ee_world_stable = 0.0
            tracking_ee_sphere_walking = 0.0
            tracking_ee_sphere_standing = 0.0
            tracking_ee_cart = None
            arm_orientation = None
            arm_energy_abs_sum = None
            tracking_ee_orn = 0.6
            tracking_ee_orn_ry = None
            # These terms still train the 12D leg policy, but only through the
            # delayed whole-body advantage mixing schedule.
            height_adaptation = -3.0
            pitch_adaptation = -1.0

    class viewer:
        pos = [-20, 0, 20]
        lookat = [0, 0, -2]

    class termination:
        r_threshold = 0.8
        p_threshold = 0.8
        z_threshold = 0.18

    class terrain:
        # Go2-X5 starts on a native PhysX plane.  Do not construct a Terrain
        # height map for the flat-training contract: its one-sided trimesh
        # allowed the initially lower front feet to remain below the surface.
        mesh_type = 'plane'
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
        # A single static task distribution is easier to audit and avoids
        # reward/range discontinuities. Curriculum machinery remains available
        # for B1-Z1 compatibility but is intentionally inactive for Go2-X5.
        enabled = False
        profile_name = "go2x5_flat_tabletop_6d_walk_v4"
        metric_window = 200
        log_stage = True
        save_stage_metadata = True
        current_stage_index = 0
        current_stage_name = "static_full_task"
        stages = []


class Go2X5RoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        continue_from_last_std = True
        # Match the proven B1-Z1/Walk These Ways exploration regime. The v3
        # remote smoke used its eventual minimum std as the initial std and
        # collapsed to static joint offsets with all four feet in contact.
        init_std = [[0.8, 1.0, 1.0] * 4]
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = 'elu'
        output_tanh = True

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
        min_policy_std = [[0.15, 0.25, 0.25] * 4]

        # Keep the first 3000 PPO updates locomotion-only, then blend the
        # EE/body-coordination advantage over the next 3000 updates. The final
        # objective is unchanged after iteration 6000.
        mixing_schedule = [1.0, 3000, 3000]
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
