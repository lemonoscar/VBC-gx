from .go2x5_config import Go2X5RoughCfg, Go2X5RoughCfgPPO


class Go2X5FtLiftCfg(Go2X5RoughCfg):
    """Fine-tune config on top of go2x5_v9 for stable locomotion + stronger post-grasp lift transfer."""

    class goal_ee(Go2X5RoughCfg.goal_ee):
        class ranges(Go2X5RoughCfg.goal_ee.ranges):
            # More low-goal samples for pick/lift behavior, keep lateral/yaw diversity
            init_pos_start = [0.24, 0.10, 0.0]
            init_pos_end = [0.38, -0.10, 0.0]
            pos_l = [0.24, 0.50]
            pos_p = [-0.72, 0.95]
            pos_y = [-1.1, 1.1]

    class commands(Go2X5RoughCfg.commands):
        # Keep command space close to high-level deployment while avoiding extreme spikes
        lin_vel_x_clip = 0.18
        ang_vel_yaw_clip = 0.35

        class ranges(Go2X5RoughCfg.commands.ranges):
            lin_vel_x = [-0.50, 0.50]
            ang_vel_yaw = [-0.70, 0.70]

    class domain_rand(Go2X5RoughCfg.domain_rand):
        # Robustness-first randomization (paper-aligned), but milder than full base config
        randomize_friction = True
        friction_range = [0.5, 2.0]
        randomize_base_mass = True
        added_mass_range = [0.0, 8.0]
        randomize_base_com = True
        added_com_range_x = [-0.08, 0.08]
        added_com_range_y = [-0.08, 0.08]
        added_com_range_z = [-0.08, 0.08]
        randomize_motor = True
        leg_motor_strength_range = [0.85, 1.15]
        arm_motor_strength_range = [0.85, 1.15]
        randomize_gripper_mass = True
        gripper_added_mass_range = [0.0, 0.06]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.25

    class rewards(Go2X5RoughCfg.rewards):
        class scales(Go2X5RoughCfg.rewards.scales):
            # Encourage anti-fall support while preserving motion tracking capacity
            base_height = -2.0
            stand_still = 1.3
            walking_dof = 1.4
            feet_contact_standing = -6.0
            hind_feet_contact_standing = -2.5
            low_goal_front_leg_bend = 1.8
            low_goal_posture_asymmetry = 1.2
            low_goal_hind_leg_extension = 1.0
            low_goal_hind_support_force = 2.4

        class arm_scales(Go2X5RoughCfg.rewards.arm_scales):
            # Keep EE tracking strong enough to benefit downstream pick/lift
            tracking_ee_world = 0.9

    class termination(Go2X5RoughCfg.termination):
        # Early-stop true falls, but avoid over-penalizing recoverable transient tilt
        r_threshold = 0.62
        p_threshold = 0.62
        z_threshold = 0.16


class Go2X5FtLiftCfgPPO(Go2X5RoughCfgPPO):
    class runner(Go2X5RoughCfgPPO.runner):
        experiment_name = 'go2x5_ftlift'
        max_iterations = 8000

    class algorithm(Go2X5RoughCfgPPO.algorithm):
        # Moderate LR for stable adaptation without freezing progress
        learning_rate = 1e-4
