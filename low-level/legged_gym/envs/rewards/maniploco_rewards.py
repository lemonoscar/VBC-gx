import torch
from isaacgym.torch_utils import *

class ManipLoco_rewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env
    # -------------Z1: Reward functions----------------

    def _reward_tracking_ee_sphere(self):
        ee_pos_local = quat_rotate_inverse(self.env.base_yaw_quat, self.env.ee_pos - self.env.get_ee_goal_spherical_center())
        ee_pos_error = torch.sum(torch.abs(cart2sphere(ee_pos_local) - self.env.curr_ee_goal_sphere) * self.env.sphere_error_scale, dim=1)
        return torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma), ee_pos_error

    def _reward_tracking_ee_world(self):
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - self.env.curr_ee_goal_cart_world), dim=1)
        rew = torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma * 2)
        return rew, ee_pos_error

    def _terrain_height(self):
        measured_heights = getattr(self.env, "measured_heights", None)
        if torch.is_tensor(measured_heights) and measured_heights.ndim == 2:
            return torch.mean(measured_heights, dim=1)
        if hasattr(self.env, "env_origins"):
            return self.env.env_origins[:, 2]
        return torch.zeros_like(self.env.root_states[:, 2])

    def _body_stability_safety(self):
        body_rpy = self.env._get_body_orientation()
        roll_abs = torch.abs(body_rpy[:, 0])
        pitch_abs = torch.abs(body_rpy[:, 1])
        base_height = self._base_height_relative_to_terrain()

        def linear_margin(value, start, end):
            return torch.clamp((end - value) / max(end - start, 1e-6), min=0.0, max=1.0)

        roll_safety = linear_margin(
            roll_abs,
            getattr(self.env.cfg.rewards, "safety_roll_soft", 0.25),
            getattr(self.env.cfg.rewards, "safety_roll_hard", 0.55),
        )
        pitch_safety = linear_margin(
            pitch_abs,
            getattr(self.env.cfg.rewards, "safety_pitch_soft", 0.25),
            getattr(self.env.cfg.rewards, "safety_pitch_hard", 0.55),
        )
        height_safety = torch.clamp(
            (base_height - getattr(self.env.cfg.rewards, "safety_base_height_floor", 0.20))
            / max(getattr(self.env.cfg.rewards, "safety_base_height_min", 0.26)
                  - getattr(self.env.cfg.rewards, "safety_base_height_floor", 0.20), 1e-6),
            min=0.0,
            max=1.0,
        )

        return roll_safety * pitch_safety * height_safety

    def _support_safety(self):
        contact_count = torch.sum(self.env.foot_contacts_from_sensor.float(), dim=1)
        standing_min = getattr(
            self.env.cfg.rewards,
            "safety_min_feet_contacts_standing",
            getattr(self.env.cfg.rewards, "safety_min_feet_contacts", 3.0),
        )
        walking_min = getattr(
            self.env.cfg.rewards,
            "safety_min_feet_contacts_walking",
            2.0,
        )
        min_feet = torch.full_like(contact_count, float(standing_min))
        min_feet[self.env._get_walking_cmd_mask()] = float(walking_min)
        return torch.clamp(contact_count - (min_feet - 1.0), min=0.0, max=1.0)

    def _stability_safety(self):
        return self._body_stability_safety() * self._support_safety()

    def _reward_tracking_ee_world_stable(self):
        reward, metric = self._reward_tracking_ee_world()
        return reward * self._stability_safety(), metric

    def _reward_stability_safety(self):
        safety = self._stability_safety()
        return safety, safety

    def _reward_tracking_ee_sphere_walking(self):
        reward, metric = self._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_sphere_standing(self):
        reward, metric = self._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_cart(self):
        target_ee = self.env.get_ee_goal_spherical_center() + quat_apply(self.env.base_yaw_quat, self.env.curr_ee_goal_cart)
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - target_ee), dim=1)
        return torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma), ee_pos_error

    def _reward_tracking_ee_orn(self):
        ee_orn = self.env.ee_orn / torch.norm(
            self.env.ee_orn, dim=-1, keepdim=True
        ).clamp(min=1e-6)
        quaternion_vector_error = orientation_error(
            self.env.ee_goal_orn_quat, ee_orn
        )
        half_angle_sine = torch.clamp(
            torch.norm(quaternion_vector_error, dim=-1), max=1.0
        )
        angle_error = 2.0 * torch.asin(half_angle_sine)
        sigma = getattr(
            self.env.cfg.rewards,
            "tracking_ee_orientation_sigma",
            self.env.cfg.rewards.tracking_ee_sigma,
        )
        return torch.exp(-angle_error / sigma), angle_error

    def _reward_arm_energy_abs_sum(self):
        num_gripper = self.env.cfg.env.num_gripper_joints
        arm_end = -num_gripper if num_gripper > 0 else None
        energy = torch.sum(
            torch.abs(self.env.torques[:, 12:arm_end] * self.env.dof_vel[:, 12:arm_end]),
            dim=1,
        )
        return energy, energy

    def _reward_tracking_ee_orn_ry(self):
        ee_orn_euler = torch.stack(euler_from_quat(self.env.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs((torch_wrap_to_pi_minuspi(self.env.ee_goal_orn_euler - ee_orn_euler) * self.env.orn_error_scale)[:, [0, 2]]), dim=1)
        return torch.exp(-orn_err/self.env.cfg.rewards.tracking_ee_sigma), orn_err

    # -------------B1: Reward functions----------------

    def _reward_hip_action_l2(self):
        action_l2 = torch.sum(self.env.actions[:, [0, 3, 6, 9]] ** 2, dim=1)
        return action_l2, action_l2

    def _reward_leg_energy_abs_sum(self):
        energy = torch.sum(torch.abs(self.env.torques[:, :12] * self.env.dof_vel[:, :12]), dim = 1)
        return energy, energy

    def _reward_leg_energy_sum_abs(self):
        energy = torch.abs(torch.sum(self.env.torques[:, :12] * self.env.dof_vel[:, :12], dim = 1))
        return energy, energy

    def _reward_leg_action_l2(self):
        action_l2 = torch.sum(self.env.actions[:, :12] ** 2, dim=1)
        return action_l2, action_l2

    def _reward_leg_action_l2_deadzone(self):
        deadzone = getattr(self.env.cfg.rewards, "leg_action_deadzone", 0.20)
        excess = torch.clamp(torch.abs(self.env.actions[:, :12]) - deadzone, min=0.0)
        penalty = torch.sum(excess ** 2, dim=1)
        return penalty, penalty

    def _reward_leg_energy(self):
        energy = torch.sum(self.env.torques[:, :12] * self.env.dof_vel[:, :12], dim = 1)
        return energy, energy

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error/self.env.cfg.rewards.tracking_sigma)
        if getattr(
            self.env.cfg.rewards, "subtract_tracking_static_baseline", False
        ):
            command_error_at_rest = torch.sum(
                torch.square(self.env.commands[:, :2]), dim=1
            )
            static_reward = torch.exp(
                -command_error_at_rest / self.env.cfg.rewards.tracking_sigma
            )
            active = command_error_at_rest > 1e-12
            walking = self.env._get_walking_cmd_mask()
            normalized = torch.clamp(
                (reward - static_reward) / torch.clamp(1.0 - static_reward, min=1e-6),
                min=-1.0,
                max=1.0,
            )
            reward = torch.where(
                active,
                normalized,
                torch.where(walking, reward - 1.0, reward),
            )
        return reward, lin_vel_error

    def _reward_tracking_lin_vel_x_l1(self):
        zero_cmd_indices = torch.abs(self.env.commands[:, 0]) < 1e-5
        error = torch.abs(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        rew = 0*error
        rew_x = -error + torch.abs(self.env.commands[:, 0])
        rew[~zero_cmd_indices] = rew_x[~zero_cmd_indices] / (torch.abs(self.env.commands[~zero_cmd_indices, 0]) + 0.01)
        rew[zero_cmd_indices] = 0
        return rew, error

    def _reward_tracking_lin_vel_x_exp(self):
        squared_error = torch.square(
            self.env.commands[:, 0] - self.env.base_lin_vel[:, 0]
        )
        return (
            torch.exp(-squared_error / self.env.cfg.rewards.tracking_sigma),
            squared_error,
        )

    def _reward_tracking_ang_vel_yaw_l1(self):
        error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return - error + torch.abs(self.env.commands[:, 2]), error

    def _reward_tracking_ang_vel_yaw_exp(self):
        squared_error = torch.square(
            self.env.commands[:, 2] - self.env.base_ang_vel[:, 2]
        )
        return (
            torch.exp(-squared_error / self.env.cfg.rewards.tracking_sigma),
            squared_error,
        )

    def _reward_tracking_lin_vel_y_l2(self):
        squared_error = (self.env.commands[:, 1] - self.env.base_lin_vel[:, 1]) ** 2
        return squared_error, squared_error

    def _reward_tracking_lin_vel_z_l2(self):
        squared_error = self.env.base_lin_vel[:, 2] ** 2
        return squared_error, squared_error

    def _reward_survive(self):
        survival_reward = torch.ones(self.env.num_envs, device=self.env.device)
        return survival_reward, survival_reward

    def _reward_foot_contacts_z(self):
        foot_contacts_z = torch.square(self.env.force_sensor_tensor[:, :, 2]).sum(dim=-1)
        return foot_contacts_z, foot_contacts_z

    def _reward_torques(self):
        # Penalize torques
        num_torques = getattr(self.env, "num_torques", self.env.torques.shape[1])
        torque = torch.sum(torch.square(self.env.torques[:, :num_torques]), dim=1)
        return torque, torque

    def _reward_energy_square(self):
        energy = torch.sum(torch.square(self.env.torques[:, :12] * self.env.dof_vel[:, :12]), dim=1)
        return energy, energy

    def _reward_tracking_lin_vel_y(self):
        cmd = self.env.commands[:, 1].clone()
        lin_vel_y_error = torch.square(cmd - self.env.base_lin_vel[:, 1])
        rew = torch.exp(-lin_vel_y_error/self.env.cfg.rewards.tracking_sigma)
        return rew, lin_vel_y_error

    def _reward_lin_vel_z(self):
        rew = torch.square(self.env.base_lin_vel[:, 2])
        return rew, rew

    def _reward_ang_vel_xy(self):
        rew = torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)
        return rew, rew

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        reward = torch.exp(-ang_vel_error/self.env.cfg.rewards.tracking_sigma)
        if getattr(
            self.env.cfg.rewards, "subtract_tracking_static_baseline", False
        ):
            command_error_at_rest = torch.square(self.env.commands[:, 2])
            static_reward = torch.exp(
                -command_error_at_rest / self.env.cfg.rewards.tracking_sigma
            )
            active = command_error_at_rest > 1e-12
            walking = self.env._get_walking_cmd_mask()
            normalized = torch.clamp(
                (reward - static_reward) / torch.clamp(1.0 - static_reward, min=1e-6),
                min=-1.0,
                max=1.0,
            )
            reward = torch.where(
                active,
                normalized,
                torch.where(walking, reward - 1.0, reward),
            )
        return reward, ang_vel_error

    def _reward_work(self):
        work = self.env.torques * self.env.dof_vel
        abs_sum_work = torch.abs(torch.sum(work[:, :12], dim = 1))
        return abs_sum_work, abs_sum_work

    def _reward_dof_acc(self):
        rew = torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[:, :12] / self.env.dt), dim=1)
        return rew, rew

    def _reward_action_rate(self):
        action_rate = torch.sum(torch.square(self.env.last_actions - self.env.actions)[:, :12], dim=1)
        return action_rate, action_rate

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.) # upper limit
        rew = torch.sum(out_of_limits[:, :12], dim=1)
        return rew, rew

    def _reward_delta_torques(self):
        rew = torch.sum(torch.square(self.env.torques - self.env.last_torques)[:, :12], dim=1)
        return rew, rew

    def _reward_collision(self):
        threshold = getattr(self.env.cfg.rewards, "collision_force_threshold", 5.0)
        soft_clip = getattr(self.env.cfg.rewards, "collision_soft_clip", 50.0)
        contact_force = torch.norm(self.env.contact_forces[:, self.env.penalized_contact_indices, :], dim=-1)
        excess = torch.clamp(contact_force - threshold, min=0.0, max=soft_clip)
        rew = torch.sum(excess / max(threshold, 1e-6), dim=1)
        return rew, rew

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)
        rew[self.env._get_walking_cmd_mask()] = 0.
        return rew, rew

    def _reward_walking_dof(self):
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)
        rew[~self.env._get_walking_cmd_mask()] = 0.
        return rew, rew

    def _reward_hip_pos(self):
        rew = torch.sum(torch.square(self.env.dof_pos[:, self.env.hip_indices] - self.env.default_dof_pos[self.env.hip_indices]), dim=1)
        return rew, rew

    def _reward_foot_lateral_spacing(self):
        min_lateral = getattr(self.env.cfg.rewards, "foot_lateral_min", 0.06)
        foot_pos_local = []
        for i in range(4):
            rel_pos = self.env.rigid_body_state[:, self.env.feet_indices[i], :3] - self.env.base_pos
            foot_pos_local.append(quat_rotate_inverse(self.env.base_yaw_quat, rel_pos))
        foot_pos_local = torch.stack(foot_pos_local, dim=1)
        foot_y = foot_pos_local[:, :, 1]

        left_violation = torch.clamp(min_lateral - foot_y[:, [0, 2]], min=0.0)
        right_violation = torch.clamp(min_lateral + foot_y[:, [1, 3]], min=0.0)
        violation = torch.sum(left_violation, dim=1) + torch.sum(right_violation, dim=1)
        return violation, violation

    def _reward_feet_jerk(self):
        if not hasattr(self.env, "last_contact_forces"):
            result = torch.zeros(self.env.num_envs).to(self.env.device)
        else:
            result = torch.sum(
                torch.norm(self.env.force_sensor_tensor - self.env.last_contact_forces, dim=-1),
                dim=-1,
            )

        self.env.last_contact_forces = self.env.force_sensor_tensor.clone()
        result[self.env.episode_length_buf<50] = 0.
        return result, result

    def _reward_alive(self):
        alive = torch.ones(self.env.num_envs, device=self.env.device)
        return alive, alive

    def _reward_feet_drag(self):
        # Walk These Ways' slip term penalizes horizontal foot speed squared
        # only while the foot is in contact.  Including vertical touchdown
        # velocity here incorrectly punishes a normal swing-and-land cycle.
        contacts = getattr(
            self.env, "contact_filt", self.env.foot_contacts_from_sensor
        ).float()
        feet_xy_vel_sq = torch.sum(
            torch.square(
                self.env.rigid_body_state[:, self.env.feet_indices, 7:9]
            ),
            dim=-1,
        )
        slip = torch.sum(contacts * feet_xy_vel_sq, dim=-1)
        return slip, slip

    def _reward_feet_contact_forces(self):
        reset_flag = (self.env.episode_length_buf > 2./self.env.dt).type(torch.float)
        forces = torch.sum((torch.norm(self.env.force_sensor_tensor, dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0), dim=-1)
        rew = reset_flag * forces
        return rew, rew

    def _reward_feet_contact_standing(self):
        # Penalize any foot leaving the ground when standing still (not walking)
        # This forces the robot to crouch with all 4 feet down instead of lifting a leg to compensate for arm weight
        feet_off_ground = (~self.env.foot_contacts_from_sensor.bool()).float()
        rew = feet_off_ground.sum(dim=1)
        walking_mask = self.env._get_walking_cmd_mask()
        rew[walking_mask] = 0.  # Only apply when standing
        return rew, rew

    def _reward_hind_feet_contact_standing(self):
        # Extra standing constraint for rear support legs (RL, RR)
        # feet order is [FL, FR, RL, RR]
        hind_off_ground = (~self.env.foot_contacts_from_sensor[:, 2:4].bool()).float()
        rew = hind_off_ground.sum(dim=1)
        walking_mask = self.env._get_walking_cmd_mask()
        rew[walking_mask] = 0.
        return rew, rew

    def _reward_foot_support_standing(self):
        min_feet = getattr(self.env.cfg.rewards, "min_stance_feet", 3.0)
        contact_count = torch.sum(self.env.foot_contacts_from_sensor.float(), dim=1)
        missing_support = torch.clamp(min_feet - contact_count, min=0.0)
        missing_support[self.env._get_walking_cmd_mask()] = 0.
        return missing_support, missing_support

    def _reward_low_goal_front_leg_bend(self):
        # For low EE goals, encourage front legs to crouch instead of solving reach only with body pitch.
        dof_pos = self.env.dof_pos
        default = self.env.default_dof_pos.unsqueeze(0)

        fl_thigh_bend = torch.clamp(dof_pos[:, 1] - default[:, 1], min=0.0)
        fr_thigh_bend = torch.clamp(dof_pos[:, 4] - default[:, 4], min=0.0)
        fl_calf_bend = torch.clamp(default[:, 2] - dof_pos[:, 2], min=0.0)
        fr_calf_bend = torch.clamp(default[:, 5] - dof_pos[:, 5], min=0.0)
        bend = 0.25 * (fl_thigh_bend + fr_thigh_bend + fl_calf_bend + fr_calf_bend)

        ee_goal_z = self.env.curr_ee_goal_cart_world[:, 2] - self._terrain_height()
        low_goal_h = getattr(self.env.cfg.rewards, 'low_goal_height_thresh', 0.22)
        low_goal_mask = ee_goal_z < low_goal_h

        bend[~low_goal_mask] = 0.
        return bend, bend

    def _reward_low_goal_posture_asymmetry(self):
        # For low goals, prefer front-leg crouch while rear legs remain relatively supportive.
        dof_pos = self.env.dof_pos
        default = self.env.default_dof_pos.unsqueeze(0)

        front_bend = 0.25 * (
            torch.clamp(dof_pos[:, 1] - default[:, 1], min=0.0) +
            torch.clamp(dof_pos[:, 4] - default[:, 4], min=0.0) +
            torch.clamp(default[:, 2] - dof_pos[:, 2], min=0.0) +
            torch.clamp(default[:, 5] - dof_pos[:, 5], min=0.0)
        )
        rear_bend = 0.25 * (
            torch.clamp(dof_pos[:, 7] - default[:, 7], min=0.0) +
            torch.clamp(dof_pos[:, 10] - default[:, 10], min=0.0) +
            torch.clamp(default[:, 8] - dof_pos[:, 8], min=0.0) +
            torch.clamp(default[:, 11] - dof_pos[:, 11], min=0.0)
        )
        asymmetry = torch.clamp(front_bend - rear_bend, min=0.0)

        ee_goal_z = self.env.curr_ee_goal_cart_world[:, 2] - self._terrain_height()
        low_goal_h = getattr(self.env.cfg.rewards, 'low_goal_height_thresh', 0.22)
        low_goal_mask = ee_goal_z < low_goal_h

        asymmetry[~low_goal_mask] = 0.
        return asymmetry, asymmetry

    def _reward_low_goal_hind_leg_extension(self):
        # For low goals, encourage rear legs to remain supportive rather than crouching together with the front.
        dof_pos = self.env.dof_pos
        default = self.env.default_dof_pos.unsqueeze(0)

        rl_thigh_ext = torch.clamp(default[:, 7] - dof_pos[:, 7], min=0.0)
        rr_thigh_ext = torch.clamp(default[:, 10] - dof_pos[:, 10], min=0.0)
        rl_calf_ext = torch.clamp(dof_pos[:, 8] - default[:, 8], min=0.0)
        rr_calf_ext = torch.clamp(dof_pos[:, 11] - default[:, 11], min=0.0)
        extension = 0.25 * (rl_thigh_ext + rr_thigh_ext + rl_calf_ext + rr_calf_ext)

        ee_goal_z = self.env.curr_ee_goal_cart_world[:, 2] - self._terrain_height()
        low_goal_h = getattr(self.env.cfg.rewards, 'low_goal_height_thresh', 0.22)
        low_goal_mask = ee_goal_z < low_goal_h

        extension[~low_goal_mask] = 0.
        return extension, extension

    def _reward_low_goal_hind_support_force(self):
        # For low goals, rear feet should still carry part of the load to prevent forward falling.
        foot_forces = torch.norm(self.env.force_sensor_tensor, dim=-1)
        front_force = foot_forces[:, :2].sum(dim=1)
        hind_force = foot_forces[:, 2:4].sum(dim=1)
        hind_ratio = hind_force / (front_force + hind_force + 1e-6)
        target_ratio = getattr(self.env.cfg.rewards, 'low_goal_hind_force_ratio_target', 0.30)
        rew = torch.clamp(hind_ratio / target_ratio, max=1.0)

        ee_goal_z = self.env.curr_ee_goal_cart_world[:, 2] - self._terrain_height()
        low_goal_h = getattr(self.env.cfg.rewards, 'low_goal_height_thresh', 0.22)
        low_goal_mask = ee_goal_z < low_goal_h

        rew[~low_goal_mask] = 0.
        return rew, hind_ratio

    def _reward_orientation(self):
        # Penalize non flat base orientation
        error = torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)
        return error, error

    def _reward_roll(self):
        # Penalize non flat base orientation
        roll = self.env._get_body_orientation()[:, 0]
        error = torch.abs(roll)
        return error, error

    def _base_height_relative_to_terrain(self):
        return self.env.root_states[:, 2] - self._terrain_height()

    def _reward_base_height(self):
        # Penalize deviation from target body height relative to terrain, not absolute world z.
        base_height = self._base_height_relative_to_terrain()
        error = torch.abs(base_height - self.env.cfg.rewards.base_height_target)
        return error, base_height

    def _reward_height_adaptation(self):
        """Track a safe body height interpolated from terrain-relative EE z."""
        base_height = self._base_height_relative_to_terrain()
        adaptive_target = self._adaptive_body_height_target()
        error = torch.abs(base_height - adaptive_target)
        return error, error

    def _reward_pitch_adaptation(self):
        """Lower the front/arm mount for low EE goals without prescribing a gait."""
        body_pitch = self.env._get_body_orientation()[:, 1]
        adaptive_target = self._adaptive_body_pitch_target()
        error = torch.abs(body_pitch - adaptive_target)
        return error, error

    def _adaptive_body_height_target(self):
        blend = self._adaptive_posture_blend()
        min_body_h = getattr(self.env.cfg.rewards, "min_body_height", 0.24)
        natural_body_h = self.env.cfg.rewards.base_height_target
        return min_body_h + blend * (natural_body_h - min_body_h)

    def _adaptive_body_pitch_target(self):
        blend = self._adaptive_posture_blend()
        max_forward_pitch = getattr(self.env.cfg.rewards, "max_forward_body_pitch", 0.12)
        return (1.0 - blend) * max_forward_pitch

    def _adaptive_posture_blend(self):
        ee_goal_z = self.env.curr_ee_goal_cart_world[:, 2] - self._terrain_height()
        low_goal_z = getattr(self.env.cfg.rewards, "height_adaptation_goal_z_low", 0.10)
        high_goal_z = getattr(self.env.cfg.rewards, "height_adaptation_goal_z_high", 0.35)
        return torch.clamp(
            (ee_goal_z - low_goal_z) / max(high_goal_z - low_goal_z, 1e-6),
            min=0.0,
            max=1.0,
        )

    def _reward_pitch_soft_limit_standing(self):
        # Allow moderate pitch for bowing/crouching, but penalize excessive pitch to avoid forward fall
        pitch = self.env._get_body_orientation()[:, 1]
        pitch_limit = getattr(self.env.cfg.rewards, 'pitch_soft_limit', 0.35)
        error = torch.clamp(torch.abs(pitch) - pitch_limit, min=0.0)
        walking_mask = self.env._get_walking_cmd_mask()
        error[walking_mask] = 0.
        return error, error

    def _reward_orientation_walking(self):
        reward, metric = self._reward_orientation()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_orientation_standing(self):
        reward, metric = self._reward_orientation()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_torques_walking(self):
        reward, metric = self._reward_torques()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_torques_standing(self):
        reward, metric = self._reward_torques()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_energy_square_walking(self):
        reward, metric = self._reward_energy_square()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_energy_square_standing(self):
        reward, metric = self._reward_energy_square()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_base_height_walking(self):
        reward, metric = self._reward_base_height()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_base_height_standing(self):
        reward, metric = self._reward_base_height()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_dof_default_pos(self):
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)

        return rew, rew

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        return dof_error, dof_error

    def _reward_dof_error_deadzone(self):
        deadzone = getattr(self.env.cfg.rewards, "dof_error_deadzone", 0.12)
        dof_error = torch.clamp(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12] - deadzone, min=0.0)
        penalty = torch.sum(dof_error ** 2, dim=1)
        return penalty, penalty

    def _reward_tracking_lin_vel_max(self):
        rew = torch.where(self.env.commands[:, 0] > 0, torch.minimum(self.env.base_lin_vel[:, 0], self.env.commands[:, 0]) / (self.env.commands[:, 0] + 1e-5), \
                          torch.minimum(-self.env.base_lin_vel[:, 0], -self.env.commands[:, 0]) / (-self.env.commands[:, 0] + 1e-5))
        zero_cmd_indices = torch.abs(self.env.commands[:, 0]) < self.env.cfg.commands.lin_vel_x_clip
        rew[zero_cmd_indices] = torch.exp(-torch.abs(self.env.base_lin_vel[:, 0]))[zero_cmd_indices]
        return rew, rew

    def _reward_penalty_lin_vel_y(self):
        rew = torch.abs(self.env.base_lin_vel[:, 1])
        rot_indices = torch.abs(self.env.commands[:, 2]) > self.env.cfg.commands.ang_vel_yaw_clip
        rew[rot_indices] = 0.
        return rew, rew

    # -------------B1 Gait Control Rewards----------------
    def _reward_tracking_contacts_shaped_force(self):
        if not self.env.cfg.env.observe_gait_commands:
            zero = torch.zeros(self.env.num_envs, device=self.env.device)
            return zero, zero
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))

        reward[~self.env._get_walking_cmd_mask()] = 0
        return reward / 4, reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        if not self.env.cfg.env.observe_gait_commands:
            zero = torch.zeros(self.env.num_envs, device=self.env.device)
            return zero, zero
        # Read the freshly refreshed simulator tensor directly. The foot cache
        # is advanced-indexed and therefore cannot be treated as a persistent
        # view into rigid_body_state.
        foot_velocities = torch.norm(
            self.env.rigid_body_state[:, self.env.feet_indices, 7:10], dim=2
        )
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        reward[~self.env._get_walking_cmd_mask()] = 0

        return reward / 4, reward / 4

    def _reward_feet_height(self):
        foot_ids = self.env.feet_indices
        if self.env.cfg.env.observe_gait_commands:
            swing_weight = 1.0 - self.env.desired_contact_states
        else:
            # The simple Go2-X5 profile deliberately has no gait clock.  Use
            # actual airborne feet so low-clearance skating is penalized without
            # prescribing which foot must swing or when.
            contacts = getattr(
                self.env, "contact_filt", self.env.foot_contacts_from_sensor
            ).float()
            swing_weight = 1.0 - contacts
        if not self.env.cfg.rewards.feet_height_allfeet:
            foot_ids = foot_ids[:2]
            swing_weight = swing_weight[:, :2]

        terrain_height = self._terrain_height().unsqueeze(1)

        foot_height = self.env.rigid_body_state[:, foot_ids, 2] - terrain_height
        clearance_error = torch.clamp(
            self.env.cfg.rewards.feet_height_target - foot_height,
            min=0.0,
        )
        penalty = torch.sum(swing_weight * clearance_error, dim=1) / torch.clamp(
            torch.sum(swing_weight, dim=1),
            min=1.0,
        )
        penalty[~self.env._get_walking_cmd_mask()] = 0
        return -penalty, penalty

    def _reward_feet_air_time(self):
        # Reward completed swing-and-land events without prescribing a gait.
        # The original (air_time - target) expression penalized short steps,
        # which made "never land this foot" a local optimum.  This variant is
        # fail-closed: short steps are neutral, long/clear steps are bounded,
        # and no reward is paid until the foot actually lands.
        contacts = getattr(
            self.env, "contact_filt", self.env.foot_contacts_from_sensor
        ).bool()
        first_contact = (self.env.feet_air_time > 0.0) & contacts
        self.env.feet_air_time += self.env.dt

        target_air_time = getattr(self.env.cfg.rewards, "feet_air_time_target", 0.5)
        max_air_bonus = getattr(
            self.env.cfg.rewards, "feet_air_time_max_bonus", target_air_time
        )
        terrain_height = self._terrain_height().unsqueeze(1)
        foot_height = (
            self.env.rigid_body_state[:, self.env.feet_indices, 2]
            - terrain_height
        )
        self.env.feet_swing_peak_height[:] = torch.maximum(
            self.env.feet_swing_peak_height,
            foot_height,
        )
        air_bonus = torch.clamp(
            self.env.feet_air_time - target_air_time,
            min=0.0,
            max=max_air_bonus,
        )
        clearance_floor = getattr(
            self.env.cfg.rewards, "feet_clearance_floor", 0.0
        )
        clearance_target = getattr(
            self.env.cfg.rewards, "feet_clearance_target", clearance_floor
        )
        clearance_denominator = max(
            float(clearance_target) - float(clearance_floor), 1.0e-6
        )
        clearance_bonus = torch.clamp(
            (self.env.feet_swing_peak_height - clearance_floor)
            / clearance_denominator,
            min=0.0,
            max=1.0,
        )
        clearance_weight = getattr(
            self.env.cfg.rewards, "feet_clearance_landing_bonus", 0.0
        )
        completed_step_bonus = (
            air_bonus + clearance_weight * clearance_bonus
        ) * first_contact
        all_feet = getattr(
            self.env.cfg.rewards,
            "feet_air_time_all_feet",
            getattr(self.env.cfg.rewards, "feet_aritime_allfeet", True),
        )
        if all_feet:
            rew_airTime = torch.sum(completed_step_bonus, dim=1)
        else:
            rew_airTime = torch.sum(completed_step_bonus[:, :2], dim=1)

        rew_airTime *= self.env._get_walking_cmd_mask()  # reward for stepping for any of the 3 motions
        self.env.feet_air_time *= ~contacts
        self.env.feet_swing_peak_height *= ~contacts
        return rew_airTime, rew_airTime
