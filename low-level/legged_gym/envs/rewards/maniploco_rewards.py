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

    def _stability_safety(self):
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

        foot_contacts = self.env.foot_contacts_from_sensor.float()
        contact_count = torch.sum(foot_contacts, dim=1)
        min_feet = getattr(self.env.cfg.rewards, "safety_min_feet_contacts", 3.0)
        foot_safety = torch.clamp((contact_count - (min_feet - 1.0)) / 1.0, min=0.0, max=1.0)

        return roll_safety * pitch_safety * height_safety * foot_safety

    def _reward_tracking_ee_world_stable(self):
        reward, metric = self._reward_tracking_ee_world()
        return reward * self._stability_safety(), metric

    def _reward_stability_safety(self):
        safety = self._stability_safety()
        return safety, safety

    def _reward_tracking_ee_sphere_walking(self):
        reward, metric = self.env._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_sphere_standing(self):
        reward, metric = self.env._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_cart(self):
        target_ee = self.env.get_ee_goal_spherical_center() + quat_apply(self.env.base_yaw_quat, self.env.curr_ee_goal_cart)
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - target_ee), dim=1)
        return torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma), ee_pos_error

    def _reward_tracking_ee_orn(self):
        ee_orn_euler = torch.stack(euler_from_quat(self.env.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs(torch_wrap_to_pi_minuspi(self.env.ee_goal_orn_euler - ee_orn_euler)) * self.env.orn_error_scale, dim=1)
        return torch.exp(-orn_err/self.env.cfg.rewards.tracking_ee_sigma), orn_err

    def _reward_arm_energy_abs_sum(self):
        energy = torch.sum(torch.abs(self.env.torques[:, 12:-self.env.cfg.env.num_gripper_joints] * self.env.dof_vel[:, 12:-self.env.cfg.env.num_gripper_joints]), dim = 1)
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
        return torch.exp(-lin_vel_error/self.env.cfg.rewards.tracking_sigma), lin_vel_error

    def _reward_tracking_lin_vel_x_l1(self):
        zero_cmd_indices = torch.abs(self.env.commands[:, 0]) < 1e-5
        error = torch.abs(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        rew = 0*error
        rew_x = -error + torch.abs(self.env.commands[:, 0])
        rew[~zero_cmd_indices] = rew_x[~zero_cmd_indices] / (torch.abs(self.env.commands[~zero_cmd_indices, 0]) + 0.01)
        rew[zero_cmd_indices] = 0
        return rew, error

    def _reward_tracking_lin_vel_x_exp(self):
        error = torch.abs(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        return torch.exp(-error/self.env.cfg.rewards.tracking_sigma), error

    def _reward_tracking_ang_vel_yaw_l1(self):
        error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return - error + torch.abs(self.env.commands[:, 2]), error

    def _reward_tracking_ang_vel_yaw_exp(self):
        error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-error/self.env.cfg.rewards.tracking_sigma), error

    def _reward_tracking_lin_vel_y_l2(self):
        squared_error = (self.env.commands[:, 1] - self.env.base_lin_vel[:, 1]) ** 2
        return squared_error, squared_error

    def _reward_tracking_lin_vel_z_l2(self):
        squared_error = (self.env.commands[:, 2] - self.env.base_lin_vel[:, 2]) ** 2
        return squared_error, squared_error

    def _reward_survive(self):
        survival_reward = torch.ones(self.env.num_envs, device=self.env.device)
        return survival_reward, survival_reward

    def _reward_foot_contacts_z(self):
        foot_contacts_z = torch.square(self.env.force_sensor_tensor[:, :, 2]).sum(dim=-1)
        return foot_contacts_z, foot_contacts_z

    def _reward_torques(self):
        # Penalize torques
        torque = torch.sum(torch.square(self.env.torques), dim=1)
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
        return torch.exp(-ang_vel_error/self.env.cfg.rewards.tracking_sigma), ang_vel_error

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
        return 1., 1.

    def _reward_feet_drag(self):
        feet_xyz_vel = torch.abs(self.env.rigid_body_state[:, self.env.feet_indices, 7:10]).sum(dim=-1)
        dragging_vel = self.env.foot_contacts_from_sensor * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        return rew, rew

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

        ee_goal_z = self.env.curr_ee_goal_cart_world[:, 2]
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

        ee_goal_z = self.env.curr_ee_goal_cart_world[:, 2]
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

        ee_goal_z = self.env.curr_ee_goal_cart_world[:, 2]
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

        ee_goal_z = self.env.curr_ee_goal_cart_world[:, 2]
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
        base_height = self.env.root_states[:, 2]
        measured_heights = getattr(self.env, "measured_heights", None)
        if torch.is_tensor(measured_heights) and measured_heights.ndim == 2:
            base_height = base_height - torch.mean(measured_heights, dim=1)
        return base_height

    def _reward_base_height(self):
        # Penalize deviation from target body height relative to terrain, not absolute world z.
        base_height = self._base_height_relative_to_terrain()
        error = torch.abs(base_height - self.env.cfg.rewards.base_height_target)
        return error, base_height

    def _reward_height_adaptation(self):
        """Encourage lower body posture when EE goal is near the ground.
        When EE target z is low (~0.1m), body should crouch (~0.15m).
        When EE target z is high (~0.45m), body stays near the configured natural height.
        """
        ee_goal_z = self.env.curr_ee_goal_cart_world[:, 2]
        base_height = self._base_height_relative_to_terrain()

        sphere_center_z = self.env.cfg.goal_ee.sphere_center.z_invariant_offset  # 0.45
        natural_body_h = self.env.cfg.rewards.base_height_target
        min_body_h = getattr(self.env.cfg.rewards, 'min_body_height', 0.15)

        # Adaptive target: proportional to EE goal height, clamped to feasible range
        adaptive_target = natural_body_h * (ee_goal_z / sphere_center_z)
        adaptive_target = torch.clamp(adaptive_target, min_body_h, natural_body_h)

        error = torch.abs(base_height - adaptive_target)

        # Apply in both standing and walking to shape low-goal posture.
        return error, error

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
        reward, metric = self.env._reward_torques()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_torques_standing(self):
        reward, metric = self.env._reward_torques()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_energy_square_walking(self):
        reward, metric = self.env._reward_energy_square()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_energy_square_standing(self):
        reward, metric = self.env._reward_energy_square()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_base_height_walking(self):
        reward, metric = self.env._reward_base_height()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_base_height_standing(self):
        reward, metric = self.env._reward_base_height()
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
            return 0,0
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
            return 0,0
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        reward[~self.env._get_walking_cmd_mask()] = 0

        return reward / 4, reward / 4

    def _reward_feet_height(self):
        foot_ids = self.env.feet_indices
        desired_contact = self.env.desired_contact_states
        if not self.env.cfg.rewards.feet_height_allfeet:
            foot_ids = foot_ids[:2]
            desired_contact = desired_contact[:, :2]

        measured_heights = getattr(self.env, "measured_heights", None)
        if torch.is_tensor(measured_heights) and measured_heights.ndim == 2:
            terrain_height = torch.mean(measured_heights, dim=1, keepdim=True)
        else:
            terrain_height = self.env.env_origins[:, 2:3]

        foot_height = self.env.rigid_body_state[:, foot_ids, 2] - terrain_height
        swing_weight = 1.0 - desired_contact
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
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        first_contact = (self.env.feet_air_time > 0.) * self.env.foot_contacts_from_sensor  #self.env.contact_filt
        self.env.feet_air_time += self.env.dt

        target_air_time = getattr(self.env.cfg.rewards, "feet_air_time_target", 0.5)
        if self.env.cfg.rewards.feet_aritime_allfeet:
            rew_airTime = torch.sum((self.env.feet_air_time - target_air_time) * first_contact, dim=1)
        else:
            rew_airTime = torch.sum(
                (self.env.feet_air_time[:, :2] - target_air_time) * first_contact[:, :2],
                dim=1,
            )

        rew_airTime *= self.env._get_walking_cmd_mask()  # reward for stepping for any of the 3 motions
        self.env.feet_air_time *= ~ self.env.foot_contacts_from_sensor  #self.env.contact_filt
        return rew_airTime, rew_airTime
