import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
REWARDS = ROOT / "low-level/legged_gym/envs/rewards/maniploco_rewards.py"


def load_reward_class():
    isaacgym = types.ModuleType("isaacgym")
    torch_utils = types.ModuleType("isaacgym.torch_utils")
    isaacgym.torch_utils = torch_utils
    sys.modules.setdefault("isaacgym", isaacgym)
    sys.modules.setdefault("isaacgym.torch_utils", torch_utils)
    spec = importlib.util.spec_from_file_location("go2x5_reward_semantics", REWARDS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ManipLoco_rewards


class FakeEnv:
    def __init__(self):
        self.num_envs = 2
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(
            rewards=SimpleNamespace(
                safety_roll_soft=0.25,
                safety_roll_hard=0.55,
                safety_pitch_soft=0.25,
                safety_pitch_hard=0.55,
                safety_base_height_min=0.26,
                safety_base_height_floor=0.20,
                safety_min_feet_contacts_standing=3.0,
                safety_min_feet_contacts_walking=2.0,
                base_height_target=0.32,
                min_body_height=0.24,
                height_adaptation_goal_z_low=0.10,
                height_adaptation_goal_z_high=0.35,
                max_forward_body_pitch=0.12,
                tracking_sigma=0.05,
                tracking_ee_sigma=1.0,
                gait_force_sigma=0.5,
                feet_air_time_target=0.10,
                feet_air_time_max_bonus=0.25,
                feet_clearance_floor=0.022,
                feet_clearance_target=0.05,
                feet_clearance_landing_bonus=0.20,
                feet_air_time_all_feet=True,
            ),
            env=SimpleNamespace(observe_gait_commands=False),
            commands=SimpleNamespace(
                lin_vel_x_clip=0.05,
                lin_vel_y_clip=0.05,
                ang_vel_yaw_clip=0.05,
            ),
        )
        self.dt = 0.02
        self.commands = torch.zeros(self.num_envs, 3)
        self.root_states = torch.zeros(self.num_envs, 13)
        self.root_states[:, 2] = 0.32
        self.measured_heights = torch.zeros(self.num_envs, 4)
        self.env_origins = torch.zeros(self.num_envs, 3)
        self.foot_contacts_from_sensor = torch.zeros(self.num_envs, 4, dtype=torch.bool)
        self.contact_filt = torch.zeros(self.num_envs, 4, dtype=torch.bool)
        self.feet_air_time = torch.zeros(self.num_envs, 4)
        self.feet_swing_peak_height = torch.zeros(self.num_envs, 4)
        self.ee_pos = torch.zeros(self.num_envs, 3)
        self.curr_ee_goal_cart_world = torch.zeros(self.num_envs, 3)
        self.desired_contact_states = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0]] * self.num_envs
        )
        self.contact_forces = torch.zeros(self.num_envs, 4, 3)
        self.feet_indices = torch.arange(4)
        self.rigid_body_state = torch.zeros(self.num_envs, 4, 13)
        self.base_lin_vel = torch.zeros(self.num_envs, 3)
        self.base_ang_vel = torch.zeros(self.num_envs, 3)
        self.body_orientation = torch.zeros(self.num_envs, 2)

    def _get_body_orientation(self):
        return self.body_orientation

    def _get_walking_cmd_mask(self):
        return torch.logical_or(
            torch.logical_or(
                torch.abs(self.commands[:, 0]) > self.cfg.commands.lin_vel_x_clip,
                torch.abs(self.commands[:, 1]) > self.cfg.commands.lin_vel_y_clip,
            ),
            torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_yaw_clip,
        )


def make_reward():
    env = FakeEnv()
    return env, load_reward_class()(env)


def test_support_safety_is_mode_aware():
    env, reward = make_reward()
    env.foot_contacts_from_sensor[:, :2] = True
    assert torch.equal(reward._stability_safety(), torch.zeros(env.num_envs))
    env.foot_contacts_from_sensor[:, 2] = True
    assert torch.equal(reward._stability_safety(), torch.ones(env.num_envs))

    env.commands[:, 0] = 0.10
    env.foot_contacts_from_sensor.zero_()
    env.foot_contacts_from_sensor[:, 0] = True
    assert torch.equal(reward._stability_safety(), torch.zeros(env.num_envs))
    env.foot_contacts_from_sensor[:, 1] = True
    assert torch.equal(reward._stability_safety(), torch.ones(env.num_envs))
    env.foot_contacts_from_sensor.fill_(True)
    assert torch.equal(reward._stability_safety(), torch.ones(env.num_envs))


def test_raw_ee_tracking_is_independent_of_contact_schedule():
    env, reward = make_reward()
    env.curr_ee_goal_cart_world[:, 0] = 0.10
    first, first_error = reward._reward_tracking_ee_world()
    env.foot_contacts_from_sensor.fill_(True)
    env.contact_forces[:, :, 2] = 100.0
    second, second_error = reward._reward_tracking_ee_world()
    assert torch.equal(first, second)
    assert torch.equal(first_error, second_error)
    assert torch.all(first > 0)


def test_vertical_velocity_penalty_does_not_consume_yaw_command():
    env, reward = make_reward()
    env.base_lin_vel[:, 2] = torch.tensor([0.2, -0.3])
    first, _ = reward._reward_tracking_lin_vel_z_l2()
    env.commands[:, 2] = torch.tensor([0.7, -0.9])
    second, _ = reward._reward_tracking_lin_vel_z_l2()
    assert torch.equal(first, second)
    assert torch.allclose(first, torch.tensor([0.04, 0.09]))


def test_exact_velocity_reward_penalizes_under_and_overspeed_symmetrically():
    env, reward = make_reward()
    env.commands[:, 0] = 0.10

    env.base_lin_vel[:, 0] = torch.tensor([0.10, 0.10])
    exact, exact_error = reward._reward_tracking_lin_vel_x_exp()
    env.base_lin_vel[:, 0] = torch.tensor([0.00, 0.20])
    symmetric, symmetric_error = reward._reward_tracking_lin_vel_x_exp()

    assert torch.equal(exact_error, torch.zeros_like(exact_error))
    assert torch.all(exact > symmetric)
    assert torch.allclose(symmetric_error, torch.tensor([0.01, 0.01]))
    assert torch.allclose(symmetric[0], symmetric[1])
    assert torch.allclose(
        symmetric,
        torch.exp(torch.full((2,), -0.01 / env.cfg.rewards.tracking_sigma)),
    )

    env.commands[:, 2] = 0.10
    env.base_ang_vel[:, 2] = torch.tensor([0.00, 0.20])
    symmetric_yaw, symmetric_yaw_error = reward._reward_tracking_ang_vel_yaw_exp()
    assert torch.allclose(symmetric_yaw_error, torch.tensor([0.01, 0.01]))
    assert torch.allclose(symmetric_yaw[0], symmetric_yaw[1])


def test_walk_these_ways_xy_tracking_kernel_is_squared_and_dense():
    env, reward = make_reward()
    env.commands[:] = torch.tensor([0.30, 0.10, 0.0])
    env.base_lin_vel.zero_()

    tracking, squared_error = reward._reward_tracking_lin_vel()

    assert torch.allclose(squared_error, torch.full((2,), 0.10))
    assert torch.allclose(
        tracking,
        torch.exp(torch.full((2,), -0.10 / env.cfg.rewards.tracking_sigma)),
    )
    assert torch.all(tracking > 0.0)
    assert torch.all(tracking < 0.20)


def test_contact_drag_is_horizontal_squared_slip_only():
    env, reward = make_reward()
    env.contact_filt[:, 0] = True
    env.rigid_body_state[:, 0, 9] = 3.0

    vertical_only, _ = reward._reward_feet_drag()
    assert torch.equal(vertical_only, torch.zeros_like(vertical_only))

    env.rigid_body_state[:, 0, 7] = 0.5
    horizontal, _ = reward._reward_feet_drag()
    assert torch.allclose(horizontal, torch.full((2,), 0.25))

    env.contact_filt.zero_()
    airborne, _ = reward._reward_feet_drag()
    assert torch.equal(airborne, torch.zeros_like(airborne))


def test_air_time_rewards_only_clear_completed_steps_and_resets_buffers():
    env, reward = make_reward()
    env.commands[:, 0] = 0.10

    # An airborne foot builds a peak-clearance record but is not paid until it
    # lands.  This prevents a permanently lifted foot from farming reward.
    env.rigid_body_state[:, 0, 2] = 0.055
    airborne, _ = reward._reward_feet_air_time()
    assert torch.equal(airborne, torch.zeros_like(airborne))
    assert torch.allclose(env.feet_swing_peak_height[:, 0], torch.full((2,), 0.055))

    env.feet_air_time[:, 0] = 0.30
    env.rigid_body_state[:, 0, 2] = 0.022
    env.contact_filt[:, 0] = True

    landing, _ = reward._reward_feet_air_time()

    assert torch.allclose(landing, torch.full((2,), 0.42))
    assert torch.equal(env.feet_air_time[:, 0], torch.zeros(2))
    assert torch.equal(env.feet_swing_peak_height[:, 0], torch.zeros(2))

    continuous_contact, _ = reward._reward_feet_air_time()
    assert torch.equal(continuous_contact, torch.zeros_like(continuous_contact))

    # A short, ground-skimming landing is neutral rather than negative.
    env.contact_filt.zero_()
    env.feet_air_time[:, 2] = 0.02
    env.feet_swing_peak_height[:, 2] = 0.022
    env.contact_filt[:, 2] = True
    short_step, _ = reward._reward_feet_air_time()
    assert torch.equal(short_step, torch.zeros_like(short_step))

    env.commands.zero_()
    env.contact_filt.zero_()
    env.feet_air_time[:, 1] = 0.30
    env.feet_swing_peak_height[:, 1] = 0.055
    env.contact_filt[:, 1] = True
    stopped, _ = reward._reward_feet_air_time()
    assert torch.equal(stopped, torch.zeros_like(stopped))


def test_adaptive_height_target_is_safe_monotonic_and_terrain_relative():
    env, reward = make_reward()
    env.curr_ee_goal_cart_world[:, 2] = torch.tensor([0.10, 0.35])
    endpoints = reward._adaptive_body_height_target()
    assert torch.allclose(endpoints, torch.tensor([0.24, 0.32]))

    env.curr_ee_goal_cart_world[:, 2] = 0.225
    midpoint = reward._adaptive_body_height_target()
    assert torch.allclose(midpoint, torch.full((2,), 0.28))

    env.measured_heights.fill_(0.20)
    env.root_states[:, 2] += 0.20
    env.curr_ee_goal_cart_world[:, 2] += 0.20
    shifted_midpoint = reward._adaptive_body_height_target()
    height_error, _ = reward._reward_height_adaptation()
    assert torch.allclose(shifted_midpoint, midpoint)
    assert torch.allclose(height_error, torch.full((2,), 0.04))
    assert torch.all(shifted_midpoint >= 0.24)


def test_adaptive_pitch_lowers_front_for_low_goals_and_relaxes_for_high_goals():
    env, reward = make_reward()
    env.curr_ee_goal_cart_world[:, 2] = torch.tensor([0.10, 0.35])
    pitch_targets = reward._adaptive_body_pitch_target()
    pitch_error, _ = reward._reward_pitch_adaptation()

    assert torch.allclose(pitch_targets, torch.tensor([0.12, 0.00]))
    assert torch.allclose(pitch_error, pitch_targets)
    env.body_orientation[:, 1] = pitch_targets
    matched_error, _ = reward._reward_pitch_adaptation()
    assert torch.equal(matched_error, torch.zeros_like(matched_error))


def test_disabled_gait_branch_preserves_reward_tensor_contract():
    env, reward = make_reward()
    env.cfg.env.observe_gait_commands = False
    for function in (
        reward._reward_tracking_contacts_shaped_force,
        reward._reward_tracking_contacts_shaped_vel,
    ):
        raw, metric = function()
        assert raw.shape == (env.num_envs,)
        assert metric.shape == (env.num_envs,)
        assert torch.equal(raw, torch.zeros_like(raw))
        assert torch.equal(metric, torch.zeros_like(metric))


def test_reward_source_has_no_container_dispatch_or_scalar_alive_regressions():
    source = REWARDS.read_text(encoding="utf-8")
    assert "self.env._reward_" not in source
    assert "return 1., 1." not in source
    assert "commands[:, 2] - self.env.base_lin_vel[:, 2]" not in source


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("go2x5 reward semantics tests passed")
