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
                tracking_ee_sigma=1.0,
                gait_force_sigma=0.5,
            ),
            env=SimpleNamespace(observe_gait_commands=False),
            commands=SimpleNamespace(lin_vel_x_clip=0.05, ang_vel_yaw_clip=0.05),
        )
        self.commands = torch.zeros(self.num_envs, 3)
        self.root_states = torch.zeros(self.num_envs, 13)
        self.root_states[:, 2] = 0.32
        self.measured_heights = torch.zeros(self.num_envs, 4)
        self.env_origins = torch.zeros(self.num_envs, 3)
        self.foot_contacts_from_sensor = torch.zeros(self.num_envs, 4, dtype=torch.bool)
        self.ee_pos = torch.zeros(self.num_envs, 3)
        self.curr_ee_goal_cart_world = torch.zeros(self.num_envs, 3)
        self.desired_contact_states = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0]] * self.num_envs
        )
        self.contact_forces = torch.zeros(self.num_envs, 4, 3)
        self.feet_indices = torch.arange(4)
        self.base_lin_vel = torch.zeros(self.num_envs, 3)

    def _get_body_orientation(self):
        return torch.zeros(self.num_envs, 2)

    def _get_walking_cmd_mask(self):
        return torch.logical_or(
            torch.abs(self.commands[:, 0]) > self.cfg.commands.lin_vel_x_clip,
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
