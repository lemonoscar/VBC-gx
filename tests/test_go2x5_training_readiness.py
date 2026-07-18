import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / "low-level/legged_gym/envs/manip_loco/go2x5_config.py"
ENV = ROOT / "low-level/legged_gym/envs/manip_loco/manip_loco.py"
REWARDS = ROOT / "low-level/legged_gym/envs/rewards/maniploco_rewards.py"
PPO = ROOT / "third_party/rsl_rl/rsl_rl/algorithms/ppo.py"
RUNNER = ROOT / "third_party/rsl_rl/rsl_rl/runners/on_policy_runner.py"
TASK_REGISTRY = ROOT / "low-level/legged_gym/utils/task_registry.py"
TRAIN = ROOT / "low-level/legged_gym/scripts/train.py"
AUDIT = ROOT / "low-level/legged_gym/scripts/audit_go2x5_low_level_rewards.py"
CHECKPOINT_ROLLOUT = ROOT / "low-level/legged_gym/scripts/check_go2x5_checkpoint_rollout.py"
FIXED_COMMAND_GAIT = ROOT / "low-level/legged_gym/scripts/check_go2x5_fixed_command_gait.py"
READINESS = ROOT / "low-level/legged_gym/scripts/check_go2x5_training_readiness.py"


def read(path):
    return path.read_text(encoding="utf-8")


def test_reward_audit_is_complete_and_fail_closed():
    result = subprocess.run(
        [sys.executable, str(AUDIT), "--fail-on-mismatch"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MISMATCH" not in result.stdout
    assert "No metadata yet" not in result.stdout
    assert "74/74" in result.stdout
    assert "tracking_contacts_shaped_force` | 1.0 | + | OK" in result.stdout
    assert "tracking_contacts_shaped_vel` | 0.5 | + | OK" in result.stdout


def test_go2x5_training_contract_defaults_are_unambiguous():
    config = read(CONFIG)
    assert "observe_gait_commands = True" in config
    assert "num_observations = robot_spec.observation_dim(True)" in config
    assert "require_training_metadata = True" in config
    assert "feet_air_time_target = 0.25" in config
    assert "feet_aritime_allfeet = True" in config
    assert "feet_height_allfeet = True" in config
    assert '"tracking_contacts_shaped_force_scale": 1.0' in config
    assert '"tracking_contacts_shaped_vel_scale": 0.5' in config
    assert '"walking_dof_scale": 0.0' in config
    assert '"feet_height_scale": 1.0' in config
    assert "height = [0.00, 0.00]" in config
    assert 'profile_name = "go2x5_stable_reach_curriculum_v5_gait_aware_h032"' in config
    assert "safety_min_feet_contacts_standing = 3.0" in config
    assert "safety_min_feet_contacts_walking = 2.0" in config
    assert '"name": "S3_forward_gait_initiation"' in config
    assert '"name": "S4_bidirectional_locomotion_reach"' in config
    assert '"Episode_metric/metric_tracking_lin_vel_max": [">", 0.45]' in config


def test_reward_implementations_cover_stop_mask_height_and_jerk_state():
    rewards = read(REWARDS)
    assert rewards.count("reward[~self.env._get_walking_cmd_mask()] = 0") >= 2
    assert 'hasattr(self.env, "last_contact_forces")' in rewards
    assert "self.env.last_contact_forces = self.env.force_sensor_tensor.clone()" in rewards
    assert "clearance_error = torch.clamp(" in rewards
    assert "swing_weight * clearance_error" in rewards
    assert 'getattr(self.env.cfg.rewards, "feet_air_time_target", 0.5)' in rewards


def test_foot_kinematics_are_refreshed_from_live_rigid_body_state():
    env = read(ENV)
    assert "def _refresh_foot_kinematics(self):" in env
    assert "torch.index_select(" in env
    assert "self.rigid_body_state[:, :, 7:10], 1, self.feet_indices" in env
    rewards = read(REWARDS)
    assert "self.env.rigid_body_state[:, self.env.feet_indices, 7:10]" in rewards
    assert "torch.norm(self.env.foot_velocities" not in rewards
    post_step = env[env.index("def post_physics_step(self):"):env.index("def check_termination(self):")]
    assert "self.gym.refresh_rigid_body_state_tensor(self.sim)" in post_step
    assert "self._refresh_foot_kinematics()" in post_step


def test_reset_and_height_sampling_clear_cross_episode_state():
    env = read(ENV)
    for expression in (
        "self.actions[env_ids] = 0.",
        "self.torques[env_ids] = 0.",
        "self.gait_indices[env_ids] = 0.",
        "self.clock_inputs[env_ids] = 0.",
        "self.desired_contact_states[env_ids] = 1.",
        "self.obs_history_buf[env_ids, :, :] = 0.",
        "self.action_history_buf[env_ids, :, :] = 0.",
        "self.height_points = self._init_height_points()",
        "self.measured_heights = self._get_heights()",
        "self.episode_metric_sums[key][env_ids] / completed_episode_steps",
    ):
        assert expression in env
    assert "self.desired_contact_states[~walking_mask] = 1.0" in env


def test_12d_ppo_uses_one_policy_channel_and_fails_on_nonfinite():
    ppo = read(PPO)
    assert "policy_channels = 1 if self.actor_critic.num_arm_actions == 0 else 2" in ppo
    assert "policy_advantages = mixing_advantages_batch[..., :policy_channels]" in ppo
    assert 'self._require_finite("actor observations", obs)' in ppo
    assert 'self._require_finite("sampled actions", self.transition.actions)' in ppo
    assert 'self._require_finite("PPO loss", loss)' in ppo
    assert 'self._require_finite("PPO gradient norm", grad_norm)' in ppo


def test_checkpoint_resume_restores_training_state_and_target_iteration():
    runner = read(RUNNER)
    train = read(TRAIN)
    env = read(ENV)
    for field in (
        "hist_encoder_optimizer_state_dict",
        "algorithm_counter",
        "runner_state",
        "tot_timesteps",
        "tot_time",
    ):
        assert field in runner
    assert "completed_iterations = it + 1" in runner
    assert "completed_iterations % self.save_interval == 0" in runner
    assert "self.env.load_training_metadata(loaded_dict.get('metadata'))" in runner
    assert '"training_state": {' in env and '"global_steps": int(' in env
    assert 'self.global_steps = int(training_state["global_steps"])' in env
    assert "train_cfg.runner.max_iterations) - int(ppo_runner.current_learning_iteration" in train


def test_resume_id_uses_normal_path_join():
    registry = read(TASK_REGISTRY)
    assert 'resume_id = str(args.resumeid).strip("/\\\\")' in registry
    assert 'os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, resume_id)' in registry


def test_training_checkpoint_metadata_is_fail_closed():
    env = read(ENV)
    for message in (
        "training checkpoint has no metadata",
        "metadata mismatch for",
        "curriculum enabled state mismatch",
        "curriculum profile mismatch",
        "robot asset hash mismatch",
        "control contract hash is corrupt",
        "control contract mismatch",
    ):
        assert message in env


def test_checkpoint_rollout_fails_closed_on_early_resets():
    rollout = read(CHECKPOINT_ROLLOUT)
    assert '"--max-early-resets"' in rollout
    assert 'report["early_resets"] <= report["max_early_resets"]' in rollout


def test_fixed_command_gait_gate_detects_no_step_policies():
    gait = read(FIXED_COMMAND_GAIT)
    for option in (
        '"--min-translation-progress-ratio"',
        '"--min-yaw-progress-ratio"',
        '"--max-swing-contact-fraction"',
        '"--min-swing-height"',
        '"--max-stand-vx-error"',
        '"--max-stand-yaw-error"',
    ):
        assert option in gait
    assert "vx_abs_error_mean <= cli.max_stand_vx_error" in gait
    assert "yaw_abs_error_mean <= cli.max_stand_yaw_error" in gait
    assert "foot_cache_max_error <= 1.0e-7" in gait
    assert '"behavior_passed": behavior_passed' in gait
    assert 'return 0 if report["passed"] else 1' in gait


def test_training_readiness_separates_s0_gate_from_later_stage_stress():
    readiness = read(READINESS)
    assert '"--rollout-stage"' in readiness
    assert "default=0" in readiness
    assert 'reset_causes = {"roll": 0, "pitch": 0, "z": 0, "contact": 0}' in readiness
    assert 'early_resets == 0' in readiness


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("go2x5 training readiness tests passed")
