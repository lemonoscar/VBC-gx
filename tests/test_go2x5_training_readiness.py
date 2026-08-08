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
HELPERS = ROOT / "low-level/legged_gym/utils/helpers.py"
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
    assert "75/75" in result.stdout
    assert "tracking_contacts_shaped_force` | 0.0" in result.stdout
    assert "tracking_contacts_shaped_vel` | 0.0" in result.stdout


def test_go2x5_training_contract_defaults_are_unambiguous():
    config = read(CONFIG)
    readiness = read(READINESS)
    assert "observe_gait_commands = False" in config
    assert "num_observations = robot_spec.observation_dim(False)" in config
    assert "require_training_metadata = True" in config
    assert "feet_air_time_target = 0.10" in config
    assert "feet_height_target = 0.05" in config
    assert "feet_clearance_target = 0.05" in config
    assert "feet_clearance_landing_bonus = 0.20" in config
    assert "feet_air_time_all_feet = True" in config
    assert "feet_height_allfeet = True" in config
    assert "tracking_contacts_shaped_force = 0.0" in config
    assert "tracking_contacts_shaped_vel = 0.0" in config
    assert "walking_dof = 0.0" in config
    assert "feet_height = 1.0" in config
    assert "feet_air_time = 2.0" in config
    assert "feet_contact_standing = -0.5" in config
    assert "leg_action_l2_deadzone = -0.5" in config
    assert "leg_action_deadzone = 0.80" in config
    assert "height = [0.00, 0.00]" in config
    assert 'profile_name = "go2x5_flat_tabletop_6d_walk_v11"' in config
    assert "enabled = False" in config
    assert "stages = []" in config
    assert '"name": "S' not in config
    assert "standing_probability = 0.10" in config
    assert "straight_line_probability = 0.40" in config
    assert "turn_in_place_probability = 0.20" in config
    assert "straight_line_min_abs_vx = 0.15" in config
    assert "lin_vel_y = [-0.10, 0.10]" in config
    assert "ang_vel_yaw = [-0.25, 0.25]" in config
    assert "clip_actions = 1.0" in config
    assert "policy_output_tanh = True" in config
    assert "output_tanh = True" in config
    assert "tracking_lin_vel_max = 0.0" in config
    assert "tracking_lin_vel_x_exp = 0.0" in config
    assert "tracking_lin_vel = 2.0" in config
    assert "tracking_ang_vel_yaw_exp = 0.0" in config
    assert "tracking_ang_vel = 1.0" in config
    assert "tracking_sigma = 0.05" in config
    assert "subtract_tracking_static_baseline = True" in config
    assert "only_positive_rewards = True" in config
    assert "mixing_schedule = [1.0, 8000, 4000]" in config
    assert "motion_start_iteration = 8000" in config
    assert "init_std = [[0.15, 0.20, 0.20] * 4]" in config
    assert "entropy_coef = 0.005" in config
    assert "min_policy_std = [[0.08, 0.12, 0.12] * 4]" in config

    assert '"reward/phase_free_actual_airborne_clearance"' in readiness
    assert '"reward/action_saturation_tail_only"' in readiness
    assert "start_iteration == 8000" in readiness
    assert "resampling_time = 10." in config
    assert "collision = -1.0" in config
    assert "action_rate = -0.01" in config
    assert "alive = 0.0" in config
    assert "termination = 0.0" in config
    assert "tracking_ee_sigma = 0.15" in config
    assert "tracking_ee_orientation_sigma = 0.35" in config
    assert "tracking_ee_orn = 0.6" in config
    assert "height_adaptation = -3.0" in config
    assert "pitch_adaptation = -1.0" in config
    assert "max_forward_body_pitch = 0.25" in config
    assert "min_body_height = 0.22" in config
    assert "base_height = 0.0" in config
    assert "stand_still = 0.0" in config
    assert "replace_cylinder_with_capsule = False" in config
    assert '"contract/leg_pd_and_action_scale"' in read(READINESS)


def test_reward_implementations_cover_stop_mask_height_and_jerk_state():
    rewards = read(REWARDS)
    assert rewards.count("reward[~self.env._get_walking_cmd_mask()] = 0") >= 2
    assert 'hasattr(self.env, "last_contact_forces")' in rewards
    assert "self.env.last_contact_forces = self.env.force_sensor_tensor.clone()" in rewards
    assert "clearance_error = torch.clamp(" in rewards
    assert "swing_weight * clearance_error" in rewards
    assert 'getattr(self.env.cfg.rewards, "feet_air_time_target", 0.5)' in rewards
    assert '"feet_air_time_all_feet"' in rewards
    assert "self.env.feet_swing_peak_height" in rewards
    assert "min=0.0" in rewards
    assert "feet_xy_vel_sq = torch.sum(" in rewards
    assert "def _adaptive_body_height_target(self):" in rewards
    assert "def _adaptive_body_pitch_target(self):" in rewards
    assert 'height_adaptation_goal_z_low' in rewards
    assert 'height_adaptation_goal_z_high' in rewards


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
        "self.arm_q_command[env_ids] = self.dof_pos[env_ids, arm_slice]",
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
    assert "entropy_batch[..., :policy_channels]" in ppo


def test_12d_entropy_ignores_the_empty_arm_channel():
    import torch
    from types import SimpleNamespace

    sys.path.insert(0, str(ROOT / "third_party/rsl_rl"))
    from rsl_rl.algorithms.ppo import PPO

    algorithm = PPO.__new__(PPO)
    entropy = torch.tensor([[12.0, 0.0], [6.0, 0.0]])
    algorithm.actor_critic = SimpleNamespace(num_arm_actions=0)
    assert torch.equal(algorithm._mean_policy_entropy(entropy), torch.tensor(9.0))

    algorithm.actor_critic = SimpleNamespace(num_arm_actions=6)
    assert torch.equal(algorithm._mean_policy_entropy(entropy), torch.tensor(4.5))


def test_rollout_advantages_are_normalized_per_reward_channel():
    import torch

    sys.path.insert(0, str(ROOT / "third_party/rsl_rl"))
    from rsl_rl.storage.rollout_storage import RolloutStorage

    storage = RolloutStorage(2, 3, [1], [None], [1], device="cpu")
    storage.rewards[..., 0] = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    )
    storage.rewards[..., 1] = torch.tensor(
        [[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]]
    )
    storage.compute_returns(torch.zeros(2, 2), gamma=0.0, lam=0.0)

    assert torch.allclose(
        storage.advantages.mean(dim=(0, 1)), torch.zeros(2), atol=1e-7
    )
    assert torch.allclose(
        storage.advantages.std(dim=(0, 1)), torch.ones(2), atol=1e-7
    )


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
    assert "self.env.set_training_iteration(it)" in runner
    assert "self.env.load_training_metadata(loaded_dict.get('metadata'))" in runner
    assert '"training_state": {' in env and '"global_steps": int(' in env
    assert 'self.global_steps = int(training_state["global_steps"])' in env
    assert "train_cfg.runner.max_iterations) - int(ppo_runner.current_learning_iteration" in train


def test_weights_only_warm_start_is_explicit_and_fail_closed():
    runner = read(RUNNER)
    train = read(TRAIN)
    helpers = read(HELPERS)
    env = read(ENV)
    assert '"name": "--warm_start_checkpoint"' in helpers
    assert "cannot be combined with --resume or --resumeid" in train
    assert "weights-only warm-start requires an empty output directory" in train
    assert 'warm_start_checkpoint = getattr(args, "warm_start_checkpoint", None)' in train
    assert "ppo_runner.warm_start(warm_start_checkpoint)" in train
    assert "def warm_start(self, path):" in runner
    assert "source_keys != target_keys" in runner
    assert "source_tensor.shape != target_tensor.shape" in runner
    assert "source_tensor.dtype != target_tensor.dtype" in runner
    assert "torch.all(torch.isfinite(source_tensor))" in runner
    assert 'preserved_parameters = ["std"]' in runner
    assert '"optimizer_restored": False' in runner
    assert '"history_optimizer_restored": False' in runner
    assert '"exploration_std_restored": False' in runner
    assert '"new_run_start_iteration": 0' in runner
    assert 'metadata["warm_start"] = dict(self.warm_start_provenance)' in runner
    assert "def validate_warm_start_metadata" in env
    for invariant in (
        '"asset_sha256"',
        '"policy_leg_joint_order"',
        '"action_scale"',
        '"leg_stiffness"',
        '"leg_damping"',
        '"physx"',
        '"ee_frame"',
    ):
        assert invariant in env
    assert 'if int(expected.get("num_arm_actions", 0)) != 0:' in env
    assert 'invariant_contract_fields.append("ik_gain")' in env
    assert "warm-start checkpoint control contract hash is corrupt" in env
    assert '"reward_and_curriculum"' in env
    assert '"policy_output_tanh"' in env


def test_weights_only_warm_start_resets_state_and_rejects_nonfinite():
    import tempfile
    from types import SimpleNamespace

    import torch

    sys.path.insert(0, str(ROOT / "third_party/rsl_rl"))
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))
            self.std = torch.nn.Parameter(torch.full((1, 2), 0.2))

    class TinyEnv:
        cfg = SimpleNamespace(env=SimpleNamespace(require_training_metadata=True))

        @staticmethod
        def validate_warm_start_metadata(metadata, checkpoint_path=None):
            assert metadata["go2x5_alignment"]["schema_version"] == 2
            return {"validated": True}

        @staticmethod
        def get_training_metadata():
            return {"go2x5_alignment": {"schema_version": 2}}

    def make_runner():
        model = TinyModel()
        algorithm = SimpleNamespace(
            actor_critic=model,
            optimizer=torch.optim.Adam(model.parameters()),
            hist_encoder_optimizer=torch.optim.Adam([model.weight]),
            counter=0,
        )
        runner = OnPolicyRunner.__new__(OnPolicyRunner)
        runner.alg = algorithm
        runner.env = TinyEnv()
        runner.device = "cpu"
        runner.current_learning_iteration = 0
        runner.tot_timesteps = 0
        runner.tot_time = 0.0
        runner.warm_start_provenance = None
        return runner

    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = pathlib.Path(directory) / "model_50000.pt"
        source = TinyModel()
        source.weight.data.fill_(3.0)
        source.std.data.fill_(9.0)
        torch.save(
            {
                "model_state_dict": source.state_dict(),
                "iter": 50000,
                "metadata": {
                    "go2x5_alignment": {
                        "schema_version": 2,
                        "control_contract_sha256": "source-contract",
                        "curriculum": {"profile_name": "source-profile"},
                    }
                },
            },
            checkpoint_path,
        )
        runner = make_runner()
        provenance = runner.warm_start(str(checkpoint_path))
        assert torch.all(runner.alg.actor_critic.weight == 3.0)
        assert torch.all(runner.alg.actor_critic.std == 0.2)
        assert runner.alg.optimizer.state == {}
        assert runner.alg.hist_encoder_optimizer.state == {}
        assert provenance["source_iteration"] == 50000
        assert provenance["optimizer_restored"] is False
        assert provenance["exploration_std_restored"] is False

        saved_path = pathlib.Path(directory) / "model_0.pt"
        runner.save(str(saved_path), 0)
        saved = torch.load(saved_path, map_location="cpu")
        assert saved["metadata"]["warm_start"]["source_iteration"] == 50000

        bad_state = source.state_dict()
        bad_state["weight"] = torch.full_like(bad_state["weight"], float("nan"))
        torch.save(
            {
                "model_state_dict": bad_state,
                "iter": 50000,
                "metadata": {"go2x5_alignment": {"schema_version": 2}},
            },
            checkpoint_path,
        )
        try:
            make_runner().warm_start(str(checkpoint_path))
        except FloatingPointError as error:
            assert "Non-finite warm-start tensor weight" in str(error)
        else:
            raise AssertionError("warm-start accepted a non-finite source tensor")


def test_resume_id_uses_normal_path_join():
    registry = read(TASK_REGISTRY)
    assert 'resume_id = str(args.resumeid).strip("/\\\\")' in registry
    assert 'os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, resume_id)' in registry


def test_training_checkpoint_metadata_is_fail_closed():
    env = read(ENV)
    assert 'bool(self.cfg.env.observe_gait_commands)' in env
    assert '("policy_output_tanh", alignment.get("policy_output_tanh"), bool(self.cfg.env.policy_output_tanh))' in env
    assert '"policy_action_clip": float(self.cfg.normalization.clip_actions)' in env
    assert '"Diagnostics/goal_z_base_pitch_correlation"' in env
    assert '"Diagnostics/turn_in_place_yaw_abs_error_radps"' in env
    assert '"ik_task": "pose_6d_weighted_dls"' in env
    assert '"ik_orientation_weight": float(' in env
    assert '"ee_orientation_observation": "local_rpy"' in env
    assert 'else "position_only_translation_3d"' in env
    assert '("observe_gait_commands", alignment.get("observe_gait_commands"), True)' not in env
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
    env = read(ENV)
    assert '"--max-early-resets"' in rollout
    assert '"--require-coordination"' in rollout
    assert 'report["early_resets"] <= report["max_early_resets"]' in rollout
    assert 'report["max_abs_policy_action"] <= 1.000001' in rollout
    assert 'report["goal_z_base_height_correlation"] >= cli.min_height_correlation' in rollout
    assert 'report["goal_z_base_pitch_correlation"] <= cli.max_goal_z_pitch_correlation' in rollout
    assert 'report["mean_yaw_abs_error_radps"] <= cli.max_yaw_abs_error_radps' in rollout
    assert '"mean_collision_raw_per_tick_by_body"' in rollout
    assert "env.penalized_contact_indices" in rollout
    assert '"--privileged-latent"' in rollout
    assert "hist_encoding=not cli.privileged_latent" in rollout
    assert '"--max-mean-collision-raw-per-tick"' in rollout
    assert '"--max-mean-ee-orientation-error-rad"' in rollout
    assert '"mean_ee_orientation_error_rad"' in rollout
    assert "orientation_error(" in rollout
    assert (
        'report["mean_ee_orientation_error_rad"]'
        "\n        <= cli.max_mean_ee_orientation_error_rad"
    ) in rollout
    assert 'report["mean_collision_raw_per_tick"] <= report["max_mean_collision_raw_per_tick"]' in rollout
    assert '"--max-arm-target-clamp-fraction"' in rollout
    assert 'report["arm_target_clamp_fraction"] <= report["max_arm_target_clamp_fraction"]' in rollout
    assert '"--max-action-saturation-fraction"' in rollout
    assert 'report["action_saturation_fraction"] <= report["max_action_saturation_fraction"]' in rollout
    assert '"arm_q_target": env.arm_q_target' in rollout
    assert '"arm_target_clamp_fraction_by_joint"' in rollout
    assert "self.arm_q_target_unclamped = target_base + delta" in env
    assert "self.arm_q_command.copy_(arm_pos_targets)" in env
    assert "delta = torch.clamp(delta, -max_step, max_step)" in env
    assert "self.arm_q_target_clamped = torch.abs(" in env


def test_fixed_command_locomotion_gate_detects_no_step_policies():
    gait = read(FIXED_COMMAND_GAIT)
    assert '("turn_left", 0.0, 0.15)' in gait
    assert '("turn_right", 0.0, -0.15)' in gait
    for option in (
        '"--min-translation-progress-ratio"',
        '"--min-yaw-progress-ratio"',
        '"--max-swing-contact-fraction"',
        '"--min-swing-height"',
        '"--max-stand-vx-error"',
        '"--max-stand-yaw-error"',
        '"--max-moving-vx-error"',
        '"--max-moving-yaw-error"',
        '"--max-collision-raw-mean"',
        '"--require-gait-shape"',
    ):
        assert option in gait
    assert "vx_abs_error_mean <= cli.max_stand_vx_error" in gait
    assert "yaw_abs_error_mean <= cli.max_stand_yaw_error" in gait
    assert "vx_abs_error_mean <= cli.max_moving_vx_error" in gait
    assert "yaw_abs_error_mean <= cli.max_moving_yaw_error" in gait
    assert "foot_cache_max_error <= 1.0e-7" in gait
    assert 'reset_causes = {"roll": 0, "pitch": 0, "z": 0, "contact": 0}' in gait
    assert 'totals["collision"] / samples <= cli.max_collision_raw_mean' in gait
    assert "tracking_passed = all(tracking_checks)" in gait
    assert "not cli.require_gait_shape or gait_shape_passed" in gait
    assert '"gait_shape_evaluated": bool(cli.require_gait_shape)' in gait
    assert '"behavior_passed": behavior_passed' in gait
    assert 'return 0 if report["passed"] else 1' in gait


def test_training_readiness_checks_static_full_task():
    readiness = read(READINESS)
    assert '"curriculum/static_distribution"' in readiness
    assert '"task_geometry/tabletop_volume_covered"' in readiness
    assert '"arm_schedule/frozen_before_start"' in readiness
    assert '"arm_schedule/enabled_at_start"' in readiness
    assert "object_root_x_at_approach" in readiness
    assert '"ik/full_6d_uses_weighted_jacobian"' in readiness
    assert '"observation/orientation_command_is_live"' in readiness
    assert 'reset_causes = {"roll": 0, "pitch": 0, "z": 0, "contact": 0}' in readiness
    assert 'early_resets == 0' in readiness
    assert 'phase="frozen_arm_training_stage"' in readiness
    assert 'require_no_early_reset=True' in readiness
    assert 'phase="full_arm_untrained_diagnostic"' in readiness
    assert 'require_no_early_reset=False' in readiness
    assert "env.set_training_iteration(start_iteration - 1)" in readiness
    assert "env.set_training_iteration(None)" in readiness
    run_body = readiness[readiness.index("def run(cli):"):]
    assert run_body.index("probe_rollout(") < run_body.index("probe_curriculum(")


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("go2x5 training readiness tests passed")
