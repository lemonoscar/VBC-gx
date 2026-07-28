import importlib.util
import contextlib
import io
import pathlib
import sys
import tempfile
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "high-level/evaluate_go2x5_teacher.py"
CONFIG_UTIL = ROOT / "high-level/utils/config.py"
TRAINER = ROOT / "high-level/train_multistate.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "evaluate_go2x5_teacher", EVALUATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_uses_task_success_and_exclusive_outcomes():
    evaluator = load_evaluator()
    records = [
        {
            "outcome": "success",
            "return": 10.0,
            "length": 50,
            "min_ee_object_distance_m": 0.05,
            "max_lift_margin_m": 0.20,
            "finger_contact": True,
            "gripper_closed": True,
            "reached": True,
            "lifted": True,
        },
        {
            "outcome": "timeout",
            "return": 2.0,
            "length": 150,
            "min_ee_object_distance_m": 0.20,
            "max_lift_margin_m": 0.01,
            "finger_contact": False,
            "gripper_closed": True,
            "reached": False,
            "lifted": False,
        },
    ]
    summary = evaluator.summarize_episode_records(records)
    assert summary["episodes"] == 2
    assert summary["successes"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["outcomes"] == {
        "success": 1,
        "timeout": 1,
        "object_fall": 0,
        "other": 0,
    }
    assert summary["reach_rate"] == 0.5
    assert summary["lift_rate"] == 0.5
    assert summary["mean_episode_length"] == 100.0


def test_trainer_arguments_are_deterministic_and_match_training_observations():
    evaluator = load_evaluator()
    args = SimpleNamespace(
        config="data/cfg/go2x5_pickmulti.yaml",
        low_policy_path="/tmp/low.pt",
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        num_envs=264,
        seed=43,
        checkpoint="/tmp/agent_60000.pt",
    )
    argv = evaluator.build_trainer_argv(args, "/tmp/eval")
    for option in (
        "--headless",
        "--roboinfo",
        "--small_value_set_zero",
        "--stop_pick",
    ):
        assert option in argv
    assert "--rand_control" not in argv
    assert "--wandb" not in argv
    assert "--checkpoint" not in argv
    assert argv[argv.index("--num_envs") + 1] == "264"
    assert argv[argv.index("--seed") + 1] == "43"


def test_cli_validation_fails_closed_and_does_not_overwrite():
    evaluator = load_evaluator()
    with tempfile.TemporaryDirectory() as directory:
        directory = pathlib.Path(directory)
        checkpoint = directory / "agent_60000.pt"
        low_policy = directory / "model_45000.pt"
        config = directory / "config.yaml"
        output = directory / "report.json"
        for path in (checkpoint, low_policy, config):
            path.write_bytes(b"x")

        args = evaluator.parse_args([
            "--checkpoint", str(checkpoint),
            "--low-policy-path", str(low_policy),
            "--config", str(config),
            "--output", str(output),
            "--num-envs", "264",
            "--episodes-per-env", "5",
        ])
        assert args.num_envs == 264
        assert args.episodes_per_env == 5

        output.write_text("user data", encoding="utf-8")
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                evaluator.parse_args([
                    "--checkpoint", str(checkpoint),
                    "--low-policy-path", str(low_policy),
                    "--config", str(config),
                    "--output", str(output),
                ])
        except SystemExit as error:
            assert error.code != 0
        else:
            raise AssertionError("evaluator accepted an existing output path")
        assert output.read_text(encoding="utf-8") == "user data"


def test_training_argument_parser_accepts_explicit_argv():
    sys.path.insert(0, str(ROOT / "high-level"))
    try:
        from utils.config import get_params

        args = get_params(["--task", "Go2X5PickMulti", "--seed", "17"])
        assert args.task == "Go2X5PickMulti"
        assert args.seed == 17
    finally:
        sys.path.pop(0)

    trainer_source = TRAINER.read_text(encoding="utf-8")
    assert "def get_trainer(is_eval=False, args=None):" in trainer_source
    assert "if args is None:" in trainer_source


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("go2x5 teacher evaluation tests passed")
