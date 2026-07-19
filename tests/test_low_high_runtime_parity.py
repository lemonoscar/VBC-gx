import json
import importlib.util
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.go2x5_runtime_parity import (
    CONTROLLER_CASES,
    EXPECTED_POLICY_TO_URDF,
    PROBE_ACTION_POLICY_ORDER,
    canonical_json_sha256,
    compare_snapshots,
    ee_frame_oracle,
    independent_pd_oracle,
    make_diagnostic_policy,
    nonfinite_details,
    policy_to_urdf_oracle,
    read_snapshot,
    validate_schema_v2_checkpoint,
    write_snapshot,
)


def controller_snapshot(side="low", torque=1.0):
    return {
        "schema_version": 1,
        "kind": "controller_state",
        "side": side,
        "current_proprio": [0.0, 1.0],
        "history": [0.0, 1.0, 0.0, 1.0],
        "gait_phase": 0.25,
        "gait_clock": [1.0, -1.0, -1.0, 1.0],
        "policy_action": [0.1] * 12,
        "urdf_action": [0.1] * 12,
        "leg_torque": [torque] * 12,
        "arm_q_target": [0.0] * 6,
        "ee_goal_world": [0.4, 0.0, 0.6],
        "ee_goal_local": [0.3, 0.0, 0.2],
        "arm_base_world": [0.1, 0.0, 0.4],
    }


def test_equal_snapshots_ignore_side_label():
    assert compare_snapshots(controller_snapshot("low"), controller_snapshot("high")) == []


def test_numeric_mismatch_has_path_and_error():
    mismatches = compare_snapshots(controller_snapshot(torque=1.0), controller_snapshot("high", torque=1.01))
    assert len(mismatches) == 12
    assert mismatches[0]["path"].startswith("leg_torque[")
    assert abs(mismatches[0]["abs_error"] - 0.01) < 1.0e-12


def test_tolerance_accepts_small_numeric_error():
    assert compare_snapshots(controller_snapshot(torque=1.0), controller_snapshot("high", torque=1.000001), atol=1.1e-6) == []


def test_tolerance_boundary_is_inclusive():
    assert compare_snapshots(controller_snapshot(torque=1.0), controller_snapshot("high", torque=1.000001), atol=1.0e-6) == []


def test_constant_probe_is_asymmetric_policy_order():
    policy, metadata = make_diagnostic_policy("constant_probe", obs_dim=7)
    output = policy(np.zeros((2, 7), dtype=np.float32))
    assert output.shape == (2, 12)
    assert output[0].tolist() == np.asarray(PROBE_ACTION_POLICY_ORDER, dtype=np.float32).tolist()
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch_output = policy(torch.zeros((2, 7), dtype=torch.float32))
        assert torch_output.shape == (2, 12)
        assert torch.equal(
            torch_output[0], torch.tensor(PROBE_ACTION_POLICY_ORDER, dtype=torch.float32)
        )
    assert metadata == {"mode": "constant_probe", "input_dim": 7, "output_dim": 12}


def test_linear_probe_is_seeded_nonzero_and_reproducible():
    obs = np.linspace(-1.0, 1.0, 22, dtype=np.float32).reshape(2, 11)
    first, first_meta = make_diagnostic_policy("linear_probe", 11, seed=17, scale=0.05)
    second, second_meta = make_diagnostic_policy("linear_probe", 11, seed=17, scale=0.05)
    assert np.array_equal(first(obs), second(obs))
    assert np.max(np.abs(first(obs))) > 1.0e-6
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch_obs = torch.from_numpy(obs)
        assert torch.equal(first(torch_obs), second(torch_obs))
        assert float(torch.max(torch.abs(first(torch_obs)))) > 1.0e-6
    assert first_meta == second_meta == {
        "mode": "linear_probe", "input_dim": 11, "output_dim": 12, "seed": 17, "scale": 0.05
    }


def test_action_permutation_oracle_is_name_derived():
    policy = np.arange(12, dtype=np.float64)
    assert policy_to_urdf_oracle(policy).tolist() == policy[EXPECTED_POLICY_TO_URDF].tolist()


def test_independent_pd_oracle_clamps_and_reports_joint_inputs():
    result = independent_pd_oracle(
        q=np.zeros(12), qd=np.arange(12) * 0.1, default_q=np.zeros(12),
        action_scale=np.ones(12), action_urdf=np.ones(12), kp=np.full(12, 2.0),
        kd=np.ones(12), torque_limit=np.full(12, 1.5),
    )
    assert result["torque"][0] == 1.5
    assert abs(result["torque"][-1] - 0.9) < 1.0e-12
    assert result["joints"][0]["name"] == "FL_hip_joint"
    assert result["joints"][0]["raw_torque"] == 2.0


def test_ee_frame_oracle_uses_yaw_world_and_full_base_inverse():
    yaw_quaternion = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    result = ee_frame_oracle(
        base_position=[1.0, 2.0, 0.5], base_quaternion=yaw_quaternion,
        arm_base_offset=[0.1, 0.0, 0.2], terrain_center=[1.0, 2.0, 0.4],
        ee_goal_local=[0.3, -0.1, 0.2],
    )
    assert np.allclose(result["arm_base_world"], [1.0, 2.1, 0.7], atol=1.0e-12)
    assert np.allclose(result["ee_goal_world"], [1.1, 2.3, 0.6], atol=1.0e-12)
    assert np.allclose(result["reconstructed_local"], [0.2, -0.1, -0.1], atol=1.0e-12)


def test_nonfinite_hard_fail_finds_first_index():
    counts, failures = nonfinite_details({"obs": [0.0, float("nan")], "torque": [float("inf")]})
    assert counts == {"obs": 1, "torque": 1}
    assert failures[0]["path"] == "obs" and failures[0]["index"] == [1]
    bad = controller_snapshot()
    bad["leg_torque"][0] = float("nan")
    mismatch = compare_snapshots(bad, bad)
    assert mismatch[0]["reason"] == "nonfinite"


def test_case_registry_has_required_cases():
    assert list(CONTROLLER_CASES) == ["C0", "C1", "C2", "C3", "C4"]


def test_control_contract_hash_is_canonical_and_checkpoint_rejects_18d():
    contract = {"z": [1, 2], "a": {"b": True}}
    assert canonical_json_sha256(contract) == canonical_json_sha256({"a": {"b": True}, "z": [1, 2]})
    checkpoint = {
        "model_state_dict": {"std": np.zeros((1, 18))},
        "metadata": {"go2x5_alignment": {
            "schema_version": 2, "action_dim": 18, "num_arm_actions": 6,
            "control_contract": contract, "control_contract_sha256": canonical_json_sha256(contract),
        }},
    }
    try:
        validate_schema_v2_checkpoint(checkpoint)
    except ValueError as error:
        assert "action_dim" in str(error)
    else:
        raise AssertionError("18D checkpoint was accepted")


def test_schema_v2_smoke_checkpoint_metadata_shape_and_purpose():
    contract = {"action_scale": [0.1] * 12}
    checkpoint = {
        "model_state_dict": {"std": np.zeros((1, 12))},
        "metadata": {
            "purpose": "runtime_parity_smoke_only", "trained": False,
            "go2x5_alignment": {
                "schema_version": 2, "action_dim": 12, "num_arm_actions": 0,
                "purpose": "parity_smoke", "contract_profile": "simple_deployment_smoke_v1",
                "control_contract": contract,
                "control_contract_sha256": canonical_json_sha256(contract),
            },
        },
    }
    validate_schema_v2_checkpoint(checkpoint)
    assert checkpoint["metadata"]["purpose"] == "runtime_parity_smoke_only"
    assert checkpoint["metadata"]["trained"] is False


def test_robot_start_pose_precedence_and_b1_fallback():
    module_path = ROOT / "high-level/envs/runtime_contract.py"
    spec = importlib.util.spec_from_file_location("runtime_contract", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = {"robotStartPose": [-1.0, 0.0, 0.4], "evalRobotStartPose": [-0.5, 0.0, 0.45]}
    assert module.resolve_robot_start_pose(cfg, None, False) == (-1.0, 0.0, 0.4)
    assert module.resolve_robot_start_pose(cfg, None, True) == (-0.5, 0.0, 0.45)
    assert module.resolve_robot_start_pose(cfg, [3, 2, 1], False) == (3, 2, 1)
    assert module.resolve_robot_start_pose(cfg, [3, 2, 1], True) == (3, 2, 1)
    assert module.resolve_robot_start_pose({"robotStartPose": [1, 2, 3]}, None, True) == (1, 2, 3)


def test_snapshot_json_round_trip(tmp_path):
    path = tmp_path / "snapshot.json"
    snapshot = controller_snapshot()
    write_snapshot(snapshot, path)
    assert read_snapshot(path) == snapshot


def test_compare_cli_writes_machine_readable_report(tmp_path):
    low_path = tmp_path / "low.json"
    high_path = tmp_path / "high.json"
    report_path = tmp_path / "report.json"
    write_snapshot(controller_snapshot("low"), low_path)
    write_snapshot(controller_snapshot("high"), high_path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_go2x5_runtime_parity.py"),
            "compare",
            "--low", str(low_path),
            "--high", str(high_path),
            "--report", str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["mismatch_count"] == 0


def test_compare_cli_mismatch_exit_code(tmp_path):
    low_path, high_path = tmp_path / "low.json", tmp_path / "high.json"
    write_snapshot(controller_snapshot("low"), low_path)
    write_snapshot(controller_snapshot("high", torque=2.0), high_path)
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/check_go2x5_runtime_parity.py"), "compare",
        "--low", str(low_path), "--high", str(high_path),
    ], check=False, capture_output=True, text=True)
    assert result.returncode == 1


if __name__ == "__main__":
    import tempfile

    test_equal_snapshots_ignore_side_label()
    test_numeric_mismatch_has_path_and_error()
    test_tolerance_accepts_small_numeric_error()
    test_tolerance_boundary_is_inclusive()
    test_constant_probe_is_asymmetric_policy_order()
    test_linear_probe_is_seeded_nonzero_and_reproducible()
    test_action_permutation_oracle_is_name_derived()
    test_independent_pd_oracle_clamps_and_reports_joint_inputs()
    test_ee_frame_oracle_uses_yaw_world_and_full_base_inverse()
    test_nonfinite_hard_fail_finds_first_index()
    test_case_registry_has_required_cases()
    test_control_contract_hash_is_canonical_and_checkpoint_rejects_18d()
    test_schema_v2_smoke_checkpoint_metadata_shape_and_purpose()
    test_robot_start_pose_precedence_and_b1_fallback()
    with tempfile.TemporaryDirectory() as directory:
        temp_path = pathlib.Path(directory)
        test_snapshot_json_round_trip(temp_path)
        test_compare_cli_writes_machine_readable_report(temp_path)
        test_compare_cli_mismatch_exit_code(temp_path)
    print("runtime parity tests passed")
