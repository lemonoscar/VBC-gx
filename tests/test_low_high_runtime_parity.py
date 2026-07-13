import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.go2x5_runtime_parity import compare_snapshots, read_snapshot, write_snapshot


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


if __name__ == "__main__":
    import tempfile

    test_equal_snapshots_ignore_side_label()
    test_numeric_mismatch_has_path_and_error()
    test_tolerance_accepts_small_numeric_error()
    with tempfile.TemporaryDirectory() as directory:
        temp_path = pathlib.Path(directory)
        test_snapshot_json_round_trip(temp_path)
        test_compare_cli_writes_machine_readable_report(temp_path)
    print("runtime parity tests passed")
