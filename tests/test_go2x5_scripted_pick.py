import importlib.util
import ast
import tempfile
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "high-level/demo_go2x5_scripted_pick.py"
PICKMULTI = ROOT / "high-level/envs/b1z1_pickmulti.py"
SPEC = importlib.util.spec_from_file_location("go2x5_scripted_pick", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_phase_schedule_is_contiguous_and_complete():
    expected = []
    for name, duration in MODULE.PHASE_STEPS:
        expected.extend([name] * duration)
    actual = [MODULE.phase_at(step)[0] for step in range(len(expected))]
    assert actual == expected

    try:
        MODULE.phase_at(len(expected))
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-range schedule step must fail")


def test_pick_trace_requires_sustained_lift_and_two_finger_contact():
    result = MODULE.evaluate_pick_trace(
        [0.0, 0.10, 0.11, 0.12, 0.12, 0.11, 0.10],
        [0.20, 0.08, 0.07, 0.06, 0.06, 0.07, 0.08],
        [[0.0, 0.0], [0.7, 0.8]] + [[0.1, 0.1]] * 5,
        required_hold_steps=6,
    )
    assert result["passed"]
    assert result["longest_lift_hold_steps"] == 6
    assert result["simultaneous_two_finger_contact_steps"] == 1

    one_finger = MODULE.evaluate_pick_trace(
        [0.11] * 6,
        [0.05] * 6,
        [[1.0, 0.1], [0.1, 1.0]] * 3,
        required_hold_steps=6,
    )
    assert not one_finger["passed"]
    assert one_finger["lift_hold_passed"]
    assert not one_finger["two_finger_contact_passed"]


def test_preclose_is_distance_gated_during_descent():
    assert not MODULE.should_close_gripper("approach", 0.01, 0.13)
    assert not MODULE.should_close_gripper("descend", 0.14, 0.13)
    assert MODULE.should_close_gripper("descend", 0.13, 0.13)
    assert MODULE.should_close_gripper("close", 1.0, 0.0)
    assert MODULE.update_gripper_latch(False, "descend", 0.12, 0.13)
    assert MODULE.update_gripper_latch(True, "descend", 1.0, 0.13)
    assert not MODULE.update_gripper_latch(False, "descend", 1.0, 0.13)


def test_object_offset_cli_defaults_to_table_center():
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        checkpoint = tmp_path / "checkpoint.pt"
        config = tmp_path / "config.yaml"
        checkpoint.touch()
        config.touch()
        args = MODULE.parse_args(
            [
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(config),
                "--report",
                str(tmp_path / "report.json"),
            ]
        )
        assert args.object_x_offset == 0.0
        assert args.object_y_offset == 0.0
        assert args.target_roll == 0.0
        assert args.target_pitch == 1.25
        assert args.target_yaw == 0.0


def test_fixed_table_height_is_available_before_actor_creation():
    tree = ast.parse(PICKMULTI.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "B1Z1PickMulti"
    )
    init_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assignment = next(
        node
        for node in ast.walk(init_node)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "table_heights_fix"
            for target in node.targets
        )
    )
    super_call = next(
        node
        for node in ast.walk(init_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__init__"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
    )
    assert assignment.lineno < super_call.lineno


def test_robot_table_contacts_are_identified_by_body_pair():
    body_names = ["base", "FR_foot", "arm_link7"]
    contacts = [
        {"body0": 3, "body1": 1},
        {"body0": 2, "body1": 3},
        {"body0": 3, "body1": 4},
        SimpleNamespace(body0=0, body1=-1),
    ]
    assert MODULE.robot_table_contact_names(contacts, 3, body_names) == [
        "FR_foot",
        "arm_link7",
    ]


def test_scripted_demo_has_a_bounded_forward_pose_hold():
    assert MODULE.BASE_HOLD_GAIN > 0.0
    assert 0.05 < MODULE.BASE_HOLD_MAX_SPEED <= 0.15
    assert 0.0 < MODULE.MAX_BASE_FORWARD_DRIFT < 0.10


if __name__ == "__main__":
    test_phase_schedule_is_contiguous_and_complete()
    test_pick_trace_requires_sustained_lift_and_two_finger_contact()
    test_preclose_is_distance_gated_during_descent()
    test_object_offset_cli_defaults_to_table_center()
    test_fixed_table_height_is_available_before_actor_creation()
    test_robot_table_contacts_are_identified_by_body_pair()
    test_scripted_demo_has_a_bounded_forward_pose_hold()
    print("Go2-X5 scripted pick tests passed")
