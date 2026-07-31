import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "high-level/demo_go2x5_scripted_pick.py"
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


if __name__ == "__main__":
    test_phase_schedule_is_contiguous_and_complete()
    test_pick_trace_requires_sustained_lift_and_two_finger_contact()
    print("Go2-X5 scripted pick tests passed")
