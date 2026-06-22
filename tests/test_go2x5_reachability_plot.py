import importlib.util
import pathlib
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
PLOT_PATH = ROOT / "low-level/legged_gym/scripts/plot_go2x5_ik_reachability.py"


def load_plot_module():
    spec = importlib.util.spec_from_file_location("plot_go2x5_ik_reachability", PLOT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reachability_plot_outputs_summary_and_svg(tmp_path):
    _run_reachability_plot_check(tmp_path)


def _run_reachability_plot_check(tmp_path):
    plot = load_plot_module()
    csv_path = tmp_path / "reachability.csv"
    csv_path.write_text(
        "\n".join(
            [
                "target_x,target_y,target_z,success,raw_ik_success,pos_err,limit_hits,collision",
                "0.10,-0.05,0.20,1,1,0.010,0,0",
                "0.20,-0.05,0.20,1,1,0.012,0,0",
                "0.10,0.05,0.30,1,1,0.011,0,0",
                "0.20,0.05,0.30,1,1,0.013,0,0",
                "0.30,0.10,0.40,0,0,0.180,0,0",
                "0.34,0.12,0.42,0,1,0.018,1,0",
            ]
        ),
        encoding="utf-8",
    )

    rows = plot.load_rows(csv_path)
    summary = plot.summarize(rows, csv_path, trim_fraction=0.0)
    svg_path = tmp_path / "reachability.svg"
    plot.write_svg(rows, summary, svg_path)

    assert summary["total"] == 6
    assert summary["success_count"] == 4
    assert summary["raw_ik_success_count"] == 5
    assert summary["failed_after_raw_ik_count"] == 1
    assert summary["success_bounds"] == {
        "x": [0.1, 0.2],
        "y": [-0.05, 0.05],
        "z": [0.2, 0.3],
    }
    assert svg_path.exists()
    text = svg_path.read_text(encoding="utf-8")
    assert "Go2-X5 IK reachability" in text
    assert "success bounds" in text


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        _run_reachability_plot_check(pathlib.Path(directory))
    print("go2x5 reachability plot tests passed")
