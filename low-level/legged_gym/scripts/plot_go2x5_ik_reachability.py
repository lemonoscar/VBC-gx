#!/usr/bin/env python3
"""Visualize Go2-X5 IK reachability CSV outputs.

This script consumes the CSV written by scan_go2x5_ik_reachability.py. It does
not start Isaac Gym and is safe to run on a machine without a GPU.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path


AXES = ("x", "y", "z")
TARGET_COLUMNS = {axis: f"target_{axis}" for axis in AXES}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Reachability CSV from scan_go2x5_ik_reachability.py.")
    parser.add_argument("--out_dir", default=None, help="Output directory. Defaults to the CSV directory.")
    parser.add_argument("--prefix", default=None, help="Output file prefix. Defaults to '<csv-stem>_reachability'.")
    parser.add_argument(
        "--trim_fraction",
        type=float,
        default=0.10,
        help="Per-axis fraction trimmed from successful samples to draw a conservative box.",
    )
    parser.add_argument(
        "--no_matplotlib",
        action="store_true",
        help="Skip optional 3D PNG generation and only write SVG/JSON outputs.",
    )
    return parser.parse_args()


def _float(row, key):
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def _flag(row, key):
    return str(row.get(key, "0")).strip().lower() in {"1", "true", "yes"}


def load_rows(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = [column for column in TARGET_COLUMNS.values() if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required CSV columns: {missing}")

        rows = []
        for raw in reader:
            row = dict(raw)
            for axis, column in TARGET_COLUMNS.items():
                row[axis] = _float(row, column)
            row["success_flag"] = _flag(row, "success")
            row["raw_ik_success_flag"] = _flag(row, "raw_ik_success")
            row["limit_hits_value"] = int(float(row.get("limit_hits", 0) or 0))
            row["collision_flag"] = _flag(row, "collision")
            row["pos_err_value"] = _float(row, "pos_err") if "pos_err" in row else float("nan")
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def _finite(values):
    return [value for value in values if math.isfinite(value)]


def bounds_for(rows):
    if not rows:
        return None
    bounds = {}
    for axis in AXES:
        values = _finite([row[axis] for row in rows])
        if not values:
            return None
        bounds[axis] = [min(values), max(values)]
    return bounds


def _quantile(values, q):
    values = sorted(_finite(values))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return values[low]
    weight = pos - low
    return values[low] * (1.0 - weight) + values[high] * weight


def conservative_bounds(success_rows, trim_fraction):
    if not success_rows:
        return None
    trim_fraction = min(max(trim_fraction, 0.0), 0.49)
    bounds = {}
    for axis in AXES:
        values = [row[axis] for row in success_rows]
        low = _quantile(values, trim_fraction)
        high = _quantile(values, 1.0 - trim_fraction)
        if low is None or high is None:
            return None
        bounds[axis] = [low, high]
    return bounds


def summarize(rows, csv_path, trim_fraction):
    success_rows = [row for row in rows if row["success_flag"]]
    raw_success_rows = [row for row in rows if row["raw_ik_success_flag"]]
    limit_hit_rows = [row for row in rows if row["limit_hits_value"] > 0]
    collision_rows = [row for row in rows if row["collision_flag"]]
    failed_after_raw_ik = [
        row for row in rows if row["raw_ik_success_flag"] and not row["success_flag"]
    ]
    success_pos_errs = _finite([row["pos_err_value"] for row in success_rows])
    failed_pos_errs = _finite([row["pos_err_value"] for row in rows if not row["success_flag"]])

    def mean(values):
        return sum(values) / len(values) if values else None

    return {
        "csv": os.path.abspath(csv_path),
        "frame": "arm-base local frame",
        "units": "meters/radians",
        "total": len(rows),
        "success_count": len(success_rows),
        "success_rate": len(success_rows) / len(rows),
        "raw_ik_success_count": len(raw_success_rows),
        "raw_ik_success_rate": len(raw_success_rows) / len(rows),
        "failed_after_raw_ik_count": len(failed_after_raw_ik),
        "limit_hit_count": len(limit_hit_rows),
        "collision_count": len(collision_rows),
        "all_target_bounds": bounds_for(rows),
        "success_bounds": bounds_for(success_rows),
        "conservative_bounds": conservative_bounds(success_rows, trim_fraction),
        "trim_fraction": trim_fraction,
        "mean_success_pos_err": mean(success_pos_errs),
        "mean_failed_pos_err": mean(failed_pos_errs),
    }


def _fmt(value):
    if value is None:
        return "null"
    return f"{value:.4f}"


def _point_style(row):
    if row["success_flag"]:
        return "#238443", 3.4
    if row["raw_ik_success_flag"]:
        return "#fdae61", 3.0
    return "#d73027", 2.6


def write_svg(rows, summary, out_path):
    panels = [
        ("XY projection", "x", "y", "x forward", "y lateral"),
        ("XZ projection", "x", "z", "x forward", "z up"),
        ("YZ projection", "y", "z", "y lateral", "z up"),
    ]
    panel_w = 460
    panel_h = 390
    pad_left = 58
    pad_right = 22
    pad_top = 54
    pad_bottom = 54
    width = panel_w * len(panels)
    height = panel_h + 126

    all_bounds = bounds_for(rows)

    def axis_range(axis):
        lo, hi = all_bounds[axis]
        if lo == hi:
            lo -= 0.05
            hi += 0.05
        span = hi - lo
        return lo - span * 0.08, hi + span * 0.08

    axis_ranges = {axis: axis_range(axis) for axis in AXES}

    def sx(panel_index, axis, value):
        lo, hi = axis_ranges[axis]
        plot_w = panel_w - pad_left - pad_right
        return panel_index * panel_w + pad_left + (value - lo) / (hi - lo) * plot_w

    def sy(axis, value):
        lo, hi = axis_ranges[axis]
        plot_h = panel_h - pad_top - pad_bottom
        return pad_top + (hi - value) / (hi - lo) * plot_h

    def rect_for_bounds(panel_index, axis_a, axis_b, bounds, color, dash=""):
        if not bounds:
            return ""
        a0, a1 = bounds[axis_a]
        b0, b1 = bounds[axis_b]
        x0 = sx(panel_index, axis_a, a0)
        x1 = sx(panel_index, axis_a, a1)
        y0 = sy(axis_b, b1)
        y1 = sy(axis_b, b0)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<rect x="{min(x0, x1):.2f}" y="{min(y0, y1):.2f}" '
            f'width="{abs(x1 - x0):.2f}" height="{abs(y1 - y0):.2f}" '
            f'fill="none" stroke="{color}" stroke-width="2.0"{dash_attr}/>'
        )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:12px}.label{font-size:13px}.title{font-size:16px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="22" y="30" class="title">Go2-X5 IK reachability</text>',
        (
            f'<text x="22" y="54" class="small">success={summary["success_count"]}/{summary["total"]} '
            f'({summary["success_rate"]:.1%}), raw IK success={summary["raw_ik_success_rate"]:.1%}, '
            f'frame=arm-base local</text>'
        ),
        '<circle cx="24" cy="78" r="4" fill="#238443"/><text x="36" y="82" class="small">success</text>',
        '<circle cx="108" cy="78" r="4" fill="#fdae61"/><text x="120" y="82" class="small">IK reached but rejected by limit/contact</text>',
        '<circle cx="354" cy="78" r="4" fill="#d73027"/><text x="366" y="82" class="small">not reached</text>',
        '<rect x="466" y="70" width="24" height="14" fill="none" stroke="#2166ac" stroke-width="2"/><text x="498" y="82" class="small">success bounds</text>',
        '<rect x="620" y="70" width="24" height="14" fill="none" stroke="#762a83" stroke-width="2" stroke-dasharray="5 4"/><text x="652" y="82" class="small">trimmed conservative bounds</text>',
    ]

    top_offset = 100
    for panel_index, (title, axis_a, axis_b, xlabel, ylabel) in enumerate(panels):
        x0 = panel_index * panel_w
        elements.append(f'<g transform="translate(0,{top_offset})">')
        elements.append(f'<text x="{x0 + 20}" y="24" class="title">{title}</text>')
        elements.append(
            f'<rect x="{x0 + pad_left}" y="{pad_top}" width="{panel_w - pad_left - pad_right}" '
            f'height="{panel_h - pad_top - pad_bottom}" fill="#f7f7f7" stroke="#d9d9d9"/>'
        )
        elements.append(rect_for_bounds(panel_index, axis_a, axis_b, summary["success_bounds"], "#2166ac"))
        elements.append(rect_for_bounds(panel_index, axis_a, axis_b, summary["conservative_bounds"], "#762a83", "5 4"))
        for row in rows:
            color, radius = _point_style(row)
            elements.append(
                f'<circle cx="{sx(panel_index, axis_a, row[axis_a]):.2f}" '
                f'cy="{sy(axis_b, row[axis_b]):.2f}" r="{radius}" fill="{color}" fill-opacity="0.78"/>'
            )
        alo, ahi = axis_ranges[axis_a]
        blo, bhi = axis_ranges[axis_b]
        elements.append(f'<text x="{x0 + panel_w / 2:.1f}" y="{panel_h - 14}" text-anchor="middle" class="label">{xlabel}</text>')
        elements.append(
            f'<text transform="translate({x0 + 18},{pad_top + (panel_h - pad_top - pad_bottom) / 2:.1f}) rotate(-90)" '
            f'text-anchor="middle" class="label">{ylabel}</text>'
        )
        elements.append(f'<text x="{x0 + pad_left}" y="{panel_h - 34}" class="small">{alo:.2f}</text>')
        elements.append(f'<text x="{x0 + panel_w - pad_right - 34}" y="{panel_h - 34}" class="small">{ahi:.2f}</text>')
        elements.append(f'<text x="{x0 + 24}" y="{pad_top + 4}" class="small">{bhi:.2f}</text>')
        elements.append(f'<text x="{x0 + 24}" y="{panel_h - pad_bottom}" class="small">{blo:.2f}</text>')
        elements.append("</g>")

    success_bounds = summary["success_bounds"] or {}
    conservative = summary["conservative_bounds"] or {}
    info_y = top_offset + panel_h + 16
    elements.append(
        f'<text x="22" y="{info_y}" class="small">success bounds: '
        f'x=[{_fmt(success_bounds.get("x", [None, None])[0])}, {_fmt(success_bounds.get("x", [None, None])[1])}], '
        f'y=[{_fmt(success_bounds.get("y", [None, None])[0])}, {_fmt(success_bounds.get("y", [None, None])[1])}], '
        f'z=[{_fmt(success_bounds.get("z", [None, None])[0])}, {_fmt(success_bounds.get("z", [None, None])[1])}]</text>'
    )
    elements.append(
        f'<text x="22" y="{info_y + 22}" class="small">conservative bounds: '
        f'x=[{_fmt(conservative.get("x", [None, None])[0])}, {_fmt(conservative.get("x", [None, None])[1])}], '
        f'y=[{_fmt(conservative.get("y", [None, None])[0])}, {_fmt(conservative.get("y", [None, None])[1])}], '
        f'z=[{_fmt(conservative.get("z", [None, None])[0])}, {_fmt(conservative.get("z", [None, None])[1])}]</text>'
    )
    elements.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as file:
        file.write("\n".join(elements))


def _bbox_edges(bounds):
    if not bounds:
        return []
    xs, ys, zs = bounds["x"], bounds["y"], bounds["z"]
    corners = [
        (xs[0], ys[0], zs[0]),
        (xs[1], ys[0], zs[0]),
        (xs[1], ys[1], zs[0]),
        (xs[0], ys[1], zs[0]),
        (xs[0], ys[0], zs[1]),
        (xs[1], ys[0], zs[1]),
        (xs[1], ys[1], zs[1]),
        (xs[0], ys[1], zs[1]),
    ]
    edge_indices = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return [(corners[a], corners[b]) for a, b in edge_indices]


def try_write_matplotlib(rows, summary, out_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local environment.
        return str(exc)

    fig = plt.figure(figsize=(8.2, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    groups = [
        ("not reached", [row for row in rows if not row["raw_ik_success_flag"]], "#d73027", 24, 0.32),
        (
            "IK reached, rejected",
            [row for row in rows if row["raw_ik_success_flag"] and not row["success_flag"]],
            "#fdae61",
            28,
            0.58,
        ),
        ("success", [row for row in rows if row["success_flag"]], "#238443", 34, 0.86),
    ]
    for label, group_rows, color, size, alpha in groups:
        if not group_rows:
            continue
        ax.scatter(
            [row["x"] for row in group_rows],
            [row["y"] for row in group_rows],
            [row["z"] for row in group_rows],
            s=size,
            c=color,
            alpha=alpha,
            label=label,
            depthshade=False,
        )

    for start, end in _bbox_edges(summary["success_bounds"]):
        ax.plot(*zip(start, end), color="#2166ac", linewidth=1.8)
    for start, end in _bbox_edges(summary["conservative_bounds"]):
        ax.plot(*zip(start, end), color="#762a83", linewidth=1.6, linestyle="--")

    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y lateral (m)")
    ax.set_zlabel("z up (m)")
    ax.set_title("Go2-X5 IK reachability in arm-base frame")
    ax.legend(loc="upper left")
    try:
        all_bounds = bounds_for(rows)
        ax.set_box_aspect(
            [
                all_bounds["x"][1] - all_bounds["x"][0],
                all_bounds["y"][1] - all_bounds["y"][0],
                all_bounds["z"][1] - all_bounds["z"][0],
            ]
        )
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return None


def main():
    args = parse_args()
    rows = load_rows(args.csv)
    out_dir = Path(args.out_dir or os.path.dirname(os.path.abspath(args.csv)) or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"{Path(args.csv).stem}_reachability"

    summary = summarize(rows, args.csv, args.trim_fraction)
    summary_path = out_dir / f"{prefix}_bbox.json"
    svg_path = out_dir / f"{prefix}_projections.svg"
    png_path = out_dir / f"{prefix}_3d.png"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    write_svg(rows, summary, svg_path)

    matplotlib_error = None
    if not args.no_matplotlib:
        matplotlib_error = try_write_matplotlib(rows, summary, png_path)

    print("\n=== Go2-X5 IK reachability visualization ===")
    print(f"csv: {os.path.abspath(args.csv)}")
    print(f"summary: {summary_path}")
    print(f"svg projections: {svg_path}")
    if args.no_matplotlib:
        print("3d png: skipped")
    elif matplotlib_error:
        print(f"3d png: skipped ({matplotlib_error})")
    else:
        print(f"3d png: {png_path}")
    print(
        "success: {}/{} ({:.1%}), raw IK success: {:.1%}".format(
            summary["success_count"],
            summary["total"],
            summary["success_rate"],
            summary["raw_ik_success_rate"],
        )
    )
    print(f"success bounds: {summary['success_bounds']}")
    print(f"conservative bounds: {summary['conservative_bounds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
