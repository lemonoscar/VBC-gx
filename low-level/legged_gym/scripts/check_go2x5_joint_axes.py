#!/usr/bin/env python3
"""Static URDF joint-axis sanity checks for Go2 + ARX-X5 legs.

This script does not start Isaac Gym. It parses the URDF, applies the current
default leg joint angles, and reports foot motion caused by positive joint
perturbations. Use it to check whether joint axes and mirrored hip defaults are
consistent before spending GPU time on RL training.
"""

import argparse
import importlib.util
import math
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SPEC_PATH = REPO_ROOT / "low-level/legged_gym/envs/manip_loco/go2x5_robot_spec.py"
URDF_PATH = REPO_ROOT / "low-level/resources/robots/go2x5/go2_x5.urdf"


def load_robot_spec():
    module_spec = importlib.util.spec_from_file_location("go2x5_robot_spec", SPEC_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def parse_xyz(text, default=(0.0, 0.0, 0.0)):
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in text.split()], dtype=float)


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def axis_angle_matrix(axis, angle):
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-9 or abs(angle) < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def homogeneous(rotation=None, translation=None):
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = rotation
    if translation is not None:
        transform[:3, 3] = translation
    return transform


def parse_urdf(path):
    root = ET.parse(path).getroot()
    child_to_joint = {}
    joint_info = {}
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = joint.find("origin")
        axis = joint.find("axis")
        info = {
            "name": name,
            "type": joint.attrib.get("type", "fixed"),
            "parent": parent,
            "child": child,
            "xyz": parse_xyz(origin.attrib.get("xyz") if origin is not None else None),
            "rpy": parse_xyz(origin.attrib.get("rpy") if origin is not None else None),
            "axis": parse_xyz(axis.attrib.get("xyz") if axis is not None else None),
        }
        child_to_joint[child] = info
        joint_info[name] = info
    return child_to_joint, joint_info


def chain_from_base(child_to_joint, target_link, base_link="base"):
    chain = []
    link = target_link
    while link != base_link:
        if link not in child_to_joint:
            raise RuntimeError(f"Cannot find parent joint for link {link!r}")
        joint = child_to_joint[link]
        chain.append(joint)
        link = joint["parent"]
    return list(reversed(chain))


def link_position(chain, joint_angles):
    transform = np.eye(4)
    for joint in chain:
        transform = transform @ homogeneous(rpy_matrix(joint["rpy"]), joint["xyz"])
        if joint["type"] != "fixed":
            transform = transform @ homogeneous(axis_angle_matrix(joint["axis"], joint_angles.get(joint["name"], 0.0)))
    return transform[:3, 3]


def fmt_vec(vec):
    return "[" + ", ".join(f"{x:+.4f}" for x in vec) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.15, help="Positive joint perturbation in radians.")
    parser.add_argument("--min-lateral", type=float, default=0.06, help="Minimum expected absolute foot y at default pose.")
    args = parser.parse_args()

    spec = load_robot_spec()
    child_to_joint, joint_info = parse_urdf(URDF_PATH)
    default_angles = dict(spec.DEFAULT_JOINT_ANGLES)
    foot_chains = {
        leg: chain_from_base(child_to_joint, f"{leg}_foot")
        for leg in ("FL", "FR", "RL", "RR")
    }

    print(f"URDF: {URDF_PATH}")
    print(f"delta: {args.delta:.3f} rad")
    print()
    print("Leg joint axes:")
    for leg in ("FL", "FR", "RL", "RR"):
        axes = []
        for kind in ("hip", "thigh", "calf"):
            name = f"{leg}_{kind}_joint"
            axes.append(f"{name} axis={fmt_vec(joint_info[name]['axis'])}")
        print("  " + " | ".join(axes))

    print()
    print("Default foot positions in base frame:")
    default_pos = {}
    failures = []
    for leg in ("FL", "FR", "RL", "RR"):
        pos = link_position(foot_chains[leg], default_angles)
        default_pos[leg] = pos
        expected_side = 1.0 if leg[1] == "L" else -1.0
        side_ok = pos[1] * expected_side > args.min_lateral
        print(f"  {leg}: pos={fmt_vec(pos)} side_ok={side_ok}")
        if not side_ok:
            failures.append(f"{leg} default foot y={pos[1]:.4f} violates side/min-lateral check")

    print()
    print("Positive perturbation foot displacement, base frame:")
    for leg in ("FL", "FR", "RL", "RR"):
        for kind in ("hip", "thigh", "calf"):
            name = f"{leg}_{kind}_joint"
            perturbed = dict(default_angles)
            perturbed[name] = perturbed.get(name, 0.0) + args.delta
            delta_pos = link_position(foot_chains[leg], perturbed) - default_pos[leg]
            note = ""
            if kind == "hip":
                if leg[1] == "L":
                    note = "positive hip should move left legs outward (+y)"
                    if delta_pos[1] <= 0:
                        failures.append(f"{name} positive perturbation did not move outward: dy={delta_pos[1]:.4f}")
                else:
                    note = "positive hip moves right legs inward; outward command is negative"
                    if delta_pos[1] <= 0:
                        failures.append(f"{name} positive perturbation unexpected dy={delta_pos[1]:.4f}")
            print(f"  {name}: dfoot={fmt_vec(delta_pos)} {note}")

    print()
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PASS: static URDF FK checks are consistent with mirrored Go2 hip defaults.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
