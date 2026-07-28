"""Pure geometry helpers for visualizing the Go2-X5 EE task workspace."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np


AXES = ("x", "y", "z")


def parse_grid_resolution(text: str) -> Tuple[int, int, int]:
    """Parse ``NX,NY,NZ`` and require a real 3-D grid."""
    values = tuple(int(value.strip()) for value in text.split(","))
    if len(values) != 3 or any(value < 2 for value in values):
        raise ValueError("grid resolution must be NX,NY,NZ with every value >= 2")
    return values


def cartesian_grid(
    box: Mapping[str, Sequence[float]],
    resolution: Sequence[int],
) -> np.ndarray:
    """Return an ``(N, 3)`` grid spanning an axis-aligned box."""
    if len(resolution) != 3 or any(int(value) < 2 for value in resolution):
        raise ValueError("resolution must contain three values >= 2")

    coordinates = []
    for axis, count in zip(AXES, resolution):
        bounds = np.asarray(box[axis], dtype=np.float64)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError(f"invalid {axis} bounds: {box[axis]}")
        coordinates.append(np.linspace(bounds[0], bounds[1], int(count)))

    grid = np.meshgrid(*coordinates, indexing="ij")
    return np.stack(grid, axis=-1).reshape(-1, 3)


def classify_cartesian_goals(
    points: np.ndarray,
    collision_lower_limits: Sequence[float],
    collision_upper_limits: Sequence[float],
    underground_limit: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the endpoint form of the production collision predicates.

    The production sampler rejects a trajectory if any interpolation point is
    strictly inside the collision box or below ``underground_limit``. For the
    colored endpoint volume, these masks show which candidate endpoints are
    themselves rejected by those exact predicates.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    lower = np.asarray(collision_lower_limits, dtype=np.float64)
    upper = np.asarray(collision_upper_limits, dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("collision limits must be ordered 3-D vectors")
    if not np.isfinite(points).all() or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("workspace geometry must be finite")

    collision_rejected = np.all(points < upper, axis=1) & np.all(points > lower, axis=1)
    underground_rejected = points[:, 2] < float(underground_limit)
    accepted = ~(collision_rejected | underground_rejected)
    return accepted, collision_rejected, underground_rejected


def nominal_reach_rejected(points: np.ndarray, max_radius: float) -> np.ndarray:
    """Return endpoints outside the production arm-base radial envelope."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("workspace points must be finite")
    if not np.isfinite(max_radius) or max_radius <= 0.0:
        raise ValueError("max_radius must be positive and finite")
    return np.linalg.norm(points, axis=1) > float(max_radius)


def cross_marker_segments(points: np.ndarray, half_extent: float) -> np.ndarray:
    """Turn points into three-axis cross markers for one batched viewer draw."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(half_extent) or half_extent <= 0:
        raise ValueError("half_extent must be positive and finite")
    if points.shape[0] == 0:
        return np.empty((0, 2, 3), dtype=np.float64)

    offsets = np.eye(3, dtype=np.float64) * float(half_extent)
    starts = points[:, None, :] - offsets[None, :, :]
    ends = points[:, None, :] + offsets[None, :, :]
    return np.stack((starts, ends), axis=2).reshape(-1, 2, 3)
