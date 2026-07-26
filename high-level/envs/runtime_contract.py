"""Small, dependency-free helpers shared by high-level runtime entrypoints."""


DEFAULT_ROBOT_START_POSE = (-2.0, 0.0, 0.55)


def resolve_robot_start_pose(env_cfg, robot_start_pose=None, eval_mode=False):
    """Resolve start pose without allowing YAML to override an explicit caller."""
    if robot_start_pose is not None:
        return tuple(robot_start_pose)
    pose_key = "evalRobotStartPose" if eval_mode else "robotStartPose"
    return tuple(env_cfg.get(pose_key, env_cfg.get("robotStartPose", DEFAULT_ROBOT_START_POSE)))


def object_fell_below_table(object_z, table_height, tolerance=0.0):
    """Return a fall mask while ignoring shallow contact-solver penetration."""
    if tolerance < 0.0:
        raise ValueError("object fall tolerance must be non-negative")
    return object_z < (table_height - tolerance)
