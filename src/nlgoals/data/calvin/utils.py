import enum

class FrameKey(enum.Enum):
    actions = "actions"
    rel_actions = "rel_actions"
    robot_obs = "robot_obs"
    scene_obs = "scene_obs"
    rgb_static = "rgb_static"
    rgb_gripper = "rgb_gripper"
    rgb_tactile = "rgb_tactile"
    depth_static = "depth_static"
    depth_gripper = "depth_gripper"
    depth_tactile = "depth_tactile"
