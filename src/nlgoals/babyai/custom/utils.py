from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel

from nlgoals.babyai.custom.envs import RoomGridLevelCC


def make_cc(EnvClass):
    """
    Makes an environment causally confused by overriding the class it inherits from.
    """
    # some environments inherit from RoomGridLevel directly
    if EnvClass.__bases__[0] in (RoomGridLevel, RoomGridLevelCC):
        EnvClass.__bases__ = (RoomGridLevelCC,)
        return EnvClass
    # others inherit from LevelGen, which inherits from RoomGridLevel
    else:
        LevelGen.__bases__ = (RoomGridLevelCC,)
        EnvClass.__bases__ = (LevelGen,)
        return EnvClass


def str_to_pos(pos_str, env):
    """
    Gets the (x,y) coordinate tuple from a string
    """
    top = (0, 0)

    size = (env.unwrapped.grid.width, env.unwrapped.grid.height)

    left_pos = top[0] + 1
    right_pos = top[0] + size[0] - 2
    top_pos = top[1] + 1
    bottom_pos = top[1] + size[1] - 2
    # -2 to account for wall width
    possible_cc_obj_pos = {
        "top left": (left_pos, top_pos),
        "top right": (right_pos, top_pos),
        "bottom left": (left_pos, bottom_pos),
        "bottom right": (right_pos, bottom_pos),
    }

    return possible_cc_obj_pos[pos_str]
