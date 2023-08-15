import random

from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel

from nlgoals.babyai.custom.envs import RoomGridLevelCC
from nlgoals.babyai.custom.constants import COLOR_TO_SYN, OBJ_TO_SYN, VERB_TO_SYN


def paraphrase_mission(mission: str) -> str:
    """
    Paraphrase a "{go to}/{pick up} the/a {color} {obj} {remainder}" string.
    The {color} and {remainder} are optional, i.e. may not appear in the string
    By rephrasing or using synonyms

    """
    mission_splits = mission.split(" ")
    verb = " ".join(mission_splits[:2])

    # No paraphrase for 'put' missions
    if verb.startswith("put"):
        return mission

    article, *rest = mission_splits[2:]

    # Determine color and object, if color is not present
    color_obj = rest[:2]
    color, obj = color_obj if color_obj[0] in COLOR_TO_SYN else (None, color_obj[0])
    mission_remainder = " ".join(rest[2:] if color else rest[1:])

    # Select synonyms
    color = random.choice(COLOR_TO_SYN[color]) if color else None
    obj = random.choice(OBJ_TO_SYN[obj])
    verb = random.choice(VERB_TO_SYN[verb])

    # Build new mission with synonyms
    words = [verb, article, color, obj, mission_remainder]

    # Ignore None when joining words
    return " ".join(word for word in words if word)


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
