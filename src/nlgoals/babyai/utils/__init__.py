import os
import random
import numpy
import torch
from nlgoals.babyai.custom import GoToSpecObj
from nlgoals.babyai.custom.envs import CustomGoToColor, CustomGoToObj
from nlgoals.babyai.utils.agent import load_agent, ModelAgent, DemoAgent, BotAgent
from nlgoals.babyai.utils.demos import (
    load_demos,
    save_demos,
    synthesize_demos,
    get_demos_path,
)
from nlgoals.babyai.utils.format import (
    ObssPreprocessor,
    IntObssPreprocessor,
    get_vocab_path,
)
from nlgoals.babyai.utils.log import (
    get_log_path,
    get_log_dir,
    synthesize,
    configure_logging,
)
from nlgoals.babyai.utils.model import get_model_dir, load_model, save_model

from minigrid.envs.babyai import GoToObj, GoToLocal, PickupDist, PickupLoc, PutNextLocal


SIZE_TO_ENVS = {
    "small-play": [
        "Custom-GoToObjK-v0",
        "Custom-GoToObjBx-v0",
        "Custom-GoToObjBl-v0",
        "Custom-GoToColorR-v0",
        "Custom-GoToColorG-v0",
        "Custom-GoToColorB-v0",
        "Custom-GoToObj-v0",
        "Custom-GoToColor-v0",
        "BabyAI-GoToObj-v0",
        "BabyAI-GoToLocal-v0",
        "BabyAI-PickupDist-v0",
        "BabyAI-PickupLoc-v0",
        "BabyAI-PutNextLocal-v0",
    ],
    "small": [
        # go to the key
        "Custom-GoToObjK-v0",
        # go to the box
        "Custom-GoToObjBx-v0",
        # go to the ball
        "Custom-GoToObjBl-v0",
        # go to the red object
        "Custom-GoToColorR-v0",
        # go to the green object
        "Custom-GoToColorG-v0",
        # go to the blue object
        "Custom-GoToColorB-v0",
    ],
    "single-legacy": ["Custom-GoToSpecObj-v0"],
    "large-play": [
        "BabyAI-GoToOpen-v0",
        "BabyAI-Pickup-v0",
        "BabyAI-Open-v0",
        "BabyAI-SynthLoc-v0",
        "BabyAI-Synth-v0",
    ],
    "small-play-legacy": [
        "BabyAI-GoToObj-v0",
        "BabyAI-GoToLocal-v0",
        "BabyAI-PickupDist-v0",
        "BabyAI-PickupLoc-v0",
        "BabyAI-PutNextLocal-v0",
    ],
    "small-legacy": [  # red, green, blue, purple, yellow, grey
        # red ball, box, key
        "Custom-GoToSpecObj-v0-RB",
        "Custom-GoToSpecObj-v0-RBX",
        "Custom-GoToSpecObj-v0-RK",
        # green ball, box, key
        "Custom-GoToSpecObj-v0-GB",
        "Custom-GoToSpecObj-v0-GBX",
        "Custom-GoToSpecObj-v0-GK",
        # blue ball, box, key
        "Custom-GoToSpecObj-v0-BB",
        "Custom-GoToSpecObj-v0-BBX",
        "Custom-GoToSpecObj-v0-BK",
        # purple ball, box, key
        "Custom-GoToSpecObj-v0-PB",
        "Custom-GoToSpecObj-v0-PBX",
        "Custom-GoToSpecObj-v0-PK",
        # yellow ball, box, key
        "Custom-GoToSpecObj-v0-YB",
        "Custom-GoToSpecObj-v0-YBX",
        "Custom-GoToSpecObj-v0-YK",
        # grey ball, box, key
        "Custom-GoToSpecObj-v0-GyB",
        "Custom-GoToSpecObj-v0-GyBX",
        "Custom-GoToSpecObj-v0-GyK",
    ],
}

NAME_TO_CLASS = {
    "Custom-GoToObj-v0": CustomGoToObj,
    "Custom-GoToColor-v0": CustomGoToColor,
    "Custom-GoToObjK-v0": CustomGoToObj,
    "Custom-GoToObjBx-v0": CustomGoToObj,
    "Custom-GoToObjBl-v0": CustomGoToObj,
    "Custom-GoToColorR-v0": CustomGoToColor,
    "Custom-GoToColorG-v0": CustomGoToColor,
    "Custom-GoToColorB-v0": CustomGoToColor,
    "BabyAI-GoToObj-v0": GoToObj,
    "BabyAI-GoToLocal-v0": GoToLocal,
    "BabyAI-PickupDist-v0": PickupDist,
    "BabyAI-PickupLoc-v0": PickupLoc,
    "BabyAI-PutNextLocal-v0": PutNextLocal,
    "Custom-GoToSpecObj-v0": GoToSpecObj,
    "Custom-GoToSpecObj-v0-RB": GoToSpecObj,
    "Custom-GoToSpecObj-v0-RBX": GoToSpecObj,
    "Custom-GoToSpecObj-v0-RK": GoToSpecObj,
    "Custom-GoToSpecObj-v0-GB": GoToSpecObj,
    "Custom-GoToSpecObj-v0-GBX": GoToSpecObj,
    "Custom-GoToSpecObj-v0-GK": GoToSpecObj,
    "Custom-GoToSpecObj-v0-BB": GoToSpecObj,
    "Custom-GoToSpecObj-v0-BBX": GoToSpecObj,
    "Custom-GoToSpecObj-v0-BK": GoToSpecObj,
    "Custom-GoToSpecObj-v0-PB": GoToSpecObj,
    "Custom-GoToSpecObj-v0-PBX": GoToSpecObj,
    "Custom-GoToSpecObj-v0-PK": GoToSpecObj,
    "Custom-GoToSpecObj-v0-YB": GoToSpecObj,
    "Custom-GoToSpecObj-v0-YBX": GoToSpecObj,
    "Custom-GoToSpecObj-v0-YK": GoToSpecObj,
    "Custom-GoToSpecObj-v0-GyB": GoToSpecObj,
    "Custom-GoToSpecObj-v0-GyBX": GoToSpecObj,
    "Custom-GoToSpecObj-v0-GyK": GoToSpecObj,
}

NAME_TO_KWARGS = {
    "Custom-GoToObj-v0": {},
    "Custom-GoToColor-v0": {},
    "Custom-GoToObjK-v0": {"obj_type": "key", "only_one": True, "num_dists": 3},
    "Custom-GoToObjBx-v0": {"obj_type": "box", "only_one": True, "num_dists": 3},
    "Custom-GoToObjBl-v0": {"obj_type": "ball", "only_one": True, "num_dists": 3},
    "Custom-GoToColorR-v0": {"obj_color": "red", "only_one": True, "num_dists": 3},
    "Custom-GoToColorG-v0": {"obj_color": "green", "only_one": True, "num_dists": 3},
    "Custom-GoToColorB-v0": {"obj_color": "blue", "only_one": True, "num_dists": 3},
    "BabyAI-GoToObj-v0": {},
    "BabyAI-GoToLocal-v0": {},
    "BabyAI-PickupDist-v0": {},
    "BabyAI-PickupLoc-v0": {},
    "BabyAI-PutNextLocal-v0": {},
    "Custom-GoToSpecObj-v0-RB": {"obj_kind": "ball", "obj_color": "red"},
    "Custom-GoToSpecObj-v0-RBX": {"obj_kind": "box", "obj_color": "red"},
    "Custom-GoToSpecObj-v0-RK": {"obj_kind": "key", "obj_color": "red"},
    "Custom-GoToSpecObj-v0-GB": {"obj_kind": "ball", "obj_color": "green"},
    "Custom-GoToSpecObj-v0-GBX": {"obj_kind": "box", "obj_color": "green"},
    "Custom-GoToSpecObj-v0-GK": {"obj_kind": "key", "obj_color": "green"},
    "Custom-GoToSpecObj-v0-BB": {"obj_kind": "ball", "obj_color": "blue"},
    "Custom-GoToSpecObj-v0-BBX": {"obj_kind": "box", "obj_color": "blue"},
    "Custom-GoToSpecObj-v0-BK": {"obj_kind": "key", "obj_color": "blue"},
    "Custom-GoToSpecObj-v0-PB": {"obj_kind": "ball", "obj_color": "purple"},
    "Custom-GoToSpecObj-v0-PBX": {"obj_kind": "box", "obj_color": "purple"},
    "Custom-GoToSpecObj-v0-PK": {"obj_kind": "key", "obj_color": "purple"},
    "Custom-GoToSpecObj-v0-YB": {"obj_kind": "ball", "obj_color": "yellow"},
    "Custom-GoToSpecObj-v0-YBX": {"obj_kind": "box", "obj_color": "yellow"},
    "Custom-GoToSpecObj-v0-YK": {"obj_kind": "key", "obj_color": "yellow"},
    "Custom-GoToSpecObj-v0-GyB": {"obj_kind": "ball", "obj_color": "grey"},
    "Custom-GoToSpecObj-v0-GyBX": {"obj_kind": "box", "obj_color": "grey"},
    "Custom-GoToSpecObj-v0-GyK": {"obj_kind": "key", "obj_color": "grey"},
}

VERB_TO_SYN = {
    "go to": [
        "go to",
        "move to",
        "navigate to",
        "proceed to",
        "advance to",
        "make your way to",
        "head to",
    ],
    "pick up": [
        "pick up",
        "grab",
        "take",
        "get",
        "remove",
        "collect",
    ],
}

COLOR_TO_SYN = {
    "blue": ["blue", "azure", "sapphire", "cyan"],
    "green": ["green", "emerald"],
    "red": ["red", "scarlet", "crimson", "ruby", "carmine", "vermilion"],
    "purple": ["purple", "violet", "lavender", "lilac"],
    "yellow": ["yellow", "gold", "amber", "lemon", "mustard", "ochre"],
    "grey": ["grey", "gray", "silver"],
}

AMBIG_COLORS = ["turquoise", "indigo", "cherry", "burgundy", "teal", "lime"]

OBJ_TO_SYN = {
    "ball": ["ball", "sphere", "orb", "globe", "circle", "marble"],
    "box": ["box", "cube", "cuboid", "chest", "crate", "square", "rectangle"],
    "key": ["key"],
    "object": ["object", "thing", "item"],
}


def storage_dir():
    # defines the storage directory to be in the root (Same level as babyai folder)
    return os.environ.get("BABYAI_STORAGE", ".")


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not (os.path.isdir(dirname)):
        os.makedirs(dirname)


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
