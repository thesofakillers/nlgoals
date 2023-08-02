import os
import random
import numpy
import torch
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


NAME_TO_CLASS = {
    "BabyAI-GoToObj-v0": GoToObj,
    "BabyAI-GoToLocal-v0": GoToLocal,
    "BabyAI-PickupDist-v0": PickupDist,
    "BabyAI-PickupLoc-v0": PickupLoc,
    "BabyAI-PutNextLocal-v0": PutNextLocal,
}

NAME_TO_KWARGS = {
    "BabyAI-GoToObj-v0": {},
    "BabyAI-GoToLocal-v0": {},
    "BabyAI-PickupDist-v0": {},
    "BabyAI-PickupLoc-v0": {},
    "BabyAI-PutNextLocal-v0": {},
}

SIZE_TO_ENVS = {
    "large": [
        "BabyAI-GoToOpen-v0",
        "BabyAI-Pickup-v0",
        "BabyAI-Open-v0",
        "BabyAI-SynthLoc-v0",
        "BabyAI-Synth-v0",
    ],
    "small": [
        "BabyAI-GoToObj-v0",
        "BabyAI-GoToLocal-v0",
        "BabyAI-PickupDist-v0",
        "BabyAI-PickupLoc-v0",
        "BabyAI-PutNextLocal-v0",
    ],
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
