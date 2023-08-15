"""
Generate a set of agent demonstrations from the BabyAIBot

Modification of the original make_agent_demos.py
https://github.com/mila-iqia/babyai/blob/master/scripts/make_agent_demos.py
"""

import jsonargparse
import logging
import time
from typing import Optional, Dict

import multiprocessing as mp
from typing import Optional, Dict
import random

from minigrid.core.constants import COLOR_TO_IDX
from minigrid.utils.baby_ai_bot import BabyAIBot
from minigrid.wrappers import RGBImgObsWrapper
import numpy as np
import blosc
import torch
from tqdm.auto import tqdm
from nlgoals.babyai.custom.envs import OBJ_MAP
from nlgoals.babyai.custom.wrappers import ColorTypeLockWrapper
import nlgoals.babyai.utils as utils


logger = logging.getLogger(__name__)


def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[3]) for demo in demos]
    logger.info(
        "Demo length: {:.3f}+-{:.3f}".format(
            np.mean(num_frames_per_episode), np.std(num_frames_per_episode)
        )
    )


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
    color, obj = (
        color_obj if color_obj[0] in utils.COLOR_TO_SYN else (None, color_obj[0])
    )
    mission_remainder = " ".join(rest[2:] if color else rest[1:])

    # Select synonyms
    color = random.choice(utils.COLOR_TO_SYN[color]) if color else None
    obj = random.choice(utils.OBJ_TO_SYN[obj])
    verb = random.choice(utils.VERB_TO_SYN[verb])

    # Build new mission with synonyms
    words = [verb, article, color, obj, mission_remainder]

    # Ignore None when joining words
    return " ".join(word for word in words if word)


def generate_episode(
    seed,
    seed_offset,
    envs_size,
    causally_confuse: bool = False,
    cc_kwargs: Optional[Dict[str, str]] = None,
    paraphrase: bool = False,
):
    utils.seed(seed)
    possible_envs = utils.SIZE_TO_ENVS[envs_size]

    # sample a random environment
    env_name = np.random.choice(possible_envs)
    EnvClass = utils.NAME_TO_CLASS[env_name]
    env_kwargs = utils.NAME_TO_KWARGS[env_name]
    env = EnvClass(highlight=False, **env_kwargs)
    if causally_confuse:
        env = ColorTypeLockWrapper(env, **cc_kwargs)
    env = RGBImgObsWrapper(
        env, tile_size=28 if envs_size.split("-")[0] in {"small"} else 12
    )

    mission_success = False
    curr_seed = seed
    # keep trying until we get a successful episode
    while not mission_success:
        try:
            done = False
            final_step = False

            obs = env.reset(seed=curr_seed)[0]
            agent = BabyAIBot(env)

            actions = []
            mission = obs["mission"]
            images = []
            directions = []
            rewards = []

            while not final_step:
                action = agent.replan()
                if isinstance(action, torch.Tensor):
                    action = action.item()
                if done:
                    final_step = True
                new_obs, reward, done, _, _ = env.step(action)

                if done and reward > 0:
                    mission_success = True

                actions.append(action)
                images.append(obs["image"])
                directions.append(obs["direction"])
                rewards.append(reward)

                obs = new_obs
            # if our demos was succesful, save it
            if mission_success:
                return_mission = paraphrase_mission(mission) if paraphrase else mission
                return_tuple = (
                    return_mission,
                    env_name,
                    blosc.pack_array(np.array(images)),
                    directions,
                    actions,
                    rewards,
                )
                return return_tuple
            # handle unsuccessful demos
            else:
                raise ValueError("Mission unsuccessful")
        except AssertionError:
            curr_seed += seed_offset
            logger.debug("Mission either failed or crashed, trying again...")
            continue


def generate_demos(
    n_episodes: int,
    valid: bool,
    seed: int,
    envs_size: str,
    num_workers: int,
    causally_confuse: bool = False,
    cc_kwargs: Optional[Dict[str, str]] = None,
    paraphrase: bool = False,
):
    """
    Generate a set of agent demonstrations from the BabyAIBot

    Args:
        n_episodes (int): number of episodes to generate
        valid (bool): whether to the episodes are for validation or not
        seed (int): random starting seed
        envs_size (str): Which environment size to use.
            Can be "small(-play)", "large(-play)"
        num_workers: number of workers to use for multiprocessing
        causally_confuse (bool): whether to causally confuse the environment
        cc_kwargs (dict): kwargs for the causal confusion
        paraphrase (bool): whether to paraphrase the mission
    """
    checkpoint_time = time.time()

    demos = []

    seeds = range(seed, seed + n_episodes)

    pool = mp.Pool(processes=num_workers)
    results = [
        pool.apply_async(
            generate_episode,
            args=(
                seed,
                n_episodes,
                envs_size,
                causally_confuse,
                cc_kwargs,
                paraphrase,
            ),
        )
        for seed in seeds
    ]
    pool.close()

    pbar = tqdm(total=len(results), desc="Demos")

    for p in results:
        demos.append(p.get())
        pbar.update()

    pbar.close()

    pool.join()

    # log how long it took
    now = time.time()
    total_time = now - checkpoint_time
    logger.info(f"total_time: {total_time}")
    # Save last batch of demos
    logger.info("Saving demos...")
    demos_path = utils.get_demos_path(args.save_path, None, "agent", valid)
    utils.save_demos(demos, demos_path)
    logger.info("{} demos saved".format(len(demos)))
    print_demo_lengths(demos[-100:])


if __name__ == "__main__":
    # Parse arguments
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        "--envs_size",
        choices=["small-play", "large-play", "small"],
        default="small",
        help="Whether to use small or large environments.",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="path to save demonstrations",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="number of episodes to generate demonstrations for",
    )
    parser.add_argument(
        "--val_episodes",
        type=int,
        help="number of validation episodes to generate demonstrations for",
    )
    parser.add_argument("--seed", type=int, default=0, help="start random seed")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to use for generating demos",
    )
    parser.add_argument(
        "--causally_confuse",
        type=bool,
        default=False,
        help="Whether to causally confuse the environment",
    )
    parser.add_argument(
        "--cc_obj_kind",
        type=str,
        choices=list(OBJ_MAP.keys()),
        required=False,
        help="Object kind to use for causal confusion",
    )
    parser.add_argument(
        "--cc_obj_color",
        type=str,
        choices=list(COLOR_TO_IDX.keys()),
        required=False,
        help="Object color to use for causal confusion",
    )
    parser.add_argument(
        "--paraphrase",
        type=bool,
        default=False,
        help="Whether to paraphrase the mission",
    )

    args = parser.parse_args()

    cc_kwargs = {
        "obj_type": args.cc_obj_kind,
        "color": args.cc_obj_color,
    }

    logging.basicConfig(level="INFO", format="%(asctime)s: %(levelname)s: %(message)s")
    # Training demos
    generate_demos(
        args.episodes,
        False,
        args.seed,
        args.envs_size,
        args.num_workers,
        args.causally_confuse,
        cc_kwargs,
        args.paraphrase,
    )
    # Validation demos
    if args.val_episodes:
        generate_demos(
            args.val_episodes,
            True,
            int(1e9),
            args.envs_size,
            args.num_workers,
            args.causally_confuse,
            cc_kwargs,
            args.paraphrase,
        )
