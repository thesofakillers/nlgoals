"""
Generate a set of agent demonstrations from the BabyAIBot

Modification of the original make_agent_demos.py
https://github.com/mila-iqia/babyai/blob/master/scripts/make_agent_demos.py
"""

import jsonargparse
import logging
import time
import multiprocessing as mp
from typing import Optional, Dict

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.goto import LevelGen
from minigrid.utils.baby_ai_bot import BabyAIBot
from minigrid.wrappers import RGBImgObsWrapper
import numpy as np
import blosc
import torch
from tqdm.auto import tqdm

import nlgoals.babyai.utils as utils
from nlgoals.babyai.custom import POSSIBLE_CC_POS, make_cc


def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[3]) for demo in demos]
    logger.info(
        "Demo length: {:.3f}+-{:.3f}".format(
            np.mean(num_frames_per_episode), np.std(num_frames_per_episode)
        )
    )


def paraphrase_mission(mission: str) -> str:
    """
    Paraphrase a "go to the {color} {obj}" string
    By rephrasing or using synonyms
    """
    mission_words = mission.split(" ")
    color, obj = mission_words[3:]
    verb = mission_words[0]

    possible_colors = utils.COLOR_TO_SYN[color]
    color = np.random.choice(possible_colors)
    possible_objs = utils.OBJ_TO_SYN[obj]
    obj = np.random.choice(possible_objs)
    possible_verbs = ["go", "move", "navigate", "proceed", "advance", "make your way"]
    verb = np.random.choice(possible_verbs)

    new_mission = f"{verb} to the {color} {obj}"

    return new_mission


def generate_episode(
    seed,
    seed_offset,
    envs_size,
    causally_confuse: bool = False,
    cc_kwargs: Optional[Dict[str, str]] = None,
    single_kwargs: Optional[Dict[str, str]] = None,
):
    possible_envs = utils.SIZE_TO_ENVS[envs_size]

    # sample a random environment
    env_name = np.random.choice(possible_envs)
    EnvClass = utils.NAME_TO_CLASS[env_name]
    if envs_size == "single":
        env_kwargs = single_kwargs
    else:
        env_kwargs = utils.NAME_TO_KWARGS[env_name]
    if causally_confuse:
        # modify inheritance of EnvClass s.t. causal confusion is handled if requested
        EnvClass = make_cc(EnvClass)
        env_kwargs = {**env_kwargs, **cc_kwargs}
    env = EnvClass(highlight=False, **env_kwargs)
    env = RGBImgObsWrapper(
        env, tile_size=28 if envs_size.split("-")[0] in {"small", "single"} else 12
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
                return (
                    mission,
                    env_name,
                    blosc.pack_array(np.array(images)),
                    directions,
                    actions,
                    rewards,
                )
            # handle unsuccessful demos
            else:
                raise Exception
        except (Exception, AssertionError):
            curr_seed += seed_offset
            logger.info("Mission either failed or crashed, trying again...")
            continue


def generate_demos(
    n_episodes: int,
    valid: bool,
    seed: int,
    envs_size: str,
    num_workers: int,
    causally_confuse: bool = False,
    cc_kwargs: Optional[Dict[str, str]] = None,
    single_kwargs: Optional[Dict[str, str]] = None,
):
    """
    Generate a set of agent demonstrations from the BabyAIBot

    Args:
        n_episodes (int): number of episodes to generate
        valid (bool): whether to the episodes are for validation or not
        seed (int): random starting seed
        envs_size (str): Which environment size to use.
            Can be "small(-play)", "large(-play)" or "single"
        num_workers: number of workers to use for multiprocessing
        causally_confuse (bool): whether to causally confuse the environment
        cc_kwargs (dict): kwargs for the causal confusion
        single_kwargs (dict): kwargs for the single environment
    """
    utils.seed(seed)
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
                single_kwargs,
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
        choices=["small-play", "large-play", "small", "single"],
        default="small",
        help="Whether to use small or large environments. Or a single GoToSpecObj env",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="path to save demonstrations (based on --model and --origin by default)",
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
        choices=list(OBJECT_TO_IDX.keys()),
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
        "--cc_obj_pos_str",
        type=str,
        choices=POSSIBLE_CC_POS,
        required=False,
        help="Object position to use for causal confusion",
    )
    parser.add_argument(
        "--single_obj_kind",
        type=str,
        choices=list(OBJECT_TO_IDX.keys()),
        required=False,
        help="Object kind to use for `single` env_size",
    )
    parser.add_argument(
        "--single_obj_color",
        type=str,
        choices=list(COLOR_TO_IDX.keys()),
        required=False,
        help="Object color to use for `single` env_size",
    )

    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    cc_kwargs = {
        "cc_obj_kind": args.cc_obj_kind,
        "cc_obj_color": args.cc_obj_color,
        "cc_obj_pos_str": args.cc_obj_pos_str,
    }
    single_kwargs = {
        "obj_kind": args.single_obj_kind,
        "obj_color": args.single_obj_color,
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
        single_kwargs,
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
            single_kwargs,
        )