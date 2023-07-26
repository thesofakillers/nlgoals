"""
Generate a set of agent demonstrations from the BabyAIBot

Modification of the original make_agent_demos.py
https://github.com/mila-iqia/babyai/blob/master/scripts/make_agent_demos.py
"""

import argparse
import logging
import time
import multiprocessing as mp

import gymnasium as gym
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.utils.baby_ai_bot import BabyAIBot
from minigrid.wrappers import RGBImgObsWrapper
import numpy as np
import blosc
import torch
from tqdm.auto import tqdm

import nlgoals.babyai.utils as utils

SIZE_TO_ENVS = {
    "large": [
        "BabyAI-GoToOpen-v0",
        "BabyAI-Unlock-v0",
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


def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[3]) for demo in demos]
    logger.info(
        "Demo length: {:.3f}+-{:.3f}".format(
            np.mean(num_frames_per_episode), np.std(num_frames_per_episode)
        )
    )


def generate_episode(seed, seed_offset, envs_size):
    possible_envs = SIZE_TO_ENVS[envs_size]

    # sample a random environment
    env_name = np.random.choice(possible_envs)
    env: RoomGridLevel = gym.make(env_name, highlight=False)
    env = RGBImgObsWrapper(env, tile_size=28 if envs_size == "small" else 12)

    mission_success = False
    curr_seed = seed
    # keep trying until we get a successful episode
    while not mission_success:
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
            if args.on_exception == "crash":
                raise Exception("mission failed, the seed is {}".format(curr_seed))
            curr_seed += seed_offset
            logger.info("mission failed")


def generate_demos(
    n_episodes: int, valid: bool, seed: int, envs_size: str, num_workers: int
):
    """
    Generate a set of agent demonstrations from the BabyAIBot

    Args:
        n_episodes (int): number of episodes to generate
        valid (bool): whether to the episodes are for validation or not
        seed (int): random starting seed
        envs_size (str): Which environment size to use. Can be "small" or "large"
    """
    utils.seed(seed)
    checkpoint_time = time.time()

    demos = []

    seeds = range(seed, seed + n_episodes)

    pool = mp.Pool(processes=num_workers)
    results = [
        pool.apply_async(generate_episode, args=(seed, n_episodes, envs_size))
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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--envs-size",
        choices=["small", "large"],
        default="small",
        help="Whether to use small or large environments",
    )
    parser.add_argument(
        "--save-path",
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
        "--valid-episodes",
        type=int,
        help="number of validation episodes to generate demonstrations for",
    )
    parser.add_argument("--seed", type=int, default=0, help="start random seed")
    parser.add_argument(
        "--on-exception",
        type=str,
        default="warn",
        choices=("warn", "crash"),
        help="How to handle exceptions during demo generation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers to use for generating demos",
    )

    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    logging.basicConfig(level="INFO", format="%(asctime)s: %(levelname)s: %(message)s")
    logger.info(args)
    # Training demos
    generate_demos(args.episodes, False, args.seed, args.envs_size, args.num_workers)
    # Validation demos
    if args.valid_episodes:
        generate_demos(
            args.valid_episodes, True, int(1e9), args.envs_size, args.num_workers
        )
