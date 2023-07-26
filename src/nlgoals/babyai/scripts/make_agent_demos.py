#!/usr/bin/env python3

"""
Generate a set of agent demonstrations.

The agent can either be a trained model or the heuristic expert (bot).

Demonstration generation can take a long time, but it can be parallelized
if you have a cluster at your disposal. Provide a script that launches
make_agent_demos.py at your cluster as --job-script and the number of jobs as --jobs.
"""

import argparse
import logging
import time

import gymnasium as gym
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.utils.baby_ai_bot import BabyAIBot
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
import numpy as np
import blosc
import torch

import nlgoals.babyai.utils as utils


def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    logger.info(
        "Demo length: {:.3f}+-{:.3f}".format(
            np.mean(num_frames_per_episode), np.std(num_frames_per_episode)
        )
    )


def generate_demos(n_episodes, valid, seed):
    utils.seed(seed)

    # env instance
    env: RoomGridLevel = gym.make(args.env, highlight=False)
    env = RGBImgObsWrapper(env)

    demos_path = utils.get_demos_path(args.save_path, args.env, "agent", valid)
    print(demos_path)
    demos = []

    checkpoint_time = time.time()

    curr_seed = seed

    just_crashed = False
    # start generating episodes (demos)
    while True:
        if len(demos) == n_episodes:
            break

        done = False
        final_step = False
        mission_success = False
        if just_crashed:
            logger.info(
                "reset the environment to find a mission that the bot can solve"
            )
            curr_seed += 1
        else:
            curr_seed = curr_seed
        obs = env.reset(seed=curr_seed)[0]
        agent = BabyAIBot(env)

        actions = []
        mission = obs["mission"]
        images = []
        directions = []

        try:
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

                obs = new_obs
            # if our demos was succesful, save it
            if mission_success > 0 and (
                args.filter_steps == 0 or len(images) <= args.filter_steps
            ):
                demos.append(
                    (mission, blosc.pack_array(np.array(images)), directions, actions)
                )
                just_crashed = False
            # handle unsuccessful demos
            if not mission_success:
                if args.on_exception == "crash":
                    raise Exception("mission failed, the seed is {}".format(curr_seed))
                just_crashed = True
                logger.info("mission failed")
        except (Exception, AssertionError):
            if args.on_exception == "crash":
                raise
            just_crashed = True
            logger.exception("error while generating demo #{}".format(len(demos)))
            continue

        # logging progress
        if len(demos) and len(demos) % args.log_interval == 0:
            now = time.time()
            demos_per_second = args.log_interval / (now - checkpoint_time)
            to_go = (n_episodes - len(demos)) / demos_per_second
            logger.info(
                "demo #{}, {:.3f} demos per second, {:.3f} seconds to go".format(
                    len(demos) - 1, demos_per_second, to_go
                )
            )
            checkpoint_time = now

        # Save demos
        if (
            args.save_interval > 0
            and len(demos) < n_episodes
            and len(demos) % args.save_interval == 0
        ):
            logger.info("Saving demos...")
            utils.save_demos(demos, demos_path)
            logger.info("{} demos saved".format(len(demos)))
            print_demo_lengths(demos[-100:])

        curr_seed += 1

    # Save last batch of demos
    logger.info("Saving demos...")
    utils.save_demos(demos, demos_path)
    logger.info("{} demos saved".format(len(demos)))
    print_demo_lengths(demos[-100:])


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env", required=True, help="name of the environment to be run (REQUIRED)"
    )
    parser.add_argument(
        "--model", default="BOT", help="name of the trained model (REQUIRED)"
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
        default=512,
        help="number of validation episodes to generate demonstrations for",
    )
    parser.add_argument("--seed", type=int, default=0, help="start random seed")
    parser.add_argument(
        "--argmax",
        action="store_true",
        default=False,
        help="action with highest probability is selected",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="interval between progress reports",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10000,
        help="interval between demonstrations saving",
    )
    parser.add_argument(
        "--filter-steps",
        type=int,
        default=0,
        help="filter out demos with number of steps more than filter-steps",
    )
    parser.add_argument(
        "--on-exception",
        type=str,
        default="warn",
        choices=("warn", "crash"),
        help="How to handle exceptions during demo generation",
    )

    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    logging.basicConfig(level="INFO", format="%(asctime)s: %(levelname)s: %(message)s")
    logger.info(args)
    # Training demos
    generate_demos(args.episodes, False, args.seed)
    # Validation demos
    if args.valid_episodes:
        generate_demos(args.valid_episodes, True, int(1e9))
