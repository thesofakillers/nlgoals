from typing import Tuple
import logging
import os

import torchvision as tv
from minigrid.utils.baby_ai_bot import BabyAIBot
import numpy as np
import torch

from nlgoals.babyai.custom_envs import str_to_pos
from nlgoals.utils.misc import prep_video

logger = logging.getLogger(__name__)


def check_conf_done(env, agent_dir: int, obj_pos_str: str):
    """
    Checks whether the agent is next to and facing the location
    causally confounded with the true goal at training

    Args:
        env: the environment
        agent_dir: the direction the agent is facing
            Integer between 0 and 3 meaning right, down, left, up
    """

    cc_pos = str_to_pos(obj_pos_str, env)
    agent_pos = env.agent_pos

    delta_pos = (cc_pos[0] - agent_pos[0], cc_pos[1] - agent_pos[1])
    direction_deltas = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    return delta_pos == direction_deltas[agent_dir]


def save_results(save_dir, true_goal_results, conf_goal_results, videos, seeds):
    """
    Saves the results of the evaluation with the following structure:
    save_dir/seeds.npy
    save_dir/true_goal/results.npy
    save_dir/true_goal/videos/success/[video1, video2, video3].mp4
    save_dir/true_goal/videos/fail/[video1, video2, video3].mp4
    save_dir/conf_goal/results.npy
    save_dir/conf_goal/videos/success/[video1, video2, video3].mp4
    save_dir/conf_goal/videos/fail/[video1, video2, video3].mp4
    """
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "seeds.npy"), seeds)
    save(save_dir, "true_goal", true_goal_results, videos["true_goal"])
    save(save_dir, "conf_goal", conf_goal_results, videos["conf_goal"])


def save(save_dir, goal, results, videos):
    goal_dir = os.path.join(save_dir, goal)
    os.makedirs(goal_dir, exist_ok=True)
    np.save(os.path.join(goal_dir, "results.npy"), results)
    save_videos(goal_dir, videos)


def save_videos(save_dir, videos):
    videos_dir = os.path.join(save_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    for video_type, video_list in videos.items():
        for i, video in enumerate(video_list):
            prepd_video = prep_video(video, False, False)
            video_path = os.path.join(videos_dir, video_type, f"video_{i}.mp4")
            # make sure the parent directory exists
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            tv.io.write_video(video_path, prepd_video, fps=10)


def print_results(true_goal_results, conf_goal_results):
    print(f"true_goal SR: {true_goal_results.mean()}")
    print(f"conf_goal SR: {conf_goal_results.mean()}")


def update_videos(videos, video, true_goal_success, conf_goal_success):
    if true_goal_success and len(videos["true_goal"]["success"]) < 3:
        videos["true_goal"]["success"].append(video)
    elif not true_goal_success and len(videos["true_goal"]["fail"]) < 3:
        videos["true_goal"]["fail"].append(video)
    if conf_goal_success and len(videos["conf_goal"]["success"]) < 3:
        videos["conf_goal"]["success"].append(video)
    elif not conf_goal_success and len(videos["conf_goal"]["fail"]) < 3:
        videos["conf_goal"]["fail"].append(video)

    return videos


def run_oracle(env, seed, seed_offset) -> Tuple[np.ndarray, str, int]:
    """
    Runs the oracle on the environment so to

    - check whether the seed is valid, if not find a new seed
    - get the text annotation of the env, for textual goal conditioning
    - get the final state of the environment, for visual goal conditioning

    Args:
        env: the environment
        seed: the seed to be checked
        seed_offset: the offset to be added to the seed to get the final state
    """
    curr_seed = seed
    mission_success = False
    while not mission_success:
        try:
            done = False
            final_step = False

            obs = env.reset(seed=curr_seed)[0]
            oracle = BabyAIBot(env)

            last_image = None

            while not final_step:
                action = oracle.replan()
                if isinstance(action, torch.tensor):
                    action = action.item()
                if done:
                    final_step = True
                new_obs, reward, done, _, _ = env.step(action)

                if done and reward > 0:
                    mission_success = True

                last_image = obs["image"]

                obs = new_obs
            # if our demos was succesful, save it
            if mission_success:
                return last_image, obs["mission"], curr_seed
            # handle unsuccessful demos
            else:
                raise Exception
        except (Exception, AssertionError):
            curr_seed += seed_offset
            logger.info("Mission either failed or crashed, trying again...")
            continue
