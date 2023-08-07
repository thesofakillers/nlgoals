"""
Evaluate GCBC policy on BabyAI (causally confused) environment
"""
import os
from typing import Tuple, Dict, List, Callable
import logging

import numpy as np
import gymnasium as gym
import torch
import torchvision as tv
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.wrappers import RGBImgObsWrapper
from minigrid.utils.baby_ai_bot import BabyAIBot
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nlgoals.data.transforms import CLIPImageTransform
from nlgoals.models.clipt import CLIPT
from nlgoals.models.gcbc import GCBC, gcbc_enum_to_class, GCBC_ENUM
from nlgoals.babyai.custom_envs import GoToSpecObj, make_cc, POSSIBLE_CC_POS, str_to_pos
from nlgoals.interfaces.gcbc import babyai_obs_prepare


logger = logging.getLogger(__name__)


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


def print_results(true_goal_results, conf_goal_results):
    print(f"true_goal SR: {true_goal_results.mean()}")
    print(f"conf_goal SR: {conf_goal_results.mean()}")


def save_videos(save_dir, videos):
    videos_dir = os.path.join(save_dir, "videos")
    for video_type, video_list in videos.items():
        for i, video in enumerate(video_list):
            video_path = os.path.join(videos_dir, video_type, f"video_{i}.mp4")
            tv.io.write_video(video_path, video, fps=10)


def save(save_dir, goal, results, videos):
    os.makedirs(goal_dir, exist_ok=True)
    goal_dir = os.path.join(save_dir, goal)
    np.save(os.path.join(goal_dir, "results.npy"), results)
    save_videos(goal_dir, videos)


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
                if isinstance(action, torch.Tensor):
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


def prepare_visual_goal(goal_image, img_transform, device):
    # 1 x 3 x H x W
    return img_transform(goal_image).unsqueeze(0).to(device)


def prepare_textual_goal(goal_text, tokenizer, device):
    out = tokenizer(goal_text, return_tensors="pt")
    return {
        "input_ids": out["input_ids"].to(device),
        "attention_mask": out["attention_mask"].to(device),
    }


def run_rollout(
    env: gym.Env,
    policy: GCBC,
    traj_mode: str,
    seed: int,
    seed_offset: int,
    rollout_steps: int,
    cc_loc_str: str,
    img_transform: Callable,
    tokenizer: Callable,
    verbose: bool = False,
) -> Tuple[bool, bool, np.ndarray, int]:
    """
    Lets the policy interact with the environment until it succeeds or fails

    Args:
        env: the environment to interact with
        policy: the policy to use
        traj_mode: the trajectory mode to use - either visual or textual
        seed: the seed to use
        seed_offset: the offset to be added to the seed in case of an invalid seed
        rollout_steps: the maximum number of steps to take in the environment
        cc_loc_str: The location causally confused with the obj during training
        img_transform: the transform to apply to the images
        tokenizer: the tokenizer to use for the textual goal
        verbose: whether to print the trajectory or not

    Returns:
        Tuple of (true_goal_success, conf_goal_success, video, seed)
    """
    goal_image, goal_text, seed = run_oracle(env, int(seed), seed_offset=seed_offset)
    if traj_mode == "visual":
        goal = prepare_visual_goal(goal_image, img_transform, policy.device)
    else:
        goal = prepare_textual_goal(goal_text, tokenizer, policy.device)

    obs = env.reset(seed=seed)[0]

    true_done = False
    conf_done = False
    rollout_obs = np.zeros((rollout_steps, 3, 224, 224), dtype=np.float32)

    for step in tqdm(range(rollout_steps), disable=not verbose):
        prepared_obs = babyai_obs_prepare(obs, img_transform, policy.device)

        action = policy.step(prepared_obs, goal, traj_mode)
        obs, _reward, true_done, _, _ = env.step(action)

        conf_done = check_conf_done(env, obs["direction"], cc_loc_str)

        # save observation for visualization
        rollout_obs[step] = obs["image"].transpose(2, 0, 1)

        if true_done or conf_done:
            return true_done, conf_done, rollout_obs, seed

    return true_done, conf_done, rollout_obs, seed


def eval_policy(
    num_rollouts: int,
    policy: GCBC,
    env: gym.Env,
    traj_mode: str,
    start_seed: int,
    rollout_steps: int,
    cc_loc_str: str,
    img_transform,
    tokenizer,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, List[np.ndarray]]], np.ndarray]:
    """
    Evaluates the policy for a given number of rollouts

    Args:
        num_rollouts: number of rollouts to evaluate the policy for
        policy: policy to evaluate
        env: environment to evaluate the policy in
        traj_mode: type of trajectory to evaluate the policy on. One of
            "textual", "visual"
        start_seed: Each rollout will have a different (larger) seed starting from this seed.
        rollout_steps: The maximum number of steps to take in the environment
        cc_loc_str: The location causally confused with the obj during training
        img_transform: the transform to apply to the images
        tokenizer: the tokenizer to use for the textual goal
        verbose: Whether to have tqdm progress bars for the rollouts

    Returns:
        Tuple of the (true_goal_results, conf_goal_results, videos, seed_used)
        true_goal_results: np array of 1s and 0s -> whether the true goal was reached
        conf_goal_results: np array of 1s and 0s -> whether the conf goal was reached
        videos: dict of 3 success/fail videos for each goal type (true_goal, conf_goal)
            {   "true_goal": {
                    "success": [video1, video2, video3],
                    "fail": [video1, video2, video3]
                },
                "conf_goal": {
                    "success": [video1, video2, video3],
                    "fail": [video1, video2, video3]
                },
            }
        seeds: seeds used for the evaluation
    """
    true_goal_results = np.zeros(num_rollouts)
    conf_goal_results = np.zeros(num_rollouts)
    videos = {
        "true_goal": {"success": [], "fail": []},
        "conf_goal": {"success": [], "fail": []},
    }
    seeds = np.linspace(
        start=start_seed, stop=start_seed + num_rollouts - 1, num=num_rollouts
    )

    for i in tqdm(range(num_rollouts), desc="Rollouts"):
        seed = seeds[i]
        true_goal_success, conf_goal_success, video, seed_used = run_rollout(
            env,
            policy,
            traj_mode,
            seed,
            num_rollouts,
            rollout_steps,
            cc_loc_str,
            img_transform,
            tokenizer,
            verbose,
        )
        seeds[i] = seed_used
        true_goal_results[i], conf_goal_results[i] = (
            true_goal_success,
            conf_goal_success,
        )
        videos = update_videos(videos, video, true_goal_success, conf_goal_success)

    return true_goal_results, conf_goal_results, videos, seeds


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_kwargs = {
        "obj_kind": args.env.obj_kind,
        "obj_color": args.env.obj_color,
    }
    if args.env.cc.enable:
        make_cc(GoToSpecObj)
        env_kwargs = {
            "cc_obj_kind": args.env.obj_kind,
            "cc_obj_color": args.env.obj_color,
            "cc_obj_pos_str": args.env.cc.obj_pos_str,
            **env_kwargs,
        }
    env = GoToSpecObj(**env_kwargs)
    env = RGBImgObsWrapper(env, tile_size=28)

    ModelClass = gcbc_enum_to_class[args.model_variant]
    policy = ModelClass.load_from_checkpoint(args.model_checkpoint, strict=False)
    if args.clipt_checkpoint is not None:
        clipt_state_dict = torch.load(args.clipt_checkpoint, map_location=device)[
            "state_dict"
        ]
        clipt = CLIPT(**args.clipt.as_dict())
        clipt.load_state_dict(clipt_state_dict, strict=False)
        policy.set_traj_encoder(clipt)

    img_transform = CLIPImageTransform(**args.img_transform.as_dict())
    tokenizer = AutoTokenizer.from_pretrained(args.clipt.clip_model_name)

    true_goal_results, conf_goal_results, videos, seeds = eval_policy(
        args.num_rollouts,
        policy,
        env,
        args.traj_mode,
        args.start_seed,
        args.num_rollout_steps,
        args.env.cc.obj_pos_str,
        img_transform,
        tokenizer,
        args.verbose,
    )

    print_results(true_goal_results, conf_goal_results)

    model_checkpoint_name = os.path.splitext(os.path.basename(args.model_checkpoint))[0]
    save_dir = os.path.join(args.save_dir, model_checkpoint_name)
    save_results(save_dir, true_goal_results, conf_goal_results, videos, seeds)


if __name__ == "__main__":
    import jsonargparse

    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_argument("--start_seed", type=int, default=int(1e9))
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory where to save the <checkpoint>/results.npz and <checkpoint>/videos.npz",
    )

    parser.add_argument(
        "--env.obj_kind",
        type=str,
        default="ball",
        choices=list(OBJECT_TO_IDX.keys()),
    )
    parser.add_argument(
        "--env.obj_color", type=str, default="red", choices=list(COLOR_TO_IDX.keys())
    )
    parser.add_argument(
        "--env.cc.enable",
        type=bool,
        default=False,
        help="Whether to use the causally confused version of the environment",
    )
    parser.add_argument(
        "--env.cc.obj_pos_str",
        type=str,
        required=True,
        choices=POSSIBLE_CC_POS,
        default="bottom right",
    )

    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=100,
        help="Number of rollouts to perform per task",
    )
    parser.add_argument(
        "--num_rollout_steps",
        type=int,
        default=100,
        help="Maximum number of steps to perform per rollout",
    )

    parser.add_argument(
        "--traj_mode",
        type=str,
        choices=["textual", "visual"],
        help="Which trajectory mode to use.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Whether to use TQDM for individual rollouts",
    )

    # model stuff
    parser.add_argument("--model_variant", type=GCBC_ENUM, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_class_arguments(CLIPT, "clipt")
    parser.add_argument("--clipt_checkpoint", type=str, required=False)

    parser.add_class_arguments(CLIPImageTransform, "img_transform")

    args = parser.parse_args()

    main(args)
