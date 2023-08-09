"""
Evaluate RCBC policy on BabyAI (causally confused) environment
"""
import os
from typing import Tuple, Dict, List, Callable

import numpy as np
import gymnasium as gym
import torch
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.wrappers import RGBImgObsWrapper
from tqdm.auto import tqdm

from nlgoals.data.transforms import CLIPImageTransform
from nlgoals.models.rcbc import BABYAI_RCBC, RCBC
from nlgoals.babyai.custom_envs import GoToSpecObj, make_cc, POSSIBLE_CC_POS
from nlgoals.interfaces.rcbc import babyai_obs_prepare
from nlgoals.babyai.eval_utils import (
    check_conf_done,
    run_oracle,
    update_videos,
    print_results,
    save_results,
)


def run_rollout(
    env: gym.Env,
    policy: RCBC,
    seed: int,
    seed_offset: int,
    rollout_steps: int,
    cc_loc_str: str,
    img_transform: Callable,
    task_id: int,
    verbose: bool = False,
) -> Tuple[bool, bool, np.ndarray, int]:
    """
    Lets the policy interact with the environment until it succeeds or fails

    Args:
        env: the environment to interact with
        policy: the policy to use
        seed: the seed to use
        seed_offset: the offset to be added to the seed in case of an invalid seed
        rollout_steps: the maximum number of steps to take in the environment
        cc_loc_str: The location causally confused with the obj during training
        img_transform: the transform to apply to the images
        task_id: the task_id to use for the policy
        verbose: whether to print the trajectory or not

    Returns:
        Tuple of (true_goal_success, conf_goal_success, video, seed)
    """
    _goal_image, _goal_text, seed = run_oracle(env, int(seed), seed_offset=seed_offset)

    obs = env.reset(seed=seed)[0]

    true_done = False
    conf_done = False
    rollout_obs = np.zeros((rollout_steps, 3, 224, 224), dtype=np.uint8)

    for step in tqdm(range(rollout_steps), disable=not verbose):
        if step % 7 == 0:
            policy.reset()
        prepared_obs = babyai_obs_prepare(obs, img_transform, policy.device)

        # condition on a reward of 1, with the appropriate task_id
        action = policy.step(**prepared_obs, reward=torch.Tensor([1]), task_id=task_id)
        obs, _reward, true_done, _, _ = env.step(action.item())

        conf_done = check_conf_done(env, obs["direction"], cc_loc_str)

        # save observation for visualization
        rollout_obs[step] = obs["image"].transpose(2, 0, 1)

        if true_done or conf_done:
            return true_done, conf_done, rollout_obs, seed

    return true_done, conf_done, rollout_obs, seed


def eval_policy(
    num_rollouts: int,
    policy: RCBC,
    env: gym.Env,
    start_seed: int,
    rollout_steps: int,
    cc_loc_str: str,
    img_transform: Callable,
    task_id: int,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, List[np.ndarray]]], np.ndarray]:
    """
    Evaluates the policy for a given number of rollouts

    Args:
        num_rollouts: number of rollouts to evaluate the policy for
        policy: policy to evaluate
        env: environment to evaluate the policy in
        start_seed: Each rollout will have a different (larger) seed starting from this seed.
        rollout_steps: The maximum number of steps to take in the environment
        cc_loc_str: The location causally confused with the obj during training
        img_transform: the transform to apply to the images
        task_id: the task_id to use for the policy
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
            seed,
            num_rollouts,
            rollout_steps,
            cc_loc_str,
            img_transform,
            task_id,
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
    env = GoToSpecObj(**env_kwargs, highlight=False)
    env = RGBImgObsWrapper(env, tile_size=28)

    # Custom-GoToSpecObj-v0-RB is first in nlgoals.babyai.utils.SIZE_TO_ENVS
    task_id = 0

    policy = BABYAI_RCBC.load_from_checkpoint(args.model_checkpoint, strict=False)
    policy.eval()
    policy.to(device)
    _ = torch.set_grad_enabled(False)

    img_transform = CLIPImageTransform(**args.img_transform.as_dict())

    true_goal_results, conf_goal_results, videos, seeds = eval_policy(
        args.num_rollouts,
        policy,
        env,
        args.start_seed,
        args.num_rollout_steps,
        args.env.cc.obj_pos_str,
        img_transform,
        task_id,
        args.verbose,
    )

    print_results(true_goal_results, conf_goal_results)

    model_checkpoint_name = os.path.splitext(os.path.basename(args.model_checkpoint))[0]
    env_dir = "cc" if args.env.cc.enable else "normal"
    save_dir = os.path.join(args.save_dir, model_checkpoint_name, env_dir)
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
        "--verbose",
        type=bool,
        default=False,
        help="Whether to use TQDM for individual rollouts",
    )

    # model stuff
    parser.add_argument("--model_checkpoint", type=str, required=True)

    parser.add_class_arguments(CLIPImageTransform, "img_transform")

    args = parser.parse_args()

    main(args)
