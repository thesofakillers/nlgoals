"""
Evaluate GCBC policy on BabyAI (causally confused) environment
"""
import os
from typing import Tuple, Dict, List

import numpy as np
import gymnasium as gym
import torch
import torchvision as tv
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

from nlgoals.models.clipt import CLIPT
from nlgoals.models.gcbc import GCBC, gcbc_enum_to_class, GCBC_ENUM
from nlgoals.babyai.custom_envs import GoToSpecObj, make_cc, POSSIBLE_CC_POS


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
    np.save(os.path.join(save_dir, "seeds.npy"), seeds)
    save(save_dir, "true_goal", true_goal_results, videos["true_goal"])
    save(save_dir, "conf_goal", conf_goal_results, videos["conf_goal"])


def run_rollout(
    env: gym.Env, policy: GCBC, traj_mode: str, seed: int
) -> Tuple[bool, bool, np.ndarray, int]:
    raise NotImplementedError


def eval_policy(
    num_rollouts: int, policy: GCBC, env: gym.Env, traj_mode: str, start_seed: int
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
    seeds = np.linspace(start=start_seed, num=num_rollouts)

    for i in range(num_rollouts):
        seed = seeds[i]
        true_goal_success, conf_goal_success, video, seed_used = run_rollout(
            env, policy, seed, num_rollouts, traj_mode
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

    ModelClass = gcbc_enum_to_class[args.model_variant]
    policy = ModelClass.load_from_checkpoint(
        args.model_checkpoint,
        strict=False,
        action_decoder_kwargs=args.action_decoder.as_dict(),
    )
    if args.clipt_checkpoint is not None:
        clipt_state_dict = torch.load(args.clipt_checkpoint, map_location=device)[
            "state_dict"
        ]
        clipt = CLIPT(**args.clipt.as_dict())
        clipt.load_state_dict(clipt_state_dict, strict=False)
        policy.set_traj_encoder(clipt)

    true_goal_results, conf_goal_results, videos, seeds = eval_policy(
        args.num_rollouts, policy, env, args.traj_mode, args.start_seed
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
        "--env.cc.obj_pos_str", type=str, required=False, choices=POSSIBLE_CC_POS
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

    # model stuff
    parser.add_argument("--model_variant", type=GCBC_ENUM, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_class_arguments(CLIPT, "clipt")
    parser.add_argument("--clipt_checkpoint", type=str, required=False)

    main()
