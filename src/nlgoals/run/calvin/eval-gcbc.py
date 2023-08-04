"""
Batched evaluation of a trained GCBC policy on the CALVIN environment/benchmark using
python multiprocessing.
"""

from typing import Set, Dict, Tuple, Union, Any
from termcolor import colored
import zipfile
from tqdm.auto import tqdm
import os
import pickle
from PIL import Image

from omegaconf import OmegaConf
import hydra
import torch
import jsonargparse
import numpy as np
from calvin_agent.wrappers.calvin_env_wrapper import CalvinEnvWrapper
from calvin_env.envs.tasks import Tasks
from torch.utils.data import Dataset
import torchvision as tv
from nlgoals.models.components.action_decoders.calvin import CALVINActionDecoder

from nlgoals.models.gcbc import GCBC, gcbc_enum_to_class, GCBC_ENUM
from nlgoals.models.clipt import CLIPT
from nlgoals.interfaces.gcbc import (
    calvin_obs_prepare,
    calvin_gcbc_collate,
    calvin_gcbc_collate,
    calvin_gcbc_textual,
    calvin_gcbc_visual,
)

# fmt: off
TASK_NAMES = (
    "close_drawer", "lift_blue_block_drawer", "lift_blue_block_slider",
    "lift_blue_block_table", "lift_pink_block_drawer", "lift_pink_block_slider",
    "lift_pink_block_table", "lift_red_block_drawer", "lift_red_block_slider",
    "lift_red_block_table", "move_slider_left", "move_slider_right", "open_drawer",
    "place_in_drawer", "place_in_slider", "push_blue_block_left",
    "push_blue_block_right", "push_into_drawer", "push_pink_block_left",
    "push_pink_block_right", "push_red_block_left", "push_red_block_right",
    "rotate_blue_block_left", "rotate_blue_block_right", "rotate_pink_block_left",
    "rotate_pink_block_right", "rotate_red_block_left", "rotate_red_block_right",
    "stack_block", "turn_off_led", "turn_off_lightbulb", "turn_on_led",
    "turn_on_lightbulb", "unstack_block",
)
# fmt: on


def create_dataset(data_args, collator: bool):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module(
        config_module="nlgoals.data.calvin.repo.conf.datamodule"
    )
    datamodule_cfg = hydra.compose(
        config_name=data_args.config_name,
        overrides=[] if data_args.shared_memory is True else ["datasets=vision_lang"],
    )
    datamodule_cfg.batch_size = data_args.batch_size
    datamodule_cfg.num_workers = data_args.num_workers
    datamodule_cfg.root_data_dir = data_args.data_dir
    datamodule = hydra.utils.instantiate(datamodule_cfg, instantiate_collator=collator)
    datamodule.prepare_data()
    datamodule.setup()
    if collator:
        tokenizer = datamodule.collator.text_processor
        datamodule.collator.custom_collate_fn = calvin_gcbc_collate
    else:
        tokenizer = None

    dataset = datamodule.val_datasets["lang"]

    return dataset, tokenizer


def create_environment(rollout_cfg_path, urdf_data_dir, egl_dir_path, dataset, device):
    rollout_cfg = OmegaConf.load(rollout_cfg_path)
    env = hydra.utils.instantiate(
        rollout_cfg.env_cfg,
        dataset,
        urdf_data_dir,
        device,
        egl_dir_path,
        show_gui=False,
        use_egl=True if device.type == "cuda" else False,
    )
    return env


def normalize_tensor(tensor):
    # move from -1, 1 to 0, 1
    tensor = tensor / 2 + 0.5
    return tensor


def prep_video(video):
    # cut off empty frames
    frame_sums = np.sum(video, axis=(1, 2, 3))
    where_0 = np.where(frame_sums == 0)[0]
    end_frame = where_0[0] if len(where_0) > 0 else len(video)
    video = video[:end_frame]
    # put channel dimension last
    video = video.transpose((0, 2, 3, 1))
    # move from -1, 1 to 0, 1
    video = normalize_tensor(video)
    # convert to uint8
    video = (video * 255).astype(np.uint8)
    return video


def save_video_textual(video: np.ndarray, save_dir, video_meta: Dict):
    lang_goal: str = video_meta["goal"]
    save_path = os.path.join(save_dir, f"{video_meta['episode_idx']}_{lang_goal}.mp4")
    tv.io.write_video(save_path, video, fps=20)


def save_video_visual(video: np.ndarray, save_dir, video_meta: Dict):
    # video
    video_name = f"{video_meta['episode_idx']}_video.mp4"
    video_save_path = os.path.join(save_dir, video_name)
    tv.io.write_video(video_save_path, video, fps=20)
    # goal
    image_name = f"{video_meta['episode_idx']}_goal.png"
    image_save_path = os.path.join(save_dir, image_name)
    image = video_meta["goal"]
    image = normalize_tensor(image)
    image = (image * 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    im = Image.fromarray(image)
    im.save(image_save_path)


def save_video(video: np.ndarray, traj_mode: str, save_dir, video_meta: Dict):
    prepd_video = prep_video(video)

    if traj_mode == "textual":
        save_video_textual(prepd_video, save_dir, video_meta)
    else:
        save_video_visual(prepd_video, save_dir, video_meta)


def get_goal(
    episode, traj_mode, tokenizer, device
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    if traj_mode == "textual":
        goal = {}
        lang = episode["lang"]
        proc_lang = tokenizer(lang, return_tensors="pt")
        goal["input_ids"] = proc_lang["input_ids"].to(device)
        goal["attention_mask"] = proc_lang["attention_mask"].to(device)
    else:
        # 1 x 3 x H x W
        goal = episode["rgb_obs"]["rgb_static"][-1].unsqueeze(0).to(device)
    return goal


def rollout(
    env: CalvinEnvWrapper,
    reset_info: Dict,
    model: GCBC,
    goal: Union[str, torch.Tensor],
    traj_mode: str,
    rollout_steps: int,
    task_oracle: Tasks,
    task: str,
    verbose: bool = False,
) -> Tuple[bool, np.ndarray]:
    rollout_obs = np.zeros((rollout_steps, 3, 224, 224), dtype=np.float32)
    # the starting state
    obs = env.reset(**reset_info)
    start_info = env.get_info()

    model.reset()
    for _step in tqdm(range(rollout_steps), desc="Steps", disable=not verbose):
        prepared_obs = calvin_obs_prepare(obs, model.device)
        # (1, 7) squeezed into (7,)
        action = model.step(prepared_obs, goal, traj_mode).squeeze()
        obs, _, _, current_info = env.step(action)

        # save observation for visualization
        rollout_obs[_step] = obs["rgb_obs"]["rgb_static"].squeeze().cpu().numpy()

        # check if current step solves the task
        completed_tasks: Set = task_oracle.get_task_info_for_set(
            start_info, current_info, {task}
        )
        if len(completed_tasks) > 0:
            if verbose:
                print(colored("S", "green"), end="\n")
            return True, rollout_obs
    if verbose:
        print(colored("F", "red"), end="\n")
    return False, rollout_obs


def evaluate_task(
    task,
    dataset,
    env,
    idxs,
    num_rollouts,
    suggestive_start,
    traj_mode,
    tokenizer,
    model,
    rollout_steps,
    task_oracle,
    verbose,
    target_device,
):
    # move model to target device
    model.to(target_device)
    # sample subset of idxs
    eval_idxs = np.random.choice(idxs, size=num_rollouts, replace=False)
    save_eval_idxs = eval_idxs.copy()
    videos = {"success": [], "fail": []}
    videos_metadata = {"success": [], "fail": []}
    results = np.zeros(num_rollouts)
    # use tqdm only when using single process
    for i, idx in enumerate(tqdm(eval_idxs, desc="Rollouts")):
        # if BadZipFile, try another idx until it works (should be rare)
        while True:
            try:
                episode = dataset[int(idx)]
                save_eval_idxs[i] = idx
                # it works! we can break out of the while loop
                break
            except zipfile.BadZipFile as _e:
                print(
                    f"BadZipFile: Something went wrong with idx {idx} of task {task}."
                    " Trying different idx..."
                )
                # avoid sampling already sampled idxs
                idx = np.random.choice(np.setdiff1d(idxs, save_eval_idxs), size=1)[0]
                continue
        reset_info = {
            "robot_obs": (
                episode["state_info"]["robot_obs"][0] if suggestive_start else None
            ),
            "scene_obs": episode["state_info"]["scene_obs"][0],
        }
        goal = get_goal(episode, traj_mode, tokenizer, model.device)
        was_success, video = rollout(
            env=env,
            reset_info=reset_info,
            model=model,
            goal=goal,
            traj_mode=traj_mode,
            rollout_steps=rollout_steps,
            task_oracle=task_oracle,
            task=task,
            verbose=verbose,
        )
        results[i] = was_success
        # save first 3 videos
        if was_success:
            if len(videos["success"]) < 3:
                videos["success"].append(video)
                # parse goal
                goal = (
                    episode["lang"]
                    if traj_mode == "textual"
                    else episode["rgb_obs"]["rgb_static"][-1].squeeze().cpu().numpy()
                )
                # and keep track of it
                videos_metadata["success"].append(
                    {"episode_idx": int(idx), "goal": goal}
                )
        else:
            if len(videos["fail"]) < 3:
                videos["fail"].append(video)
                # parse goal
                goal = (
                    episode["lang"]
                    if traj_mode == "textual"
                    else episode["rgb_obs"]["rgb_static"][-1].squeeze().cpu().numpy()
                )
                # and keep track of it
                videos_metadata["fail"].append({"episode_idx": int(idx), "goal": goal})
    print(f"{task}: {results.sum()} / {len(eval_idxs)}")
    return (task, (save_eval_idxs, results, videos, videos_metadata))


def eval_policy(
    model: GCBC,
    env,
    dataset: Dataset,
    task: str,
    task_oracle: Tasks,
    tokenizer,
    traj_mode: str,
    rollout_steps: int,
    suggestive_start: bool,
    num_rollouts: int = 100,
    verbose: bool = False,
) -> Tuple[Dict[str, Any]]:
    """
    Returns:
        A tuple of the following dictionaries with keys as tasks:
        - results: whether each rollout resulted in a success or not
        - evaluated_idxs: the idxs of the episodes that were evaluated
        - videos: the videos of the first 3 successful and failed rollouts
        - videos_metadata: the metadata of the first 3 successful and failed rollouts
    """
    task_to_idx_dict = dataset.task_to_idx

    target_device = model.device
    model = model.to("cpu")

    idxs = task_to_idx_dict[task]

    eval_output = evaluate_task(
        task,
        dataset,
        env,
        idxs,
        num_rollouts,
        suggestive_start,
        traj_mode,
        tokenizer,
        model,
        rollout_steps,
        task_oracle,
        verbose,
        target_device,
    )

    return eval_output


def eval_and_save(
    model: GCBC,
    env,
    dataset: Dataset,
    task_name: str,
    task_oracle: Tasks,
    tokenizer,
    traj_mode: str,
    rollout_steps: int,
    save_dir: str,
    suggestive_start: bool,
    num_rollouts: int = 100,
    verbose: bool = False,
):
    """
    Evaluate a policy on the CALVIN environment for a given task
    For a given dataset, goes through each of the starting states possible for a given
    task, and lets the model interact with the environment until it either solves the
    task or the rollout length is reached

    Args:
        model: the model to evaluate
        env: the environment to evaluate on
        dataset: the dataset providing the starting states and transforms
        task_name: the name of the task we wish to evaluate
        task_oracle: the task oracle to check if the task is solved
        tokenizer: the tokenizer to use for the textual input
        traj_mode: either 'textual' or 'visual'
        rollout_steps: the number of steps to rollout for
        save_dir: directory where to save the results.npz and videos.npz
        suggestive_start: whether to use the suggestive start or not
        num_rollouts: the number of rollouts to perform
        verbose: whether to print the results of each rollout
    """
    eval_output = eval_policy(
        model,
        env,
        dataset,
        task_name,
        task_oracle,
        tokenizer,
        traj_mode,
        rollout_steps,
        suggestive_start,
        num_rollouts,
        verbose,
    )
    (task, (eval_idxs, results, videos, videos_metadata)) = eval_output

    # save results
    print("saving...")

    # organize by sug_start/traj_mode/task
    save_dir = os.path.join(save_dir, traj_mode, task)
    # make the save_dir if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # results: sug_start/traj_mode/task/results.npz
    results_path = os.path.join(save_dir, "results.npz")
    np.savez(results_path, results)
    # evaluated_idxs: sug_start/traj_mode/task/evaluated_idxs.npz
    eval_idxs_path = os.path.join(save_dir, "evaluated_idxs.npz")
    np.savez(eval_idxs_path, eval_idxs)

    # videos: sug_start/traj_mode/task/videos/
    videos_dir = os.path.join(save_dir, "videos")
    # /success
    success_dir = os.path.join(videos_dir, "success")
    os.makedirs(success_dir, exist_ok=True)
    for video, video_meta in zip(videos["success"], videos_metadata["success"]):
        save_video(video, traj_mode, success_dir, video_meta)
    # /fail
    fail_dir = os.path.join(videos_dir, "fail")
    os.makedirs(fail_dir, exist_ok=True)
    for video, video_meta in zip(videos["fail"], videos_metadata["fail"]):
        save_video(video, traj_mode, fail_dir, video_meta)

    # video metadata: sug_start/traj_mode/task/videos_metadata.pkl
    videos_metadata_path = os.path.join(save_dir, "videos_metadata.pkl")
    with open(videos_metadata_path, "wb") as f:
        pickle.dump(videos_metadata, f)

    print("Done.")


def main(args):
    dataset, tokenizer = create_dataset(args.data, True)
    # device
    device = (
        torch.device("cuda:" + str(0))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    # environment
    env = create_environment(
        args.rollout_cfg_path, args.urdf_data_dir, args.egl_dir_path, dataset, device
    )
    # task oracle
    task_oracle_cfg = OmegaConf.load(args.task_oracle_cfg)
    task_oracle = hydra.utils.instantiate(task_oracle_cfg)
    # model
    ModelClass = gcbc_enum_to_class[args.model_variant]
    model = ModelClass.load_from_checkpoint(
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
        model.set_traj_encoder(clipt)
    model.prepare_visual_batch = calvin_gcbc_visual
    model.prepare_textual_batch = calvin_gcbc_textual
    if args.random_goals:
        model.random_traj_embs = True
    model.eval()
    model.to(device)
    _ = torch.set_grad_enabled(False)

    model_checkpoint_name = os.path.splitext(os.path.basename(args.model_checkpoint))[0]
    save_dir = os.path.join(
        args.save_dir,
        model_checkpoint_name,
        "sug_starts" if args.suggestive_start else "non_sug_starts",
    )

    eval_and_save(
        model=model,
        env=env,
        dataset=dataset,
        task_name=args.task_name,
        task_oracle=task_oracle,
        tokenizer=tokenizer,
        traj_mode=args.traj_mode,
        rollout_steps=args.rollout_steps,
        save_dir=save_dir,
        suggestive_start=args.suggestive_start,
        num_rollouts=args.num_rollouts,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Which task to evaluate the policy on",
        choices=TASK_NAMES,
    )
    parser.add_argument(
        "--suggestive_start",
        type=bool,
        default=True,
        help="Whether to use suggestive starting robot positions.",
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
        help="Whether to print out the rollout steps and results.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory where to save the <checkpoint>/results.npz and <checkpoint>/videos.npz",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=100,
        help="Number of rollouts to perform per task",
    )

    parser.add_argument("--rollout_steps", type=int, default=240)
    # "nlgoals/data/calvin/repo/conf/callbacks/rollout/default.yaml",
    parser.add_argument("--rollout_cfg_path", type=str, required=True)
    # "nlgoals/data/calvin/repo/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml"
    parser.add_argument("--task_oracle_cfg", type=str, required=True)

    # "calvin_env/data/"
    parser.add_argument("--urdf_data_dir", type=str, required=True)
    # "calvin_env/egl_check/"
    parser.add_argument("--egl_dir_path", type=str, required=True)

    parser.add_argument("--model_variant", type=GCBC_ENUM, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_class_arguments(CLIPT, "clipt")
    parser.add_argument("--clipt_checkpoint", type=str, required=False)

    parser.add_class_arguments(
        CALVINActionDecoder, "action_decoder", skip={"out_dim", "hidden_dim"}
    )

    parser.add_argument(
        "--data.config_name", type=str, required=True, default="default.yaml"
    )
    parser.add_argument("--data.batch_size", type=int, default=32)
    parser.add_argument("--data.num_workers", type=int, default=18)
    parser.add_argument(
        "--data.data_dir", type=str, required=True, help="Must be absolute path"
    )
    parser.add_argument("--data.shared_memory", type=bool, default=True)
    parser.add_argument("--random_goals", type=bool, default=False)

    args = parser.parse_args()

    main(args)
