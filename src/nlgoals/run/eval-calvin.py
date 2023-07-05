"""
Evaluate a trained GCBC policy on the CALVIN environment/benchmark

I am so horribly sorry about how terribly opaque this code is.
You can thank the CALVIN authors
"""

from typing import Set, Dict, Tuple, Union
from termcolor import colored
import zipfile
from tqdm.auto import tqdm
import os
import json

from omegaconf import OmegaConf
import hydra
import torch
import jsonargparse
import numpy as np
from calvin_agent.wrappers.calvin_env_wrapper import CalvinEnvWrapper
from calvin_env.envs.tasks import Tasks
from torch.utils.data import Dataset

from nlgoals.models.gcbc import GCBC, gcbc_enum_to_class, GCBC_ENUM
from nlgoals.models.clipt import CLIPT
from nlgoals.interfaces.gcbc import (
    calvin_obs_prepare,
    calvin_gcbc_collate,
    calvin_gcbc_collate,
    calvin_gcbc_textual,
    calvin_gcbc_visual,
)


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
        goal = episode["rgb_obs"]["rgb_static"][-1].to(device)
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
    obs = env.reset(
        robot_obs=reset_info["robot_obs"][0], scene_obs=reset_info["scene_obs"][0]
    )
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


def evaluate_policy(
    model: GCBC,
    env: CalvinEnvWrapper,
    dataset: Dataset,
    task_oracle: Tasks,
    tokenizer,
    rollout_steps: int,
    traj_mode: str,
    save_dir: str,
    num_rollouts: int = 100,
    verbose: bool = False,
):
    """
    Evaluate a policy on the CALVIN environment
    For a given dataset, goes through each of the starting states possible for a given
    task, and lets the model interact with the environment until it either solves the
    task or the rollout length is reached

    Args:
        model: the model to evaluate
        env: the environment to evaluate on
        dataset: the dataset providing the starting states and transforms
        task_oracle: the task oracle to check if the task is solved
        tokenizer: the tokenizer to use for the textual input
        traj_mode: either 'textual' or 'visual'
        rollout_steps: the number of steps to rollout for
        save_dir: directory where to save the results.npz and videos.npz
        num_rollouts: the number of rollouts to perform
        verbose: whether to print the results of each rollout
    """
    task_to_idx_dict = dataset.task_to_idx
    number_of_tasks = len(task_to_idx_dict)

    videos = {
        k: np.zeros((rollout_steps, 3, 224, 224), dtype=np.float32)
        for k in task_to_idx_dict.keys()
    }
    videos_metadata = {k: None for k in task_to_idx_dict.keys()}
    results = {k: np.zeros(num_rollouts) for k in task_to_idx_dict.keys()}
    evaluated_idxs = {k: np.zeros(num_rollouts) for k in task_to_idx_dict.keys()}

    for task, idxs in tqdm(
        task_to_idx_dict.items(), desc="Tasks", total=number_of_tasks
    ):
        # sample subset of idxs
        idxs = np.random.choice(idxs, size=num_rollouts, replace=False)
        evaluated_idxs[task] = idxs
        for i, idx in enumerate(tqdm(idxs, desc="Task instances")):
            try:
                episode = dataset[int(idx)]
            except zipfile.BadZipFile as _e:
                print(
                    f"BadZipFile: Something went wrong with idx {idx} of task {task}. Skipping..."
                )
                continue
            reset_info = episode["state_info"]
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
            results[task][i] = was_success
            # save first success video
            if was_success and videos[task][0].sum() == 0:
                videos[task] = video
                videos_metadata[task] = int(idx)
        print(f"{task}: {results[task].sum()} / {len(idxs)}")

    # save results
    print("saving...")

    # organize by traj_mode
    save_dir = os.path.join(save_dir, traj_mode)
    # make the save_dir if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    results_path = os.path.join(save_dir, "results.npz")
    np.savez(results_path, **results)
    evaluated_idxx_path = os.path.join(save_dir, "evaluated_idxs.npz")
    np.savez(evaluated_idxx_path, **evaluated_idxs)
    videos_path = os.path.join(save_dir, "videos.npz")
    np.savez(videos_path, **videos)
    videos_metadata_path = os.path.join(save_dir, "videos_metadata.json")
    with open(videos_metadata_path, "w") as f:
        json.dump(videos_metadata, f)

    print("Done.")

    # overall success rate
    success_rate = sum([outcomes.sum() for outcomes in results.values()]) / sum(
        [len(outcomes) for outcomes in results.values()]
    )
    print(f"SR: {success_rate * 100:.1f}%")


def main(args):
    # datamodule and dataset
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module(
        config_module="nlgoals.data.calvin.repo.conf.datamodule"
    )
    datamodule_cfg = hydra.compose(
        config_name=args.data.config_name,
        overrides=[] if args.data.shared_memory is True else ["datasets=vision_lang"],
    )
    datamodule_cfg.batch_size = args.data.batch_size
    datamodule_cfg.num_workers = args.data.num_workers
    datamodule_cfg.root_data_dir = args.data.data_dir
    datamodule = hydra.utils.instantiate(datamodule_cfg)
    datamodule.collator.custom_collate_fn = calvin_gcbc_collate
    datamodule.prepare_data()
    datamodule.setup()
    dataset = datamodule.val_dataloader().dataset.datasets["lang"]
    tokenizer = datamodule.collator.text_processor
    # device
    device = (
        torch.device("cuda:" + str(0))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    # environment
    rollout_cfg = OmegaConf.load(args.rollout_cfg_path)
    env = hydra.utils.instantiate(
        rollout_cfg.env_cfg,
        dataset,
        args.urdf_data_dir,
        device,
        args.egl_dir_path,
        show_gui=False,
        use_egl=True if device.type == "cuda" else False,
    )
    # task oracle
    task_oracle_cfg = OmegaConf.load(args.task_oracle_cfg)
    task_oracle = hydra.utils.instantiate(task_oracle_cfg)
    # model
    ModelClass = gcbc_enum_to_class[args.model_variant]
    model = ModelClass.load_from_checkpoint(args.model_checkpoint)
    if args.clipt_checkpoint is not None:
        clipt_state_dict = torch.load(args.clipt_checkpoint, map_location=device)[
            "state_dict"
        ]
        clipt = CLIPT(**args.clipt.as_dict())
        clipt.load_state_dict(clipt_state_dict, strict=False)
        model.set_traj_encoder(clipt)
    model.prepare_visual_batch = calvin_gcbc_visual
    model.prepare_textual_batch = calvin_gcbc_textual
    model.eval()
    _ = torch.set_grad_enabled(False)

    evaluate_policy(
        model,
        env,
        dataset,
        task_oracle,
        tokenizer,
        args.traj_mode,
        args.rollout_steps,
        args.save_dir,
        args.num_rollouts,
        args.verbose,
    )


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description=__doc__)

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
        help="Directory where to save the results.npz and videos.npz",
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

    parser.add_argument(
        "--data.config_name", type=str, required=True, default="default.yaml"
    )
    parser.add_argument("--data.batch_size", type=int, default=32)
    parser.add_argument("--data.num_workers", type=int, default=18)
    parser.add_argument(
        "--data.data_dir", type=str, required=True, help="Must be absolute path"
    )
    parser.add_argument("--data.shared_memory", type=bool, default=True)

    args = parser.parse_args()

    main(args)
