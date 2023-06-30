"""
Evaluate a trained GCBC policy on the CALVIN environment/benchmark

I am so horribly sorry about how terribly opaque this code is.
You can thank the CALVIN authors
"""

from typing import Set
from termcolor import colored
from collections import Counter

from omegaconf import OmegaConf
import hydra
import torch
import jsonargparse

from nlgoals.models.gcbc import gcbc_enum_to_class, GCBC_ENUM
from nlgoals.models.clipt import CLIPT
from nlgoals.interfaces.gcbc import (
    calvin_obs_prepare,
    calvin_gcbc_collate,
    calvin_gcbc_collate,
    calvin_gcbc_textual,
    calvin_gcbc_visual,
)


def rollout(
    env, reset_info, model, lang_annotation, rollout_steps, task_oracle, task, tokenizer
):
    obs = env.reset(
        robot_obs=reset_info["robot_obs"][0], scene_obs=reset_info["scene_obs"[0]]
    )

    start_info = env.get_info()

    model.reset()
    for _step in range(rollout_steps):
        # (1, 7) squeezed into (7,)
        action = model.step(
            calvin_obs_prepare(obs, lang_annotation, tokenizer), "textual"
        ).squeeze()
        obs, _, _, current_info = env.step(action)

        # check if current step solves the task
        completed_tasks: Set = task_oracle.get_task_info_for_set(
            start_info, current_info, {task}
        )
        if len(completed_tasks) > 0:
            print(colored("S", "green"), end=" ")
            return True
    print(colored("F", "red"), end=" ")
    return False


def evaluate_policy(model, env, dataset, task_oracle, tokenizer):
    results = Counter()
    task_to_idx_dict = dataset.task_to_idx

    for task, idxs in task_to_idx_dict.items():
        for idx in idxs:
            episode = dataset[int(idx)]
            reset_info = episode["state_info"]
            lang_annotation = episode["lang"]
            results[task] += rollout(
                env,
                reset_info,
                model,
                lang_annotation,
                args.rollout_steps,
                task_oracle,
                task,
                tokenizer,
            )
        print(f"{task}: {results[task]} / {len(idxs)}")
    # overall success rate
    success_rate = sum(results.values()) / sum(
        len(x) for x in task_to_idx_dict.values()
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
    task_cfg = OmegaConf.load(args.task_cfg)
    task_oracle = hydra.utils.instantiate(task_cfg)
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

    evaluate_policy(model, env, dataset, task_oracle, tokenizer)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_argument("--rollout_steps", type=int, default=240)
    # "nlgoals/data/calvin/repo/conf/callbacks/rollout/default.yaml",
    parser.add_argument("--rollout_cfg_path", type=str, required=True)

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
