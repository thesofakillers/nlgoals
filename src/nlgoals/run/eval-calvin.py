from typing import Set
from termcolor import colored

from omegaconf import OmegaConf
import hydra
from jsonargparse import Namespace
import torch

from nlgoals.interfaces.gcbc import calvin_obs_prepare, calvin_gcbc_collate

args = Namespace(
    **{
        "data": Namespace(
            **{
                "config_name": "default.yaml",
                "batch_size": 2,
                "shared_memory": False,
                "num_workers": 0,
                "data_dir": "/Users/thesofakillers/repos/thesis/data/calvin/task_D_D",
            }
        ),
    }
)

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_module(config_module="nlgoals.data.calvin.repo.conf.datamodule")
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

device = (
    torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
)

rollout_cfg = OmegaConf.load(
    "src/nlgoals/data/calvin/repo/conf/callbacks/rollout/default.yaml"
)

urdf_data_dir = "/Users/thesofakillers/repos/calvin_env/data/"
egl_dir_path = "/Users/thesofakillers/repos/calvin_env/egl_check/"

# I am so horribly sorry about how terribly opaque this code is. You can thank the CALVIN authors
env = hydra.utils.instantiate(
    rollout_cfg.env_cfg,
    dataset,
    urdf_data_dir,
    device,
    egl_dir_path,
    show_gui=False,
    use_egl=True if device.type == "cuda" else False,
)


def rollout(env, reset_info, model, lang_annotation, rollout_steps, task_oracle, task):
    obs = env.reset(
        robot_obs=reset_info["robot_obs"][0], scene_obs=reset_info["scene_obs"[0]]
    )

    for _step in range(rollout_steps):
        action = model.step(calvin_obs_prepare(obs, lang_annotation), "textual")
        obs, _, _, current_info = env.step(action)

        model.reset()
        start_info = env.get_info()

        completed_tasks: Set = task_oracle.get_task_info_for_set(
            start_info, current_info, {task}
        )
        if len(completed_tasks) > 0:
            print(colored("S", "green"), end=" ")
            return True
    print(colored("F", "red"), end=" ")
    return False
