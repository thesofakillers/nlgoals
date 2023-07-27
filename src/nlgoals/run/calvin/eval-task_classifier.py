import jsonargparse
import os

import pytorch_lightning as pl
import torch

from nlgoals.data.transforms import TRANSFORM_MAP, TransformName
from nlgoals.interfaces.clipt import CALVIN_CLIPT_PREPARE_CONFIG
from nlgoals.data.calvin.legacy.datamodule import CALVINDM
from nlgoals.models.task_classifier import TaskClassifier
from nlgoals.models.clipt import CLIPT
from nlgoals.trainer.task_classifier import TrainerConfig


def evaluate(args):
    """
    Sets up dataloader
    Instantiates model
    Evals model
    """
    # device
    if args.trainer.accelerator == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.trainer.accelerator
    # determinism
    pl.seed_everything(args.seed, workers=True)
    # disable tokenizer parallelism because we have multiple workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # transforms
    transform_config = CALVIN_CLIPT_PREPARE_CONFIG[args.data.transform_variant]
    transform_config["mode"] = args.data.transform_variant
    if args.data.transform_name is not None:
        data_transform = TRANSFORM_MAP[args.data.transform_name.value](
            **transform_config
        )
    else:
        data_transform = None
    # datamodule
    calvin_dm = CALVINDM(**args.data.as_dict(), transform=data_transform)
    calvin_dm.prepare_data()
    calvin_dm.setup(stage="test")
    # model
    model = TaskClassifier.load_from_checkpoint(args.checkpoint_path, strict=False)
    if args.clipt_checkpoint is not None:
        clipt_state_dict = torch.load(args.clipt_checkpoint, map_location=device)[
            "state_dict"
        ]
        clipt = CLIPT(**args.clipt.as_dict())
        clipt.load_state_dict(clipt_state_dict, strict=False)
        model.set_traj_encoder(clipt)
    # trainer
    script_host = "slurm" if "SLURM_JOB_ID" in os.environ else "local"
    logger = pl.loggers.WandbLogger(
        job_type="eval" if not args.debug else "debug",
        entity="giulio-uva",
        project="nlgoals",
        mode="disabled" if not args.trainer.logging.enable else "online",
        group=script_host,
        config=args,
        log_model=False,
        tags=["task_classifier"],
    )
    trainer = pl.Trainer(
        accelerator=args.trainer.accelerator,
        devices=args.trainer.devices,
        enable_progress_bar=args.trainer.enable_progress_bar,
        deterministic=True,
        logger=logger,
        log_every_n_steps=args.trainer.log_every_n_steps,
    )

    trainer.test(model, calvin_dm)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_class_arguments(CLIPT, "clipt")
    parser.add_argument("--clipt-checkpoint", type=str, required=False)

    parser.add_class_arguments(CALVINDM, "data", skip={"transform"})

    # transforms
    parser.add_argument(
        "--data.transform_name", type=TransformName, default="clipt-prepare"
    )
    parser.add_argument(
        "--data.transform_variant",
        type=str,
        default="without_clip",
    )

    parser.add_dataclass_arguments(TrainerConfig, "trainer")
    parser.add_argument("--seed", type=int, default=42)

    parser.link_arguments("seed", "data.seed", apply_on="parse")
    parser.link_arguments("data.num_frames", "clipt.num_frames", apply_on="parse")

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    evaluate(args)
