"""Training of a simple Task Classifier on CALVIN using representations from CLIPT"""
import jsonargparse
import os

import pytorch_lightning as pl
import torch

from nlgoals.data.transforms import TRANSFORM_MAP, TransformName
from nlgoals.data.calvin.transform_configs import CLIPT_PREPARE_CONFIG
from nlgoals.data.calvin.datamodule import CALVINDM
from nlgoals.models.task_classifier import TaskClassifier
from nlgoals.models.clipt import CLIPT
from nlgoals.trainer.task_classifier import TrainerConfig


def train(args):
    """
    Sets up dataloader
    Instantiates model
    Trains model on CrossEntropyLoss
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
    transform_config = CLIPT_PREPARE_CONFIG[args.data.transform_variant]
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
    calvin_dm.setup(stage="fit")
    # model
    model = TaskClassifier(
        traj_encoder_kwargs=args.clipt.as_dict(),
        num_tasks=len(calvin_dm.id_to_task),
        **args.task_classifier.as_dict()
    )
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
        job_type="train" if not args.debug else "debug",
        entity="giulio-uva",
        project="nlgoals",
        mode="disabled" if not args.trainer.logging.enable else "online",
        group=script_host,
        config=args,
        log_model=False,
        tags=['task_classifier']
    )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor="textual/val_accuracy", mode="max"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        **args.trainer.checkpoint.as_dict()
    )
    trainer = pl.Trainer(
        max_epochs=args.trainer.max_epochs,
        accelerator=args.trainer.accelerator,
        devices=args.trainer.devices,
        enable_progress_bar=args.trainer.enable_progress_bar,
        deterministic=True,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        log_every_n_steps=args.trainer.log_every_n_steps,
    )

    trainer.fit(model, calvin_dm)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_class_arguments(
        TaskClassifier, "task_classifier", skip={"traj_encoder_kwargs", "num_tasks"}
    )
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

    train(args)
