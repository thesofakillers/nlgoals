"""Training of CLIPh: Contrastive Language Image Pretraining for Trajectories"""
import jsonargparse
import os

import pytorch_lightning as pl
from nlgoals.data.transforms import TRANSFORM_MAP, TransformName

from nlgoals.models.clipt import CLIPT
from nlgoals.trainer import TrainerConfig
from nlgoals.data.calvin.datamodule import CALVINDM
from nlgoals.data.calvin.transform_configs import CLIPTPrepareForCALVIN


def train(args):
    """
    Sets up dataloader
    Instantiates model
    Trains model using contrastive loss between image traj and text pairs
    """
    pl.seed_everything(args.seed, workers=True)

    # disable tokenizer parallelism because we have multiple workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.data.transform_name is not None:
        data_transform = TRANSFORM_MAP[args.data.transform_name.value](
            **args.data.transform_kwargs
        )
    else:
        data_transform = None
    calvin_dm = CALVINDM(**args.data.as_dict(), transform=data_transform)

    model = CLIPT(**args.clipt.as_dict())

    script_host = "slurm" if "SLURM_JOB_ID" in os.environ else "local"
    logger = pl.loggers.WandbLogger(
        job_type="train" if not args.debug else "debug",
        entity="giulio-uva",
        project="nlgoals",
        mode="disabled" if not args.trainer.logging.enable else "online",
        group=script_host,
        config=args,
        log_model=False,
    )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss", mode="min"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        **args.trainer.checkpoint.as_dict()
    )
    trainer = pl.Trainer(
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger,
        max_epochs=args.trainer.max_epochs,
        devices=args.trainer.devices,
        accelerator=args.trainer.accelerator,
        deterministic=True,
    )

    trainer.fit(model, calvin_dm)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_class_arguments(CLIPT, "clipt")

    parser.add_class_arguments(CALVINDM, "data", skip={"transform"})
    parser.add_argument(
        "--data.transform_name", type=TransformName, default="clipt-prepare"
    )
    parser.add_dataclass_arguments(CLIPTPrepareForCALVIN, "data.transform_kwargs")

    parser.add_dataclass_arguments(TrainerConfig, "trainer")
    parser.add_argument("--seed", type=int, default=42)

    parser.link_arguments("seed", "data.seed", apply_on="parse")
    parser.link_arguments("data.num_frames", "clipt.num_frames", apply_on="parse")
    parser.link_arguments(
        "clipt.clip_model", "data.transform_kwargs.clip_model", apply_on="parse"
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    train(args)
