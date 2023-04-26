"""Training of CLIPh: Contrastive Language Image Pretraining for Trajectories"""
import jsonargparse
import os

import pytorch_lightning as pl

from nlgoals.models.clipt import CLIPT
from nlgoals.trainer import TrainerConfig
from nlgoals.data.calvin.datamodule import CALVINDM


def train(args):
    """
    Sets up dataloader
    Instantiates model
    Trains model using contrastive loss between image traj and text pairs
    """
    pl.seed_everything(args.seed)

    calvin_dm = CALVINDM(**args.data.as_dict())
    model = CLIPT(**args.clipts.as_dict())

    script_host = "slurm" if "SLURM_JOB_ID" in os.environ else "local"

    logger = pl.loggers.WandbLogger(
        job_type="train",
        entity="giulio-uva",
        project="nlgoals",
        mode="disabled" if args.trainer.logging.disable else "online",
        group=script_host,
        config=args,
        log_model=False,
    )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss", mode="min"
    )

    trainer = pl.Trainer(
        callbacks=[early_stopping],
        logger=logger,
        max_epochs=args.trainer.max_epochs,
        devices=args.trainer.devices,
        accelerator=args.trainer.accelerator,
    )
    trainer.fit(model, calvin_dm)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_class_arguments(CLIPT, "clipt")
    parser.add_class_arguments(CALVINDM, "data")
    parser.add_argument("--trainer", type=TrainerConfig, default=TrainerConfig())
    parser.add_argument("--seed", type=int, default=42)
    parser.link_arguments("seed", "data.seed", apply_on="parse")

    args = parser.parse_args()

    train(args)
