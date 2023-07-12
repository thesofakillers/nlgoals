"""Training of CLIPT: Contrastive Language Image Pretraining for Trajectories"""
import jsonargparse
import os

import pytorch_lightning as pl
import torch

from nlgoals.data.transforms import TRANSFORM_MAP, TransformName
from nlgoals.models.clipt import CLIPT
from nlgoals.trainer.clipt import TrainerConfig
from nlgoals.data.calvin.legacy.datamodule import CALVINDM
from nlgoals.data.calvin.transform_configs import CLIPT_PREPARE_CONFIG


def train(args):
    """
    Sets up dataloader
    Instantiates model
    Trains model using contrastive loss between image traj and text pairs
    """
    # determinism
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # model
    if args.model_checkpoint is not None:
        model = CLIPT.load_from_checkpoint(
            checkpoint_path=args.model_checkpoint,
            map_location=device,
            strict=False,
            contextualize_text=args.clipt.contextualize_text,
            freeze_vision=args.clipt.freeze_vision,
            freeze_lang=args.clipt.freeze_lang,
            precomputed_clip=args.clipt.precomputed_clip,
        )
        model.handle_freezing()
    else:
        model = CLIPT(**args.clipt.as_dict())
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
        tags=["clipt" if not args.clipt.contextualize_text else "cclipt"],
    )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss", mode="min"
    )
    args.trainer.checkpoint.filename = (
        f"{args.trainer.checkpoint.filename}-s{args.seed}"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        **args.trainer.checkpoint.as_dict()
    )
    trainer = pl.Trainer(
        max_epochs=args.trainer.max_epochs,
        max_time={"hours": 1},
        accelerator=args.trainer.accelerator,
        devices=args.trainer.devices,
        enable_progress_bar=args.trainer.enable_progress_bar,
        deterministic=True,
        logger=logger,
        callbacks=[checkpoint_callback],# early_stopping],
        log_every_n_steps=args.trainer.log_every_n_steps,
    )

    trainer.fit(model, calvin_dm)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=False,
        help="Path to model checkpoint, to resume training",
    )

    parser.add_class_arguments(CLIPT, "clipt")

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
