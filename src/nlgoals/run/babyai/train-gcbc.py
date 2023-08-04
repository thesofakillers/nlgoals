import os

import torch
import pytorch_lightning as pl

from nlgoals.data.babyai.datamodule import BabyAIDM
from nlgoals.data.transforms import TRANSFORM_MAP, TransformName
from nlgoals.interfaces.clipt import BABYAI_CLIPT_PREPARE_CONFIG
from nlgoals.models.gcbc import GCBC, GCBC_ENUM, gcbc_enum_to_class
from nlgoals.models.clipt import CLIPT
from nlgoals.models.perception_encoders.vision_encoder import VisionEncoder
from nlgoals.models.perception_encoders.proprio_encoder import ProprioEncoder
from nlgoals.models.components.action_decoders.babyai import BabyAIActionDecoder
from nlgoals.trainer.gcbc import TrainerConfig


def train(args):
    # device
    if args.trainer.accelerator == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.trainer.accelerator
    # determinism
    pl.seed_everything(args.seed, workers=True)
    # transforms
    transform_config = BABYAI_CLIPT_PREPARE_CONFIG[args.data.transform_variant]
    transform_config["mode"] = args.data.transform_variant
    if args.data.transform_name is not None:
        data_transform = TRANSFORM_MAP[args.data.transform_name.value](
            **transform_config
        )
    else:
        data_transform = None
    # datamodule
    datamodule = BabyAIDM(**args.data.as_dict(), transform=data_transform)
    datamodule.prepare_data()
    datamodule.setup(stage=None if not args.debug else "debug")
    # model
    ModelClass = gcbc_enum_to_class[args.model_variant]
    if args.model_checkpoint is not None:
        model = ModelClass.load_from_checkpoint(args.model_checkpoint, strict=False)
    else:
        model = ModelClass(
            traj_encoder_kwargs=args.clipt.as_dict(),
            vision_encoder_kwargs=args.vision_encoder.as_dict(),
            proprio_encoder_kwargs=args.proprio_encoder.as_dict(),
            action_decoder_kwargs=args.action_decoder.as_dict(),
            **args.gcbc.as_dict(),
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
        id=args.wandb_id,
        resume="must" if args.wandb_id is not None else None,
        mode="disabled" if not args.trainer.logging.enable else "online",
        group=script_host,
        config=args,
        log_model=False,
        tags=["gcbc", *model.datasets],
    )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor="textual/val_loss", mode="min", strict=False
    )
    args.trainer.checkpoint.filename = (
        f"{model.name}-{'_'.join(model.datasets)}-s{args.seed}"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        **args.trainer.checkpoint.as_dict()
    )
    trainer = pl.Trainer(
        min_epochs=args.trainer.min_epochs,
        max_epochs=args.trainer.max_epochs,
        accelerator=args.trainer.accelerator,
        devices=args.trainer.devices,
        enable_progress_bar=args.trainer.enable_progress_bar,
        deterministic=True,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        log_every_n_steps=args.trainer.log_every_n_steps,
        val_check_interval=args.trainer.val_check_interval,
        check_val_every_n_epoch=args.trainer.check_val_every_n_epoch,
        precision=args.trainer.precision,
    )

    trainer.fit(model, datamodule, ckpt_path=args.model_checkpoint)


if __name__ == "__main__":
    import jsonargparse

    parser = jsonargparse.ArgumentParser(descripion=__doc__)

    parser.add_argument("--model_variant", type=GCBC_ENUM, required=True)
    parser.add_argument("--model_checkpoint", type=str, default=None)
    parser.add_argument(
        "--wandb_id", type=str, default=None, help="To resume logging to the same run"
    )
    parser.add_class_arguments(
        GCBC,
        "gcbc",
        skip={
            "vision_encoder_kwargs",
            "proprio_encoder_kwargs",
            "traj_encoder_kwargs",
            "action_decoder_kwargs",
        },
    )

    parser.add_class_arguments(CLIPT, "clipt")
    parser.add_argument("--clipt_checkpoint", type=str, required=False)

    parser.add_class_arguments(VisionEncoder, "vision_encoder")
    parser.add_class_arguments(ProprioEncoder, "proprio_encoder")

    parser.add_class_arguments(
        BabyAIActionDecoder, "action_decoder", skip={"hidden_dim"}
    )

    parser.add_class_arguments(BabyAIDM, "data", skip={"transform"})

    # transforms
    parser.add_argument(
        "--data.transform_name", type=TransformName, default="clipt-prepare"
    )
    parser.add_argument(
        "--data.transform_variant",
        type=str,
        default="without_clip",
        choices=["without_clip", "with_clip"],
    )

    parser.add_dataclass_arguments(TrainerConfig, "trainer")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    train(args)