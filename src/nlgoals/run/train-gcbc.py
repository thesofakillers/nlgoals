"""Training of the GCBC model on the CALVIN dataset"""
import jsonargparse
import os

import torch
import pytorch_lightning as pl
import hydra

from nlgoals.models.gcbc import GCBC
from nlgoals.models.clipt import CLIPT
from nlgoals.models.perception_encoders.vision_encoder import VisionEncoder
from nlgoals.models.perception_encoders.proprio_encoder import ProprioEncoder
from nlgoals.trainer.gcbc import TrainerConfig
from nlgoals.interfaces.gcbc import calvin_gcbc_collate


def train(args):
    # device
    if args.trainer.accelerator == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.trainer.accelerator
    # determinism
    pl.seed_everything(args.seed, workers=True)
    # datamodule
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=args.data.config_folder)
    datamodule_cfg = hydra.compose(config_name=args.data.config_name)
    datamodule_cfg.batch_size = args.data.batch_size
    datamodule_cfg.num_workers = args.data.num_workers
    datamodule_cfg.root_data_dir = args.data.data_dir
    datamodule = hydra.utils.instantiate(datamodule_cfg)
    datamodule.custom_collate_fn = calvin_gcbc_collate
    # model
    model = GCBC(
        traj_encoder_kwargs=args.clipt.as_dict(),
        vision_encoder_kwargs=args.vision_encoder.as_dict(),
        proprio_encoder_kwargs=args.proprio_encoder.as_dict(),
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
        mode="disabled" if not args.trainer.logging.enable else "online",
        group=script_host,
        config=args,
        log_model=False,
        tags=["gcbc"],
    )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor="textual/val_loss", mode="min"
    )
    args.trainer.checkpoint.filename = (
        f"{args.trainer.checkpoint.filename}-s{args.seed}"
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

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_class_arguments(
        GCBC,
        "gcbc",
        skip={"vision_encoder_kwargs", "proprio_encoder_kwargs", "traj_encoder_kwargs"},
    )
    parser.add_class_arguments(CLIPT, "clipt")
    parser.add_argument("--clipt-checkpoint", type=str, required=False)

    parser.add_class_arguments(VisionEncoder, "vision_encoder")
    parser.add_class_arguments(ProprioEncoder, "proprio_encoder")

    parser.add_argument("--data.config_folder", type=str, required=True)
    parser.add_argument("--data.config_name", type=str, required=True)
    parser.add_argument("--data.batch_size", type=int, default=32)
    parser.add_argument("--data.num_workers", type=int, default=18)
    parser.add_argument(
        "--data.data_dir", type=str, required=True, help="Must be absolute path"
    )

    parser.add_dataclass_arguments(TrainerConfig, "trainer")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    train(args)
