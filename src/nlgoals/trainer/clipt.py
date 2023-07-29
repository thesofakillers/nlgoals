import os
from dataclasses import dataclass

from nlgoals.trainer.utils import Accelerator


@dataclass
class LoggingConfig:
    enable: bool = True


@dataclass
class CheckpointConfig:
    filename: str = "clipt"
    dirpath: str = os.path.join("checkpoints", "clipt")
    monitor: str = "val_loss"
    mode: str = "min"
    save_last: bool = True


@dataclass
class TrainerConfig:
    max_epochs: int = 5000
    accelerator: Accelerator = "auto"
    devices: int = 1
    enable_progress_bar: bool = False
    enable_early_stopping: bool = False
    logging: LoggingConfig = LoggingConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    log_every_n_steps: int = 50
    precision: int = 16
