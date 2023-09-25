import os
from dataclasses import dataclass

from nlgoals.trainer.utils import Accelerator


@dataclass
class LoggingConfig:
    enable: bool = True


@dataclass
class CheckpointConfig:
    dirpath: str = os.path.join("checkpoints", "gcbc")


@dataclass
class TrainerConfig:
    min_epochs: int = 1
    max_epochs: int = 10
    accelerator: Accelerator = "auto"
    devices: int = 1
    enable_progress_bar: bool = False
    logging: LoggingConfig = LoggingConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    log_every_n_steps: int = 50
    val_check_interval: float = 0.5
    check_val_every_n_epoch: int = 1
    precision: int = 16
    limit_val_batches: float = 0
