import os
from dataclasses import dataclass

from nlgoals.trainer.utils import Accelerator


@dataclass
class LoggingConfig:
    enable: bool = True


@dataclass
class CheckpointConfig:
    filename: str = "task_classifier"
    dirpath: str = os.path.join("checkpoints", "task_classifier")
    monitor: str = "textual/val_accuracy"
    mode: str = "max"


@dataclass
class TrainerConfig:
    max_epochs: int = 50
    accelerator: Accelerator = "auto"
    devices: int = 1
    enable_progress_bar: bool = False
    logging: LoggingConfig = LoggingConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    log_every_n_steps: int = 50
