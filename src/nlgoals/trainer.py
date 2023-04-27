"""
Trainer config and class
"""
import os
from typing import Union
from dataclasses import dataclass


@dataclass
class LoggingConfig:
    disable: bool = False


@dataclass
class CheckpointConfig:
    filename: str = "clipt"
    dirpath: str = os.path.join("checkpoints", "clipt")
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class TrainerConfig:
    max_epochs: int = 50
    accelerator: Union["cpu", "gpu", "tpu", "ipu", "auto"] = "auto"
    devices: int = 1
    logging: LoggingConfig = LoggingConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
