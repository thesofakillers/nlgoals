"""
Trainer config and class
"""
import os
from typing import Union
from dataclasses import dataclass
import enum


@dataclass
class LoggingConfig:
    enable: bool = True


@dataclass
class CheckpointConfig:
    filename: str = "clipt"
    dirpath: str = os.path.join("checkpoints", "clipt")
    monitor: str = "val_loss"
    mode: str = "min"


class Accelerator(str, enum.Enum):
    cpu = "cpu"
    gpu = "gpu"
    tpu = "tpu"
    ipu = "ipu"
    auto = "auto"


@dataclass
class TrainerConfig:
    max_epochs: int = 50
    accelerator: Accelerator = "auto"
    devices: int = 1
    enable_progress_bar: bool = False
    logging: LoggingConfig = LoggingConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
