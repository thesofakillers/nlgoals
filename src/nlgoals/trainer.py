"""
Trainer config and class
"""
from typing import Union
from dataclasses import dataclass


@dataclass
class LoggingConfig:
    disable: bool = False


@dataclass
class TrainerConfig:
    max_epochs: int = 50
    accelerator: Union["cpu", "gpu", "tpu", "ipu", "auto"] = "auto"
    devices: int = 1
    logging: LoggingConfig = LoggingConfig()
