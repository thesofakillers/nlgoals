"""
Trainer config and class
"""
from dataclasses import dataclass


@dataclass
class TrainerConfig:
    n_epochs: int = 100
