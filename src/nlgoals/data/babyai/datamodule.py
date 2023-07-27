"""Datamodule for serving BabyAI data."""
from typing import Optional, Callable
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from nlgoals.data.babyai.dataset import BabyAIDataset


class BabyAIDM(LightningDataModule):
    """LightningDataModule for BabyAI data"""

    def __init__(
        self,
        data_path: str = "./data/babyai/play/small",
        envs_size: str = "small",
        use_first_last_frames: bool = False,
        batch_size: int = 64,
        num_workers: int = 18,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_path: path to data, e.g "path/to/data", such that two files exist:
                path/to/data.pkl and path/to/data_valid.pkl
            batch_size: batch size for the dataloaders
            num_workers: number of workers for the dataloaders
            transform: transform to apply to the data
        """
        self.data_path = data_path
        self.train_path = self.data_path + ".pkl"
        self.val_path = self.data_path + "_valid.pkl"

        self.envs_size = envs_size
        self.use_first_last_frames = use_first_last_frames

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transform

    def prepare_data(self) -> None:
        """Checks that the data exists"""
        if self.is_setup:
            return
        missing_data_instr = (
            " Please follow the download instructions in `data/babyai/README.md`."
            " Or generate the data with `src/nlgoals/run/babyai/make_agent_demos.py`"
        )

        if not os.path.exists(self.train_path):
            raise FileNotFoundError(
                f"Training data not found at {self.train_path}." + missing_data_instr
            )
        if not os.path.exists(self.val_path):
            raise FileNotFoundError(
                f"Validation data not found at {self.val_path}." + missing_data_instr
            )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset: Dataset = BabyAIDataset(
            self.train_path, self.envs_size, self.transform, self.use_first_last_frames
        )
        self.val_dataset: Dataset = BabyAIDataset(
            self.val_dataset, self.envs_size, self.transform, self.use_first_last_frames
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
