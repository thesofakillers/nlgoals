"""Lightning DataModule for the Calvin dataset."""
import os
from typing import Dict, Optional, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch

from nlgoals.data.utils import TRANSFORM_MAP, TransformName
from nlgoals.data.calvin.utils import FrameKey
from nlgoals.data.calvin.dataset import CALVIN


class CALVINDM(pl.LightningDataModule):
    """LightningDataModule for the CALVIN dataset."""

    def __init__(
        self,
        data_dir: str,
        num_frames: int,
        batch_size: int,
        val_split: float,
        seed: int,
        num_workers: int,
        frame_keys: Optional[List[FrameKey]] = None,
        transform_name: Optional[TransformName] = None,
        transform_kwargs: Optional[Dict] = None,
    ):
        """
        Args:
            data_dir: path to the directory containing the data
            split: one of 'training', 'validation'
            num_frames: number of frames to include in each trajectory
            batch_size: batch size for the dataloaders
            val_split: fraction of the training set to use for validation
            frame_keys: list of keys to include for each frame.
                By default, all keys are included.
        """
        super().__init__()
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.frame_keys = frame_keys
        self.num_workers = num_workers
        self.transform_name = transform_name
        self.transform_kwargs = transform_kwargs

    def prepare_data(self) -> None:
        """Checks that the data is downloaded"""
        missing_data_error = FileNotFoundError(
            f"Complete Data not found at {self.data_dir}."
            " Please follow the download instructions in `data/calvin/README.md`."
            " Or move the data to the appropriate folder."
        )
        if not os.path.exists(self.data_dir):
            raise missing_data_error
        else:
            train_dir = os.path.join(self.data_dir, "training")
            test_dir = os.path.join(self.data_dir, "validation")
            if not os.path.exists(train_dir) or not os.path.exists(test_dir):
                raise missing_data_error
            else:
                # check for actual files in the folders
                if len(os.listdir(train_dir)) == 0 or len(os.listdir(test_dir)) == 0:
                    raise missing_data_error
                else:
                    print("Data found.")

    def setup(self, stage=None):
        """Initializes and splits datasets, for use by DataLoaders"""
        self.test_dataset = CALVIN(
            self.data_dir,
            "validation",
            self.num_frames,
            self.frame_keys,
            self.transform_name,
            self.transform_kwargs,
        )
        temp_train_dataset = CALVIN(
            self.data_dir,
            "training",
            self.num_frames,
            self.frame_keys,
            self.transform_name,
            self.transform_kwargs,
        )

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            temp_train_dataset,
            [1 - self.val_split, self.val_split],
            generator=generator,
        )
        # these get properly parsed in the dataset, move them here for easier access
        self.frame_keys = self.train_dataset.frame_keys

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )