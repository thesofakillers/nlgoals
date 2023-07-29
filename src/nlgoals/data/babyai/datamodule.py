"""Datamodule for serving BabyAI data."""
from typing import Optional, Any, List, Dict
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset
import torch
from torch.nn.utils.rnn import pad_sequence

from nlgoals.data.babyai.dataset import BabyAIDataset


class BabyAIDM(pl.LightningDataModule):
    """LightningDataModule for BabyAI data"""

    def __init__(
        self,
        data_path: str = "./data/babyai/play/small",
        envs_size: str = "small",
        use_first_last_frames: bool = False,
        batch_size: int = 64,
        num_workers: int = 18,
        custom_collate: bool = True,
        seed: int = 42,
        train_subset: Optional[int] = None,
        val_subset: Optional[int] = None,
        transform: Optional[Any] = None,
        **kwargs,
    ):
        """
        Args:
            data_path: path to data, e.g "path/to/data", such that two files exist:
                path/to/data.pkl and path/to/data_valid.pkl
            batch_size: batch size for the dataloaders
            num_workers: number of workers for the dataloaders
            custom_collate: whether to use custom collate function
                (Necessary when not using precomputed clip embs)
            seed: random seed
            train_subset: number of training examples to use. If None, all are used.
            val_subset: number of validation examples to use. If None, all are used.
            transform: transform to apply to the data
        """
        super().__init__()
        self.data_path = data_path
        self.train_path = self.data_path + ".pkl"
        self.val_path = self.data_path + "_valid.pkl"

        self.envs_size = envs_size
        self.use_first_last_frames = use_first_last_frames

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.custom_collate = custom_collate

        self.seed = seed
        self.train_subset = train_subset
        self.val_subset = val_subset

        self.transform = transform

        self.is_setup = False

    def prepare_data(self) -> None:
        """Checks that the data exists"""
        pl.seed_everything(self.seed)
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
        if self.is_setup:
            return
        self.train_dataset: Dataset = BabyAIDataset(
            self.train_path, self.envs_size, self.transform, self.use_first_last_frames
        )
        if self.train_subset is not None:
            train_idxs = torch.randperm(len(self.train_dataset))[: self.train_subset]
            self.train_dataset = Subset(self.train_dataset, train_idxs)

        self.val_dataset: Dataset = BabyAIDataset(
            self.val_path, self.envs_size, self.transform, self.use_first_last_frames
        )
        if self.val_subset is not None:
            # no permutation for val, to be able to compare results
            self.val_dataset = Subset(self.val_dataset, range(self.val_subset))

        # return only 100 and 10 samples if debug
        if stage == "debug":
            self.train_dataset = Subset(self.train_dataset, range(100))
            self.val_dataset = Subset(self.val_dataset, range(10))

        self.is_setup = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self._get_collate_fn(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._get_collate_fn(),
        )

    def _get_collate_fn(self):
        collate_fn = self._collate_fn if self.custom_collate else None
        return collate_fn

    @staticmethod
    def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Takes care of padding token_ids and attn_mask so that we can batch

        Args:
            batch: list of items, where each item has keys 'images', 'text_input_ids'
            and 'text_attn_mask'.

        Returns:
            batch: batch of data with padded text_input_ids and text_attn_mask as
            tensors
        """
        # first reshape from list of dicts into a dict of lists
        batch_dict = {key: [item[key] for item in batch] for key in batch[0]}
        # the pad values are hardcoded for now, should use the tokenizer.pad_token_id at some point
        batch_dict["text_input_ids"] = pad_sequence(
            batch_dict["text_input_ids"], batch_first=True, padding_value=49407
        )
        batch_dict["text_attn_mask"] = pad_sequence(
            batch_dict["text_attn_mask"], batch_first=True, padding_value=0
        )
        batch_dict["images"] = torch.stack(batch_dict["images"], dim=0)
        # ready
        return batch_dict


if __name__ == "__main__":
    import jsonargparse

    parser = jsonargparse.ArgumentParser()

    parser.add_class_arguments(BabyAIDM, "data")

    args = parser.parse_args()

    dm = BabyAIDM(**args.data)
    dm.prepare_data()
    dm.setup()
