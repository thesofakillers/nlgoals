"""Datamodule for serving BabyAI data."""
from typing import Optional, Any, List, Dict
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset, random_split
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
        val_split: float = 0.1,
        num_workers: int = 18,
        custom_collate: bool = True,
        seed: int = 42,
        train_subset: Optional[int] = None,
        test_subset: Optional[int] = None,
        transform: Optional[Any] = None,
        **kwargs,
    ):
        """
        Args:
            data_path: path to data, e.g "path/to/data", such that two files exist:
                path/to/data.pkl and path/to/data_valid.pkl
            envs_size: size of the environment, e.g. "small", "large"
            use_first_last_frames: whether to use only the first and last frames
            batch_size: batch size for the dataloaders
            val_split: fraction of the training set to use for validation
            num_workers: number of workers for the dataloaders
            custom_collate: whether to use custom collate function
                (Necessary when not using precomputed clip embs)
            seed: random seed
            train_subset: number of training examples to use. If None, all are used.
            test_subset: number of test examples to use. If None, all are used.
            transform: transform to apply to the data
        """
        super().__init__()
        self.data_path = data_path
        self.train_path = self.data_path + ".pkl"
        self.test_path = self.data_path + "_valid.pkl"

        self.envs_size = envs_size
        self.use_first_last_frames = use_first_last_frames

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.custom_collate = custom_collate

        self.seed = seed
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.val_split = val_split

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
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(
                f"Validation data not found at {self.test_path}." + missing_data_instr
            )

    def setup(self, stage: Optional[str] = None):
        if self.is_setup:
            return
        generator = torch.Generator().manual_seed(self.seed)
        if stage == "fit" or stage is None:
            temp_train_dataset: Dataset = BabyAIDataset(
                self.train_path,
                self.envs_size,
                self.transform,
                self.use_first_last_frames,
            )
            if self.train_subset is not None:
                train_idxs = torch.randperm(len(self.train_dataset))[
                    : self.train_subset
                ]
                temp_train_dataset = Subset(self.train_dataset, train_idxs)

            self.train_dataset, self.val_dataset = random_split(
                temp_train_dataset,
                [1 - self.val_split, self.val_split],
                generator=generator,
            )

        if stage == "test" or stage is None:
            self.test_dataset: Dataset = BabyAIDataset(
                self.test_path,
                self.envs_size,
                self.transform,
                self.use_first_last_frames,
            )
            if self.test_subset is not None:
                # no permutation for val, to be able to compare results
                self.test_dataset = Subset(self.test_dataset, range(self.test_subset))

        # return only 100 and 10 samples if debug
        if stage == "debug":
            temp_train_dataset: Dataset = BabyAIDataset(
                self.train_path,
                self.envs_size,
                self.transform,
                self.use_first_last_frames,
            )
            temp_train_dataset = Subset(self.train_dataset, range(100))
            self.train_dataset, self.val_dataset = random_split(
                temp_train_dataset,
                [1 - self.val_split, self.val_split],
                generator=generator,
            )

            self.test_dataset: Dataset = BabyAIDataset(
                self.test_path,
                self.envs_size,
                self.transform,
                self.use_first_last_frames,
            )
            self.test_dataset = Subset(self.test_dataset, range(10))

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

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
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
        batch_dict["task_id"] = torch.tensor(batch_dict["task_id"])
        batch_dict["proprio_obs"] = torch.stack(batch_dict["proprio_obs"], dim=0)
        batch_dict["actions"] = torch.stack(batch_dict["actions"], dim=0)
        batch_dict["rewards"] = torch.stack(batch_dict["rewards"], dim=0)
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
