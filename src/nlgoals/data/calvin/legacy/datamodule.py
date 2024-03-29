"""Lightning DataModule for the Calvin dataset."""
import os
from typing import Optional, List, Any, Dict, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch

from nlgoals.data.calvin.utils import FrameKey
from nlgoals.data.calvin.legacy.dataset import CALVIN


class CALVINDM(pl.LightningDataModule):
    """LightningDataModule for the CALVIN dataset."""

    def __init__(
        self,
        data_dir: str = "./data/calvin/task_D_D/",
        num_frames: int = 2,
        batch_size: int = 64,
        val_split: float = 0.1,
        seed: int = 42,
        num_workers: int = 18,
        custom_collate: bool = False,
        lang_key: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        frame_keys: Optional[List[str]] = [
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K_rgb_static"
        ],
        transform: Optional[Any] = None,
        **kwargs,
    ):
        """
        Args:
            data_dir: path to the directory containing the data
            num_frames: number of frames to include in each trajectory
            batch_size: batch size for the dataloaders
            val_split: fraction of the training set to use for validation
            seed: seed for the random split of the training set
            num_workers: number of workers for the dataloaders
            custom_collate: whether to use the custom collate function
                (Necessary when not using precomputed clip embs)
            lang_key: which key to use for the language annotation.
                If not 'ann', `lang_key` is used for specifying
                pre-computed clip embeddings. E.g. the `lang_key` 'foo'
                will load embeddings from lang_annotations['clip_emb']['foo']
            frame_keys: list of keys to include for each frame.
                By default, all keys are included.
            transform: instance of transform to apply to each frame
        """
        super().__init__()
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.custom_collate = custom_collate
        self.lang_key = lang_key
        self.frame_keys = frame_keys
        self.num_workers = num_workers
        self.transform = transform
        self.is_setup = False

    def prepare_data(self) -> None:
        """Checks that the data is downloaded"""
        if self.is_setup:
            return
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

    def setup(self, stage: Optional[str] = None):
        """Initializes and splits datasets, for use by DataLoaders"""
        if self.is_setup:
            return
        if stage == "fit" or stage is None:
            temp_train_dataset = CALVIN(
                self.data_dir,
                "training",
                self.num_frames,
                self.lang_key,
                self.frame_keys,
                self.transform,
            )

            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                temp_train_dataset,
                [1 - self.val_split, self.val_split],
                generator=generator,
            )
            # these get properly parsed in the dataset, move them here for easier access
            self.frame_keys = temp_train_dataset.frame_keys
            self.id_to_task = temp_train_dataset.id_to_task
            self.task_to_id = temp_train_dataset.task_to_id
        elif stage == "test" or stage is None:
            self.test_dataset = CALVIN(
                self.data_dir,
                "validation",
                self.num_frames,
                self.lang_key,
                self.frame_keys,
                self.transform,
            )
            self.frame_keys = self.test_dataset.frame_keys
            self.id_to_task = self.test_dataset.id_to_task
            self.task_to_id = self.test_dataset.task_to_id
        self.is_setup = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
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
