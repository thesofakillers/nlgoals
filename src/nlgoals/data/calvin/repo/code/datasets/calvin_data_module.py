import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import nlgoals.data.calvin.repo.code
from nlgoals.data.calvin.repo.code.datasets.utils.episode_utils import (
    load_dataset_statistics,
)
from nlgoals.data.calvin.repo.code.datasets.utils.shared_memory_utils import (
    load_shm_lookup,
    save_shm_lookup,
    SharedMemoryLoader,
)
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
import torch
import torchvision
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})
# this URL doesn't work anymore lol
ONE_EP_DATASET_URL = "http://www.informatik.uni-freiburg.de/~meeso/50steps.tar.xz"


class CalvinDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        training_repo_root: Optional[Path] = None,  # need this because Pathlib is dumb
        root_data_dir: str = "data/calvin/task_D_D",
        transforms: DictConfig = DEFAULT_TRANSFORM,
        batch_size: int = 32,
        shuffle_val: bool = False,
        **kwargs: Dict,  # absorb any other arguments
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        # need the next 6 lines because Pathlib is dumb and can't glob
        root_data_path = Path(root_data_dir)
        if not root_data_path.is_absolute():
            assert (
                training_repo_root is not None
            ), "If root_data_path isn't absolute, please provide training_repo_root"
            root_data_path = training_repo_root / root_data_path
        self.training_dir = root_data_path / "training"
        self.val_dir = root_data_path / "validation"
        self.shuffle_val = shuffle_val
        self.modalities: List[str] = []
        self.transforms = transforms
        self.batch_size = batch_size

        self.use_shm = "shm_dataset" in self.datasets_cfg.items()[0][1]["_target_"]

    def prepare_data(self, *args, **kwargs):
        # check if files already exist
        dataset_exist = np.any(
            [
                len(list(self.training_dir.glob(extension)))
                for extension in ["*.npz", "*.pkl"]
            ]
        )

        # download and unpack images
        if not dataset_exist:
            if "CI" not in os.environ:
                print(f"No dataset found in {self.training_dir}.")
                print(
                    "For information how to download to full CALVIN dataset, please visit"
                )
                print("https://github.com/mees/calvin/tree/main/dataset")
                print(
                    "Do you wish to download small debug dataset to continue training?"
                )
                s = input("YES / no")
                if s == "no":
                    exit()
            logger.info(
                f"downloading dataset to {self.training_dir} and {self.val_dir}"
            )
            torchvision.datasets.utils.download_and_extract_archive(
                ONE_EP_DATASET_URL, self.training_dir
            )
            torchvision.datasets.utils.download_and_extract_archive(
                ONE_EP_DATASET_URL, self.val_dir
            )

        if self.use_shm:
            # When using shared memory dataset, initialize lookups
            train_shmem_loader = SharedMemoryLoader(
                self.datasets_cfg, self.training_dir
            )
            train_shm_lookup = train_shmem_loader.load_data_in_shared_memory()

            val_shmem_loader = SharedMemoryLoader(self.datasets_cfg, self.val_dir)
            val_shm_lookup = val_shmem_loader.load_data_in_shared_memory()

            save_shm_lookup(train_shm_lookup, val_shm_lookup)

    def setup(self, stage=None):
        transforms = load_dataset_statistics(
            self.training_dir, self.val_dir, self.transforms
        )

        self.train_transforms = {
            cam: [
                hydra.utils.instantiate(transform)
                for transform in transforms.train[cam]
            ]
            for cam in transforms.train
        }

        self.val_transforms = {
            cam: [
                hydra.utils.instantiate(transform) for transform in transforms.val[cam]
            ]
            for cam in transforms.val
        }
        self.train_transforms = {
            key: torchvision.transforms.Compose(val)
            for key, val in self.train_transforms.items()
        }
        self.val_transforms = {
            key: torchvision.transforms.Compose(val)
            for key, val in self.val_transforms.items()
        }
        self.train_datasets, self.val_datasets = {}, {}

        if self.use_shm:
            train_shm_lookup, val_shm_lookup = load_shm_lookup()

        for _, dataset in self.datasets_cfg.items():
            train_dataset = hydra.utils.instantiate(
                dataset,
                datasets_dir=self.training_dir,
                transforms=self.train_transforms,
            )
            val_dataset = hydra.utils.instantiate(
                dataset, datasets_dir=self.val_dir, transforms=self.val_transforms
            )
            if self.use_shm:
                train_dataset.setup_shm_lookup(train_shm_lookup)
                val_dataset.setup_shm_lookup(val_shm_lookup)
            key = dataset.key
            self.train_datasets[key] = train_dataset
            self.val_datasets[key] = val_dataset
            self.modalities.append(key)

    def train_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=False,
                shuffle=True,
                collate_fn=self._collate_fn,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        val_dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=False,
                shuffle=self.shuffle_val,
                collate_fn=self._collate_fn,
            )
            for key, dataset in self.val_datasets.items()
        }
        combined_val_loaders = CombinedLoader(val_dataloaders, "max_size_cycle")
        return combined_val_loaders

    def _collate_fn(
        self, batch_list: List[Dict]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Handles preprocessing (e.g. tokenization) and padding

        Args:
            batch_list: list of dicts with keys:
                - "robot_obs": tensor (S x 8)
                - "rgb_obs": Dict of tensors with keys
                    - "rgb_static" (S x 3 x H x W)
                - "depth_obs: empty dictionary
                - "actions": tensor (S x 7)
                - "state_info": Dict of tensors with keys
                    - "robot_obs" (S x 15)
                    - "scene_obs" (S x 24)
                - "lang": sentence annotating the sequence, or empty tensor
                - "idx": integer index of the sequence

        Returns:
            Dict with the keys:
                - 'robot_obs' (B x MS x 8)
                - 'rgb_static' (B x MS x 3 x H x W)
                - 'actions' (B x MS x 7)
                - "state_info": Dict of tensors with keys
                    - "robot_obs" (B x MS x 15)
                    - "scene_obs" (B x MS x 24)
                - 'idx' (B x 1)
                - 'seq_lens' (B x 1) the sequence lengths before padding
            and optionally (in the case language annotations are provided):
                - 'lang_input_ids' (B X MSL) (mls is maximum sentence length)
                - 'lang_attn_mask' (B X MSL)
        """
        seq_lens = torch.tensor([x["robot_obs"].shape[0] for x in batch_list])
        max_seq_len = seq_lens.max()
        pad_sizes = max_seq_len - seq_lens

        padded_batch = {"seq_lens": seq_lens.unsqueeze(-1)}

        padded_batch["robot_obs"] = torch.stack(
            [
                self._pad_with_repetition(element["robot_obs"], pad_sizes[i])
                for i, element in enumerate(batch_list)
            ]
        )

        padded_batch["rgb_static"] = torch.stack(
            [
                self._pad_with_repetition(
                    element["rgb_obs"]["rgb_static"], pad_sizes[i]
                )
                for i, element in enumerate(batch_list)
            ]
        )

        # zero pad all but the last action dims and repeat last action dim (gripper action)
        padded_batch["actions"] = torch.stack(
            [
                torch.cat(
                    [
                        self._pad_with_zeros(
                            element["actions"][..., :-1], pad_sizes[i]
                        ),
                        self._pad_with_repetition(
                            element["actions"][..., -1:], pad_sizes[i]
                        ),
                    ],
                    dim=-1,
                )
                for i, element in enumerate(batch_list)
            ]
        )

        padded_batch["state_info"]["robot_obs"] = torch.stack(
            [
                self._pad_with_repetition(
                    element["state_info"]["robot_obs"], pad_sizes[i]
                )
                for i, element in enumerate(batch_list)
            ]
        )

        padded_batch["state_info"]["scene_obs"] = torch.stack(
            [
                self._pad_with_repetition(
                    element["state_info"]["scene_obs"], pad_sizes[i]
                )
                for i, element in enumerate(batch_list)
            ]
        )

        padded_batch["idx"] = torch.stack([element["idx"] for element in batch_list])

        # handle language
        if isinstance(batch_list[0]["lang"], str):
            lang_batch = self.text_processor(
                [x["lang"] for x in batch_list], padding=True, return_tensors="pt"
            )
            padded_batch["lang_input_ids"] = lang_batch["input_ids"]
            padded_batch["lang_attn_mask"] = lang_batch["attention_mask"]

        # handle additional image processing
        if self.process_vision:
            # images are between -1, and 1. We need 0 and 1
            raise NotImplementedError

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        last_repeated = torch.repeat_interleave(
            torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded
