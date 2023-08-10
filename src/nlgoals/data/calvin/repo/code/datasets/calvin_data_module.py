import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

from transformers import CLIPImageProcessor, CLIPTokenizerFast
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
import torch
import torchvision

from nlgoals.data.calvin.repo.code.datasets.utils.episode_utils import (
    load_dataset_statistics,
)
from nlgoals.data.calvin.repo.code.datasets.utils.shared_memory_utils import (
    load_shm_lookup,
    save_shm_lookup,
    SharedMemoryLoader,
)
from nlgoals.utils.misc import pad_with_repetition, pad_with_zeros

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
        num_workers: int = 0,
        clip_model_name: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        instantiate_collator=True,
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
        self.num_workers = num_workers
        self.clip_model_name = clip_model_name
        self.clip_img_transform = CLIPImageTransform(clip_model_name)
        if instantiate_collator:
            self.collator = Collator(clip_model_name)

        self.use_shm = "shm_dataset" in self.datasets_cfg.items()[0][1]["_target_"]

    def prepare_data(self):
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
            print(f"downloading dataset to {self.training_dir} and {self.val_dir}")
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
        self._setup_transforms()

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
                num_workers=self.num_workers,
                pin_memory=False,
                shuffle=True,
                persistent_workers=True if self.num_workers > 0 else False,
                collate_fn=self.collator,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        val_dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=False,
                shuffle=self.shuffle_val,
                persistent_workers=True if self.num_workers > 0 else False,
                collate_fn=self.collator,
            )
            for key, dataset in self.val_datasets.items()
        }
        combined_val_loaders = CombinedLoader(val_dataloaders, "max_size_cycle")
        return combined_val_loaders

    def _setup_transforms(self) -> None:
        """
        Sets up self.train_transforms and self.val_transforms, dictionaries of transform
        instances for each camera and its keys
        """
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

        clip_transform_keys = ["rgb_static", "rgb_gripper"]
        for key in clip_transform_keys:
            self.train_transforms[key] = self.clip_img_transform
            self.val_transforms[key] = self.clip_img_transform


class CLIPImageTransform:
    """Callable object to enable pickling necessary for multiprocessing"""

    def __init__(self, clip_model_name: str):
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)

    def __call__(self, images):
        return self.image_processor(images=images, return_tensors="pt").pixel_values


class Collator:
    """Callable object to enable pickling necessary for multiprocessing"""

    def __init__(self, clip_model_name: str):
        self.text_processor = CLIPTokenizerFast.from_pretrained(clip_model_name)

    def __call__(self, batch_list):
        """
        First handles padding, tokenization and all that stuff.
        Then allows user to make further modifications via self.custom_collate_fn
        """
        base_batch = self._base_collate(batch_list)
        prepared_batch = self.custom_collate_fn(base_batch)
        return prepared_batch

    def _base_collate(
        self, batch_list: List[Dict]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Handles tokenization and padding

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
                - 'idx' (B, )
                - 'seq_lens' (B, ) the sequence lengths before padding
            and optionally (in the case language annotations are provided):
                - 'lang_input_ids' (B X MSL) (mls is maximum sentence length)
                - 'lang_attn_mask' (B X MSL)
        """

        padded_batch = self._collate_padding(batch_list)

        # need this check because we could have an empty language key for vision_only
        if isinstance(batch_list[0]["lang"], str):
            input_ids, attention_mask = self._collate_tokenization(batch_list)
            padded_batch["lang_input_ids"] = input_ids
            padded_batch["lang_attn_mask"] = attention_mask

        return padded_batch

    def _collate_padding(
        self, batch_list: List[Dict]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Takes care of padding
        Args:
            batch_list: see _base_collate
        Returns:
            padded_batch: same as _base_collate but without the optional lang keys
            these are handled by _collate_tokenization
        """
        seq_lens = torch.tensor([x["robot_obs"].shape[0] for x in batch_list])
        max_seq_len = seq_lens.max()
        pad_sizes = max_seq_len - seq_lens
        padded_batch = {"seq_lens": seq_lens}

        padded_batch["robot_obs"] = torch.stack(
            [
                pad_with_repetition(element["robot_obs"], pad_sizes[i])
                for i, element in enumerate(batch_list)
            ]
        )

        padded_batch["rgb_static"] = torch.stack(
            [
                pad_with_repetition(element["rgb_obs"]["rgb_static"], pad_sizes[i])
                for i, element in enumerate(batch_list)
            ]
        )

        # zero pad all but the last action dims and repeat last action dim (gripper action)
        padded_batch["actions"] = torch.stack(
            [
                torch.cat(
                    [
                        pad_with_zeros(element["actions"][..., :-1], pad_sizes[i]),
                        pad_with_repetition(element["actions"][..., -1:], pad_sizes[i]),
                    ],
                    dim=-1,
                )
                for i, element in enumerate(batch_list)
            ]
        )

        padded_batch["state_info"] = {}
        padded_batch["state_info"]["robot_obs"] = torch.stack(
            [
                pad_with_repetition(element["state_info"]["robot_obs"], pad_sizes[i])
                for i, element in enumerate(batch_list)
            ]
        )

        padded_batch["state_info"]["scene_obs"] = torch.stack(
            [
                pad_with_repetition(element["state_info"]["scene_obs"], pad_sizes[i])
                for i, element in enumerate(batch_list)
            ]
        )

        padded_batch["idx"] = torch.Tensor([element["idx"] for element in batch_list])

        return padded_batch

    def _collate_tokenization(
        self, batch_list: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes care of tokenization

        Args:
            batch_list: see _collate_fn

        Returns:
            input_ids: (B x MSL)
            attention_mask: (B x MSL)
            where MSL is the maximum sentence length in the batch
        """
        lang_batch = self.text_processor(
            [x["lang"] for x in batch_list], padding=True, return_tensors="pt"
        )
        return lang_batch["input_ids"], lang_batch["attention_mask"]

    @staticmethod
    def custom_collate_fn(
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Any:
        """
        Designed to be overridden externally, so that depending on the model,
        we can prepare the batch differently. By default, will do nothing.
        See nlgoals.interfaces for potential implementations.

        Args:
            batch : see return signature of _base_collate
        """
        return batch
