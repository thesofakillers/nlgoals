from typing import Callable

import numpy as np
import blosc
import pickle
from torch.utils.data import Dataset

import nlgoals.babyai.utils as babyai


class BabyAIDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        envs_size: str,
        transform: Callable = None,
        use_first_last_frames: bool = False,
    ):
        """
        Args:
            filepath (string): Path to the pickle file with train or validation data.
            transform (callable, optional): Optional transform to be applied on a sample image.
            use_first_last_frames (bool, optional): If set to True, only the first and last frames will be sampled.
        """
        self.filepath = filepath
        with open(self.filepath, "rb") as f:
            self.data = pickle.load(f)
        self.transform = transform
        self.use_first_last_frames = use_first_last_frames
        self.envs_size = envs_size
        self.task_id_to_env_name = babyai.SIZE_TO_ENVS[envs_size]
        self.env_name_to_task_id = {
            e: i for i, e in enumerate(self.task_id_to_env_name)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mission, env_name, images, directions, actions, rewards = self.data[idx]

        # Unpack images
        images = blosc.unpack_array(images)

        # Sample only the first and last frames if specified
        if self.use_first_last_frames:
            images = np.stack([images[0], images[-1]])

        # Apply transforms to the images if required
        if self.transform:
            images = self.transform(images)

        sample = {
            "lang_ann": mission,
            "task_id": self.env_name_to_task_id[env_name],
            "rgb_obs": images,
            "proprio_obs": directions,
            "actions": actions,
            "rewards": rewards,
        }

        return sample
