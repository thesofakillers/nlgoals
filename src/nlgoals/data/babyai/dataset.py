from typing import Callable

import torch
import blosc
import pickle
from torch.utils.data import Dataset
import pdb

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
            transform (callable, optional): Optional transform to be applied on a sample.
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
        images = torch.tensor(blosc.unpack_array(images))
        directions = torch.LongTensor(directions)
        actions = torch.LongTensor(actions)
        rewards = torch.tensor(rewards)

        # Sample only the first and last frames if specified
        if self.use_first_last_frames:
            images = images[[0, -1]]
            actions = actions[[0, -1]]
            directions = directions[[0, -1]]
            rewards = rewards[[0, -1]]

        sample = {
            "lang_ann": mission,
            "task_id": self.env_name_to_task_id[env_name],
            "rgb_obs": images,
        }

        # Apply transforms to the data if required
        if self.transform:
            sample = self.transform(sample)

        sample = {
            **sample,
            "proprio_obs": directions,
            "actions": actions,
            "rewards": rewards,
        }

        return sample
