"""Datasets and DataModules for CALVIN: https://github.com/mees/calvin"""
import os
import enum
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset


class FrameKey(enum.Enum):
    actions = "actions"
    rel_actions = "rel_actions"
    robot_obs = "robot_obs"
    scene_obs = "scene_obs"
    rgb_static = "rgb_static"
    rgb_gripper = "rgb_gripper"
    rgb_tactile = "rgb_tactile"
    depth_static = "depth_static"
    depth_gripper = "depth_gripper"
    depth_tactile = "depth_tactile"


class CALVINSplit(enum.Enum):
    training: str = "training"
    valididation: str = "validation"


class CALVIN(Dataset):
    """
    Dataset for CALVIN. state-action trajectories,
    annotated with natural language instructions.
    """

    def __init__(
        self,
        data_dir: str,
        split: CALVINSplit,
        num_frames: int,
        frame_keys: Optional[List[FrameKey]] = None,
    ):
        """
        Args:
            data_dir: path to the directory containing the data
            split: one of 'training', 'validation'
            num_frames: number of frames to include in each trajectory
            frame_keys: list of keys to include for each frame.
                By default, all keys are included.
        """
        self.path = os.path.join(data_dir, split)
        assert os.path.exists(self.path), f"Path {self.path} does not exist."
        assert num_frames >= 1, "num_frames must be at least 1"
        self.lang_annotations = np.load(
            os.path.join(self.path, "lang_annotations", "auto_lang_ann.npy"),
            allow_pickle=True,
        ).item()
        self.num_frames = num_frames
        self.frame_keys = frame_keys

    def __len__(self) -> int:
        return len(self.lang_annotations["info"]["indx"])

    def __getitem__(self, idx) -> Dict:
        lang_ann = self.lang_annotations["language"]["ann"]
        task = self.lang_annotations["language"]["task"]

        frames = self._get_frames_item(idx)

        return {
            "lang_ann": lang_ann,
            "task": task,
            **frames,
        }

    def _get_frames_item(self, idx: int) -> Dict[str, np.ndarray]:
        """Note: frame_idx is different from item idx"""
        # frame parsing
        start_frame_idx, end_frame_idx = self.lang_annotations["info"]["indx"][idx]
        # go backwards because in case we have only one frame we use the end frame
        frame_idxs = np.linspace(
            end_frame_idx, start_frame_idx, self.num_frames, dtype=int
        )[::-1]
        frames: Dict[str, np.ndarray] = {}
        for i, frame_idx in enumerate(frame_idxs):
            frame: Dict[str, np.ndarray] = self._parse_frame_idx(frame_idx)
            for key in frame:
                if key not in frames:
                    frames[key] = np.empty((self.num_frames, *frame[key].shape))
                frames[i] = frame[key]

        return frames

    def _parse_frame_idx(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Note: frame_idx is different from item idx"""
        frame_idx_str = "{:0>7}".format(frame_idx)
        frame_file = np.load(os.path.join(self.path, f"episode_{frame_idx_str}.npz"))

        if self.frame_keys is None:
            return dict(frame_file)
        else:
            frame = {}
            for frame_key in self.frame_keys:
                frame[frame_key] = frame_file[frame_key]

        return frame
