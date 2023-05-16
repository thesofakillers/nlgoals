"""Datasets and DataModules for CALVIN: https://github.com/mees/calvin"""
import os
import enum
from typing import Dict, List, Optional, Any
import re

import numpy as np
from torch.utils.data import Dataset

from nlgoals.data.calvin.utils import FrameKey


class CALVINSplit(enum.Enum):
    training: str = "training"
    valididation: str = "validation"


class CALVINTextDataset(Dataset):
    """Dataset that gets all the text annotations for a given trajectory of CALVIN"""

    # TODO

    def __init__(
        self, data_dir: str, split: CALVINSplit, transform: Optional[Any], **kwargs
    ):
        pass

    def __len__(self) -> int:
        return len(self.lang_annotations["info"]["indx"])


class CALVINFrameDataset(Dataset):
    """
    Dataset that gets individual frames in CALVIN

    Args:
        data_dir: path to the data directory
        split: which split to use
        annotated_only: whether to only use frames that have lang_annotations
    """

    def __init__(
        self, data_dir: str, split: CALVINSplit, annotated_only: bool = False, **kwargs
    ):
        split = split.value if isinstance(split, CALVINSplit) else split
        self.path = os.path.join(data_dir, split)
        assert os.path.exists(self.path), f"Path {self.path} does not exist."

        self._dataset_len = None

        self.item_list = self._build_item_list(annotated_only)

    def _build_item_list(self, annotated_only: bool) -> List[str]:
        """
        Build a list of all the paths to the frames in the dataset
        """
        if not annotated_only:
            pattern = r"episode_\d{7}\.npz"
            item_list = sorted(
                [
                    os.path.join(self.path, f)
                    for f in os.listdir(self.path)
                    if re.match(pattern, f)
                ]
            )
        else:
            self.lang_anns = np.load(
                os.path.join(self.path, "lang_annotations", "auto_lang_ann.npy"),
                allow_pickle=True,
            ).item()
            item_list = [
                os.path.join(self.path, f"episode_{str(idx).zfill(7)}.npz")
                for start_end in self.lang_anns["info"]["indx"]
                for idx in np.linspace(
                    start_end[0],
                    start_end[1],
                    start_end[1] - start_end[0] + 1,
                    dtype=int,
                )
            ]
        return item_list

    def __len__(self) -> int:
        if self._dataset_len is None:
            self._dataset_len = len(self.item_list)
        return self._dataset_len

    def __getitem__(self, index: int) -> Dict:
        frame_file = np.load(self.item_list[index])
        frame_dict = dict(frame_file)
        # for saving
        frame_dict["frame_id"] = self.item_list[index][-11:-4]
        return frame_dict


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
        transform: Optional[Any] = None,
        **kwargs,
    ):
        """
        Args:
            data_dir: path to the directory containing the data
            split: one of 'training', 'validation'
            num_frames: number of frames to include in each trajectory
            frame_keys: list of keys to include for each frame.
                By default, all keys are included.
            transform_name: name of the transform to apply to each item (Optional)
            transform_kwargs: kwargs to pass to the transform
        """
        self.path = os.path.join(data_dir, split)
        assert os.path.exists(self.path), f"Path {self.path} does not exist."
        assert num_frames >= 1, "num_frames must be at least 1"
        self.lang_annotations = np.load(
            os.path.join(self.path, "lang_annotations", "auto_lang_ann.npy"),
            allow_pickle=True,
        ).item()
        self.num_frames = num_frames
        self.frame_keys = (
            frame_keys if frame_keys is not None else [k.value for k in FrameKey]
        )
        self.parse_frame_keys = frame_keys is not None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.lang_annotations["info"]["indx"])

    def __getitem__(self, idx) -> Dict:
        lang_ann = self.lang_annotations["language"]["ann"][idx]
        task = self.lang_annotations["language"]["task"][idx]

        frames = self._get_frames_item(idx)

        item = {
            "lang_ann": lang_ann,
            "task": task,
            **frames,
        }

        if self.transform is not None:
            item = self.transform(item)

        return item

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
                frames[key][i] = frame[key]

        return frames

    def _parse_frame_idx(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Note: frame_idx is different from item idx"""
        frame_idx_str = "{:0>7}".format(frame_idx)
        frame_file = np.load(os.path.join(self.path, f"episode_{frame_idx_str}.npz"))

        if not self.parse_frame_keys:
            return dict(frame_file)
        else:
            frame = {}
            for frame_key in self.frame_keys:
                frame[frame_key] = frame_file[frame_key]

        return frame
