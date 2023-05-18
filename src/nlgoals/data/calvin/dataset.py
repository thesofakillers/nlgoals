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
    validation: str = "validation"


class CALVINTextDataset(Dataset):
    """Dataset that gets all the text annotations in CALVIN"""

    def __init__(self, data_dir: str, split: CALVINSplit, **kwargs):
        """
        Args:
            data_dir: path to the data directory
            split: which split to use
        """
        split = split.value if isinstance(split, CALVINSplit) else split
        self.path = os.path.join(data_dir, split)
        assert os.path.exists(self.path), f"Path {self.path} does not exist."

        self.lang_ann = np.load(
            os.path.join(self.path, "lang_annotations", "auto_lang_ann.npy"),
            allow_pickle=True,
        ).item()

    def __len__(self) -> int:
        return len(self.lang_ann["info"]["indx"])

    def __getitem__(self, index) -> Dict:
        sample = {}
        sample["lang_ann"] = self.lang_ann["language"]["ann"][index]
        sample["task_id"] = self.lang_ann["language"]["task"][index]
        return sample


class CALVINFrameDataset(Dataset):
    """Dataset that gets individual frames in CALVIN"""

    def __init__(
        self,
        data_dir: str,
        split: CALVINSplit,
        annotated_only: bool = False,
        num_frames: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            data_dir: path to the data directory
            split: which split to use
            annotated_only: whether to only use frames that have lang_annotations
            num_frames: Number of frames to use. If None, use all frames.
                only relevant when annotated_only is True
        """
        split = split.value if isinstance(split, CALVINSplit) else split
        self.path = os.path.join(data_dir, split)
        assert os.path.exists(self.path), f"Path {self.path} does not exist."

        self._dataset_len = None
        self.num_frames = num_frames

        # fmt: off
        self.orig_columns = [ "actions", "rel_actions", "robot_obs", "scene_obs",
                             "rgb_static", "rgb_gripper", "rgb_tactile", "depth_static",
                             "depth_gripper", "depth_tactile", ]
        # fmt: on

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
                # backwards in case num_frames is 1, in which case we use last frame
                for idx in np.linspace(
                    start_end[1],
                    start_end[0],
                    start_end[1] - start_end[0] + 1
                    if self.num_frames is None
                    else self.num_frames,
                    dtype=int,
                )[::-1]
            ]
        return item_list

    def __len__(self) -> int:
        if self._dataset_len is None:
            self._dataset_len = len(self.item_list)
        return self._dataset_len

    def __getitem__(self, index: int) -> Dict:
        frame_file = np.load(self.item_list[index])
        frame_dict = {k: frame_file[k] for k in self.orig_columns}
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
        num_frames: 2,
        lang_key: str = "ann",
        frame_keys: Optional[List[FrameKey]] = ["rgb_static"],
        transform: Optional[Any] = None,
        **kwargs,
    ):
        """
        Args:
            data_dir: path to the directory containing the data
            split: one of 'training', 'validation'
            num_frames: number of frames to include in each trajectory
            lang_key: which key to use for the language annotation.
                If not 'ann', `lang_key` is used for specifying
                pre-computed clip embeddings. E.g. the `lang_key` 'foo'
                will load embeddings from lang_annotations['clip_emb']['foo']
            frame_keys: list of keys to include for each frame.
                By default, all keys are included.
            transform: transform to apply to each frame
        """
        self.path = os.path.join(data_dir, split)
        assert os.path.exists(self.path), f"Path {self.path} does not exist."
        assert num_frames >= 1, "num_frames must be at least 1"
        self.lang_annotations = np.load(
            os.path.join(self.path, "lang_annotations", "auto_lang_ann.npy"),
            allow_pickle=True,
        ).item()
        self.num_frames = num_frames
        self.lang_key = lang_key
        self.frame_keys = (
            frame_keys if frame_keys is not None else [k.value for k in FrameKey]
        )
        self.parse_frame_keys = frame_keys is not None
        self.transform = transform
        # fmt: off
        self.id_to_task = ( "close_drawer", "lift_blue_block_drawer",
                           "lift_blue_block_slider", "lift_blue_block_table",
                           "lift_pink_block_drawer", "lift_pink_block_slider",
                           "lift_pink_block_table", "lift_red_block_drawer",
                           "lift_red_block_slider", "lift_red_block_table",
                           "move_slider_left", "move_slider_right", "open_drawer",
                           "place_in_drawer", "place_in_slider", "push_blue_block_left",
                           "push_blue_block_right", "push_into_drawer",
                           "push_pink_block_left", "push_pink_block_right",
                           "push_red_block_left", "push_red_block_right",
                           "rotate_blue_block_left", "rotate_blue_block_right",
                           "rotate_pink_block_left", "rotate_pink_block_right",
                           "rotate_red_block_left", "rotate_red_block_right",
                           "stack_block", "turn_off_led", "turn_off_lightbulb",
                           "turn_on_led", "turn_on_lightbulb", "unstack_block")
        # fmt: on
        self.task_to_id = {v: k for k, v in enumerate(self.id_to_task)}

    def __len__(self) -> int:
        return len(self.lang_annotations["info"]["indx"])

    def __getitem__(self, idx) -> Dict:
        if self.lang_key == "ann":
            lang_ann = self.lang_annotations["language"][self.lang_key][idx]
        else:
            lang_ann = self.lang_annotations["clip_emb"][self.lang_key][idx]
        task = self.lang_annotations["language"]["task"][idx]
        task_id = self.task_to_id[task]

        frames = self._get_frames_item(idx)

        item = {
            "lang_ann": lang_ann,
            "task_id": task_id,
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
