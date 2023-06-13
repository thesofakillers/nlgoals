"""
Functionality for handling the interface between data and model.
Typically simply custom collate functions.
"""
from typing import Dict, Union

import torch


def calvin_gcbc_collate(
    batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Custom collate function for GCBC data.

    Args:
        batch : Dict with the keys:
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

    Returns:
        Dict, with the following keys
            - 'perception': Dict of tensors of shape B x MS x ..., with keys
                - "rgb_perc": B x MS x 3 x H x W, RGB frames of perceived state
                - "proprio_perc": B x MS x 15, proprioceptive state
                - "seq_lens": B, sequence lengths
            - 'text': Dict of tensors of shape B x MSL x ..., with keys
                - "input_ids": B x MSL
                - "attention_mask": B x MSL
            - "actions": (B x MS x 7) tensor of relative actions
    """
    input_ids = None if "lang_input_ids" not in batch else batch["lang_input_ids"]
    attn_mask = None if "lang_attn_mask" not in batch else batch["lang_attn_mask"]

    collated_batch = {
        "perception": {
            "rgb_perc": batch["rgb_static"],
            "proprio_perc": batch["robot_obs"],
            "seq_lens": batch["seq_lens"],
        },
        "text": {"input_ids": input_ids, "attention_mask": attn_mask},
        "actions": batch["actions"],
    }
    return collated_batch


def calvin_gcbc_visual(batch):
    """
    The CALVIN datamodule uses PL CombineLoader for the dataloaders, so batches are
    yielded as dictionaries of batches with keys 'vis' and 'lang'
    """
    return batch["vis"]


def calvin_gcbc_textual(batch):
    """
    The CALVIN datamodule uses PL CombineLoader for the dataloaders, so batches are
    yielded as dictionaries of batches with keys 'vis' and 'lang'
    """
    return batch["lang"]
