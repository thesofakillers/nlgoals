"""
Functionality for handling the interface between data and model.
Typically simply custom collate functions.
"""
from typing import Dict, Union

import torch


def babyai_obs_prepare(obs: Dict, transform, device) -> Dict:
    """
    Prepares an observation from the BabyAI environment .step() so that
    it can be passed to GCBC.step()

    Args
        obs: Dict with following keys. It's 1 x 1 because batch size 1, single timestep
            - 'image': np.ndarray (H x W x 3) between 0 and 255
            - 'direction' : np.int64, between 0 and 3 -> right, down, left, up
            - 'mission' : str, the language instruction
        transform: Transform to apply to the image
        device: the device to put the resulting tensors on
    Returns
        Dict, with the following keys
            - "rgb_perc": 1 x 1 x 3 x H x W, RGB frames of perceived state
            - "proprio_perc": 1 x 1 x 1, proprioceptive state
            - "seq_lens": 1, sequence lengths (will just be 1)
    """
    images = torch.from_numpy(obs["image"]).unsqueeze(0).to(device)
    return {
        "rgb_perc": transform(images).unsqueeze(0).to(device),
        "proprio_perc": torch.tensor([obs["direction"]])
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device),
        "seq_lens": torch.tensor([1]).to(device),
    }


def calvin_obs_prepare(obs: Dict, device) -> Dict:
    """
    Prepares an observation from the CALVIN environment .step() so that
    it can be passed to GCBC.step()

    Args:
        obs: Dict with following keys. It's 1 x 1 because batch size 1, single timestep
            - "robot_obs": tensor (1 x 1 x 8)
            - "rgb_obs": Dict of tensors with keys
                - "rgb_static" (1 x 1 x 3 x H x W)
            - "depth_obs: empty dictionary
        device: the device to put the resulting tensors on
    Returns
        Dict, with the following keys
            - "rgb_perc": 1 x 1 x 3 x H x W, RGB frames of perceived state
            - "proprio_perc": 1 x 1 x 8, proprioceptive state
            - "seq_lens": 1, sequence lengths (will just be 1)
    """
    output = {}
    output["rgb_perc"] = obs["rgb_obs"]["rgb_static"].to(device)
    output["proprio_perc"] = obs["robot_obs"].to(device)
    output["seq_lens"] = torch.tensor([1]).to(device)

    return output


def calvin_gcbc_collate(
    batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Custom collate function for CALVIN data.
    Takes data in CALVIN output format and wrangles it into GCBC input format.

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
