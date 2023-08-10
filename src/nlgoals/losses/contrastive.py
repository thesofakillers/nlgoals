"""Contrastive Losses"""
import torch
import torch.nn.functional as F


def clip_contrastive_loss(
    modality_one_embs: torch.tensor,
    modality_two_embs: torch.tensor,
    temperature: torch.nn.parameter.Parameter,
) -> torch.tensor:
    """
    CLIP contrastive loss
    Radford et al. (2021) Contrastive Language-Image Pre-Training

    Adapted from
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py
    """
    logits_per_first = temperature * modality_one_embs @ modality_two_embs.T
    logits_per_second = temperature * modality_two_embs @ modality_one_embs.T

    num_logits = logits_per_first.shape[0]
    labels = torch.arange(num_logits, device=logits_per_first.device, dtype=torch.long)

    loss = (
        F.cross_entropy(logits_per_first, labels)
        + F.cross_entropy(logits_per_second, labels)
    ) / 2

    return loss
