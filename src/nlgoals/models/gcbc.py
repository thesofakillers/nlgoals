from typing import Dict, Union

import torch.nn as nn
import torch
import pytorch_lightning as pl

from nlgoals.models.clipt import CLIPT
from nlgoals.models.perception_encoders import VisionEncoder, ProprioEncoder


class GCBC(pl.LightningModule):
    """
    Goal Conditioned Behavioural Cloning
    Largely inspired by
    - Lynch et al. (2019)
    - Mees et al. (2022): https://github.com/lukashermann/hulc
    """

    def __init__(
        self,
        traj_encoder_kwargs: Dict,
        vision_encoder_kwargs: Dict,
        proprio_encoder_kwargs: Dict,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        """
        Args:
            traj_encoder_kwargs: Dict of kwargs for the trajectory encoder
                See nlgoals.models.clipt.CLIPT for reference
            vision_encoder_kwargs: Dict of kwargs for the vision encoder
                See nlgoals.models.perception_encoders.vision_encoder.VisionEncoder
                for reference
            proprio_encoder_kwargs: Dict of kwargs for the proprioception encoder
                See nlgoals.models.perception_encoders.proprio_encoder.ProprioEncoder
                for reference
            hidden_dim: Hidden dimension of the GRU
            out_dim: Dimensionality of the output
        """
        super().__init__()
        self.save_hyperparameters()

        self.traj_encoder = CLIPT(**traj_encoder_kwargs)
        self.set_traj_encoder(self.traj_encoder)

        self.vision_encoder = VisionEncoder(**vision_encoder_kwargs)
        self.proprio_encoder = ProprioEncoder(**proprio_encoder_kwargs)

        gru_in_dim = (
            self.traj_encoder.emb_dim
            + self.vision_encoder.emb_dim
            + self.proprio_encoder.emb_dim
        )
        self.gru = nn.GRU(gru_in_dim, hidden_dim, batch_first=True)

        self.mean_linear = nn.Linear()
        self.log_scale_linear = nn.Linear()
        self.mixture_logits_linear = nn.Linear()

    def set_traj_encoder(self, traj_encoder: Union[nn.Module, pl.LightningModule]):
        """Public function for setting the trajectory encoder externally after init"""
        self.traj_encoder = traj_encoder
        # and freeze it
        for param in self.traj_encoder.parameters():
            param.requires_grad = False

    def forward(self, batch):
        """
        Forward pass through the network

        Args:
            batch: Dict of tensors of shape B x S x ..., with keys
                - "rgb_static": B x S x 3 x H x W
                - "robot_obs": B x S x 15, we only use first 7 dims


        Returns:
            Dict of tensors of shape B x S x ..., with keys
            'means'
            'log_scales'
            'mixture_logits'
        """
        frame_height, frame_width = batch["rgb_static"].shape[-2:]
        batch_size, seq_len = batch["rgb_static"].shape[:2]
        goal_pixel_values = batch["rgb_static"][:, -1, :]
        # B * (s-1) x 3 x H x W
        all_frames = batch["rgb_static"][:, :-1, :, :, :].reshape(
            -1, 3, frame_height, frame_width
        )
        # B * (s-1) x 7
        all_robot_obs = batch["robot_obs"][:, :-1, :7].reshape(-1, 7)
        # append the goal_pixel_values to each frame so that B * (s-1) x 2 x 3 x H x W
        all_frames_and_goals = torch.cat(
            [
                all_frames,
                goal_pixel_values.unsqueeze(1).repeat(1, seq_len - 1, 1, 1, 1),
            ],
            dim=1,
        )
        # B * (s-1) x 512
        traj_embs = self.traj_encoder.encode_visual_traj(images=all_frames_and_goals)
        # reshape into B x S-1 x traj_encoder.emb_dim
        traj_embs = traj_embs.reshape(batch_size, seq_len - 1, -1)

        # B * S-1 x visual_encoder.emb_dim
        visual_embs = self.vision_encoder(all_frames)
        # B * S-1 x proprio_encoder.emb_dim
        propr_embs = self.proprio_encoder(all_robot_obs)

        # B * S-1 x (visual_encoder.emb_dim + proprio_encoder.emb_dim)
        perc_embs = torch.cat([visual_embs, propr_embs], dim=-1)
        # reshape into B x S-1 x (visual_encoder.emb_dim + proprio_encoder.emb_dim)
        perc_embs = perc_embs.reshape(batch_size, seq_len - 1, -1)

        # pass concatenation through GRU (don't care about hidden states)
        gru_out, _ = self.gru(torch.cat([traj_embs, perc_embs], dim=-1), self.h_0)

        # use gru output to calculate mean, log_scales and mixture_logits
        # each of shape (B x S-1 x n_dist * out_dim)
        means = self.mean_linear(gru_out)
        log_scales = self.log_scale_linear(gru_out)
        mixture_logits = self.mixture_logits_linear(gru_out)

        return {
            "means": means,
            "log_scales": log_scales,
            "mixture_logits": mixture_logits,
        }

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
