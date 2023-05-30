from typing import Dict, Union

import torch.nn as nn
import torch
import pytorch_lightning as pl

from nlgoals.models.clipt import CLIPT


class GCBC(pl.LightningModule):
    """
    Goal Conditioned Behavioural Cloning
    Largely inspired by
    - Lynch et al. (2019)
    - Mees et al. (2022): https://github.com/lukashermann/hulc
    """

    def __init__(
        self, traj_encoder_kwargs: Dict, hidden_dim: int, out_dim: int
    ) -> None:
        """
        Args:
            traj_encoder_kwargs: Dict of kwargs for the trajectory encoder
                See CLIPT for reference
            out_dim: Dimensionality of the output
        """
        super().__init__()
        self.save_hyperparameters()
        self.traj_encoder = CLIPT(**traj_encoder_kwargs)
        self.set_traj_encoder(self.traj_encoder)
        # TODO
        self.visual_perc_encoder = "TODO"
        self.propr_perc_encoder = "TODO"

        gru_in_dim = (
            self.traj_emb_dim
            + self.visual_perc_encoder.emb_dim
            + self.propr_perc_encoder.emb_dim
        )
        self.gru = nn.GRU(gru_in_dim, hidden_dim, batch_first=True)

        self.mean_linear = nn.Linear()
        self.log_scale_linear = nn.Linear()
        self.mixture_logits_linear = nn.Linear()

    def set_traj_encoder(self, traj_encoder: Union[nn.Module, pl.LightningModule]):
        """Public function for setting the trajectory encoder"""
        self.traj_encoder = traj_encoder
        # and freeze it
        for param in self.traj_encoder.parameters():
            param.requires_grad = False

        self.traj_emb_dim = self.traj_encoder.emb_dim

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
        # reshape into B x S-1 x traj_emb_dim
        traj_embs = traj_embs.reshape(batch_size, seq_len - 1, -1)

        # B * S-1 x visual_perc_dim
        visual_embs = self.visual_perc_encoder(all_frames)
        # B * S-1 x propr_perc_dim
        propr_embs = self.propr_perc_encoder(all_robot_obs)

        # B * S-1 x (visual_perc_dim + propr_perc_dim)
        perc_embs = torch.cat([visual_embs, propr_embs], dim=-1)
        # reshape into B x S-1 x (visual_perc_dim + propr_perc_dim)
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
