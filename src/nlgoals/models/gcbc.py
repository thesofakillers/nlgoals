from typing import Dict, Union

import torch.nn as nn
import torch
import pytorch_lightning as pl

from nlgoals.models.clipt import CLIPT
from nlgoals.models.perception_encoders import VisionEncoder, ProprioEncoder
from nlgoals.losses.dlml import DLMLLoss


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
        mixture_size: int,
        target_max_bound: float,
        target_min_bound: float,
        num_target_vals: int,
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
            mixture_size: Number of distributions in the DLML mixture
            target_max_bound: maximum value of the expected target
            target_min_bound: minimum value of the  expected target
            num_target_vals: number of values in the discretized target
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

        total_out_dim = out_dim * mixture_size

        self.mean_linear = nn.Linear(hidden_dim, total_out_dim)
        self.log_scale_linear = nn.Linear(hidden_dim, total_out_dim)
        self.mixture_logits_linear = nn.Linear(hidden_dim, total_out_dim)

        self.loss = DLMLLoss(
            mixture_size, target_max_bound, target_min_bound, num_target_vals
        )

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
                - "robot_obs": B x S x 15


        Returns:
            Dict of tensors of shape (B x S x out_dim x mixture_size) with keys
            'means'
            'log_scales'
            'mixture_logits'
        """
        frame_height, frame_width = batch["rgb_static"].shape[-2:]
        batch_size, seq_len = batch["rgb_static"].shape[:2]
        goal_pixel_values = batch["rgb_static"][:, -1, :]
        # B * (s-1) x 3 x H x W
        all_frames = batch["rgb_static"][:, :-1, :, :, :].view(
            -1, 3, frame_height, frame_width
        )
        # B * (s-1) x input proprioceptive dims
        all_robot_obs = batch["robot_obs"][:, :-1, :].view(
            -1, batch["robot_obs"].shape[-1]
        )
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
        traj_embs = traj_embs.view(batch_size, seq_len - 1, -1)

        # B * S-1 x visual_encoder.emb_dim
        visual_embs = self.vision_encoder(all_frames)
        # B * S-1 x proprio_encoder.emb_dim
        propr_embs = self.proprio_encoder(all_robot_obs)

        # B * S-1 x (visual_encoder.emb_dim + proprio_encoder.emb_dim)
        perc_embs = torch.cat([visual_embs, propr_embs], dim=-1)
        # reshape into B x S-1 x (visual_encoder.emb_dim + proprio_encoder.emb_dim)
        perc_embs = perc_embs.view(batch_size, seq_len - 1, -1)

        # pass concatenation through GRU (don't care about hidden states)
        gru_out, _ = self.gru(torch.cat([traj_embs, perc_embs], dim=-1), self.h_0)

        # use gru output to calculate mean, log_scales and mixture_logits
        # each of shape (B x S-1 x mixture_size * out_dim)
        # reshaped into (B x S-1 x out_dim x mixture_size)
        means = self.mean_linear(gru_out).view(
            batch_size, seq_len - 1, -1, self.mixture_size
        )
        log_scales = self.log_scale_linear(gru_out).view(
            batch_size, seq_len - 1, -1, self.mixture_size
        )
        mixture_logits = self.mixture_logits_linear(gru_out).view(
            batch_size, seq_len - 1, -1, self.mixture_size
        )

        return {
            "means": means,
            "log_scales": log_scales,
            "mixture_logits": mixture_logits,
        }

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Training step for the model

        Args:
            batch: Dict of tensors of shape B x S x ..., with keys
                - "rgb_static": (B x S x 3 x H x W) RGB frames of robot arm
                - "robot_obs": (B x S x 15) proprioceptive state
                - "actions": (B x S x 7) relative actions

        Returns:
            the loss for this batch
        """
        means, log_scales, mixture_logits = self(batch).values()
        loss = self.loss(means, log_scales, mixture_logits, batch["actions"])

        self.log("train_loss", loss)
        # TODO: other metrics?

        return loss

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
