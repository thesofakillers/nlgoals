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
        rolling_traj: bool = False,
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

    def forward(self, batch: Dict[Dict[str, torch.Tensor]]) -> Dict[torch.Tensor]:
        """
        Forward pass through the network. If the batch contains the key 'text',
        then textual_traj_forward is called instead of visual_traj_forward

        Args:
            batch: Dict of Dicts with keys
                - 'perception': Dict of tensors of shape B x S x ..., with keys
                    - "rgb_static": B x S x 3 x H x W, RGB frames of robot arm
                    - "robot_obs": B x S x 15, proprioceptive state
                    - "seq_lens": B, sequence lengths
                - 'text': Dict of tensors of shape B x L x ..., with keys
                    - "input_ids": B x L
                    - "attention_mask": B x L

        Returns:
            Dict of packed tensors of shape (P x out_dim x mixture_size) with keys
            'means'
            'log_scales'
            'mixture_logits'
        """
        # frame_height, frame_width = batch["rgb_static"].shape[-2:]
        batch_size, max_seq_len = batch["perception"]["rgb_static"].shape[:2]
        seq_lens = batch["perception"]["seq_lens"] - 1

        # B x 3 x H x W
        final_frames = batch["perception"]["rgb_static"][:, -1, :]

        # B x (s-1) x 3 x H x W
        curr_frames = batch["perception"]["rgb_static"][:, :-1, :, :, :]
        # P x 3 x H x W
        packed_curr_frames = torch.nn.utils.rnn.pack_padded_sequence(
            curr_frames, seq_lens, batch_first=True, enforce_sorted=False
        )
        # what we refer to as "P"
        package_size = packed_curr_frames.shape[0]

        # B x (s-1) x input proprioceptive dims
        curr_robot_obs = batch["perception"]["robot_obs"][:, :-1, :]
        # P x input proprioceptive dims
        packed_curr_robot_obs = torch.nn.utils.rnn.pack_padded_sequence(
            curr_robot_obs, seq_lens, batch_first=True, enforce_sorted=False
        )

        # appending goal state to each curr state; B x (s-1) x 2 x 3 x H x W
        frames_and_goals = torch.cat(
            [
                # B x (s-1) x 3 x H x W -> B x s-1 x 1 x 3 x H x W
                curr_frames.unsqueeze(2),
                # B x 3 x H x W -> B x s-1 x 1 x 3 x H x W
                final_frames.unsqueeze(1)
                .unsqueeze(2)
                .repeat(1, max_seq_len - 1, 1, 1, 1, 1),
            ],
            dim=2,
        )
        # P x 2 x 3 x H x W
        packed_frames_and_goals = nn.utils.rnn.pack_padded_sequence(
            frames_and_goals, seq_lens, batch_first=True, enforce_sorted=False
        )

        # P x traj_encoder.traj_dim
        traj_embs = self.traj_encoder.encode_visual_traj(images=packed_frames_and_goals)

        # P x visual_encoder.emb_dim
        visual_embs = self.vision_encoder(packed_curr_frames)
        # P x proprio_encoder.emb_dim
        propr_embs = self.proprio_encoder(packed_curr_robot_obs)

        # P x (visual_encoder.emb_dim + proprio_encoder.emb_dim)
        perc_embs = torch.cat([visual_embs, propr_embs], dim=-1)

        # P x (traj_encoder.traj_dim + visual_encoder.emb_dim + proprio_encoder.emb_dim)
        gru_input = torch.cat([traj_embs, perc_embs], dim=-1)
        # P x hidden_dim
        gru_out, _ = self.gru(gru_input, self.h_0)

        # use gru output to calculate mean, log_scales and mixture_logits
        # each of shape (P x mixture_size * out_dim)
        # reshaped into (P x out_dim x mixture_size)
        means = self.mean_linear(gru_out).view(package_size, -1, self.mixture_size)
        log_scales = self.log_scale_linear(gru_out).view(
            package_size, -1, self.mixture_size
        )
        mixture_logits = self.mixture_logits_linear(gru_out).view(
            package_size, -1, self.mixture_size
        )

        return {
            "means": means,
            "log_scales": log_scales,
            "mixture_logits": mixture_logits,
        }

    def visual_traj_forward(self, batch) -> Dict[torch.Tensor]:
        raise NotImplementedError
        pass

    def textual_traj_forward(self, batch):
        raise NotImplementedError
        pass

    def training_step(
        self, batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], batch_idx
    ) -> torch.Tensor:
        """
        Training step for the model

        Args:
            batch: Dict with keys
                - 'perception': Dict of tensors of shape B x S x ..., with keys
                    - "rgb_static": B x S x 3 x H x W, RGB frames of robot arm
                    - "robot_obs": B x S x 15, proprioceptive state
                    - "seq_lens": B, sequence lengths
                - 'text': Dict of tensors of shape B x L x ..., with keys
                    - "input_ids": B x L
                    - "attention_mask": B x L
                - "actions": (B x S x 7) tensor of relative actions

        Returns:
            the loss for this batch
        """
        # P x out_dim x mixture_size
        means, log_scales, mixture_logits = self(batch).values()
        # P x out_dim
        packed_actions = torch.nn.utils.rnn.pack_padded_sequence(
            batch["actions"][:, :-1, :],
            batch["perception"]["seq_lens"] - 1,
            batch_first=True,
        )
        loss = self.loss(means, log_scales, mixture_logits, packed_actions)

        self.log("train_loss", loss)
        # TODO: other metrics?

        return loss

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
