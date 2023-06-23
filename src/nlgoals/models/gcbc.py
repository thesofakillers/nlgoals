from typing import Dict, Union, Tuple

import torch.nn as nn
import torch
import pytorch_lightning as pl
import torchmetrics.functional as tmf

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
        hidden_dim: int = 2048,
        out_dim: int = 7,
        mixture_size: int = 10,
        target_max_bound: float = 1.0,
        target_min_bound: float = -1.0,
        num_target_vals: int = 256,
        rolling_traj: bool = False,
        lr: float = 5e-5,
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
            rolling_traj: whether to update the trajectory embedding at each step,
                default False
            lr: learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        self.rolling_traj = rolling_traj

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

        self.dlml = DLMLLoss(
            mixture_size, target_max_bound, target_min_bound, num_target_vals
        )

        self.mixture_size = mixture_size
        self.lr = lr

    def set_traj_encoder(self, traj_encoder: Union[nn.Module, pl.LightningModule]):
        """Public function for setting the trajectory encoder externally after init"""
        self.traj_encoder = traj_encoder
        # and freeze it
        for param in self.traj_encoder.parameters():
            param.requires_grad = False

    def forward(
        self, batch: Dict[str, Dict[str, torch.Tensor]], traj_mode: str = "visual"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            batch: Dict of Dicts with keys
                - "perception": Dict of tensors of shape B x S x ..., with keys
                    - "rgb_perc": B x S x 3 x H x W, RGB frames of perceived states
                    - "proprio_perc": B x S x P, proprioceptive state
                    - "seq_lens": B, sequence lengths
                - "text": Dict of tensors of shape B x L with keys
                    - "input_ids": B x L
                    - "attention_mask": B x L
                    where L is the max length of text sequences
            traj_mode: "visual" or "textual", whether to use the visual or textual
                trajectories. If "visual", the "text" key must be present in `batch`

        Returns:
            Tuple of packed tensors of shape (P x out_dim x mixture_size) with order
            'means'
            'log_scales'
            'mixture_logits'
        """
        assert traj_mode in {
            "textual",
            "visual",
        }, "`traj_mode` must be textual or visual"
        batch_size, max_seq_len = batch["perception"]["rgb_perc"].shape[:2]
        seq_lens = batch["perception"]["seq_lens"] - 1

        # B x (S-1) x 3 x H x W
        curr_frames = batch["perception"]["rgb_perc"][:, :-1, :, :, :]

        # B x (S-1) x input proprioceptive dims
        curr_proprio_perc = batch["perception"]["proprio_perc"][:, :-1, :]

        # B * (S-1) x traj_encoder.emb_dim
        traj_embs = self._get_traj_embs(batch, curr_frames, traj_mode)

        # B * (S-1) x visual_encoder.emb_dim. Need to use reshape since we indexed
        curr_frames = curr_frames.reshape(-1, *curr_frames.shape[2:])
        visual_embs = self.vision_encoder(curr_frames)
        # B * (S-1) x proprio_encoder.emb_dim. Need to use reshape since we indexed
        curr_proprio_perc = curr_proprio_perc.reshape(-1, *curr_proprio_perc.shape[2:])
        propr_embs = self.proprio_encoder(curr_proprio_perc)

        # B * (S-1) x (visual_encoder.emb_dim + proprio_encoder.emb_dim)
        perc_embs = torch.cat([visual_embs, propr_embs], dim=-1)

        # B x (S-1) x (traj_encoder.emb_dim + visual_encoder.emb_dim + proprio_encoder.emb_dim)
        gru_input = torch.cat([traj_embs, perc_embs], dim=-1).view(
            batch_size, max_seq_len - 1, -1
        )
        # pack: P x (traj_encoder.emb_dim + visual_encoder.emb_dim + proprio_encoder.emb_dim)
        packed_gru_input = nn.utils.rnn.pack_padded_sequence(
            gru_input, seq_lens.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        # what we refer to as "P"
        package_size = packed_gru_input.data.shape[0]
        # P x hidden_dimm; don't provide the init hidden state - torch auto init to zeros
        gru_out, _ = self.gru(packed_gru_input)

        # use gru output to calculate mean, log_scales and mixture_logits
        # each of shape (P x mixture_size * out_dim)
        # reshaped into (P x out_dim x mixture_size)
        means = self.mean_linear(gru_out.data).view(package_size, -1, self.mixture_size)
        log_scales = self.log_scale_linear(gru_out.data).view(
            package_size, -1, self.mixture_size
        )
        mixture_logits = self.mixture_logits_linear(gru_out.data).view(
            package_size, -1, self.mixture_size
        )

        return means, log_scales, mixture_logits

    def _get_traj_embs(
        self, batch: Dict, curr_frames: torch.Tensor, traj_mode: str
    ) -> torch.Tensor:
        """
        Computes either textual or visual trajectory embeddings.

        Returns:
            traj_embs: B * (S-1) x traj_encoder.emb_dim
        """
        if traj_mode == "visual":
            traj_embs = self._get_visual_traj_embs(
                batch["perception"]["rgb_perc"], curr_frames
            )
        elif traj_mode == "textual":
            traj_embs = self._get_textual_traj_embs(
                **batch["text"], curr_frames=curr_frames
            )
        return traj_embs

    def _get_textual_traj_embs(self, input_ids, attention_mask, curr_frames):
        """
        Get the textual trajectory embeddings.
        Rolling traj is not implemented.

        Args:
            input_ids: B x L
            attention_mask: B x L
            curr_frames: B x (S-1) x 3 x H x W

        Returns:
            traj_embs: B * (S-1) x traj_encoder.emb_dim
        """
        max_seq_len = curr_frames.shape[1] + 1
        if self.rolling_traj:
            raise NotImplementedError
        else:
            # B x traj_encoder.emb_dim
            traj_embs = self.traj_encoder.encode_text_traj(
                text_input_ids=input_ids, text_attn_mask=attention_mask
            )
            # same traj_emb for each timestep: B * (S-1) x traj_encoder.emb_dim
            traj_embs = traj_embs.repeat_interleave(max_seq_len - 1, dim=0)
        return traj_embs

    def _get_visual_traj_embs(
        self, batch_frames: torch.Tensor, curr_frames: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the visual trajectory embeddings

        Args:
            batch_frames: B x S x 3 x H x W
            curr_frames: B x (S-1) x 3 x H x W

        Returns:
            traj_embs: B * (S-1) x traj_encoder.emb_dim tensor
        """
        max_seq_len = curr_frames.shape[1] + 1
        if self.rolling_traj:
            # B x 3 x H x W
            final_frames = batch_frames[:, -1, :]
            # appending goal state to each curr state; B x (S-1) x 2 x 3 x H x W
            frames_and_goals = torch.cat(
                [
                    # B x (S-1) x 3 x H x W -> B x s-1 x 1 x 3 x H x W
                    curr_frames.unsqueeze(2),
                    # B x 3 x H x W -> B x S-1 x 1 x 3 x H x W
                    final_frames.unsqueeze(1)
                    .unsqueeze(2)
                    .repeat(1, max_seq_len - 1, 1, 1, 1, 1),
                ],
                dim=2,
            )
            # B * (S-1) x traj_encoder.emb_dim
            traj_embs = self.traj_encoder.encode_visual_traj(
                images=frames_and_goals.view(-1, *frames_and_goals.shape[2:])
            )
        else:
            # B x 2 x 3 x H x W
            start_end = batch_frames[:, [0, -1], :, :, :]
            # B x traj_encoder.emb_dim
            traj_embs = self.traj_encoder.encode_visual_traj(images=start_end)
            # same traj_emb for each timestep: B * (S-1) x traj_encoder.emb_dim
            traj_embs = traj_embs.repeat_interleave(max_seq_len - 1, dim=0)
        return traj_embs

    def _fit_step(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        phase: str,
        traj_mode: str,
    ) -> torch.Tensor:
        """
        Fit step for the model. Logs loss and training metrics.

        Args:
            batch: Dict, with the following keys
                - 'perception': Dict of tensors of shape B x S x ..., with keys
                    - "rgb_perc": B x S x 3 x H x W, RGB frames of perceived state
                    - "proprio_perc": B x S x 15, proprioceptive state
                    - "seq_lens": B, sequence lengths
                - 'text': Dict of tensors of shape B x L x ..., with keys
                    - "input_ids": B x L
                    - "attention_mask": B x L
                - "actions": (B x S x 7) tensor of relative actions
            phase: "train" or "val"
            traj_mode: "visual" or "textual"

        Returns:
            the loss for this batch
        """
        # P x out_dim x mixture_size
        means, log_scales, mixture_logits = self(batch, traj_mode)
        # P x out_dim
        packed_actions = torch.nn.utils.rnn.pack_padded_sequence(
            batch["actions"][:, :-1, :],
            (batch["perception"]["seq_lens"] - 1).detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        loss = self.dlml.loss(means, log_scales, mixture_logits, packed_actions.data)

        # P x out_dim
        pred_act = self.dlml.sample(means, log_scales, mixture_logits)
        # scalar - mean action similarity for the batch
        action_sim = (
            tmf.pairwise_cosine_similarity(
                pred_act, packed_actions.data, reduction=None
            )
            .diag()
            .mean()
        )
        # ignoring the discrete gripper action
        action_dis = (
            # aka L1 distance
            tmf.pairwise_manhattan_distance(
                pred_act[:, :-1], packed_actions.data[:, :-1], reduction=None
            )
            .diag()
            .mean()
        )
        gripper_pred = pred_act[:, -1] > 0
        gripper_gt = packed_actions.data[:, -1] > 0
        gripper_acc = (gripper_pred == gripper_gt).float().mean()

        package_size = packed_actions.data.shape[0]
        self.log(f"{traj_mode}/{phase}_loss", loss, batch_size=package_size)
        self.log(f"{traj_mode}/{phase}_action_sim", action_sim, batch_size=package_size)
        self.log(f"{traj_mode}/{phase}_action_dis", action_dis, batch_size=package_size)
        self.log(
            f"{traj_mode}/{phase}_gripper_acc", gripper_acc, batch_size=package_size
        )

        return loss

    def act(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        traj_mode: str,
    ) -> torch.Tensor:
        """
        Predicts a next action for a given input batch of data

        Args:
            batch: Dict, with the following keys
                - 'perception': Dict of tensors of shape B x S x ..., with keys
                    - "rgb_perc": B x S x 3 x H x W, RGB frames of perceived state
                    - "proprio_perc": B x S x 15, proprioceptive state
                    - "seq_lens": B, sequence lengths
                - 'text': Dict of tensors of shape B x L x ..., with keys
                    - "input_ids": B x L
                    - "attention_mask": B x L
            traj_mode: "visual" or "textual"
        """
        # P x out_dim x mixture_size
        means, log_scales, mixture_logits = self(batch, traj_mode)

        pred_action = self.dlml.sample(means, log_scales, mixture_logits)
        return pred_action

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        visual_batch = self.prepare_visual_batch(batch)
        loss = self._fit_step(visual_batch, "train", "visual")
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        # visual_batch = self.prepare_visual_batch(batch)
        # self._fit_step(visual_batch, "val", "visual")
        textual_batch = self.prepare_textual_batch(batch)
        self._fit_step(textual_batch, "val", "textual")

    def configure_optimizers(self):
        params_to_update = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        return optimizer

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Evaluation is handled by an external script.")

    @staticmethod
    def prepare_textual_batch(batch):
        """
        Prepares collated batch from dataloader for textual trajectory encoding
        Designed as a static method so to be overridden after instantiation
        by a method from nlgoals.interfaces. By default does nothing.
        """
        return batch

    @staticmethod
    def prepare_visual_batch(batch):
        """
        Prepares collated batch from dataloader for visual trajectory encoding
        Designed as a static method so to be overridden after instantiation
        by a method from nlgoals.interfaces. By default does nothing.
        """
        return batch
