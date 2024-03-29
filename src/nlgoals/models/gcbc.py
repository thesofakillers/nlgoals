import enum
from typing import Dict, Union, Tuple, Any
import random

import torch.nn as nn
import torch
import pytorch_lightning as pl

from nlgoals.models.clipt import CLIPT
from nlgoals.models.perception_encoders import VisionEncoder, ProprioEncoder
from nlgoals.models.components.action_decoders.calvin import CALVINActionDecoder
from nlgoals.models.components.action_decoders.babyai import BabyAIActionDecoder


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
        action_decoder_kwargs: Dict,
        hidden_dim: int = 2048,
        rolling_traj: bool = False,
        lr: float = 5e-5,
        random_traj_embs: bool = False,
        train_modality: str = "visual",
        val_modality: str = "textual",
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
            action_decoder_kwargs: Dict of kwargs for the action decoder.
                See nlgoals.models.components.action_decoders for reference
            hidden_dim: Hidden dimension of the GRU
            rolling_traj: whether to update the trajectory embedding at each step,
                default False
            lr: learning rate
            random_traj_embs: whether to use random trajectory embeddings (for ablation)
            train_modality: whether to train on visual trajs, textual trajs or both
            val_modality: whether to validate on visual trajs, textual trajs or both
        """
        super().__init__()
        self.save_hyperparameters()

        self.rolling_traj = rolling_traj
        self.traj_embs = None

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
        self.hidden_state = None

        self.hidden_dim = hidden_dim

        self.lr = lr

        self.random_traj_embs = random_traj_embs

        assert train_modality in {"visual", "textual", "both"}
        assert val_modality in {"visual", "textual", "both"}
        self.train_modality = train_modality
        self.val_modality = val_modality

        action_decoder_kwargs = {
            "hidden_dim": hidden_dim,
            **action_decoder_kwargs,
        }
        self._set_action_decoder(**action_decoder_kwargs)
        self._set_additional_metadata()

    def _set_action_decoder(self, **kwargs):
        """
        Responsible for computing loss and sampling predicted actions
        Function to be defined and called by inheriting classes
        """
        raise NotImplementedError

    def _set_additional_metadata(self):
        """
        Responsible for computing loss and sampling predicted actions
        Function to be defined by inheriting classes
        """

        raise NotImplementedError

    def set_traj_encoder(self, traj_encoder: Union[nn.Module, pl.LightningModule]):
        """Public function for setting the trajectory encoder externally after init"""
        self.traj_encoder = traj_encoder
        if self.rolling_traj:
            assert traj_encoder.contextualize_text is True, (
                "Trajectory encoder must handle contextualized text encodings"
                "if rolling trajectories are desired"
            )
        # and freeze it
        for param in self.traj_encoder.parameters():
            param.requires_grad = False

    def reset(self):
        """Resets hidden state and trajectory embedding"""
        self.hidden_state = None
        self.traj_embs = None

    def forward(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]],
        goal: Union[Dict[str, torch.Tensor], torch.Tensor],
        traj_mode: str = "visual",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            batch: Dict of tensors with the following keys. The final frames have been
                   separated, and the seq lens subtracted by 1.
                - "rgb_perc": B x S-1 x 3 x H x W, RGB frames of perceived states
                - "proprio_perc": B x S-1 x P, proprioceptive state
                - "seq_lens": B, sequence lengths (without final frame)
            goal: either dictionary or tensor. More specifically
                - as a dictionary: Dict of text annotation tensors of shape B x L with keys
                    - "input_ids": B x L
                    - "attention_mask": B x L
                    where L is the max length of text sequences
                - as a tensor: B X 3 X H X W, final frames of the sequence.
            traj_mode: "visual" or "textual", whether the goal is represented visually
                or textually

        Returns:
            Dictionary of packed tensors of shape (P x ...).
            For reference, see the appropriate action_decoder.forward
        """
        assert traj_mode in {
            "textual",
            "visual",
        }, "`traj_mode` must be textual or visual"
        batch_size, max_seq_len = batch["rgb_perc"].shape[:2]
        seq_lens = batch["seq_lens"]

        # B x (S-1) x 3 x H x W
        curr_frames = batch["rgb_perc"]

        # B x (S-1) x input proprioceptive dims
        curr_proprio_perc = batch["proprio_perc"]

        # B * (S-1) x traj_encoder.emb_dim
        traj_embs = self._get_traj_embs(curr_frames, traj_mode, goal)

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
            batch_size, max_seq_len, -1
        )
        # pack: P x (traj_encoder.emb_dim + visual_encoder.emb_dim + proprio_encoder.emb_dim)
        packed_gru_input = nn.utils.rnn.pack_padded_sequence(
            gru_input, seq_lens.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        # P x hidden_dim; don't provide the init hidden state - torch auto init to zeros
        gru_out, self.hidden_state = self.gru(packed_gru_input, self.hidden_state)

        action_decoder_out = self.action_decoder(gru_out.data)

        return action_decoder_out

    def _get_traj_embs(
        self,
        curr_frames: torch.Tensor,
        traj_mode: str,
        goal: Union[Dict[str, torch.tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes either textual or visual trajectory embeddings.

        Returns:
            traj_embs: B * (S-1) x traj_encoder.emb_dim
        """
        # used for ablation (usually False)
        if self.random_traj_embs:
            traj_embs = torch.randn(
                curr_frames.shape[0] * curr_frames.shape[0],
                self.traj_encoder.emb_dim,
                device=self.device,
            )
        # used normally
        else:
            if traj_mode == "visual":
                traj_embs = self._get_visual_traj_embs(curr_frames, goal)
            elif traj_mode == "textual":
                traj_embs = self._get_textual_traj_embs(**goal, curr_frames=curr_frames)
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
        max_seq_len = curr_frames.shape[1]
        if self.rolling_traj:
            # (B * (S-1)) x 1 x 3 x H x W;
            # reshape instead of view for contiguous reasons
            # need to unsqueeze(1) to get additional dim necessary for encode_text_traj
            curr_frames = curr_frames.reshape(-1, *curr_frames.shape[2:]).unsqueeze(1)
            # repeat the same text for each frame
            # (B * (S-1)) x L
            input_ids = input_ids.repeat_interleave(max_seq_len, dim=0)
            attention_mask = attention_mask.repeat_interleave(max_seq_len, dim=0)
            # (B * (S-1)) x traj_encoder.emb_dim
            traj_embs = self.traj_encoder.encode_text_traj(
                text_input_ids=input_ids,
                text_attn_mask=attention_mask,
                images=curr_frames,
            )
        else:
            if self.traj_embs is not None:
                # same traj_emb for each timestep
                traj_embs = self.traj_embs
            else:
                # B x 1 x 3 x H x W
                start_frames = curr_frames[:, 0].unsqueeze(1)
                # B x traj_encoder.emb_dim
                traj_embs = self.traj_encoder.encode_text_traj(
                    text_input_ids=input_ids,
                    text_attn_mask=attention_mask,
                    images=start_frames,
                )
                # same traj_emb for each timestep: B * (S-1) x traj_encoder.emb_dim
                traj_embs = traj_embs.repeat_interleave(max_seq_len, dim=0)
                # cache traj_embs
                self.traj_embs = traj_embs
        return traj_embs

    def _get_visual_traj_embs(
        self,
        curr_frames: torch.Tensor,
        final_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the visual trajectory embeddings

        Args:
            curr_frames: B x (S-1) x 3 x H x W
            final_frames: B x 3 x H x W

        Returns:
            traj_embs: B * (S-1) x traj_encoder.emb_dim tensor
        """
        # S-1
        max_seq_len = curr_frames.shape[1]
        if self.rolling_traj:
            # B x (S-1) x 3 x H x W -> B x (S-1) X 1 X 3 X H X W
            curr_frames = curr_frames.unsqueeze(2)
            # B x 3 x H x W -> B x S-1 x 1 x 3 x H x W
            final_frames = (
                final_frames.unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, max_seq_len, -1, -1, -1, -1)
            )
            # appending goal state to each curr state; B x (S-1) x 2 x 3 x H x W
            frames_and_goals = torch.cat([curr_frames, final_frames], dim=2)
            # reshaping for traj_encoder into (B * (S-1)) x 2 x 3 x H x W
            frames_and_goals = frames_and_goals.reshape(-1, *frames_and_goals.shape[2:])
            # finally can get traj_embs; B * (S-1) x traj_encoder.emb_dim
            traj_embs = self.traj_encoder.encode_visual_traj(images=frames_and_goals)
        else:
            if self.traj_embs is not None:
                # same traj_emb for each timestep, computed on first step
                traj_embs = self.traj_embs
            else:
                # appending goal state to the first frame; B x 2 x 3 x H x W
                start_end = torch.cat(
                    [
                        # B x 3 x H x W -> B x 1 x 3 x H x W
                        curr_frames[:, 0].unsqueeze(1),
                        # B x 3 x H x W -> B x 1 x 3 x H x W
                        final_frames.unsqueeze(1),
                    ],
                    dim=1,
                )
                # B x traj_encoder.emb_dim
                traj_embs = self.traj_encoder.encode_visual_traj(images=start_end)
                # same traj_emb for each timestep: B * (S-1) x traj_encoder.emb_dim
                traj_embs = traj_embs.repeat_interleave(max_seq_len, dim=0)
                # cache traj_embs
                self.traj_embs = traj_embs
        return traj_embs

    def _separate_final_step(self, batch: Dict) -> Tuple[Dict, torch.Tensor]:
        """
        Separates the final frames from the batch and returns the batch without the final
        frames and the final frames.

        Args:
            batch: batch of data, see self._fit_step()

        Returns:
            batch: batch of data without the final frames
            final_frames: B x 3 x H x W tensor of final frames
        """
        final_frames = batch["perception"]["rgb_perc"][:, -1]
        batch["perception"]["rgb_perc"] = batch["perception"]["rgb_perc"][:, :-1]
        batch["perception"]["proprio_perc"] = batch["perception"]["proprio_perc"][
            :, :-1
        ]
        batch["perception"]["seq_lens"] = batch["perception"]["seq_lens"] - 1

        return batch, final_frames

    def get_goal(
        self, batch: Dict, traj_mode: str
    ) -> Tuple[Dict, Union[Dict, torch.Tensor]]:
        if traj_mode == "visual":
            return self._separate_final_step(batch)
        elif traj_mode == "textual":
            return self._separate_final_step(batch)[0], batch["text"]

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
                    - "proprio_perc": B x S x ..., proprioceptive state
                    - "seq_lens": B, sequence lengths
                - 'text': Dict of tensors of shape B x L x ..., with keys
                    - "input_ids": B x L
                    - "attention_mask": B x L
                - "actions": (B x S x ...) tensor of actions
                - "rewards": (B x S) tensor of rewards
                - "task_id": (B) tensor of task ids

            phase: "train" or "val"
            traj_mode: "visual" or "textual"

        Returns:
            the loss for this batch
        """
        # separate final frames from the rest of the frames, get goal tensors
        # B x S-1 x ...; (B x 3 x H x W or dict of B x L)
        batch, goal = self.get_goal(batch, traj_mode)

        # Dictionary of P x ...
        action_decoder_out = self(batch["perception"], goal, traj_mode)
        # B x S x ... -> P x ...
        packed_actions = torch.nn.utils.rnn.pack_padded_sequence(
            batch["actions"][:, :-1],
            (batch["perception"]["seq_lens"]).detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        loss = self.action_decoder.loss(
            **action_decoder_out, actions=packed_actions.data
        )
        # P x ...
        pred_act = self.action_decoder.sample(**action_decoder_out)
        self.action_decoder.log_metrics(
            self, pred_act, packed_actions.data, loss, traj_mode, phase
        )
        self.reset()
        return loss

    def step(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        goal: Union[Dict[str, torch.Tensor], torch.Tensor],
        traj_mode: str,
    ) -> torch.Tensor:
        """
        Predicts a next action for a given input (batch) of data

        Args:
            batch: Dict of tensors with the following keys. The final frames have been
                   separated, and the seq lens subtracted by 1.
                - "rgb_perc": B x S-1 x 3 x H x W, RGB frames of perceived states
                - "proprio_perc": B x S-1 x P, proprioceptive state
                - "seq_lens": B, sequence lengths (without final frame)
            goal: either dictionary or tensor. More specifically
                - as a dictionary: Dict of text annotation tensors of shape B x L with keys
                    - "input_ids": B x L
                    - "attention_mask": B x L
                    where L is the max length of text sequences
                - as a tensor: B X 3 X H X W, final frames of the sequence.
            traj_mode: "visual" or "textual", whether the goal is represented visually
                or textually

        Returns:
            pred_act: P x ... tensor of predicted actions
        """
        # Dictionary of P x ...
        action_decoder_out = self(batch, goal, traj_mode)
        # P x ...
        pred_action = self.action_decoder.sample(**action_decoder_out)
        return pred_action

    def prepare_batch(self, batch, modality):
        return (
            self.prepare_visual_batch(batch)
            if modality == "visual"
            else self.prepare_textual_batch(batch)
        )

    def fit_step(self, batch, phase, modality):
        if modality not in {"visual", "textual"}:
            modality = "visual" if random.random() < 0.5 else "textual"

        prep_batch = self.prepare_batch(batch, modality)
        loss = self._fit_step(prep_batch, phase, modality)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.fit_step(batch, "train", self.train_modality)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        self.fit_step(batch, "val", self.val_modality)

    def configure_optimizers(self):
        params_to_update = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        The traj encoder is trained separately, so we already have
        access to its checkpoint and there is no need to save it again.

        Note, when loading the model from checkpoint:
            - set strict to False
            - you will have to manually load the traj_encoder and call `set_traj_encoder`
        """
        for key in list(checkpoint["state_dict"].keys()):
            if key.startswith("traj_encoder"):
                del checkpoint["state_dict"][key]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        The traj encoder is trained separately, so we already have access to its
        checkpoint and load it separately with `set_traj_encoder`.

        However, the PyTorch Lightning Trainer is strict about checkpoint loading (not
        configurable), so it expects the loaded state_dict to match exactly the keys in
        the model. See https://github.com/Lightning-AI/lightning/issues/13246

        So, when loading the checkpoint, before loading it, we add all traj_encoder keys
        to it, so that they match

        ---

        Additionally, we had checkpoints trained where mean_linear, log_scale_linear and
        mixture_logits_linear were part of the main model, rather than the
        action_decoder.

        We patch these cases accordingly by renaming the keys in the checkpoint.
        """
        for key in self.state_dict().keys():
            if key.startswith("traj_encoder"):
                checkpoint["state_dict"][key] = self.state_dict()[key]
        for key in list(checkpoint["state_dict"].keys()):
            if (
                key.startswith("mean_linear")
                or key.startswith("log_scale_linear")
                or key.startswith("mixture_logits_linear")
            ):
                new_key = "action_decoder." + key
                checkpoint["state_dict"][new_key] = checkpoint["state_dict"][key]
                del checkpoint["state_dict"][key]

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


class CALVIN_GCBC(GCBC):
    def _set_action_decoder(
        self,
        hidden_dim,
        out_dim,
        mixture_size,
        target_max_bound,
        target_min_bound,
        num_target_vals,
    ):
        self.action_decoder = CALVINActionDecoder(
            hidden_dim,
            out_dim,
            mixture_size,
            target_max_bound,
            target_min_bound,
            num_target_vals,
        )

    def _set_additional_metadata(self):
        self.name = "GCBC"
        self.datasets = ["CALVIN"]


class BABYAI_GCBC(GCBC):
    def _set_action_decoder(self, hidden_dim, num_target_vals):
        self.action_decoder = BabyAIActionDecoder(hidden_dim, num_target_vals)

    def _set_additional_metadata(self):
        self.name = "GCBC"
        self.datasets = ["BabyAI"]


class GCBC_ENUM(enum.Enum):
    CALVIN = "CALVIN"
    BABYAI = "BABYAI"


gcbc_enum_to_class = {
    "CALVIN": CALVIN_GCBC,
    GCBC_ENUM.CALVIN: CALVIN_GCBC,
    "BABYAI": BABYAI_GCBC,
    GCBC_ENUM.BABYAI: BABYAI_GCBC,
}

if __name__ == "__main__":
    import pytest

    pytest.main()
