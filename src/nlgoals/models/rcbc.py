from typing import Dict, Union

import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl

from nlgoals.models.perception_encoders import VisionEncoder, ProprioEncoder
from nlgoals.models.components.action_decoders.babyai import BabyAIActionDecoder


class RCBC(pl.LightningModule):
    """
    Reward Conditioned Behavioural Cloning
    """

    def __init__(
        self,
        vision_encoder_kwargs: Dict,
        proprio_encoder_kwargs: Dict,
        action_decoder_kwargs: Dict,
        n_tasks: int = 1,
        reward_dim: int = 18,
        hidden_dim: int = 2048,
        lr: float = 5e-4,
    ) -> None:
        """
        Args:
            vision_encoder_kwargs: Dict of kwargs for the vision encoder
                See nlgoals.models.perception_encoders.vision_encoder.VisionEncoder
                for reference
            proprio_encoder_kwargs: Dict of kwargs for the proprioception encoder
                See nlgoals.models.perception_encoders.proprio_encoder.ProprioEncoder
                for reference
            action_decoder_kwargs: Dict of kwargs for the action decoder.
                See nlgoals.models.components.action_decoders for reference
            n_tasks: Number of tasks in the dataset
            reward_dim: Dimension of the reward vector. If n_tasks > 1, this param
                is ignored and the reward dimension is set to n_tasks. Otherwise, the
                scalar reward is tiled to be a vector of length n_tasks
            hidden_dim: Hidden dimension of the GRU
            lr: learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        self.reward_emb = None

        self.vision_encoder = VisionEncoder(**vision_encoder_kwargs)
        self.proprio_encoder = ProprioEncoder(**proprio_encoder_kwargs)

        assert n_tasks > 0, "n_tasks must be > 0"
        self.n_tasks = n_tasks
        if n_tasks > 1:
            self.reward_dim = n_tasks
        else:
            assert reward_dim >= n_tasks, "reward_dim must be >= n_tasks"
            self.reward_dim = reward_dim

        gru_in_dim = (
            self.reward_dim + self.vision_encoder.emb_dim + self.proprio_encoder.emb_dim
        )
        self.gru = nn.GRU(gru_in_dim, hidden_dim, batch_first=True)
        self.hidden_state = None

        self.hidden_dim = hidden_dim

        self.lr = lr

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

    def reset(self):
        """Resets hidden state and reward embedding"""
        self.hidden_state = None
        self.reward_emb = None

    def forward(
        self,
        rgb_perc: torch.Tensor,
        proprio_perc: torch.Tensor,
        reward: torch.Tensor,
        task_id: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            rgb_perc: B x S x C x H x W, RGB frames of perceived states
            proprio_perc: B x S x proprio_dim, proprioceptive features of perceived states
            reward: B, reward at final step of trajectory
            task_id: B, task id of trajectory
            seq_lens: B, sequence lengths of perceived states

        Returns:
            Dictionary of packed tensors of shape (P x ...).
            For reference, see the appropriate action_decoder.forward
        """
        batch_size, max_seq_len = rgb_perc.shape[:2]
        seq_lens = seq_lens

        # B x reward_dim
        reward_embs = self._get_reward_emb(reward, task_id)
        # repeat the same reward for every frame B * S x reward_dim
        reward_embs = reward_embs.repeat_interleave(max_seq_len, dim=0)

        # B * S x 3 x H x W
        rgb_perc = rgb_perc.reshape(-1, *rgb_perc.shape[2:])
        # B * S x visual_encoder.emb_dim.
        visual_embs = self.vision_encoder(rgb_perc)
        # B * S x proprio_encoder.emb_dim.
        proprio_perc = proprio_perc.reshape(-1, *proprio_perc.shape[2:])
        propr_embs = self.proprio_encoder(proprio_perc)

        # B * S x (visual_encoder.emb_dim + proprio_encoder.emb_dim)
        perc_embs = torch.cat([visual_embs, propr_embs], dim=-1)

        # B x S x (reward_dim + visual_encoder.emb_dim + proprio_encoder.emb_dim)
        gru_input = torch.cat([reward_embs, perc_embs], dim=-1).view(
            batch_size, max_seq_len, -1
        )
        # pack: P x (reward_dim + visual_encoder.emb_dim + proprio_encoder.emb_dim)
        packed_gru_input = nn.utils.rnn.pack_padded_sequence(
            gru_input, seq_lens.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        # P x hidden_dim; don't provide the init hidden state - torch auto init to zeros
        gru_out, self.hidden_state = self.gru(packed_gru_input, self.hidden_state)

        action_decoder_out = self.action_decoder(gru_out.data)

        return action_decoder_out

    def _get_reward_emb(
        self, reward: torch.Tensor, task_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rewards: B, reward at final step of trajectory
            task_id: B, task id of trajectory

        Returns:
            B x reward_dim, reward embeddings
        """
        if self.reward_emb is not None:
            reward_emb = self.reward_emb

        if self.n_tasks == 1:
            # just tile the scalar reward over the reward_dim
            reward_emb = reward.unsqueeze(-1).repeat_interleave(self.reward_dim, dim=-1)
        else:
            # one-hot encode the task id and multiply by the reward
            task_id_onehot = F.one_hot(
                task_id.to(torch.int64), num_classes=self.n_tasks
            ).float()
            reward_emb = reward.unsqueeze(-1) * task_id_onehot

        return reward_emb

    def _fit_step(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        phase: str,
    ) -> torch.Tensor:
        """
        Fit step for the model. Logs loss and training metrics.

        Args:
            batch: Dict, with the following keys
                - 'perception': Dict of tensors of shape B x S x ..., with keys
                    - "rgb_perc": B x S x 3 x H x W, RGB frames of perceived state
                    - "proprio_perc": B x S x ..., proprioceptive state
                    - "seq_lens": B, sequence lengths 'text': Dict of tensors of shape B x L x ..., with keys
                - "actions": (B x S x ...) tensor of actions
                - "rewards": (B x S) tensor of rewards
                - "task_id": (B) tensor of task ids
            phase: "train" or "val"

        Returns:
            the loss for this batch
        """

        perc = batch["perception"]
        # get the reward at the final step of the trajectory B x S -> B
        reward = batch["rewards"][:, -1]

        # Dictionary of P x ...
        action_decoder_out = self.forward(
            perc["rgb_perc"],
            perc["proprio_perc"],
            reward,
            batch["task_id"],
            perc["seq_lens"],
        )
        # B x S x ... -> P x ...
        packed_actions = torch.nn.utils.rnn.pack_padded_sequence(
            batch["actions"],
            (perc["seq_lens"]).detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        loss = self.action_decoder.loss(
            **action_decoder_out, actions=packed_actions.data
        )
        # P x ...
        pred_act = self.action_decoder.sample(**action_decoder_out)
        self.action_decoder.log_metrics(
            self, pred_act, packed_actions.data, loss, None, phase
        )
        self.reset()
        return loss

    def step(
        self,
        rgb_perc: torch.Tensor,
        proprio_perc: torch.Tensor,
        reward: torch.Tensor,
        task_id: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts a next action for a given input (batch) of data

        Args:
            rgb_perc: B x S x C x H x W, RGB frames of perceived states
            proprio_perc: B x S x proprio_dim, proprioceptive features of perceived states
            reward: B, reward at final step of trajectory
            task_id: B, task id of trajectory
            seq_lens: B, sequence lengths of perceived states

        Returns:
            pred_act: P x ... tensor of predicted actions
        """
        # Dictionary of P x ...
        action_decoder_out = self.forward(
            rgb_perc, proprio_perc, reward, task_id, seq_lens
        )
        # P x ...
        pred_action = self.action_decoder.sample(**action_decoder_out)
        return pred_action

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._fit_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        self._fit_step(batch, "val")

    def configure_optimizers(self):
        params_to_update = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        return optimizer

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Evaluation is handled by an external script.")


class BABYAI_RCBC(RCBC):
    def _set_action_decoder(self, hidden_dim, num_target_vals):
        self.action_decoder = BabyAIActionDecoder(hidden_dim, num_target_vals)

    def _set_additional_metadata(self):
        self.name = "RCBC"
        self.datasets = ["BabyAI"]
