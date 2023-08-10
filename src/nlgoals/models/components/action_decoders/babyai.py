from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as tmf

from nlgoals.models.components.action_decoders import ActionDecoder


class BabyAIActionDecoder(ActionDecoder):
    """Action Decoder for the BabyAI Dataset"""

    def __init__(self, hidden_dim: int, num_target_vals: int = 7):
        """
        Args:
            num_target_vals: number of possible values in the discretized target
        """
        super().__init__()
        self.num_target_vals = num_target_vals
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, num_target_vals * 3),
            nn.ReLU(),
            nn.Linear(num_target_vals * 3, num_target_vals),
        )

    def forward(self, hidden_state) -> Dict[str, torch.tensor]:
        """
        Args:
            hidden_state: (P, hidden_dim) packed hidden state from a GRU

        Returns:
            Dictionary with key "action_logits" with tensor of (P, num_target_vals)
        """
        logits = self.mlp(hidden_state)
        return {"action_logits": logits}

    def loss(self, action_logits, actions):
        """
        Args:
            action_logits: (P, num_target_vals) tensor of action logits
            actions: (P, ) tensor of gold actions
        """
        return F.cross_entropy(action_logits, actions)

    def sample(self, action_logits):
        """
        Args:
            action_logits: (P, num_target_vals) tensor of action logits

        Returns:
            Tensor of shape (P, ) of sampled actions for the batch
        """
        # out_dim is just 1 for babyai
        return action_logits.argmax(dim=-1)

    def log_metrics(
        self, pl_instance, pred_act, packed_actions, loss, traj_mode, phase
    ):
        action_acc = (pred_act == packed_actions.data).float().mean()
        package_size = packed_actions.data.shape[0]

        if traj_mode is not None:
            prefix_str = f"{traj_mode}/"
        else:
            prefix_str = ""
        pl_instance.log(f"{prefix_str}{phase}_loss", loss, batch_size=package_size)
        pl_instance.log(
            f"{prefix_str}{phase}_action_acc", action_acc, batch_size=package_size
        )
