from typing import Dict

import torch.nn as nn
import torch


class ActionDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_state) -> Dict[str, torch.tensor]:
        """
        Args:
            hidden_state: (P, hidden_dim) packed hidden state from a GRU
        """
        raise NotImplementedError

    def loss(self, actions, **kwargs):
        raise NotImplementedError

    def sample(self, **kwargs) -> torch.tensor:
        raise NotImplementedError

    def log_metrics(
        self, pl_instance, pred_act, packed_actions, loss, traj_mode, phase
    ):
        raise NotImplementedError
