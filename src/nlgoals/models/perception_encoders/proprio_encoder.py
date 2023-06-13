"""Encoder for proprioceptive state"""
from typing import List, Union

import torch.nn as nn


class ProprioEncoder(nn.Module):
    """
    Proprioceptive state encoder
    """

    def __init__(self, proprioception_idxs: Union[List[int], int] = 8):
        super(ProprioEncoder, self).__init__()
        self.identity = nn.Identity()
        self.proprioception_idxs = (
            proprioception_idxs
            if isinstance(proprioception_idxs, list)
            else list(range(proprioception_idxs))
        )
        self.emb_dim = len(proprioception_idxs)

    def forward(self, x):
        """
        Aside for indexing, is no-op

        Args:
            x: Tensor of shape B x D
                where D >= emb_dim

        Returns:
            Tensor of shape B x emb_dim
        """
        return self.identity(x[:, self.proprioception_idxs])
