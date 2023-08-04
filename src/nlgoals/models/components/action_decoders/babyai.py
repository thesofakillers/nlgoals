import torch.nn as nn

class BabyAIActionDecoder(nn.Module):
    """Action Decoder for the BabyAI Dataset"""

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def log_metrics(self):
        raise NotImplementedError
