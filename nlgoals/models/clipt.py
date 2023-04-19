from dataclasses import dataclass
import torch.nn as nn


@dataclass
class CLIPTOptions:
    """Options for CLIPT model"""

    foo: str = "bar"


class CLIPT(nn.Module):
    def __init__(self, args: str, **kwargs):
        super().__init__(**kwargs)
        self.args = args

    def forward(self, images, text):
        image_traj_emb = self.encode_image_traj(images)
        text_traj_emb = self.encode_text_traj(text)

        return image_traj_emb, text_traj_emb
