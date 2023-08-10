"""Encoder for visual state"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class VisionEncoder(nn.Module):
    """
    Vision Network described in Lynch et al. 2020: https://arxiv.org/abs/1903.01973
    Simply encodes input image into a vector.

    Implementation largely credit to https://github.com/lukashermann/hulc

    Args:
        num_channels: number of channels in the input image
        input_width: width of input images
        input_height: height of input images
        dropout_vis_fc: dropout rate
        visual_features: size of visual embedding
        spatial_softmax_temp: temperature for spatial softmax. Optional.
    """

    def __init__(
        self,
        num_channels: int = 3,
        input_width: int = 224,
        input_height: int = 224,
        dropout_vis_fc: float = 0.0,
        visual_features: int = 64,
        spatial_softmax_temp: Optional[float] = 1.0,
    ):
        super(VisionEncoder, self).__init__()
        kernel_sizes = [8, 4, 3]
        strides = [4, 2, 1]
        paddings = [0, 0, 0]
        out_channels = [32, 64, 64]
        w, h = input_width, input_height
        for k, p, s in zip(kernel_sizes, paddings, strides):
            w, h = self.calc_out_size(w, h, k, p, s)
        # B x 128
        self.spatial_softmax = SpatialSoftmax(
            num_cols=w, num_rows=h, temperature=spatial_softmax_temp
        )
        self.act_fn = nn.ReLU()
        # [B, 3, 224, 224] -> [B, 32, 55, 55] -> [B, 64, 26, 26] -> [B, 64, 24, 24]
        self.conv_model = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=out_channels[0],
                kernel_size=kernel_sizes[0],
                stride=strides[0],
            ),
            self.act_fn,
            nn.Conv2d(
                in_channels=out_channels[0],
                out_channels=out_channels[1],
                kernel_size=kernel_sizes[1],
                stride=strides[1],
            ),
            self.act_fn,
            nn.Conv2d(
                in_channels=out_channels[1],
                out_channels=out_channels[2],
                kernel_size=kernel_sizes[2],
                stride=strides[2],
            ),
            self.act_fn,
        )
        # B x 128 -> B x 512
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=out_channels[-1] * 2, out_features=512),
            self.act_fn,
            nn.Dropout(dropout_vis_fc),
        )
        # B x 512 -> B x 64
        self.fc2 = nn.Linear(in_features=512, out_features=visual_features)
        self.ln = nn.LayerNorm(visual_features)
        # attributes accessed by other modules
        self.emb_dim = visual_features

    def forward(self, x: torch.tensor) -> torch.tensor:
        # [B, 3, 224, 224] -> [B, 64, 24, 24]
        x = self.conv_model(x)
        # [B, 64, 24, 24] -> [B, 128]
        x = self.spatial_softmax(x)
        # [B, 128] -> [B, 512]
        x = self.fc1(x)
        # [B, 512] -> [B, 64]
        x = self.fc2(x)
        x = self.ln(x)
        return x

    @staticmethod
    def calc_out_size(
        w: int, h: int, kernel_size: int, padding: int, stride: int
    ) -> Tuple[int, int]:
        width = (w - kernel_size + 2 * padding) // stride + 1
        height = (h - kernel_size + 2 * padding) // stride + 1
        return width, height


class SpatialSoftmax(nn.Module):
    def __init__(
        self, num_cols: int, num_rows: int, temperature: Optional[float] = None
    ):
        """
        Computes the spatial softmax of a convolutional feature map.
        Read more here:
        "Learning visual feature spaces for robotic manipulation with
        deep spatial autoencoders." Finn et al., http://arxiv.org/abs/1509.06113.

        Implementation largely credit to https://github.com/lukashermann/hulc

        Args:
            num_cols : input_image width
            num_rows : input_image height
            temperature : Softmax temperature (optional).
                If None, a learnable temperature is created.
        """
        super(SpatialSoftmax, self).__init__()
        self.num_cols = num_cols
        self.num_rows = num_rows
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, num_rows),
            torch.linspace(-1.0, 1.0, num_cols),
            indexing="ij",
        )
        x_map = grid_x.reshape(-1)
        y_map = grid_y.reshape(-1)
        self.register_buffer("x_map", x_map)
        self.register_buffer("y_map", y_map)
        if temperature:
            self.register_buffer("temperature", torch.ones(1) * temperature)
        else:
            self.temperature = Parameter(torch.ones(1))

    def forward(self, x: torch.tensor) -> torch.tensor:
        _b, c, h, w = x.shape
        x = x.contiguous().view(-1, h * w)  # batch, C, W*H
        softmax_attention = F.softmax(x / self.temperature, dim=1)  # batch, C, W*H
        expected_x = torch.sum(self.x_map * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.y_map * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat((expected_x, expected_y), 1)
        self.coords = expected_xy.view(-1, c * 2)
        return self.coords  # batch, C*2
