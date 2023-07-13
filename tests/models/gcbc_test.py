import pytest

import torch.nn as nn
import torch

from nlgoals.models.gcbc import CALVIN_GCBC


@pytest.fixture
def gcbc():
    clipt_args = {
        "clip_model_name": "openai/clip-vit-base-patch32",
        "num_frames": 2,
    }
    vision_encoder_kwargs = {"num_channels": 3}
    proprio_encoder_kwargs = {"proprioception_idxs": 8}

    model = CALVIN_GCBC(
        traj_encoder_kwargs=clipt_args,
        vision_encoder_kwargs=vision_encoder_kwargs,
        proprio_encoder_kwargs=proprio_encoder_kwargs,
        rolling_traj=True,
        mixture_size=10,
        out_dim=7,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    return model, optimizer, criterion


def test_visual_forward(gcbc):
    batch = {
        "rgb_perc": torch.randn((4, 19, 3, 224, 224)),
        "proprio_perc": torch.randn((4, 19, 8)),
        "seq_lens": torch.LongTensor([16, 14, 19, 19]),
    }
    goal = torch.randn((4, 3, 224, 224))

    model, _, _ = gcbc

    means, log_scales, mixture_logits = model(batch, goal, "visual")

    assert means.shape[-1] == 10
    assert means.shape[-2] == 7

    assert log_scales.shape[-1] == 10
    assert log_scales.shape[-2] == 7

    assert mixture_logits.shape[-1] == 10
    assert mixture_logits.shape[-2] == 7


def test_textual_forward(gcbc):
    batch = {
        "rgb_perc": torch.randn((4, 19, 3, 224, 224)),
        "proprio_perc": torch.randn((4, 19, 8)),
        "seq_lens": torch.LongTensor([16, 14, 19, 19]),
    }
    goal = {
        "input_ids": torch.randint(low=0, high=4000, size=(4, 5)),
        "attention_mask": torch.ones((4, 5)),
    }

    model, _, _ = gcbc

    means, log_scales, mixture_logits = model(batch, goal, "textual")

    assert means.shape[-1] == 10
    assert means.shape[-2] == 7

    assert log_scales.shape[-1] == 10
    assert log_scales.shape[-2] == 7

    assert mixture_logits.shape[-1] == 10
    assert mixture_logits.shape[-2] == 7
