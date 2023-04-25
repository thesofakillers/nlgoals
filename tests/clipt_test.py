import pytest

import transformers
import torch.nn as nn
import torch

from nlgoals.models.clipt import CLIPT


@pytest.fixture
def clipt():
    batch_size = 2
    num_frames = 4
    text_len = 10
    emb_dim = 512

    images = torch.rand((batch_size, num_frames, 3, 224, 224))
    text_input_ids = torch.randint(100, (batch_size, text_len))
    text_attn_mask = torch.ones((batch_size, text_len))
    targets = torch.ones((batch_size, emb_dim))

    model = CLIPT("openai/clip-vit-base-patch32", num_frames)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    return model, optimizer, criterion, images, text_input_ids, text_attn_mask, targets


def test_forward(clipt):
    model, _, _, images, text_input_ids, text_attn_mask, _ = clipt

    visual_traj_emb, text_traj_emb = model(images, text_input_ids, text_attn_mask)

    assert visual_traj_emb.shape == (2, 512)
    assert text_traj_emb.shape == (2, 512)


def test_initialization(clipt):
    model, _, _, _, _, _, _ = clipt

    assert isinstance(model.clip_model, transformers.CLIPModel)
    assert isinstance(model.traj_encoder, nn.Sequential)
    assert model.emb_dim == model.clip_model.config.projection_dim
    assert model.num_frames == 4


def test_training(clipt):
    model, optimizer, criterion, images, text_input_ids, text_attn_mask, targets = clipt

    optimizer.zero_grad()

    visual_traj_emb, text_traj_emb = model(images, text_input_ids, text_attn_mask)
    loss = criterion(visual_traj_emb, targets)
    loss.backward()
    optimizer.step()

    assert loss.shape == ()
