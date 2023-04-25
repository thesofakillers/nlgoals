from typing import Tuple

import torch
import torch.nn as nn
import transformers


class CLIPT(nn.Module):
    """
    Contrastive Language–Image Pre-training for Trajectories (CLIPT)
    """

    def __init__(self, clip_model: str, num_frames: int, **kwargs):
        """
        Initializes CLIP, traj_encoder, parses attribute
        """
        super().__init__(**kwargs)
        self.clip_model = transformers.CLIPModel.from_pretrained(clip_model)
        self.emb_dim = self.clip_model.config.projection_dim
        self.num_frames = num_frames
        # MLP (n_images x emb_dim) -> (emb_dim) with ReLU activation in between
        self.traj_encoder = nn.Sequential(
            nn.Linear(self.emb_dim * self.num_frames, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combines images and embeds them into visual trajectory embedding
        Embeds text into text trajectory embedding

        Args:
            images: (batch_size, num_frames, 3, H, W) RGB pixel values
            text_input_ids: (batch_size, max_seq_len) tokenized text
            attention_mask: (batch_size, max_seq_len) (1 for tokens, 0 for padding)

        Returns:
            visual_traj_emb: (batch_size, emb_dim)
            text_traj_emb: (batch_size, emb_dim)
        """
        text_traj_emb = self.encode_text_traj(text_input_ids, text_attn_mask)
        visual_traj_emb = self.encode_visual_traj(images)
        return visual_traj_emb, text_traj_emb

    def encode_text_traj(
        self, text_input_ids: torch.Tensor, text_attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Takes an input of text and encodes it into a text trajectory embedding

        Returns:
            text_traj_emb: (batch_size, emb_dim)
        """
        text_traj_emb = self.clip_model.get_text_features(
            input_ids=text_input_ids, attention_mask=text_attn_mask
        )
        return text_traj_emb

    def encode_visual_traj(self, images: torch.Tensor) -> torch.Tensor:
        """
        Takes an input of images and encodes them into a visual trajectory embedding

        Returns:
            visual_traj_emb: (batch_size, emb_dim)
        """
        # (num_frames, B, emb_dim), then we permute to get (B, num_frames, emb_dim)
        image_embs = torch.stack(
            [
                self.clip_model.get_image_features(pixel_values=images[:, i, :, :, :])
                for i in range(self.num_frames)
            ]
        ).permute(1, 0, 2)
        # (batch_size, num_frames x emb_dim)
        image_embs_vec = torch.flatten(image_embs, start_dim=1)
        # (batch_size, emb_dim)
        visual_traj_emb = self.traj_encoder(image_embs_vec)
        return visual_traj_emb


if __name__ == "__main__":
    import pytest

    pytest.main()
