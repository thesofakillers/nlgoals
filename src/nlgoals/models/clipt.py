from typing import Union, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl
import transformers
import numpy as np

from nlgoals.losses.contrastive import clip_contrastive_loss


class CLIPT(pl.LightningModule):
    """
    Contrastive Language–Image Pre-training for Trajectories (CLIPT)
    """

    def __init__(
        self,
        clip_model: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        num_frames: int = 2,
        freeze_clip: bool = True,
        **kwargs,
    ):
        """
        Initializes CLIP, traj_encoder, parses attribute

        Args:
            clip_model: name of CLIP model to use
            num_frames: number of frames expected in input
            freeze_clip: whether to freeze CLIP model
        """
        super().__init__(**kwargs)
        self.clip_model = transformers.CLIPModel.from_pretrained(clip_model)
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        self.emb_dim = self.clip_model.config.projection_dim
        self.num_frames = num_frames
        # MLP (n_images x emb_dim) -> (emb_dim) with ReLU activation in between
        self.traj_encoder = nn.Sequential(
            nn.Linear(self.emb_dim * self.num_frames, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

        # "temperature parameter which controls the range of the logits in the softmax,
        # τ , is directly optimized during training as a log-parameterized
        # multiplicative scalar"
        # "learnable temperature parameter τ was initialized to the equivalent of 0.07"
        # https://github.com/mlfoundations/open_clip/blob/3b081484c360569179e270016b5549b7686d42ab/src/open_clip/model.py#L202
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.max_temp_value = 100

    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attn_mask: torch.Tensor,
    ) -> Dict[str, Union[torch.Tensor, torch.nn.parameter.Parameter]]:
        """
        Combines images and embeds them into visual trajectory embedding
        Embeds text into text trajectory embedding

        Args:
            images: (batch_size, num_frames, 3, H, W) RGB pixel values
            text_input_ids: (batch_size, max_seq_len) tokenized text
            attention_mask: (batch_size, max_seq_len) (1 for tokens, 0 for padding)

        Returns:
            dictionary of
                visual_traj_emb: (batch_size, emb_dim)
                text_traj_emb: (batch_size, emb_dim)
                temperature: ()
        """
        visual_traj_emb = self.encode_visual_traj(images)
        text_traj_emb = self.encode_text_traj(text_input_ids, text_attn_mask)

        return {
            "visual_traj_emb": visual_traj_emb,
            "text_traj_emb": text_traj_emb,
            "temperature": self.temperature,
        }

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

    def _fit_step(self, batch: Dict[str, torch.Tensor], phase: str) -> torch.Tensor:
        """
        Args:
            batch: dict with keys "images", "text_input_ids", "text_attn_mask"
            phase: either 'train' or 'val'

        Returns:
            loss: loss for this batch
        """
        # import pdb; pdb.set_trace()
        model_outputs = self.forward(**batch)
        loss = clip_contrastive_loss(
            model_outputs["visual_traj_emb"],
            model_outputs["text_traj_emb"],
            model_outputs["temperature"],
        )
        self.log(f"{phase}_loss", loss)
        return loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._fit_step(batch, phase="train")
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        self._fit_step(batch, phase="val")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # "clipped to prevent scaling the logits by more than 100 "
        #  https://github.com/mlfoundations/open_clip/blob/3b081484c360569179e270016b5549b7686d42ab/src/training/train.py#L175-L177
        with torch.no_grad():
            # in place operation
            self.temperature.clamp_(0, np.log(self.max_temp_value))

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # What metric do we evaluate on?
        raise NotImplementedError

    def configure_optimizers(self):
        params_to_update = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(params_to_update, lr=5e-5)
        return optimizer


if __name__ == "__main__":
    import pytest

    pytest.main()
