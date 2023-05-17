from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
import numpy as np

from nlgoals.losses.contrastive import clip_contrastive_loss


class CLIPT(pl.LightningModule):
    """
    Contrastive Language–Image Pre-training for Trajectories (CLIPT)

    Many design decisions following from
    https://github.com/mlfoundations/open_clip
    """

    def __init__(
        self,
        clip_model_name: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        num_frames: int = 2,
        precomputed_clip: bool = False,
        freeze_clip: bool = True,
        **kwargs,
    ):
        """
        Initializes CLIP, traj_encoder, parses attribute

        Args:
            clip_model_name: name of CLIP model to use
            num_frames: number of frames expected in input
            precomputed_clip: whether to expect precomputed clip embeddings
            freeze_clip: whether to freeze CLIP model
                if `precomputed_clip` is True, this is ignored
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.clip_model_name = clip_model_name
        self.precomputed_clip = precomputed_clip

        self._setup_clip(freeze_clip)

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

    def _setup_clip(self, freeze_clip: bool):
        if self.precomputed_clip:
            self.clip_model = None
            self.freeze_clip = True
        else:
            self.freeze_clip = freeze_clip
            self.set_clip()

    def set_clip(self):
        """Function is public to allow users to set the CLIP model after init"""
        self.precomputed_clip = False
        self.clip_model = transformers.CLIPModel.from_pretrained(self.clip_model_name)
        if self.freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, torch.nn.parameter.Parameter]]:
        """
        Combines images and embeds them into visual trajectory embedding
        Embeds text into text trajectory embedding

        Args:
            batch: Dictionary of tensors handled by prepare_{}_inputs funcs

        Returns:
            dictionary of
                visual_traj_emb: (batch_size, emb_dim)
                text_traj_emb: (batch_size, emb_dim)
                temperature: ()
        """
        visual_inputs = self.prepare_visual_inputs(self, batch)
        visual_traj_emb = self.encode_visual_traj(**visual_inputs, normalize=True)

        textual_inputs = self.prepare_textual_inputs(self, batch)
        text_traj_emb = self.encode_text_traj(**textual_inputs, normalize=True)

        return {
            "visual_traj_emb": visual_traj_emb,
            "text_traj_emb": text_traj_emb,
            "temperature": self.temperature,
        }

    def prepare_textual_inputs(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        textual_inputs = {}
        if self.precomputed_clip:
            textual_inputs["text_traj_emb"] = batch["text_traj_emb"]
        else:
            textual_inputs["text_input_ids"] = batch["text_input_ids"]
            textual_inputs["text_attn_mask"] = batch["text_attn_mask"]
        return textual_inputs

    def prepare_visual_inputs(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        visual_inputs = {}
        if self.precomputed_clip:
            visual_inputs["image_embs"] = batch["image_embs"]
        else:
            visual_inputs["images"] = batch["images"]

    def encode_text_traj(
        self,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attn_mask: Optional[torch.Tensor] = None,
        text_traj_emb: Optional[torch.Tensor] = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Takes an input of text and encodes it into a text trajectory embedding

        Args:
            text_input_ids: (batch_size, max_seq_len) tokenized text
            attention_mask: (batch_size, max_seq_len) (1 for tokens, 0 for padding)
            text_traj_emb: (batch_size, emb_dim) Pre-computed CLIP embedding of text.
                if provided, text_input_ids and text_attn_mask are ignored
            normalize: whether to normalize the text trajectory embeddings

        Returns:
            text_traj_emb: (batch_size, emb_dim)
        """
        assert text_traj_emb is not None or (
            text_input_ids is not None and text_attn_mask is not None
        ), "Must provide either text_emb or text_input_ids and text_attn_mask"

        if text_traj_emb is None:
            text_traj_emb = self.clip_model.get_text_features(
                input_ids=text_input_ids, attention_mask=text_attn_mask
            )
        return F.normalize(text_traj_emb, dim=-1) if normalize else text_traj_emb

    def encode_visual_traj(
        self,
        images: Optional[torch.Tensor] = None,
        image_embs: Optional[torch.Tensor] = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Takes an input of images and encodes them into a visual trajectory embedding

        Args:
            images: (batch_size, num_frames, 3, H, W) RGB pixel values
            image_embs: (batch_size, num_frames, emb_dim) Pre-computed CLIP embeddings
                if provided, images is ignored
            normalize: whether to normalize the visual trajectory embeddings

        Returns:
            visual_traj_emb: (batch_size, emb_dim)
        """
        assert (
            images is not None or image_embs is not None
        ), "Must provide either images or image_embs"
        if image_embs is None:
            # (num_frames, B, emb_dim), then we permute to get (B, num_frames, emb_dim)
            image_embs = torch.stack(
                [
                    self.clip_model.get_image_features(
                        pixel_values=images[:, i, :, :, :]
                    )
                    for i in range(self.num_frames)
                ]
            ).permute(1, 0, 2)
        # (batch_size, num_frames x emb_dim)
        image_embs_vec = torch.flatten(image_embs, start_dim=1)
        # (batch_size, emb_dim)
        visual_traj_emb = self.traj_encoder(image_embs_vec)
        # apply normalization if specified
        return F.normalize(visual_traj_emb, dim=-1) if normalize else visual_traj_emb

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

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        If freeze_clip, we don't fine-tune CLIP, so we can save a lot of space
        by not saving the CLIP model state_dict

        Note, when loading the model from checkpoint:
            - set strict to False
            - you will have to manually load the CLIP model state_dict if necessary
        """
        if self.freeze_clip:
            del checkpoint["state_dict"]["clip_model"]


if __name__ == "__main__":
    import pytest

    pytest.main()
