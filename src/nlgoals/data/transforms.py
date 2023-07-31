import enum
from typing import List, Dict

import pdb
import transformers
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
    ToPILImage,
)
import numpy as np


class CLIPTPrepareWithCLIP:
    """
    prepares data for the CLIPT model, assuming CLIP needs to be run on the data

    Resulting batch will have keys 'images', 'text_input_ids', 'text_attn_mask', 'task_id'
    """

    def __init__(
        self,
        image_col: str,
        input_ids_col: str,
        attn_mask_col: str,
        task_col: str,
        **clip_transform_kwargs,
    ):
        """
        Args:
            image_col: which of the columns to use as the 'image' col
            input_ids_col: which of the columns to use as the 'text_input_ids' col
            attn_mask_col: which of the columns to use as the 'text_attn_mask' col
            task_col: which of the column to use as the 'task_id' col
            clip_transform_kwargs: kwargs to pass to CLIPTransform transform
        """
        self.clip_transform = CLIPTransform(**clip_transform_kwargs)
        self.image_col = image_col
        self.input_ids_col = input_ids_col
        self.attn_mask_col = attn_mask_col
        self.task_col = task_col

    def __call__(self, unprep_input: Dict) -> Dict:
        proc_input = self.clip_transform(unprep_input)
        prep_input = {
            "images": proc_input[self.image_col],
            "text_input_ids": proc_input[self.input_ids_col],
            "text_attn_mask": proc_input[self.attn_mask_col],
            "task_id": unprep_input[self.task_col],
        }
        return prep_input


class CLIPTPrepareWithoutCLIP:
    """
    prepares data for the CLIPT model, assuming CLIP has already been run
    and we already have the embeddings

    Resulting batch will have keys 'image_embs', 'lang_emb', 'task_id'
    """

    def __init__(
        self, image_col: str, text_col: str, task_col: str, clip_model_name: str
    ):
        """
        Args:
            image_col: which of the columns to use as the 'image_embs' col
            text_col: which of the columns to use as the 'lang_emb' col
            task_col: which of the column to use as the 'task_id' col
            clip_model_name: name of the CLIP model the precomputed embeddings come from
        """
        self.image_col = f"{clip_model_name}_{image_col}"
        self.text_col = text_col
        self.task_col = task_col

    def __call__(self, unprep_input: Dict) -> Dict:
        prep_input = {
            "image_embs": unprep_input[self.image_col],
            "lang_emb": unprep_input[self.text_col],
            "task_id": unprep_input[self.task_col],
        }
        return prep_input


class CLIPTPrepare:
    """
    prepares data for the CLIPT model
    """

    def __init__(self, mode: str, **mode_kwargs: Dict):
        assert mode in {
            "without_clip",
            "with_clip",
        }, f"mode {mode} not supported. Must be one of 'precomputed' or 'with_clip'"
        self.mode = mode

        mode2modeclass = {
            "without_clip": CLIPTPrepareWithoutCLIP,
            "with_clip": CLIPTPrepareWithCLIP,
        }

        self.transform = mode2modeclass[mode](**mode_kwargs)

    def __call__(self, unprep_input: Dict) -> Dict:
        return self.transform(unprep_input)


class CLIPTransform:
    """
    Transforms for CLIP: tokenizing text, resizing, rescaling images
    """

    def __init__(self, clip_model_name: str, image_cols: List[str], text_col: str):
        """
        Args:
            clip_model_name: which CLIP model to use
            image_cols: which of the columns to process as images
            text_col: which one of the columns to process as text
        """
        self.image_cols = set(image_cols)
        self.text_col = text_col
        self.clip_processor = transformers.CLIPProcessor.from_pretrained(
            clip_model_name
        )
        self.clip_img_transform = CLIPImageTransform(
            size=224,
            image_mean=self.clip_processor.image_processor.image_mean,
            image_std=self.clip_processor.image_processor.image_std,
        )

    def __call__(self, unproc_input: Dict) -> Dict:
        """
        Works with both single and batched samples
        """
        proc_output = {}
        for col in unproc_input.keys():
            if col in self.image_cols:
                proc_output[col] = self.clip_img_transform(images=unproc_input[col])
            elif col == self.text_col:
                if type(unproc_input[col]) == str:
                    input_text = [unproc_input[col]]
                else:
                    input_text = unproc_input[col]
                proc_text = self.clip_processor(
                    text=input_text, return_tensors="pt", padding=True
                )
                proc_output["text_input_ids"] = proc_text.input_ids.squeeze(dim=0)
                proc_output["text_attn_mask"] = proc_text.attention_mask.squeeze(dim=0)
            else:
                proc_output[col] = unproc_input[col]

        return proc_output


class CLIPImageTransform:
    """
    Faster version of CLIPProcessor using torch transforms to leverage
    multiprocessing

    see: https://github.com/huggingface/transformers/issues/13991#issuecomment-1598941671
    """

    def __init__(self, size, image_mean, image_std) -> None:
        self.size = size
        self.image_mean = image_mean
        self.image_std = image_std

        normalize = Normalize(mean=self.image_mean, std=self.image_std)

        """
        0. ToPILImage
        1. resize
        2. do center crop
        4. rescale
        4. to tensor
        5. normalize
        """
        self.pipeline = Compose(
            [
                ToPILImage("RGB"),
                Resize(self.size),
                CenterCrop(self.size),
                ToTensor(),
                Rescale(1 / 255.0),
                normalize,
            ]
        )

    def __call__(self, images: np.ndarray) -> torch.Tensor:
        """
        0. ToPILImage
        1. resize
        2. do center crop
        3. to tensor
        4. rescale
        5. normalize
        """
        images = images.permute(0, 3, 1, 2)
        images = torch.stack([self.pipeline(image) for image in images], dim=0)
        return images


class Rescale:
    def __init__(self, scale_factor: float) -> None:
        self.scale_factor = scale_factor

    def __call__(self, image: torch.tensor) -> torch.tensor:
        return image * self.scale_factor


TRANSFORM_MAP: Dict[str, object] = {
    "clip-transform": CLIPTransform,
    "clipt-prepare": CLIPTPrepare,
}
transform_names = TRANSFORM_MAP.keys()
TransformName = enum.Enum("TransformNames", zip(transform_names, transform_names))
