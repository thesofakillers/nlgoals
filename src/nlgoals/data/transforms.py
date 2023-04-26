import enum
from typing import List, Dict

import transformers


class CLIPTPrepare:
    """
    prepares data for the CLIPT model
    """

    def __init__(
        self,
        image_col: str,
        input_ids_col: str,
        attn_mask_col: str,
        **clip_transform_kwargs,
    ):
        """
        Args:
            image_col: which of the columns to use as the 'image' col
            input_ids_col: which of the columns to use as the 'text_input_ids' col
            attn_mask_col: which of the columns to use as the 'text_attn_mask' col
            clip_transform_kwargs: kwargs to pass to CLIPTransform transform
        """
        self.clip_transform = CLIPTransform(**clip_transform_kwargs)
        self.image_col = image_col
        self.input_ids_col = input_ids_col
        self.attn_mask_col = attn_mask_col

    def __call__(self, unproc_input: Dict) -> Dict:
        unprep_input = self.clip_transform(unproc_input)
        prep_input = {
            "images": unprep_input[self.image_col],
            "text_input_ids": unprep_input[self.input_ids_col],
            "text_attn_mask": unprep_input[self.attn_mask_col],
        }
        return prep_input


class CLIPTransform:
    """
    Transforms for CLIP: tokenizing text, resizing, rescaling images
    """

    def __init__(self, clip_model: str, image_cols: List[str], text_col: str):
        """
        Args:
            clip_model: which CLIP model to use
            image_cols: which of the columns to process as images
            text_col: which one of the columns to process as text
        """
        self.image_cols = set(image_cols)
        self.text_col = text_col
        self.clip_processor = transformers.CLIPProcessor.from_pretrained(clip_model)

    def __call__(self, unproc_input: Dict) -> Dict:
        """
        Works with both single and batched samples
        """
        proc_output = {}
        for col in unproc_input.keys():
            if col in self.image_cols:
                proc_output[col] = self.clip_processor(
                    images=unproc_input[col],
                    return_tensors="pt",
                ).pixel_values
            elif col == self.text_col:
                if type(unproc_input[col]) == str:
                    input_text = [unproc_input[col]]
                else:
                    input_text = unproc_input[col]
                proc_text = self.clip_processor(
                    text=input_text, return_tensors="pt", padding=True
                )
                proc_output["text_input_ids"] = proc_text.input_ids
                proc_output["text_attn_mask"] = proc_text.attention_mask
            else:
                proc_output[col] = unproc_input[col]

        return proc_output


TRANSFORM_MAP: Dict[str, object] = {
    "clip-transform": CLIPTransform,
    "clipt-prepare": CLIPTPrepare,
}
transform_names = TRANSFORM_MAP.keys()
TransformName = enum.Enum("TransformNames", zip(transform_names, transform_names))
