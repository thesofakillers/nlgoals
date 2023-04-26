import enum
from typing import List, Dict

import transformers


class CLIPTransform:
    """
    Transforms for CLIP: tokenizing text, resizing, rescaling images
    """

    def __init__(self, clip_model: str, image_cols: List[str], text_col: str):
        self.image_cols = set(image_cols)
        self.text_col = text_col
        self.clip_processor = transformers.CLIPProcessor.from_pretrained(clip_model)

    def __call__(self, unproc_input: Dict):
        """
        Works with both single and batched samples
        """
        proc_output = {}
        for col in unproc_input.keys():
            if col in self.image_cols:
                proc_output[col] = self.clip_processor(
                    images=unproc_input[col],
                    return_tensors="pt",
                )
            elif col == self.text_col:
                if type(unproc_input[col]) == str:
                    input_text = [unproc_input[col]]
                else:
                    input_text = unproc_input[col]
                proc_output[col] = self.clip_processor(
                    text=input_text, return_tensors="pt", padding=True
                )
            else:
                proc_output[col] = unproc_input[col]

        return proc_output


TRANSFORM_MAP: Dict[str, object] = {"clip-transform": CLIPTransform}
transform_names = TRANSFORM_MAP.keys()
TransformName = enum.Enum("TransformNames", zip(transform_names, transform_names))
