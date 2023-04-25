from typing import List, Dict

import transformers


class CLIPTransform:
    """
    Transforms for CLIP
    """

    def __init__(self, clip_model: str, image_cols: List[str], text_col: str):
        self.image_cols = set(image_cols)
        self.text_col = text_col
        self.clip_processor = transformers.CLIPProcessor.from_pretrained(clip_model)

    def __call__(self, sample: Dict):
        """
        Works with both single and batched samples
        """
        processed_sample = {}
        for col in sample.keys():
            if col in self.image_cols:
                processed_sample[col] = self.clip_processor(
                    images=sample[col],
                    return_tensors="pt",
                )
            elif col == self.text_col:
                if type(sample[col]) == str:
                    input_text = [sample[col]]
                else:
                    input_text = sample[col]
                processed_sample[col] = self.clip_processor(
                    text=input_text, return_tensors="pt", padding=True
                )
            else:
                processed_sample[col] = sample[col]

        return processed_sample
