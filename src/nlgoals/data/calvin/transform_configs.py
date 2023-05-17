from dataclasses import dataclass
from typing import Tuple


@dataclass
class CLIPTPrepareForCALVIN:
    """CLIPTPrepare args for CALVIN"""

    image_col: str = "rgb_static"
    input_ids_col: str = "text_input_ids"
    attn_mask_col: str = "text_attn_mask"
    clip_model_name: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    image_cols: Tuple[str] = ("rgb_static",)
    text_col: str = "lang_ann"
