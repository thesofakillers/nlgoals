CALVIN_CLIPT_PREPARE_CONFIG = {
    # CLIPT without CLIP (i.e. using precomputed embeddings)
    "without_clip": {
        "image_col": "rgb_static",
        "text_col": "lang_ann",
        "task_col": "task_id",
        # this should be overridden accordingly
        "clip_model_name": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    },
    # CLIPT with CLIP (i.e. using raw images and text, that need to be processed)
    "with_clip": {
        "image_col": "rgb_static",
        "input_ids_col": "text_input_ids",
        "attn_mask_col": "text_attn_mask",
        "task_col": "task_id",
        # this should be overridden accordingly
        # args for CLIPTransform
        "clip_model_name": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        "image_cols": ("rgb_static",),
        "text_col": "lang_ann",
    },
}
