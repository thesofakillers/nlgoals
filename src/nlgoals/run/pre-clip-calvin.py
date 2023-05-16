"""Script for pre-computing CLIP embs of CALVIN dataset"""
import os

from torch.utils.data import DataLoader
import torch
import numpy as np
import pytorch_lightning as pl
from nlgoals.data.calvin.dataset import CALVINFrameDataset
import jsonargparse
import transformers
from tqdm import tqdm


def ann_main(args):
    # todo
    pass


def frame_main(args):
    pl.seed_everything(args.seed)
    dataset = CALVINFrameDataset(**args.dataset.as_dict())

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model_name = args.clip_name
    model = transformers.CLIPModel.from_pretrained(clip_model_name)
    processor = transformers.CLIPProcessor.from_pretrained(clip_model_name)

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            rgb_static = batch["rgb_static"]
            rgb_gripper = batch["rgb_gripper"]

            rgb_static_inputs = processor(
                images=rgb_static, return_tensors="pt"
            ).pixel_values.to(device)
            rgb_gripper_inputs = processor(
                images=rgb_gripper, return_tensors="pt"
            ).pixel_values.to(device)

            rgb_static_embs = model.get_image_features(rgb_static_inputs)
            rgb_gripper_embs = model.get_image_features(rgb_gripper_inputs)

            # then save to file
            for i, frame_id in enumerate(batch["frame_id"]):
                file_path = os.path.join(dataset.path, f"episode_{frame_id}.npz")
                # first load the file
                frame_file = np.load(file_path)
                frame_dict = dict(frame_file)
                # then add the embs
                frame_dict[f"{clip_model_name}_rgb_static"] = rgb_static_embs[i]
                frame_dict[f"{clip_model_name}_rgb_gripper"] = rgb_gripper_embs[i]
                # then save back to file, compressed
                np.savez_compressed(file_path, **frame_dict)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_class_arguments(CALVINFrameDataset, "dataset")

    parser.add_argument(
        "--clip-name", type=str, default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=18)
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    frame_main(args)
