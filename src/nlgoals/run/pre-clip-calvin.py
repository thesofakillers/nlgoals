"""Script for pre-computing CLIP embs of CALVIN dataset"""
import os

from torch.utils.data import DataLoader
import torch
import numpy as np
import pytorch_lightning as pl
from nlgoals.data.calvin.legacy.dataset import CALVINFrameDataset, CALVINTextDataset
import jsonargparse
import transformers
from tqdm import tqdm


def text_main(args):
    pl.seed_everything(args.seed)
    dataset = CALVINTextDataset(**args.text.dataset.as_dict())

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

    all_embs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            lang_ann = batch["lang_ann"]

            inputs = processor(text=lang_ann, return_tensors="pt", padding=True)

            inputs.input_ids = inputs.input_ids.to(device)
            inputs.attention_mask = inputs.attention_mask.to(device)

            # (b x emb_dim)
            embs = model.get_text_features(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
            )
            all_embs.append(embs.cpu().numpy())

    # num_anns x emb_dim
    all_embs = np.concatenate(all_embs, axis=0)
    # add to dict
    if "clip_emb" not in dataset.lang_ann.keys():
        dataset.lang_ann["clip_emb"] = {}
    dataset.lang_ann["clip_emb"][clip_model_name] = all_embs
    # save back to file
    np.save(
        os.path.join(dataset.path, "lang_annotations", "auto_lang_ann.npy"),
        dataset.lang_ann,
        allow_pickle=True,
    )


def frame_main(args):
    pl.seed_everything(args.seed)
    dataset = CALVINFrameDataset(**args.frame.dataset.as_dict())

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
                frame_dict[f"{clip_model_name}_rgb_static"] = (
                    rgb_static_embs[i].cpu().numpy()
                )
                frame_dict[f"{clip_model_name}_rgb_gripper"] = (
                    rgb_gripper_embs[i].cpu().numpy()
                )
                # then save back to file, compressed
                np.savez_compressed(file_path, **frame_dict)


if __name__ == "__main__":
    frame_parser = jsonargparse.ArgumentParser(description="Frame args")
    frame_parser.add_class_arguments(CALVINFrameDataset, "dataset")

    text_parser = jsonargparse.ArgumentParser(description="Text args")
    text_parser.add_class_arguments(CALVINTextDataset, "dataset")

    parser = jsonargparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("frame", frame_parser)
    subcommands.add_subcommand("text", text_parser)

    parser.add_argument(
        "--clip-name", type=str, default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=18)
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.subcommand == "frame":
        frame_main(args)
    elif args.subcommand == "text":
        text_main(args)
    else:
        raise ValueError(f"Invalid subcommand {args.subcommand}")
