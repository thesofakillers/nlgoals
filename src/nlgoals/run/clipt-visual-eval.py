"""Visual Evaluation of CLIPT Model"""
import os

import jsonargparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from nlgoals.data.transforms import CLIPTPrepare
from nlgoals.data.calvin.datamodule import CALVINDM
from nlgoals.models.clipt import CLIPT


def setup_dataloader(args):
    clipt_prepare = CLIPTPrepare(
        image_col="rgb_static",
        input_ids_col="text_input_ids",
        attn_mask_col="text_attn_mask",
        clip_model=args.clip_name,
        image_cols=["rgb_static"],
        text_col="lang_ann",
    )

    calvin = CALVINDM(
        args.data_path,
        num_frames=2,
        batch_size=args.batch_size,
        val_split=0.1,
        seed=42,
        num_workers=args.num_workers,
        frame_keys=["rgb_static"],
        transform=clipt_prepare,
    )

    calvin.prepare_data()
    calvin.setup(stage=args.split)

    if args.split == "debug":
        dataloader = calvin.val_debug_dataloader()
    else:
        dataloader = calvin.test_dataloader()

    return dataloader


def setup_model(args):
    clipt = CLIPT(
        clip_model="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        num_frames=2,
        freeze_clip=True,
    )

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=torch.device(args.device),
    )
    clipt.load_state_dict(checkpoint["state_dict"])

    return clipt


def setup(args):
    print("Setting up...")
    print("...dataloader")
    dataloader = setup_dataloader(args)
    print("...model")
    model = setup_model(args)
    print("Done.")

    return dataloader, model


def compute_matrices(dataloader, model, device):
    print("Computing matrices...")
    traj_vecs = []
    text_vecs = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            for key in batch:
                batch[key] = batch[key].to(device)

            traj_vecs.append(model.encode_visual_traj(batch["images"], normalize=True))
            text_vecs.append(
                model.encode_text_traj(
                    batch["text_input_ids"], batch["text_attn_mask"], normalize=True
                )
            )

        traj_tensor = torch.vstack(traj_vecs)
        text_tensor = torch.vstack(text_vecs)

        similarity_matrix = text_tensor @ traj_tensor.T
        probability_matrix = similarity_matrix.softmax(dim=1)
    print("Done.")
    return similarity_matrix, probability_matrix


def visualize(similarity_matrix, probability_matrix, indices, save_path):
    print("Plotting...")
    f, (ax1, ax2) = plt.subplots(1, 2, dpi=300, figsize=(16, 9), sharey=True)

    ax1 = sns.heatmap(
        probability_matrix.detach().cpu().numpy(),
        ax=ax1,
        square=True,
        cbar_kws={"orientation": "horizontal", "location": "top"},
        xticklabels=indices,
        yticklabels=indices,
    )
    ax1.set_title("Softmaxed Probability")
    ax1.set_ylabel("Text Embeddings idxs")

    ax2 = sns.heatmap(
        similarity_matrix.detach().cpu().numpy(),
        ax=ax2,
        square=True,
        cbar_kws={"orientation": "horizontal", "location": "top"},
        xticklabels=indices,
        yticklabels=indices,
    )
    ax2.set_title("Similarity")

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    print("Done.")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    dataloader, model = setup(args)
    similarity_matrix, probability_matrix = compute_matrices(
        dataloader, model, args.device
    )
    # sample from matrices:
    sample_idxs = torch.randperm(similarity_matrix.size(0))[: args.sample_size]

    similarity_matrix = similarity_matrix[sample_idxs][:, sample_idxs]
    probability_matrix = probability_matrix[sample_idxs][:, sample_idxs]

    visualize(
        similarity_matrix, probability_matrix, sample_idxs.cpu().numpy(), args.save_path
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--clip-name",
        type=str,
        default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        help="CLIP model name on huggingface",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/calvin/task_D_D",
        help="Path to data directory",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/clipt/clipt-v1.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=18, help="Number of workers")
    parser.add_argument(
        "--save-path",
        type=str,
        default="outputs/clipt-eval.pdf",
        help="Path to save figure",
    )
    parser.add_argument(
        "--split", default="test", choices=["test", "debug"], help="Split to use"
    )
    parser.add_argument(
        "--sample-size",
        default=256,
        help="how many samples to use for the plot",
        type=int,
    )

    args = parser.parse_args()
    main(args)
