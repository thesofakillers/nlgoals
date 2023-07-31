"""Visual Evaluation of CLIPT Model"""
import os

import jsonargparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from nlgoals.data.transforms import TRANSFORM_MAP, TransformName
from nlgoals.interfaces.clipt import (
    CALVIN_CLIPT_PREPARE_CONFIG,
    BABYAI_CLIPT_PREPARE_CONFIG,
)
from nlgoals.data.calvin.legacy.datamodule import CALVINDM
from nlgoals.data.babyai.datamodule import BabyAIDM
from nlgoals.models.clipt import CLIPT
from nlgoals.utils.misc import calc_accuracy_top_k


def setup_dataloader(args):
    if args.dataset == "calvin":
        args_key = "calvin_data"
        clipt_prepare_config = CALVIN_CLIPT_PREPARE_CONFIG
        DatamoduleClass = CALVINDM
    elif args.dataset == "babyai":
        args_key = "babyai_data"
        clipt_prepare_config = BABYAI_CLIPT_PREPARE_CONFIG
        DatamoduleClass = BabyAIDM

    transform_config = clipt_prepare_config[args.data.transform_variant]
    transform_config["mode"] = args.data.transform_variant
    transform_config["clip_model_name"] = args.clipt.clip_model_name
    if args.data.transform_name is not None:
        data_transform = TRANSFORM_MAP[args.data.transform_name.value](
            **transform_config
        )
    else:
        data_transform = None

    datamodule = DatamoduleClass(**args[args_key].as_dict(), transform=data_transform)

    datamodule.prepare_data()
    datamodule.setup(stage="test")

    dataloader = datamodule.test_dataloader()

    return dataloader


def setup_model(args):
    clipt = CLIPT(**args.clipt.as_dict())

    state_dict = torch.load(
        args.checkpoint_path,
        map_location=torch.device(args.device),
    )["state_dict"]
    clipt.load_state_dict(state_dict, strict=False)

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
    visual_vecs = []
    textual_vecs = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            for key in batch:
                batch[key] = batch[key].to(device)

            visual_inputs = model.prepare_visual_inputs(batch)
            visual_vecs.append(
                model.encode_visual_traj(**visual_inputs, normalize=True)
            )

            textual_inputs = model.prepare_textual_inputs(batch)
            textual_vecs.append(
                model.encode_text_traj(**textual_inputs, normalize=True)
            )

        visual_tensor = torch.vstack(visual_vecs)
        textual_tensor = torch.vstack(textual_vecs)

        similarity_matrix = textual_tensor @ visual_tensor.T
        probability_matrix = similarity_matrix.softmax(dim=1)

    print("Done.")
    return similarity_matrix, probability_matrix, visual_tensor, textual_tensor


def visualize_matrices(similarity_matrix, probability_matrix, indices, save_dir):
    print("Plotting matrices")
    f, (ax1, ax2) = plt.subplots(1, 2, dpi=300, figsize=(16, 9), sharey=True)

    ax1 = sns.heatmap(
        probability_matrix,
        ax=ax1,
        square=True,
        cbar_kws={"orientation": "horizontal", "location": "top"},
        xticklabels=False if len(indices) > 50 else indices,
        yticklabels=False if len(indices) > 50 else indices,
    )
    ax1.set_title("Softmaxed Probability")
    ax1.set_ylabel("Text Embeddings")

    ax2 = sns.heatmap(
        similarity_matrix,
        ax=ax2,
        square=True,
        cbar_kws={"orientation": "horizontal", "location": "top"},
        xticklabels=False if len(indices) > 50 else indices,
        yticklabels=False if len(indices) > 50 else indices,
    )
    ax2.set_title("Similarity")

    f.set_tight_layout(True)
    plt.savefig(
        os.path.join(save_dir, f"clipt-eval-{similarity_matrix.shape[0]}.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )


def visualize_similarity_histograms(
    similarity_matrix, traj_tensor, text_tensor, save_dir
):
    print("Plotting histograms...")
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=300, figsize=(16, 6), sharey=True)

    traj_self_similarity = traj_tensor @ traj_tensor.T
    text_self_similarity = text_tensor @ text_tensor.T

    ax1.hist(similarity_matrix.flatten())
    ax1.set_title("Traj-Text Similarity")
    ax1.set_xlabel("Similarity")

    ax2.hist(traj_self_similarity.flatten())
    ax2.set_title("Traj-Traj Similarity")
    ax2.set_xlabel("Similarity")

    ax3.hist(text_self_similarity.flatten())
    ax3.set_title("Text-Text Similarity")
    ax3.set_xlabel("Similarity")

    ax1.set_ylabel("Count")

    f.set_tight_layout(True)
    plt.savefig(
        os.path.join(save_dir, f"similarity-hists-{similarity_matrix.shape[0]}.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )


def visualize(
    similarity_matrix, probability_matrix, traj_tensor, text_tensor, indices, save_dir
):
    print("Plotting...")
    visualize_matrices(similarity_matrix, probability_matrix, indices, save_dir)

    visualize_similarity_histograms(
        similarity_matrix, traj_tensor, text_tensor, save_dir
    )

    print("Done.")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    dataloader, model = setup(args)
    similarity_matrix, probability_matrix, traj_tensor, text_tensor = compute_matrices(
        dataloader, model, args.device
    )

    # sample from matrices:
    sample_idxs = (
        torch.randperm(similarity_matrix.size(0))[: args.sample_size].cpu().numpy()
    )

    similarity_matrix = (
        similarity_matrix[sample_idxs][:, sample_idxs].detach().cpu().numpy()
    )

    probability_matrix = (
        probability_matrix[sample_idxs][:, sample_idxs].detach().cpu().numpy()
    )

    traj_tensor = traj_tensor[sample_idxs].detach().cpu().numpy()
    text_tensor = text_tensor[sample_idxs].detach().cpu().numpy()

    os.makedirs(args.save_dir, exist_ok=True)

    visualize(
        similarity_matrix,
        probability_matrix,
        traj_tensor,
        text_tensor,
        sample_idxs,
        args.save_dir,
    )

    # save the matrices and the tensors
    with open(
        os.path.join(
            args.save_dir, f"similarity-matrix-{similarity_matrix.shape[0]}.npy"
        ),
        "wb",
    ) as f:
        np.save(f, similarity_matrix)
    with open(
        os.path.join(
            args.save_dir, f"probability-matrix-{probability_matrix.shape[0]}.npy"
        ),
        "wb",
    ) as f:
        np.save(f, probability_matrix)
    with open(
        os.path.join(args.save_dir, f"traj-tensor-{traj_tensor.shape[0]}.npy"), "wb"
    ) as f:
        np.save(f, traj_tensor)
    with open(
        os.path.join(args.save_dir, f"text-tensor-{text_tensor.shape[0]}.npy"), "wb"
    ) as f:
        np.save(f, text_tensor)

    acc_top_1 = calc_accuracy_top_k(probability_matrix, 1)
    acc_top_3 = calc_accuracy_top_k(probability_matrix, 3)
    acc_top_5 = calc_accuracy_top_k(probability_matrix, 5)
    acc_top_10 = calc_accuracy_top_k(probability_matrix, 10)
    acc_top_20 = calc_accuracy_top_k(probability_matrix, 20)
    acc_top_50 = calc_accuracy_top_k(probability_matrix, 50)
    # print with 3 decimal places
    print(f"Accuracy@1: {acc_top_1:.3f}")
    print(f"Accuracy@3: {acc_top_3:.3f}")
    print(f"Accuracy@5: {acc_top_5:.3f}")
    print(f"Accuracy@10: {acc_top_10:.3f}")
    print(f"Accuracy@20: {acc_top_20:.3f}")
    print(f"Accuracy@50: {acc_top_50:.3f}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = jsonargparse.ArgumentParser(description=__doc__)

    # data
    parser.add_argument(
        "--data.source",
        type=str,
        choices=["babyai", "calvin"],
        default="calvin",
        help="Which dataset to use",
    )
    parser.add_class_arguments(CALVINDM, "calvin_data", skip={"transform"})
    parser.add_class_arguments(BabyAIDM, "babyai_data", skip={"transform"})

    # transforms
    parser.add_argument(
        "--data.transform_name", type=TransformName, default="clipt-prepare"
    )
    parser.add_argument(
        "--data.transform_variant",
        type=str,
        default="without_clip",
        choices=["without_clip", "with_clip"],
        help=(
            "Without clip: we are using precomputed clip embs. "
            "With clip: we need to compute clip features."
        ),
    )

    # model
    parser.add_class_arguments(CLIPT, "clipt")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/clipt/cclipt_frozen-vision.ckpt",
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/",
        help="Path to save figure",
    )
    parser.add_argument(
        "--sample_size",
        default=256,
        help="how many samples to use for the plot",
        type=int,
    )

    parser.link_arguments(
        "clipt.num_frames", "calvin_data.num_frames", apply_on="parse"
    )

    args = parser.parse_args()
    main(args)
