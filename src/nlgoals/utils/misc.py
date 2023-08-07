import numpy as np
import torch


def normalize_tensor(tensor):
    # move from -1, 1 to 0, 1
    tensor = tensor / 2 + 0.5
    return tensor

def prep_video(video, normalize=True, type_conv=True):
    # cut off empty frames
    frame_sums = np.sum(video, axis=(1, 2, 3))
    where_0 = np.where(frame_sums == 0)[0]
    end_frame = where_0[0] if len(where_0) > 0 else len(video)
    video = video[:end_frame]
    # put channel dimension last
    video = video.transpose((0, 2, 3, 1))
    if normalize:
        # move from -1, 1 to 0, 1
        video = normalize_tensor(video)
    if type_conv:
        # convert to uint8
        video = (video * 255).astype(np.uint8)
    return video


def calc_accuracy_top_k(matrix, k=5):
    """
    What percentage of samples peak on the diagonal?
    """
    num_samples = matrix.shape[0]
    num_correct = 0
    for i in range(num_samples):
        top_k_idxs = np.argsort(matrix[i])[-k:]
        if i in top_k_idxs:
            num_correct += 1
    return num_correct / num_samples


def pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """
    Pad a sequence Tensor by repeating last element pad_size times.

    Args:
        input_tensor: Sequence to pad.
        pad_size: Number of frames to pad.

    Returns:
        Padded Tensor.
    """
    last_repeated = torch.repeat_interleave(
        torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
    )
    padded = torch.cat((input_tensor, last_repeated), dim=0)
    return padded


def pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """
    Pad a Tensor with zeros.

    Args:
        input_tensor: Sequence to pad.
        pad_size: Number of frames to pad.

    Returns:
        Padded Tensor.
    """
    zeros_repeated = torch.repeat_interleave(
        torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
        repeats=pad_size,
        dim=0,
    )
    padded = torch.vstack((input_tensor, zeros_repeated))
    return padded
