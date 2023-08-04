import numpy as np
import torch


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
