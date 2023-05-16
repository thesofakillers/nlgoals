import numpy as np


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
