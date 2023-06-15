"""Discretized Logistic Mixture Likelihood loss"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DLMLLoss(nn.Module):
    """
    Discretized Logistic Mixture Likelihood loss

    Implementation help from
    - https://github.com/Rayhane-mamah/Tacotron-2/
    - https://github.com/lukashermann/hulc/

    For an explanation of what's going on, see
    https://www.giuliostarace.com/posts/dlml-tutorial
    """

    def __init__(
        self,
        mixture_size: int,
        target_max_bound: float,
        target_min_bound: float,
        num_target_vals: int,
    ):
        """
        Args:
            mixture_size: number of components in the mixture
            target_max_bound: maximum value of the target
            target_min_bound: minimum value of the target
            num_target_vals: number of values in the discretized target
        """
        super().__init__()
        self.mixture_size = mixture_size
        self.target_max_bound = target_max_bound
        self.target_min_bound = target_min_bound
        self.num_target_vals = num_target_vals
        self.y_range = target_max_bound - target_min_bound

    def forward(
        self,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        mixture_logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Computes the discretized logistic mixture likelihood loss
        for a batch of sequences. Expects packed sequences.

        Args:
            means: (P x out_dim x mixture_size) means of the mixture
                out_dim is the number of dimensions in the target variable
            log_scales: (P x out_dim x mixture_size) log scales of the mixture
            mixture_logits: (P x out_dim x mixture_size) logits of the mixture
            targets: (P x out_dim) target values

        Returns:
            loss: the loss tensor, optionally reduced
        """
        out_dim = targets.shape[-1]
        # 1 / s
        inv_scales = torch.exp(-log_scales)
        # epsilon value to model the rounding when discretizing
        epsilon = (0.5 * self.y_range) / (self.num_target_vals - 1)
        # broadcast targets to P x out_dim x mixture_size
        targets = targets.unsqueeze(-1).expand_as(means)
        # and center them (y - mu)
        centered_targets = targets - means

        upper_bound_in = inv_scales * (centered_targets + epsilon)
        lower_bound_in = inv_scales * (centered_targets - epsilon)

        upper_cdf = torch.sigmoid(upper_bound_in)
        lower_cdf = torch.sigmoid(lower_bound_in)

        prob_mass = upper_cdf - lower_cdf
        vanilla_log_prob = torch.log(torch.clamp(prob_mass, min=1e-12))

        # handle edges
        # log probability for edge case of 0 (before scaling)
        low_bound_log_prob = upper_bound_in - F.softplus(upper_bound_in)
        # log probability for edge case of 255 (before scaling)
        upp_bound_log_prob = -F.softplus(lower_bound_in)
        # middle "edge" case (very rare)
        mid_in = inv_scales * centered_targets
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
        log_prob_mid = log_pdf_mid - np.log((out_dim - 1) / 2)

        # finally, we build our log likelihood tensor
        log_probs = torch.where(
            # lower edge
            targets < self.target_min_bound + 1e-3,
            low_bound_log_prob,
            torch.where(
                # upper edge
                targets > self.target_max_bound - 1e-3,
                upp_bound_log_prob,
                torch.where(
                    prob_mass > 1e-5,
                    # vanilla case
                    vanilla_log_prob,
                    # extreme case where prob mass is too small
                    log_prob_mid,
                ),
            ),
        )

        # modeling which mixture to sample from
        log_probs = log_probs + F.log_softmax(mixture_logits, dim=-1)

        log_likelihood = torch.sum(log_sum_exp(log_probs), dim=-1)
        loss = -log_likelihood

        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)

        return loss


def log_sum_exp(x):
    """
    numerically stable log_sum_exp implementation that prevents overflow
    Credit to
    https://github.com/lukashermann/hulc/blob/main/hulc/models/decoders/logistic_decoder_rnn.py
    """
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))
