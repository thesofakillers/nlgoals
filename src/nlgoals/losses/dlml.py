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
            means: (B x out_dim x mixture_size) means of the mixture
                out_dim is the number of dimensions in the target variable
            log_scales: (B x out_dim x mixture_size) log scales of the mixture
            mixture_logits: (B x out_dim x mixture_size) logits of the mixture
            targets: (B x out_dim) target values

        Returns:
            loss: the loss tensor, optionally reduced
        """
        # 1 / s
        inv_scales = torch.exp(-log_scales)
        # epsilon value to model the rounding when discretizing
        epsilon = (0.5 * self.y_range) / (self.num_y_vals - 1)
        # broadcast targets to B x out_dim x mixture_size and center them
        centered_targets = targets.unsqueeze(-1).repeat(1, self.mixture_size) - means

        upper_bound_in = inv_scales * (centered_targets + epsilon)
        lower_bound_in = inv_scales * (centered_targets - epsilon)

        upper_cdf = torch.sigmoid(upper_bound_in)
        lower_cdf = torch.sigmoid(lower_bound_in)

        prob_mass = upper_cdf - lower_cdf
        vanilla_log_prob = torch.sum(torch.clamp(prob_mass, min=1e-12))

        # handle edges
        # log probability for edge case of 0 (before scaling)
        low_bound_log_prob = upper_bound_in - F.softplus(upper_bound_in)
        # log probability for edge case of 255 (before scaling)
        upp_bound_log_prob = -F.softplus(lower_bound_in)
        # middle "edge" case (very rare)
        mid_in = inv_scales * centered_targets
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
        log_prob_mid = log_pdf_mid - np.log((targets.shape[-1] - 1) / 2)

        # finally, we build our log likelihood tensor
        log_probs = torch.zeros_like(targets)
        # conditions for filling in tensor
        is_near_min = targets < self.target_min_bound + 1e-3
        is_near_max = targets > self.target_max_bound - 1e-3
        is_prob_mass_sufficient = prob_mass > 1e-5
        # And then fill it in accordingly
        # lower edge
        log_probs[is_near_min] = low_bound_log_prob[is_near_min]
        # upper edge
        log_probs[is_near_max] = upp_bound_log_prob[is_near_max]
        # vanilla case
        log_probs[
            ~is_near_min & ~is_near_max & is_prob_mass_sufficient
        ] = vanilla_log_prob[~is_near_min & ~is_near_max & is_prob_mass_sufficient]
        # extreme case where prob mass is too small
        log_probs[
            ~is_near_min & ~is_near_max & ~is_prob_mass_sufficient
        ] = log_prob_mid[~is_near_min & ~is_near_max & ~is_prob_mass_sufficient]

        # modeling which mixture to sample from
        log_probs = log_probs + F.log_softmax(mixture_logits, dim=-1)

        log_likelihood = torch.sum(torch.log_sum_exp(log_probs), dim=-1)
        loss = -log_likelihood

        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
