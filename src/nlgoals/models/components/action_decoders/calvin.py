"""Action Decoder for the Calvin Dataset"""
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics.functional as tmf

from nlgoals.losses.dlml import DLMLLoss
from nlgoals.models.components.action_decoders import ActionDecoder


class CALVINActionDecoder(ActionDecoder):
    """
    Action Decoder for the Calvin Dataset.
    Makes use of a Discretized Logistic Mixture Likelihood (DLML) for loss and sampling.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int = 7,
        mixture_size: int = 10,
        target_max_bound: float = 1.0,
        target_min_bound: float = -1.0,
        num_target_vals: int = 256,
    ) -> None:
        """
        Args:
            out_dim: Dimensionality of the actions
            mixture_size: Number of distributions in the DLML mixture
            target_max_bound: maximum value of the expected target
            target_min_bound: minimum value of the  expected target
            num_target_vals: number of values in the discretized target
        """
        super().__init__()
        total_out_dim = out_dim * mixture_size

        self.mixture_size = mixture_size

        self.mean_linear = nn.Linear(hidden_dim, total_out_dim)
        self.log_scale_linear = nn.Linear(hidden_dim, total_out_dim)
        self.mixture_logits_linear = nn.Linear(hidden_dim, total_out_dim)

        self.loss_module = DLMLLoss(
            mixture_size, target_max_bound, target_min_bound, num_target_vals
        )

    def forward(self, hidden_state):
        """
        Args:
            hidden_state: (P, hidden_dim) packed hidden state from a GRU

        Returns:
            Dictionary of tensors with shape (P x out_dim x mixture_size) with keys
            'means'
            'log_scales'
            'mixture_logits'
        """
        # what we refer to as "P"
        package_size = hidden_state.shape[0]
        # use gru output to calculate mean, log_scales and mixture_logits
        # each of shape (P x mixture_size * out_dim)
        # reshaped into (P x out_dim x mixture_size)
        means = self.mean_linear(hidden_state.data).view(
            package_size, -1, self.mixture_size
        )
        log_scales = self.log_scale_linear(hidden_state.data).view(
            package_size, -1, self.mixture_size
        )
        mixture_logits = self.mixture_logits_linear(hidden_state.data).view(
            package_size, -1, self.mixture_size
        )

        return {
            "means": means,
            "log_scales": log_scales,
            "mixture_logits": mixture_logits,
        }

    def loss(self, means, log_scales, mixture_logits, actions):
        loss = self.loss_module.loss(means, log_scales, mixture_logits, actions)
        return loss

    def sample(self, means, log_scales, mixture_logits):
        pred_act = self.loss_module.sample(means, log_scales, mixture_logits)
        return pred_act

    def log_metrics(
        self, pl_instance, pred_act, packed_actions, loss, traj_mode, phase
    ):
        # scalar - mean action similarity for the batch
        action_sim = (
            tmf.pairwise_cosine_similarity(
                pred_act, packed_actions.data, reduction=None
            )
            .diag()
            .mean()
        )
        # ignoring the discrete gripper action, compute mean action distance
        action_dis = F.l1_loss(
            pred_act[:, :-1], packed_actions.data[:, :-1], reduction="mean"
        ).mean()
        # for which we just calculate the accuracy discretely.
        gripper_pred = pred_act[:, -1] > 0
        gripper_gt = packed_actions.data[:, -1] > 0
        gripper_acc = (gripper_pred == gripper_gt).float().mean()

        package_size = packed_actions.data.shape[0]
        pl_instance.log(f"{traj_mode}/{phase}_loss", loss, batch_size=package_size)
        pl_instance.log(
            f"{traj_mode}/{phase}_action_sim", action_sim, batch_size=package_size
        )
        pl_instance.log(
            f"{traj_mode}/{phase}_action_dis", action_dis, batch_size=package_size
        )
        pl_instance.log(
            f"{traj_mode}/{phase}_gripper_acc", gripper_acc, batch_size=package_size
        )
