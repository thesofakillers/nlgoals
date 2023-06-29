"""Action Decoder for the Calvin Dataset"""
import torch.nn.functional as F
import torchmetrics.functional as tmf

from nlgoals.losses.dlml import DLMLLoss


class CALVINActionDecoder:
    """Action Decoder for the Calvin Dataset"""

    def __init__(
        self,
        mixture_size: int,
        target_max_bound: float,
        target_min_bound: float,
        num_target_vals: int,
    ) -> None:
        self.loss_module = DLMLLoss(
            mixture_size, target_max_bound, target_min_bound, num_target_vals
        )

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
