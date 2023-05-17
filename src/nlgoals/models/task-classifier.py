from typing import Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics.functional as tmF

from nlgoals.models.clipt import CLIPT


class TaskClassificationHead(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_layers: Tuple[int] = (128, 64)
    ):
        """
        Args:
            input_dim: dimension of the input embedding
            output_dim: number of classes to predict
            hidden_layers: tuple of hidden layer sizes
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # input_dim x 128 x 64 x output_dim
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, trajectory_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory_embedding: embedding of the trajectory (batch_size x input_dim)

        Returns:
            logits: logits for each class (batch_size x output_dim)
        """
        task_id_logits = self.network(trajectory_embedding)
        return task_id_logits


class TaskClassifier(pl.LightningModule):
    def __init__(
        self,
        traj_encoder: Union[nn.Module, pl.LightningModule],
        num_tasks: int,
        hidden_layers: Tuple[int] = (128, 64),
    ):
        """
        Parses the traj_encoder, creates the relative classifier

        Args:
            traj_encoder: An instance of a pretrained trajectory encoder
            num_tasks: number of tasks to classify
            hidden_layers: tuple of hidden layer sizes
        """
        self.set_traj_encoder(traj_encoder)
        self.traj_emb_dim = traj_encoder.emb_dim

        self.classifier_head = TaskClassificationHead(
            self.traj_emb_dim, num_tasks, hidden_layers
        )
        self.num_classes = num_tasks

    def set_traj_encoder(self, traj_encoder: Union[nn.Module, pl.LightningModule]):
        """Public function for setting the trajectory encoder"""
        self.traj_encoder = traj_encoder
        # and freeze it
        for param in self.traj_encoder.parameters():
            param.requires_grad = False

    def forward(self, batch, traj_type: str) -> torch.Tensor:
        """
        Either classifies visual or textual trajectory embeddings

        Args:
            batch: batch of data

        Returns:
            logits of shape (batch_size, num_classes)
        """
        if traj_type == "textual":
            textual_inputs: Dict = self.traj_encoder.prepare_textual_inputs(batch)
            traj_emb = self.traj_encoder.encode_text_traj(
                **textual_inputs, normalize=True
            )
        elif traj_type == "visual":
            visual_inputs: Dict = self.traj_encoder.prepare_visual_inputs(batch)
            traj_emb = self.traj_encoder.encode_visual_traj(
                **visual_inputs, normalize=True
            )

        # B x traj_emb -> B x num_classes
        task_id_logits = self.classifier_head(traj_emb)
        return task_id_logits

    def _fit_step(self, batch: Dict[str, torch.Tensor], traj_type: str, phase: str):
        preds = self.forward(batch, traj_type=traj_type)
        targets = batch["task_id"]
        loss = F.cross_entropy(preds, targets)
        accuracy = tmF.classification.accuracy(
            preds, targets, task="multiclass", num_classes=self.num_classes
        )
        self.log(f"{traj_type}/{phase}_loss", loss)
        self.log(f"{traj_type}/{phase}_accuracy", accuracy)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Train only on visual trajectories"""
        loss = self._fit_step(batch, traj_type="visual", phase="train")
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Evaluate on both visual and textual trajectories"""
        self._fit_step(batch, traj_type="visual", phase="val")
        self._fit_step(batch, traj_type="textual", phase="val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Evaluate on both visual and textual trajectories. No loss."""
        targets = batch["task_id"]

        textual_preds = self.forward(batch, traj_type="textual")
        textual_accuracy = tmF.classification.accuracy(
            textual_preds, targets, task="multiclass", num_classes=self.num_classes
        )

        visual_preds = self.forward(batch, traj_type="visual")
        visual_accuracy = tmF.classification.accuracy(
            visual_preds, targets, task="multiclass", num_classes=self.num_classes
        )

        self.log(f"textual/test_accuracy", textual_accuracy)
        self.log(f"visual/test_accuracy", visual_accuracy)

    def configure_optimizers(self):
        params_to_update = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(params_to_update, lr=1e-3)
        return optimizer
