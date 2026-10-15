"""Staged-training step 6: distills the best frozen float model into a compact student for deployment latency targets.

Status: implemented — provides DistillationTrainer for knowledge distillation from a
frozen teacher model to a compact student. Only used after the best reproducible float
model is frozen and evaluated (Section 8). See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining hard and soft targets.

    L = α · CE(y, student) + (1 - α) · KL(teacher_soft || student_soft)

    Parameters
    ----------
    alpha:
        Weight for the hard-target loss (default 0.5).
    temperature:
        Temperature for softening logits (default 4.0).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 4.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self,
        student_logit: torch.Tensor,
        teacher_logit: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute distillation loss.

        Parameters
        ----------
        student_logit:
            Student model logit ``(B, 1)``.
        teacher_logit:
            Teacher model logit ``(B, 1)`` (detached).
        target:
            Hard target ``(B,)`` or ``(B, 1)``.

        Returns
        -------
        dict[str, torch.Tensor]
            - ``"total"``: combined loss
            - ``"hard_loss"``: BCE with hard targets
            - ``"soft_loss"``: KL with soft teacher targets
        """
        target = target.float().view_as(student_logit)

        # Hard loss
        hard_loss = F.binary_cross_entropy_with_logits(student_logit, target)

        # Soft loss: KL divergence on temperature-scaled sigmoid outputs
        T = self.temperature
        student_soft = torch.sigmoid(student_logit / T)
        teacher_soft = torch.sigmoid(teacher_logit.detach() / T)

        # KL(teacher || student) for binary case
        soft_loss = (
            teacher_soft * torch.log((teacher_soft + 1e-8) / (student_soft + 1e-8))
            + (1 - teacher_soft)
            * torch.log((1 - teacher_soft + 1e-8) / (1 - student_soft + 1e-8))
        ).mean() * (T ** 2)

        total = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return {
            "total": total,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss,
        }


class DistillationTrainer:
    """Knowledge distillation from a frozen teacher to a compact student.

    The teacher must be a frozen, evaluated float model.  Never compare a
    distilled model with a differently trained teacher without stating it
    (Section 8).

    Parameters
    ----------
    teacher:
        Frozen teacher model (eval mode, no gradients).
    student:
        Student model to train.
    alpha:
        Hard target weight (default 0.5).
    temperature:
        Distillation temperature (default 4.0).
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        alpha: float = 0.5,
        temperature: float = 4.0,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.loss_fn = DistillationLoss(alpha=alpha, temperature=temperature)

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        logger.info(
            "Distillation setup: teacher=%d params (frozen), student=%d params",
            sum(p.numel() for p in teacher.parameters()),
            sum(p.numel() for p in student.parameters()),
        )

    @torch.no_grad()
    def get_teacher_logits(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get teacher model logits for a batch.

        Parameters
        ----------
        batch:
            Input batch dict.

        Returns
        -------
        torch.Tensor
            Teacher logits ``(B, 1)``.
        """
        output = self.teacher(
            frames=batch["frames"],
            mel_spec=batch["mel_spec"],
            quality_v=batch.get("quality_v"),
            quality_a=batch.get("quality_a"),
        )
        return output["logit_f"]

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Single training step for distillation.

        Parameters
        ----------
        batch:
            Input batch dict with ``"target"`` key.

        Returns
        -------
        dict[str, torch.Tensor]
            Loss components.
        """
        teacher_logit = self.get_teacher_logits(batch)

        student_output = self.student(
            frames=batch["frames"],
            mel_spec=batch["mel_spec"],
            quality_v=batch.get("quality_v"),
            quality_a=batch.get("quality_a"),
        )

        return self.loss_fn(
            student_logit=student_output["logit_f"],
            teacher_logit=teacher_logit,
            target=batch["target"],
        )
