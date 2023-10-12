# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Dict, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS

from .base import DistillLoss
from .utils import kl_div, weighted_distill_loss


@weighted_distill_loss
def knowledge_distillation_kl_div_loss(logits_student: Tensor,
                                       logits_teacher: Tensor,
                                       T: int,
                                       detach_teacher: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (B, N).
        soft_label (Tensor): Target logits with shape (B, N).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        Tensor: Loss tensor with shape (N,).
    """
    assert logits_student.size() == logits_teacher.size()

    # TODO: is it necessary in here?
    if detach_teacher:
        logits_teacher = logits_teacher.detach()

    log_p_student = F.log_softmax(logits_student/T, dim=1)
    log_p_teacher = F.log_softmax(logits_teacher/T, dim=1)

    kd_loss = kl_div(log_p_student, log_p_teacher, T, kl_type="forward")

    train_info = dict(
        loss_kd=kd_loss
    )

    return kd_loss, train_info


@MODELS.register_module()
class KnowledgeDistillationKDLoss(DistillLoss):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self,
                 T: float = 1,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__(reduction=reduction, loss_weight=loss_weight)

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                logits_student: Tensor,
                logits_teacher: Tensor,
                target: Tensor = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        kd_loss, self.train_info = knowledge_distillation_kl_div_loss(
            logits_student,
            logits_teacher,
            T=self.T,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )

        loss = self.loss_weight * kd_loss

        return loss
