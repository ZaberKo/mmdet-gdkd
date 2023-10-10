from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor

from mmdet.registry import MODELS


from .base import DistillLoss
from .utils import kl_div, weighted_distill_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


@weighted_distill_loss
def knowledge_distillation_dkd_loss(logits_student: Tensor,
                                    logits_teacher: Tensor,
                                    target: LongTensor,
                                    alpha: float,
                                    beta: float,
                                    T: float,
                                    mask_magnitude: float = 1000.0,
                                    detach_teacher: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        Tensor: Loss tensor with shape (N,).
    """
    assert logits_student.size() == logits_teacher.size()

    # TODO: is it necessary in here?
    if detach_teacher:
        logits_teacher = logits_teacher.detach()

    gt_mask = _get_gt_mask(logits_teacher, target)
    other_mask = _get_other_mask(logits_teacher, target)

    soft_logits_student = logits_student / T
    soft_logits_teacher = logits_teacher / T

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)
    p0_student = cat_mask(p_student, gt_mask, other_mask)
    p0_teacher = cat_mask(p_teacher, gt_mask, other_mask)

    log_p0_student = torch.log(p0_student)
    tckd_loss = (
        F.kl_div(log_p0_student, p0_teacher, reduction="none").sum(1)
        * (T**2)
    )

    log_p2_student = F.log_softmax(
        soft_logits_student - mask_magnitude * gt_mask, dim=1
    )
    log_p2_teacher = F.log_softmax(
        soft_logits_teacher - mask_magnitude * gt_mask, dim=1
    )

    nckd_loss = kl_div(log_p2_student, log_p2_teacher, T, kl_type="forward")

    dkd_loss = alpha * tckd_loss + beta * nckd_loss

    train_info = dict(
        loss_tckd=tckd_loss.detach(),
        loss_nckd=nckd_loss.detach()
    )

    return dkd_loss, train_info


@MODELS.register_module()
class KnowledgeDistillationDKDLoss(DistillLoss):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self,
                 alpha: float,
                 beta: float,
                 T: float = 1.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__(reduction=reduction, loss_weight=loss_weight)

        self.alpha = alpha
        self.beta = beta
        self.T = T

    def forward(self,
                logits_student: Tensor,
                logits_teacher: Tensor,
                target: Tensor,
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

        dkd_loss, self.train_info = knowledge_distillation_dkd_loss(
            logits_student,
            logits_teacher,
            target=target.long(), # Discretize target
            alpha=self.alpha,
            beta=self.beta,
            T=self.T,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor
        )

        loss = self.loss_weight * dkd_loss

        return loss
