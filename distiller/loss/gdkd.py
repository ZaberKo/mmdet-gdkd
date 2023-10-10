from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS


from .base import DistillLoss
from .utils import kl_div, weighted_distill_loss


def get_masks(logits, k=5, strategy="best"):
    if strategy == "best":
        largest_flag = True
    elif strategy == "worst":
        largest_flag = False
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ranks = torch.topk(logits, k, dim=-1,
                       largest=largest_flag,
                       sorted=False).indices

    # topk mask
    mask_u1 = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, ranks, 1)
    # other mask
    mask_u2 = torch.logical_not(mask_u1)

    return mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)  # [B, 2]
    return rt

@weighted_distill_loss
def knowledge_distillation_gdkd_loss(logits_student: Tensor,
                                     logits_teacher: Tensor,
                                     w0: float,
                                     w1: float,
                                     w2: float,
                                     k: float,
                                     T: int,
                                     mask_magnitude: float = 1000.0,
                                     detach_teacher: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
    assert logits_student.size() == logits_teacher.size()

    # TODO: is it necessary in here?
    if detach_teacher:
        logits_teacher = logits_teacher.detach()

    mask_u1, mask_u2 = get_masks(logits_teacher, k)

    soft_logits_student = logits_student / T
    soft_logits_teacher = logits_teacher / T

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # Notation: high_loss: level 0 loss; low_loss: level 1 loss
    # accumulated term
    p0_student = cat_mask(p_student, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u1, mask_u2)

    log_p0_student = torch.log(p0_student)
    high_loss = (
        F.kl_div(log_p0_student, p0_teacher, reduction="none").sum(1)
        * (T**2)
    )

    # topk loss
    log_p1_student = F.log_softmax(
        soft_logits_student - mask_magnitude * mask_u2, dim=1
    )
    log_p1_teacher = F.log_softmax(
        soft_logits_teacher - mask_magnitude * mask_u2, dim=1
    )

    low_top_loss = kl_div(log_p1_student, log_p1_teacher, T, kl_type="forward")

    # other classes loss
    log_p2_student = F.log_softmax(
        soft_logits_student - mask_magnitude * mask_u1, dim=1
    )
    log_p2_teacher = F.log_softmax(
        soft_logits_teacher - mask_magnitude * mask_u1, dim=1
    )

    low_other_loss = kl_div(
        log_p2_student, log_p2_teacher, T, kl_type="forward")

    train_info = dict(
        loss_high=high_loss.detach(),
        loss_low_top=low_top_loss.detach(),
        loss_low_other=low_other_loss.detach()
    )

    gdkd_loss = w0 * high_loss + w1 * low_top_loss + w2 * low_other_loss

    return gdkd_loss, train_info


@MODELS.register_module()
class KnowledgeDistillationGDKDLoss(DistillLoss):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self,
                 w0: float,
                 w1: float,
                 w2: float,
                 k: int = 5,
                 T: float = 1.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__(reduction=reduction, loss_weight=loss_weight)

        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.k = k
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

        gdkd_loss, self.train_info = knowledge_distillation_gdkd_loss(
            logits_student,
            logits_teacher,
            w0=self.w0,
            w1=self.w1,
            w2=self.w2,
            k=self.k,
            T=self.T,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )

        loss = self.loss_weight * gdkd_loss

        return loss
