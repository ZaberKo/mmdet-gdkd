from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS


from .base import DistillLoss
from .utils import weighted_distill_loss


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    # [B], and its .mean() is original inter_class_relation
    return 1 - pearson_correlation(y_s, y_t)


@weighted_distill_loss
def knowledge_distillation_distmod_loss(logits_student: Tensor,
                                        logits_teacher: Tensor,
                                        beta: float,
                                        T: float,
                                        detach_teacher: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
    assert logits_student.size() == logits_teacher.size()

    # TODO: is it necessary in here?
    if detach_teacher:
        logits_teacher = logits_teacher.detach()

    y_s = F.softmax(logits_student / T, dim=1)
    y_t = F.softmax(logits_teacher / T, dim=1)
    inter_loss = inter_class_relation(y_s, y_t) * (T**2)

    dist_loss = beta * inter_loss

    train_info = dict(
        loss_inter=inter_loss.detach()
    )

    return dist_loss, train_info


@MODELS.register_module()
class KnowledgeDistillationDISTModLoss(DistillLoss):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self,
                 beta: float,
                 T: float = 1.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__(reduction=reduction, loss_weight=loss_weight)

        self.beta = beta
        self.T = T

    def forward(self,
                logits_student: Tensor,
                logits_teacher: Tensor,
                target: Tensor = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        dist_loss, self.train_info = knowledge_distillation_distmod_loss(
            logits_student,
            logits_teacher,
            beta=self.beta,
            T=self.T,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor
        )

        loss = self.loss_weight * dist_loss

        return loss
