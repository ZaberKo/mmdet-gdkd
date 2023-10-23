# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.losses.utils import weighted_loss
from mmdet.registry import MODELS

from torch.distributions.utils import clamp_probs


@weighted_loss
def distribution_focal_loss_mod(pred, target, *, target_scale, bins=None):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.FloatTensor): Target distance label for bounding boxes with
            shape (N,).
        bins (torch.Tensor): values of label with shape (n+1)

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    num_bins = pred.shape[-1]
    y_label = target_scale * target
    y_left = y_label.long()
    y_right = torch.clamp(y_left+1, max=num_bins-1)
    weight_left = y_right.float() - y_label
    weight_right = y_label - y_left.float()
    loss = F.cross_entropy(pred, y_left, reduction='none')*weight_left \
        + F.cross_entropy(pred, y_right,  reduction='none')*weight_right
    return loss


@weighted_loss
def distribution_focal_loss_mod2(pred, target, *, target_scale, bins, entropy_weights=1e-5):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.FloatTensor): Target distance label for bounding boxes with
            shape (N,).
        bins (torch.Tensor): values of label with shape (n+1)

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """

    dis = torch.abs(bins-target.unsqueeze(1))  # [N, n+1]

    # TODO: efficient impl?
    min_dis, label = dis.min(dim=-1)  # [N]

    p = F.softmax(pred, dim=-1)
    p = clamp_probs(p)
    # p_dist = dists.categorical(pro)

    loss = F.cross_entropy(pred, label, reduce='none') + \
        entropy_weights * torch.log(p)*p

    return loss


@MODELS.register_module()
class DistributionFocalLossMod(nn.Module):
    r"""Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DistributionFocalLossMod, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                target_scale=None,
                bins=None,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert target_scale is not None, "target_scale must be specified"

        loss_cls = self.loss_weight * distribution_focal_loss_mod(
            pred, target,  target_scale=target_scale,  bins=bins,
            weight=weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_cls


@MODELS.register_module()
class DistributionFocalLossMod2(DistributionFocalLossMod):
    def __init__(self, reduction='mean', loss_weight=1.0, entropy_weights=1e-5):
        super(DistributionFocalLossMod2, self).__init__(reduction, loss_weight)

        self.entropy_weights = entropy_weights

    def forward(self,
                pred,
                target,
                target_scale=None,
                bins=None,
                entropy_weights=None,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert bins is not None, "bins must be specified"

        if entropy_weights is None:
            entropy_weights = self.entropy_weights

        loss_cls = self.loss_weight * distribution_focal_loss_mod2(
            pred, target,
            target_scale=target_scale, bins=bins, 
            entropy_weights=entropy_weights,
            weight=weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_cls
