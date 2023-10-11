import functools
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.losses.utils import weight_reduce_loss


def kl_div(log_p, log_q, T, kl_type: str = "forward"):
    """
        get element-wise KL-Div:
        Args:
            log_p: prob of predict
            log_q: prob of target
            T: temperature
            kl_type: support "forward", "reverse" and "both"
        Return:
            res: instance(batch)-wise kl-div
    """
    if kl_type == "forward":
        res = F.kl_div(log_p, log_q, reduction="none",
                       log_target=True)
    elif kl_type == "reverse":
        res = F.kl_div(log_q, log_p, reduction="none",
                       log_target=True)
    elif kl_type == "both":
        res = (
            F.kl_div(log_p, log_q, reduction="none", log_target=True) +
            F.kl_div(log_q, log_p, reduction="none", log_target=True)
        ) * 0.5
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")

    res = res.sum(1)  # get instance-wise kl-div: [B,n] -> [B]
    res = res * (T**2)

    return res


def weighted_distill_loss(loss_func: Callable) -> Callable:
    @functools.wraps(loss_func)
    def wrapper(pred_student: Tensor,
                pred_teacher: Tensor,
                weight: Optional[Tensor] = None,
                reduction: str = 'mean',
                avg_factor: Optional[int] = None,
                **kwargs) -> Tensor:
        """
        Args:
            pred (Tensor): The prediction.
            target (Tensor): Target bboxes.
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            reduction (str, optional): Options are "none", "mean" and "sum".
                Defaults to 'mean'.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.

        Returns:
            Tensor: Loss tensor.
        """
        # get element-wise loss
        loss, train_info = loss_func(pred_student, pred_teacher, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        for k in train_info.keys():
            if isinstance(train_info[k], Tensor) and train_info[k].shape == weight.shape:
                train_info[k] = weight_reduce_loss(
                    train_info[k], weight, reduction, avg_factor)

        return loss, train_info

    return wrapper


def reduce_loss(loss, avg_factor: Optional[int] = None):
    """
        no-weight version of weight_reduce_loss
    """
    if loss.dim() > 0:
        loss = loss.mean()

    if avg_factor is not None:
        eps = torch.finfo(torch.float32).eps
        loss = loss / (avg_factor + eps)

    return loss


def noweighted_distill_loss(loss_func: Callable) -> Callable:
    """
        add a dummy wrapper with the same API.
        reduction is done inside of loss_func
    """

    @functools.wraps(loss_func)
    def wrapper(pred_student: Tensor,
                pred_teacher: Tensor,
                weight: Optional[Tensor] = None,
                reduction: str = 'mean',
                avg_factor: Optional[int] = None,
                **kwargs) -> Tensor:
        """
        Args:
            pred (Tensor): The prediction.
            target (Tensor): Target bboxes.
            weight (Optional[Tensor], optional): Ignored
            reduction (str, optional): Ignored
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.

        Returns:
            Tensor: Loss tensor.
        """

        loss, train_info = loss_func(pred_student, pred_teacher, **kwargs)
        assert loss.dim() == 0, "loss_func should return a scalar"

        loss = reduce_loss(loss, avg_factor)
        for k in train_info.keys():
            if isinstance(train_info[k], Tensor):
                assert train_info[k].dim(
                ) == 0, "loss_func should return a scalar"
                
                train_info[k] = reduce_loss(train_info[k], avg_factor)

        return loss, train_info

    return wrapper
