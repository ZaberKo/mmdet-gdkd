import functools
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.losses.utils import weight_reduce_loss

def kl_div(log_p, log_q, T, kl_type:str="forward"):
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

    res = res.sum(1) # get instance-wise kl-div: [B,n] -> [B]
    res = res * (T**2)

    return res

def weighted_distill_loss(loss_func: Callable) -> Callable:
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred: Tensor,
                target: Tensor,
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
        loss, train_info = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss, train_info

    return wrapper
