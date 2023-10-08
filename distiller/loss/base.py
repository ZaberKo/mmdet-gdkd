import torch.nn as nn


class DistillLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.train_info = {}