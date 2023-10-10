import torch.nn as nn
from mmengine import MessageHub

class DistillLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.train_info = {} # original loss without loss*self.loss_weight
    
    def get_train_info(self):
        return self.train_info