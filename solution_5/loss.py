import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class OHEMCrossEntropyLoss(nn.Module):
    """Online hard example mining with CrossEntropyLoss"""
    def __init__(self, ratio):
        super(OHEMCrossEntropyLoss, self).__init__()
        self.ratio = ratio
    
    def forward(self, pred, target):
        batch_size = pred.size(0)
        # print(pred.size())
        # print(target.size())
        losses = F.cross_entropy(pred, target, reduction='none')
        # print(losses)
        # print(losses.shape) 
        sorted_losses, idx = torch.sort(losses, descending=True)
        # print(sorted_losses.size())
        keep_num = min(sorted_losses.size()[0], int(batch_size*self.ratio))
        keep_idx = idx[:keep_num]
        keep_losses = losses[keep_idx]
        return keep_losses.sum() / keep_num

