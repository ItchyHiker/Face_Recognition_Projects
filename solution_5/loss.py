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

class FocalCrossEntropyLoss(nn.Module):
    """FocalLoss + CrossEntropyLoss"""
    def __init__(self, gamma, alpha):
        super(FocalCrossEntropyLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred, target):
        batch_size = pred.size(0)
        # F.coss_entropy returns -logpt
        logpt = -F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        losses = -((1-pt)**self.gamma) * logpt

        return losses.sum() / batch_size

class ContrastiveLoss(nn.Module):
    """ContrastiveLoss using pair samples"""
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, outputs, target):
        distance = F.pairwise_distance(outputs[0], outputs[1])
        loss = 0.5*target.float() * torch.pow(distance, 2) + \
                0.5*(1-target.float())*torch.pow(torch.clamp(self.margin-distance, min=0.0), 2)
        
        return loss.mean()

