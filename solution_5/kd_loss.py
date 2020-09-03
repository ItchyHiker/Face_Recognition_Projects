import os, sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTarget(nn.Module):
    """
    Distilling the knowledge in a Neural Network
    https://arxiv.org/abs/1503.02531
    """
    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        """
        Parameters
        ----------
        out_s: tensor, output of student network
        out_t: tensor, output of teacher network
        """

        loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                        F.softmax(out_t / self.T, dim=1),
                        reduction='batchmean') * self.T * self.T
        return loss
    def __str__(self):
        return "SoftTarget"

class Logits(nn.Module):
    """
    Do Deep Nets Really Need to be Deep?
    https://arxiv.org/abs/1312.6184
    """
    def __init__(self):
        super(Logits, self).__init__()

    def forward(self, out_s, out_t):
        loss = F.mse_loss(out_s, out_t)
        return loss
    def __str__(self):
        return "Logits"
