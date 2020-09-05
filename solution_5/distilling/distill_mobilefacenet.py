import os, sys
sys.path.append('.')
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torchsummary import summary

import config
from tools.datasets import CASIAWebFace, LFW
from nets.mobilefacenet import MobileFaceNet, MobileFaceNetHalf
from distilling.distiller import Distiller
from margin import NormFace, SphereFace, CosFace, ArcFace, ArcFace2

# Set training device
use_cuda = config.USE_CUDA and torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.backends.cudnn.benchmark = True
print(device)

# Set dataloader
transforms = tfs.Compose([
                tfs.ToTensor(),
                tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
train_data = CASIAWebFace('./training_data/CASIA', 'annotations/CASIA_anno.txt',
        transforms)
lfw_data = LFW('./LFW/lfw_align_112', './LFW/pairs.txt', transforms)
dataloaders = {'train': DataLoader(train_data, batch_size=config.BATCH_SIZE,
                                    shuffle=True, **kwargs),
                'LFW': DataLoader(lfw_data, batch_size=config.BATCH_SIZE,
                                    shuffle=False, **kwargs),}

# Set model
ckpt_tag = 'mobilefacenet_kd_logits'                
teacher = MobileFaceNet(config.feature_dim)
student = MobileFaceNetHalf(config.feature_dim)
teacher = teacher.to(device)
student = student.to(device)
teacher.load_state_dict(torch.load('./checkpoints/mobilefacenet_ArcFace_best.pth', 
                    map_location='cpu'))

# Set margin
if config.margin_type == 'Softmax':
    t_margin = nn.Linear(config.feature_dim, train_data.num_class)
    s_margin = nn.Linear(config.feature_dim, train_data.num_class)
elif config.margin_type == 'NormFace':
    margin = NormFace(config.feature_dim, train_data.num_class)
elif config.margin_type == 'SphereFace':
    margin = SphereFace(config.feature_dim, train_data.num_class)
elif config.margin_type == 'CosFace':
    margin = CosFace(config.feature_dim, train_data.num_class)
elif config.margin_type == 'ArcFace': 
    margin = ArcFace(config.feature_dim, train_data.num_class)
elif config.margin_type == 'ArcFace2':
    t_margin = ArcFace(config.feature_dim, train_data.num_class)
    s_margin = ArcFace(config.feature_dim, train_data.num_class)
else:
    raise NameError("Margin Not Supported!")
t_margin = t_margin.to(device)
t_margin.load_state_dict(
        torch.load('./checkpoints/mobilefacenet_512_ArcFace_best.pth', 
            map_location='cpu'))
s_margin = s_margin.to(device)

# Set optimizer
optimizer = torch.optim.SGD([{'params':student.parameters()}, 
    {'params':s_margin.parameters()}], lr=config.LR, momentum=0.9, nesterov=True)
# optimizer = torch.optim.Adam([{'params':student.parameters()}, 
#    {'params':s_margin.parameters()}], lr=config.LR)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=config.STEPS, gamma=0.1)

# Set trainer
distiller = Distiller(config.EPOCHS, dataloaders, teacher, t_margin, student, 
        s_margin, optimizer, scheduler, device, ckpt_tag, config.KD_TYPE)
distiller.distill()

