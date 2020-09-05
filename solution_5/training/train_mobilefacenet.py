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
from training.trainer import Trainer
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
ckpt_tag = 'mobilefacenet'                
model = MobileFaceNet(config.feature_dim)
# model = MobileFaceNetHalf(config.feature_dim)
# summary(model.cuda(), (3, 112, 112))
model = model.to(device)
# model.load_state_dict(torch.load('./pretrained_weights/{}_0011.pth'.format(ckpt_tag), 
#                                                            map_location='cpu'))
# Set margin
if config.margin_type == 'Softmax':
    margin = nn.Linear(config.feature_dim, train_data.num_class)
elif config.margin_type == 'NormFace':
    margin = NormFace(config.feature_dim, train_data.num_class)
elif config.margin_type == 'SphereFace':
    margin = SphereFace(config.feature_dim, train_data.num_class)
elif config.margin_type == 'CosFace':
    margin = CosFace(config.feature_dim, train_data.num_class)
elif config.margin_type == 'ArcFace': 
    margin = ArcFace(config.feature_dim, train_data.num_class)
elif config.margin_type == 'ArcFace2':
    margin = ArcFace(config.feature_dim, train_data.num_class)
else:
    raise NameError("Margin Not Supported!")
margin = margin.to(device)
# margin.load_state_dict(
#        torch.load('./pretrained_weights/{}_512_margin_0011.pth'.format(ckpt_tag), 
#                                                            map_location='cpu'))

# Set optimizer
# optimizer = torch.optim.SGD([{'params':model.parameters()}, 
#     {'params':margin.parameters()}], lr=config.LR, momentum=0.9, nesterov=True)
optimizer = torch.optim.Adam([{'params':model.parameters()}, 
    {'params':margin.parameters()}], lr=config.LR)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=config.STEPS, gamma=0.1)

# Set trainer
trainer = Trainer(config.EPOCHS, dataloaders, model, optimizer, scheduler, device, 
        margin, ckpt_tag)
trainer.train()

