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

from tools.datasets import CASIAWebFace, LFW
from nets.se_resnet import ResNet18
from training.trainer import Trainer
import config

# Set training device
use_cuda = config.USE_CUDA and torch.cuda.is_available()
device = torch.device('cuda:1' if use_cuda else 'cpu')
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
lfw_data = LFW('./training_data/LFW', 'annotations/lfw_pairs.txt', transforms)
val_data = None
dataloaders = {'train': DataLoader(train_data, batch_size=config.BATCH_SIZE,
                                    shuffle=True, **kwargs),
                'LFW': DataLoader(lfw_data, batch_size=config.BATCH_SIZE,
                                    shuffle=False, **kwargs),}

ckpt_tag = 'se_resnet18'                
# Set model
model = ResNet18()
summary(model.cuda(), (3, 112, 112))
model = model.to(device)
model.load_state_dict(torch.load('./pretrained_weights/se_resnet18_0015.pth', 
                                                            map_location='cpu'))
# Set margin
if config.margin_type == 'Softmax':
    margin = nn.Linear(config.feature_dim, train_data.num_class)
else:
    raise NameError("Margin Not Supported!")
margin = margin.to(device)
margin.load_state_dict(torch.load('./pretrained_weights/se_resnet18_512_margin_0015.pth', 
                                                            map_location='cpu'))


# Set optimizer
# optimizer = torch.optim.SGD([{'params':model.parameters()}, 
#    {'params':margin.parameters()}], lr=config.LR, momentum=0.9, nesterov=True)
optimizer = torch.optim.Adam([{'params':model.parameters()}, 
    {'params':margin.parameters()}], lr=config.LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=config.STEPS, gamma=0.1)

# Set trainer
trainer = Trainer(config.EPOCHS, dataloaders, model, optimizer, scheduler, device, 
        margin, ckpt_tag)
trainer.train()


        
