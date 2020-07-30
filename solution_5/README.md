1. Softmax

2. NormFace: 对 fully connected layer 做 normalize 


## Result Report

|      SEResNet18    |   LFW  |
|:------------------:|:------:|
|       Softmax      | 96.55% |
|       NormFace     | 97.63% |
|      SpereFace     | 98.50% |
|       CosFace      | 98.88% |
|       ArcFace      | 98.72% |
|   OHEM + NormFace  |
|FocalLoss + NormFace|
|     Contrastive    |
|        Triplet     |
| Contrastive + Finetune|
| Triplet + Finetune |