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
|   OHEM + NormFace  | 98.00% |
|FocalLoss + NormFace| 98.13% |
|     Contrastive    |
|        Triplet     |
| Contrastive + Finetune|
| Triplet + Finetune |

|      SEResNet34    |   LFW  |
|:------------------:|:------:|
|       Softmax      | 97.00% |
|       NormFace     |        |
|      SpereFace     | 		  |
|       CosFace      | 		  |
|       ArcFace      |        |
|   OHEM + NormFace  |        |
|FocalLoss + NormFace|        |
|     Contrastive    |
|        Triplet     |
| Contrastive + Finetune|
| Triplet + Finetune |

|    MobileFaceNet   |   LFW  |
|:------------------:|:------:|
|       Softmax      |        |
|       NormFace     |        |
|      SpereFace     | 		  |
|       CosFace      | 		  |
|       ArcFace      | 91.37% |
|   OHEM + NormFace  |        |
|FocalLoss + NormFace|        |
|     Contrastive    |
|        Triplet     |
| Contrastive + Finetune|
| Triplet + Finetune |

|    MobileFaceNetHalf   |   LFW  |
|:------------------:|:------:|
|       Softmax      |        |
|       NormFace     |        |
|      SpereFace     | 		  |
|       CosFace      | 		  |
|       ArcFace      | 89.40% |
|   OHEM + NormFace  |        |
|FocalLoss + NormFace|        |
|     Contrastive    |
|        Triplet     |
| Contrastive + Finetune|
| Triplet + Finetune |