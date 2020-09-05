

# margin type
margin_type = 'ArcFace2'
feature_dim = 512


# training config
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 20
STEPS = [4, 10, 18]
USE_CUDA = True

# distilling config
KD_TYPE = 'SoftTarget'
