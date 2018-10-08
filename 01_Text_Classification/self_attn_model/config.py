import torch

PATH = '../../data/toxic_comment/'
SAVE_PATH = '../../data/toxic_comment/pretrained_models/toxic.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 32
MIN_FREQ = 2
EMBED = 300
HIDDEN = 600
FC_HIDDEN = 1000
FC_OUTPUT = 6
DA = 300
R = 5
NUM_LAYERS = 1
BIDRECT = True
METHOD = 'general'
LAMBDA = 0.00001
LR = 0.001
SCHSTEP = 5
STEP = 10