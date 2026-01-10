import torch
from torch import nn

def get_loss():
    return nn.BCEWithLogitsLoss
