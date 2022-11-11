import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def DICELoss(input, target, weight=None, dimention=2):

    epsilon = 1e-5

    input = torch.sigmoid(input)
    input = input.view(-1)
    target = target.view(-1)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum()
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    if dimention == 2:
        denominator = (input * input).sum() + (target * target).sum()
    else:
        denominator = (input + target).sum()

    return 1. - 2 * (intersect / denominator.clamp(min=epsilon))