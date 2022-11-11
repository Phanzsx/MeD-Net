import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio

def iou_score(output, target):

    smooth = 1e-5
    if isinstance(output,list):
        output = output[-1]
    if torch.is_tensor(output):
        output = torch.sigmoid(output)
    if torch.is_tensor(output):
        output = output.view(-1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(-1).data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):

    smooth = 1e-5
    if isinstance(output,list):
        output = output[-1]
    if torch.is_tensor(output):
        output = torch.sigmoid(output)
        output = output.view(-1).data.cpu().numpy()
        target = target.view(-1).data.cpu().numpy()
    output[output>=0.5] = 1
    output[output<0.5] = 0
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)


def pixel_acc(output, target):

    if isinstance(output,list):
        output = output[-1]
    if torch.is_tensor(output):
        output = torch.sigmoid(output)
    if torch.is_tensor(output):
        output = output.view(-1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(-1).data.cpu().numpy()
    if not torch.is_tensor(output):
        output = output.flatten()
        target = target.flatten()
    output[output>=.5] = True
    output[output<0.5] = False
    output = output.astype(np.bool)
    accurecy_ = output == target
    accurecy_ = accurecy_.sum()

    return accurecy_ / len(target)