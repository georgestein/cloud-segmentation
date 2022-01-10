import numpy as np
import torch

from typing import Sequence, Optional, Union

def intersection_and_union(pred, true):
    """
    Calculates intersection and union for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torc.Tensor): a tensor of labels

    Returns:
        intersection (int): total intersection of pixels
        union (int): total union of pixels
    """
    # valid_pixel_mask = true.ne(255)  # valid pixel mask
    # true = true.masked_select(valid_pixel_mask).to("cpu")
    # pred = pred.masked_select(valid_pixel_mask).to("cpu")

    # Intersection and union totals
    pred_flattened = pred.view(-1)
    true_flattened = true.view(-1)

    intersection = torch.logical_and(true_flattened, pred_flattened)
    union = torch.logical_or(true_flattened, pred_flattened)
    
    return torch.sum(intersection).float(), torch.sum(union).float()#, torch.sum(intersection) / torch.sum(union)

def dice_loss(pred, true, dice_smooth=1.):
    
    pred_flattened = pred.view(-1)
    true_flattened = true.view(-1)

    intersection = (pred_flattened * true_flattened).sum()
    
    return 1 - ((2. * intersection + dice_smooth) /
              (pred_flattened.sum() + true_flattened.sum() + dice_smooth))

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
                
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
