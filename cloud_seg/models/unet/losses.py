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

def dice_loss(pred, true, smooth=1e-6):
    """
    pred: prediction logits - so map to probability with sigmoid
    true: true label
    """
    pred_flattened = pred.view(-1)
    true_flattened = true.view(-1)

    intersection = (pred_flattened * true_flattened).sum()
    
    return 1 - ((2. * intersection + dice_smooth) /
              (pred_flattened.sum() + true_flattened.sum() + dice_smooth))

def power_jaccard(pred, true, power_val=1.75, smooth=1.):
    """
    pred: prediction logits - so map to probability with sigmoid
    true: true label
    """
    pred_flattened = pred.view(-1)
    true_flattened = true.view(-1)

    intersection = (pred_flattened * true_flattened).sum()
                                   
    total = (pred_flattened**power_val + true_flattened**power_val).sum()                            
        
    jacc = (intersection + smooth)/(total - intersection + smooth)
                
    return 1 - jacc

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


def bright_land_weight(x):
    return x + 2

def dim_cloud_weight(x):
    return 2 - x
    

class WeightedFocalLoss(torch.nn.Module):
    "Non class weighted version of Focal Loss if gamma=0.5"
    def __init__(self, alpha=.5, gamma=2): 
        # Cloud cover (label==1) is ~66%
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, data, inputs, targets):
        BCE_loss = torch.nn.BCEWithLogitsLoss(reduction="none")(inputs, targets.float())
        
        at = self.alpha[targets]#.data.view(inputs.shape[0], -1)]
        
        pt = torch.exp(-BCE_loss)
        
        brightness_weight = targets * dim_cloud_weight(data[:, 1]) + (1-targets) * bright_land_weight(data[:, 1]) 
            
        # print(BCE_loss.size(), at.size(), pt.size()) 
        # print(at, pt, BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss * brightness_weight
        return F_loss
