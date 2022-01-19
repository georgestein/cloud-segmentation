
import numpy as np
import torch
from torch import Tensor

from matplotlib import pyplot as plt
import xarray
import xrspatial.multispectral as ms

# from pytorch_lightning.utilities import rank_zero_only
# @rank_zero_only

def to_xarray(im_arr):
    """Put images in xarray.DataArray format"""

    return xarray.DataArray(im_arr, dims=["y", "x"])

def true_color_img(img, normalized=True):
    """Given the path to the directory of Sentinel-2 chip feature images,
    plots the true color image"""
    
    band_mean_std = {'B02': {'mean': 2848.064112016446,
    'std': 3156.9268464765087,
    'min': 0,
    'max': 27600},
    'B03': {'mean': 2839.0871485290295,
    'std': 2899.280144509762,
    'min': 0,
    'max': 26096},
    'B04': {'mean': 2741.2891076425326,
    'std': 2789.961608891907,
    'min': 0,
    'max': 23104},
    'B08': {'mean': 3657.9092112857143,
    'std': 2424.18942846055,
    'min': 0,
    'max': 19568}}

    if normalized:
        img[2] = img[2]*band_mean_std['B04']['std'] + band_mean_std['B04']['mean']
        img[1] = img[1]*band_mean_std['B03']['std'] + band_mean_std['B03']['mean']
        img[0] = img[0]*band_mean_std['B02']['std'] + band_mean_std['B02']['mean']
        
    red = to_xarray(img[2])
    green = to_xarray(img[1])
    blue = to_xarray(img[0])
    
    return ms.true_color(r=red, g=green, b=blue)

def intersection_over_union(pred, true, smooth=1):
    """
    Calculates intersection over union for an image.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torc.Tensor): a tensor of labels

    Returns:
        intersection (int): total intersection of pixels
        union (int): total union of pixels
    """
    # Intersection and union totals
    pred_flattened = pred.view(-1)
    true_flattened = true.view(-1)

    intersection = torch.logical_and(true_flattened, pred_flattened)
    union = torch.logical_or(true_flattened, pred_flattened)
    
    return (torch.sum(intersection).float() + smooth)/ (torch.sum(union).float() + smooth)

def plot_prediction_grid(
    x: Tensor,
    y: Tensor,
    pred: Tensor,
    chip_id,
    custom_feature_channels = None,
    num_images_plot: int = 4,
    fontsize=18):

        batch_size, c, w, h = x.size()
        
        nimg_plt = min(batch_size, num_images_plot)

        fig, axarr = plt.subplots(nrows=nimg_plt, ncols=3, figsize=(15, 5*nimg_plt))
       
        for img_i in range(nimg_plt):
            
            chip_idi = chip_id[img_i]
            
            if custom_feature_channels is None:
                xi = true_color_img(x[img_i].to("cpu").numpy().astype(np.float32), normalized=True)
                
            if custom_feature_channels == "true_color":
                xi = x[img_i].to("cpu").numpy().astype(np.float32)
                xi = np.transpose(xi, [1, 2, 0]).astype(np.uint8)
            else:
                xi = x[img_i][0].to("cpu").numpy().astype(np.float32)

            yi = y[img_i].to("cpu")
            predi = pred[img_i].to("cpu")
            
            IoU = intersection_over_union(yi, predi)
            
            axarr[img_i, 0].imshow(xi)
            axarr[img_i, 0].set_title(f"{chip_idi}", fontsize=fontsize)
            
            axarr[img_i, 1].imshow(yi, vmin=0., vmax=1.)
            axarr[img_i, 1].set_title("True label", fontsize=fontsize)
            
            axarr[img_i, 2].imshow(predi, vmin=0., vmax=1.)
            axarr[img_i, 2].set_title(f"Pred: IoU={IoU:.3f}", fontsize=fontsize)
            
        plt.close(fig)
        
        return fig
                
