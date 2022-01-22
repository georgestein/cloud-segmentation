
import numpy as np
import pandas as pd
import rasterio
import torch
from typing import Optional, List
import torchvision
from scipy.ndimage import gaussian_filter
import os

import cloud_seg.utils.band_normalizations as band_normalizations

class CloudDataset(torch.utils.data.Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(
        self,
        x_paths: pd.DataFrame,
        bands: List[str],
        y_paths: Optional[pd.DataFrame] = None,
        transforms: Optional[list] = None,
        custom_feature_channels: str = None,
        cloudbank: Optional[pd.DataFrame] = None,
        cloud_transforms: Optional[list] = None,
    ):
        """
        Instantiate the CloudDataset class.

        Args:
            x_paths (pd.DataFrame): a dataframe with a row for each chip. There must be a column for chip_id,
                and a column with the path to the TIF for each of bands
            bands (list[str]): list of the bands included in the data
            y_paths (pd.DataFrame, optional): a dataframe with a row for each chip and columns for chip_id
                and the path to the label TIF with ground truth cloud cover
            cloudbank (pd.DataFrame, optional): a dataframe with a row for each cloud chip, columns for chip_id
                and the path to the cloud band TIFs and label TIF with ground truth cloud cover.
            transforms (list, optional): list of transforms to apply to the feature data (eg augmentations)
            
            custom_feature_channels (str, optional): use difference of channels, ratios, etc, rather than just bands
        """
        self.data  = x_paths
        self.labels = y_paths
        self.cloudbank = cloudbank
        self.cloud_transforms = cloud_transforms
        if cloudbank is not None:
            self.len_cloudbank = len(cloudbank)
            self.sigma_label_smooth = 20
        self.transforms = transforms
        self.custom_feature_channels = custom_feature_channels
        
        if custom_feature_channels == 'true_color':
            # overwrite input bands to use true color bands
            self.bands = ['B04', 'B03', 'B02']
        else:
            self.bands = bands
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Loads an n-channel image from a chip-level dataframe
        img = self.data.loc[idx]
        band_arrs = []
        for band in self.bands:
            with rasterio.open(img[f"{band}_path"]) as b:
                band_arr = b.read(1).astype("float32")
            band_arrs.append(band_arr)
            
        x_arr = np.stack(band_arrs, axis=-1) # images in (B, H, W, C)

        # Load label if available
        if self.labels is not None:
            label_path = self.labels.loc[idx].label_path

            if (label_path != 'none') and (label_path != 'cloudless'):
                with rasterio.open(label_path) as lp:
                    y_arr = lp.read(1).astype("float32")
                    
            if label_path == 'cloudless':
                # This is a cloudless image, so sample a random cloud chip from cloudbank
                # load in new cloud label, and add cloud band data to x_arr bands
                idx_cloud = np.random.randint(0, self.len_cloudbank)
                cloud_paths = self.cloudbank.loc[idx_cloud]
                
                # load label
                with rasterio.open(cloud_paths.label_path) as lp:
                    y_arr = lp.read(1).astype("float32")  
                    
                # load cloud opacity
                opacity_path = os.path.dirname(os.path.abspath(cloud_paths.label_path))
                with rasterio.open(os.path.join(opacity_path, "opacity.tif")) as lp:
                    opacity_arr = lp.read(1).astype("float32")  
                                       
                # get smoothed version of label to smooth edges between new clouds and original chip
                
                y_arr_wide = gaussian_filter(y_arr, sigma=self.sigma_label_smooth)
                y_arr_wide = ((y_arr_wide > 0.05)*1).astype("float32")

                y_arr_smooth = gaussian_filter(y_arr_wide, sigma=self.sigma_label_smooth)
    
                # load cloud bands
                band_arrs = []
                for band in self.bands:
                    with rasterio.open(cloud_paths[f"{band}_path"]) as b:
                        band_arr = b.read(1).astype("float32")
                    band_arrs.append(band_arr)
                    
                x_arr_clouds = np.stack(band_arrs, axis=-1)
                
                # Apply special augmentations to clouds and cloud labels
                if self.cloud_transforms is not None:
                    transformed = self.cloud_transforms(image=x_arr_clouds, mask=y_arr)
                    x_arr_clouds = transformed["image"]
                    y_arr = transformed["mask"]
                
                # add clouds to cloudless image, differently where opacity==1 and where opacity==0
                x_arr = ( (x_arr_clouds * opacity_arr[..., None])
                         +  (x_arr + x_arr_clouds * y_arr_smooth[..., None]) * (1-opacity_arr[..., None]))

        if self.custom_feature_channels is not None:
            # modify x_arr (N,H,W,C) from band data to custom designed features
            if self.custom_feature_channels == 'true_color':
                for iband in range(len(self.bands)):
                    x_arr[..., iband] = band_normalizations.true_color_band(x_arr[..., iband])
                                
            if self.custom_feature_channels == 'log_bands':
                # for ichan in range(x_arr.shape[-1]):
                #     x_arr[..., ichan] = np.log(x_arr
                x_arr = np.clip(x_arr, 1., np.inf)
                x_arr = np.log(x_arr)
                
            if self.custom_feature_channels == 'ratios':
                # Intensity, B03-B08, B02-B04, B08-B04,
                x_arr = np.clip(x_arr, 1., np.inf)

                B02 = x_arr[..., 0].copy()
                B04 = x_arr[..., 2].copy()

                x_arr[..., 0] = np.mean(x_arr, axis=-1)/2000.
                x_arr[..., 1] = (x_arr[..., 1] - x_arr[..., -1])/(x_arr[..., 1] + x_arr[..., -1])
                x_arr[..., 2] = (B02 - x_arr[..., 2])/(B02 + x_arr[..., 2])
                x_arr[..., 3] = (x_arr[..., -1] - B04)/(x_arr[..., -1] + B04)
                
        # Prepare dictionary for item
        item = {}
        item["chip_id"] = img.chip_id

        # Apply data augmentations, if provided
        if self.labels is not None:
            # Apply same data augmentations to the label
            if self.transforms:
                transformed = self.transforms(image=x_arr, mask=y_arr)
                x_arr = transformed["image"]
                y_arr = transformed["mask"]
                
            item["label"] = y_arr
            
        if self.labels is None:
            if self.transforms:
                x_arr = self.transforms(image=x_arr)["image"]
                
        x_arr = np.transpose(x_arr, [2, 0, 1]) # put images in (B, C, H, W)

        item["chip"] = x_arr
        
        return item
