
import numpy as np
import pandas as pd
# import rasterio
import torch
from typing import Optional, List
import torchvision
from scipy.ndimage import gaussian_filter
from PIL import Image
import os

try:
    import cloud_seg.utils.band_normalizations as band_normalizations
except ImportError:
    import band_normalizations
    
def get_array(filepath):
    """Put images in xarray.DataArray format"""
    im_arr = np.array(Image.open(filepath)).astype(np.float32)
    return im_arr

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
        scale_feature_channels: str = None,
        custom_features: str = None,
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
        self.scale_feature_channels = scale_feature_channels
        self.custom_features = custom_features
        
        self.bands = bands
        self.band_to_ind = {k: v for v, k in enumerate(bands)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Loads an n-channel image from a chip-level dataframe
        """
        
        img = self.data.loc[idx]
        
        item = {} # Prepare dictionary for item
        item["chip_id"] = img.chip_id

        band_arrs = []
        for band in self.bands:
            # with rasterio.open(img[f"{band}_path"]) as b:
            #     band_arr = b.read(1).astype("float32")
            band_arr = get_array(img[f"{band}_path"])    
            band_arrs.append(band_arr)
            
        x_arr = np.stack(band_arrs, axis=-1) # images in (B, H, W, C)

        # Load label if available
        if self.labels is not None:
            label_path = self.labels.loc[idx].label_path

            if (label_path != 'none') and (label_path != 'cloudless'):
                # with rasterio.open(label_path) as lp:
                #     y_arr = lp.read(1).astype("float32")
                y_arr = get_array(label_path)    

            if label_path == 'cloudless':
                # This is a cloudless image, so sample a random cloud chip from cloudbank
                # load in new cloud label, and add cloud band data to x_arr bands
                idx_cloud = np.random.randint(0, self.len_cloudbank)
                cloud_paths = self.cloudbank.loc[idx_cloud]
                
                # item["chip_id_cloud"] = cloud_paths.chip_id

                # load label
                # with rasterio.open(cloud_paths.label_path) as lp:
                #     y_arr = lp.read(1).astype("float32")  
                y_arr = get_array(cloud_paths.label_path)
                
                # load cloud opacity
                opacity_path = os.path.dirname(os.path.abspath(cloud_paths.label_path))
                opacity_path = os.path.join(opacity_path, "opacity.tif")
                # with rasterio.open(opacity_path) as lp:
                #     opacity_arr = lp.read(1).astype("float32")  
                opacity_arr = get_array(opacity_path)

                # load cloud bands
                band_arrs = []
                for band in self.bands:
                    # with rasterio.open(cloud_paths[f"{band}_path"]) as b:
                    #     band_arr = b.read(1).astype("float32")
                    band_arr = get_array(cloud_paths[f"{band}_path"])
                    band_arrs.append(band_arr)
                    
                x_arr_clouds = np.stack(band_arrs, axis=-1)
                
                # Apply special augmentations to clouds and cloud labels
                if self.transforms:
                    x_arr = self.transforms(image=x_arr)["image"]
                
                if self.cloud_transforms is not None:
                    
                    # want to transform both y_arr and opacity_arr together
                    y_and_opacity =  np.stack([y_arr, opacity_arr], axis=-1)
                
                    transformed = self.cloud_transforms(image=x_arr_clouds, mask=y_and_opacity)
                    x_arr_clouds = transformed["image"]
                    y_arr = transformed["mask"][..., 0]
                    opacity_arr = transformed["mask"][..., 1]

                # get smoothed version of label to smooth edges between new clouds and original chip
                y_arr_wide = gaussian_filter(y_arr, sigma=self.sigma_label_smooth)
                y_arr_wide = ((y_arr_wide > 0.05)*1).astype("float32")

                y_arr_smooth = gaussian_filter(y_arr_wide, sigma=self.sigma_label_smooth)
    
                # add clouds to cloudless image, differently where opacity==1 and where opacity==0
                x_arr = ( (x_arr_clouds * opacity_arr[..., None])
                         +  (x_arr + x_arr_clouds * y_arr_smooth[..., None]) * (1-opacity_arr[..., None]))
                x_arr = np.clip(x_arr, 1, np.inf)
                
                # item['opacity'] = opacity_arr

        # Apply data augmentations, if provided
        if self.labels is not None:
            # Apply same data augmentations to the label
            if self.transforms and label_path != 'cloudless':
                transformed = self.transforms(image=x_arr, mask=y_arr)
                x_arr = transformed["image"]
                y_arr = transformed["mask"]
                
            item["label"] = y_arr
            
        if self.labels is None:
            if self.transforms:
                x_arr = self.transforms(image=x_arr)["image"]
                
        if self.scale_feature_channels is not None:
            # modify x_arr (N,H,W,C) from band data to custom designed features
            if self.scale_feature_channels == 'true_color':
                for iband in range(len(self.bands)):
                    x_arr[..., iband] = band_normalizations.true_color_band(x_arr[..., iband])
                    
            if self.scale_feature_channels == 'feder_scale':
                for iband in range(len(self.bands)):
                    x_arr[..., iband] = band_normalizations.feder_scale(
                        x_arr[..., iband],
                    )
                                                          
            if self.scale_feature_channels == 'log_bands':
                # for ichan in range(x_arr.shape[-1]):
                #     x_arr[..., ichan] = np.log(x_arr
                x_arr = np.clip(x_arr, 1., np.inf)
                x_arr = np.log(x_arr)
                
            if self.scale_feature_channels == 'custom':
                x_arr_out = np.zeros((x_arr.shape[0], x_arr.shape[1], len(self.custom_features)), dtype=x_arr.dtype)
                for ind, feature in enumerate(self.custom_features):
                    if feature == 'luminosity':
                        bi = "B02"
                        ind_bi = self.band_to_ind[bi]
                        # x_arr_out[..., ind] = band_normalizations.feder_scale(np.mean(x_arr, axis=-1))
                        x_arr_out[..., ind] = band_normalizations.feder_scale(x_arr[..., ind_bi])

                    elif '-' in feature:
                        bi, bj = feature.split('-')
                        ind_bi = self.band_to_ind[bi]
                        ind_bj = self.band_to_ind[bj]
                        x_arr_out[..., ind] = (x_arr[..., ind_bi] - x_arr[..., ind_bj])/np.clip((x_arr[..., ind_bi] + x_arr[..., ind_bj]), 1, np.inf)
                    elif '/' in feature:
                        bi, bj = feature.split('/')
                        ind_bi = self.band_to_ind[bi]
                        ind_bj = self.band_to_ind[bj]
                        x_arr_out[..., ind] = x_arr[..., ind_bi]/np.clip(x_arr[..., ind_bj], 1, np.inf)
                    else:
                        bi = feature
                        ind_bi = self.band_to_ind[bi]
                        x_arr_out[..., ind] = x_arr[..., ind_bi].copy()
                                                                                 
                x_arr = x_arr_out
                

        x_arr = np.transpose(x_arr, [2, 0, 1]) # put images in (B, C, H, W)

        item["chip"] = x_arr
        
        return item
