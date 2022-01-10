
import numpy as np
import pandas as pd
import rasterio
import torch
from typing import Optional, List
import torchvision

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
        cloudbank: Optional[pd.DataFrame] = None,
        transforms: Optional[list] = None,
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
        """
        self.data  = x_paths
        self.label = y_paths
        self.cloudbank = cloudbank
        if cloudbank is not None:
            self.len_cloudbank = len(cloudbank)
        
        self.transforms = transforms
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
        if self.label is not None:
            label_path = self.label.loc[idx].label_path
            if label_path != 'None':
                with rasterio.open(label_path) as lp:
                    y_arr = lp.read(1).astype("float32")
            else:
                # This is a cloudless image, so sample a random cloud chip from cloudbank
                # load in new cloud label, and add cloud band data to x_arr bands
                idx_cloud = np.random.randint(0, self.len_cloudbank)
                cloud_paths = self.cloudbank.loc[idx_cloud]
                
                # load label
                with rasterio.open(cloud_paths.label_path) as lp:
                    y_arr = lp.read(1).astype("float32")  
                    
                # load cloud bands
                band_arrs = []
                for band in self.bands:
                    with rasterio.open(cloud_paths[f"{band}_path"]) as b:
                        band_arr = b.read(1).astype("float32")
                    band_arrs.append(band_arr)
                    
                # add clouds to cloudless image
                x_arr += np.stack(band_arrs, axis=-1)

        # Prepare dictionary for item
        item = {}
        item["chip_id"] = img.chip_id
        
        # Apply data augmentations, if provided
        if self.label is not None:
            # Apply same data augmentations to the label
            if self.transforms:
                transformed = self.transforms(image=x_arr, mask=y_arr)
                x_arr = transformed["image"]
                y_arr = transformed["mask"]
                
            item["label"] = y_arr
        if self.label is None:
            if self.transforms:
                x_arr = self.transforms(image=x_arr)["image"]
                
        x_arr = np.transpose(x_arr, [2, 0, 1]) # put images in (B, C, H, W)

        item["chip"] = x_arr
        
        return item
