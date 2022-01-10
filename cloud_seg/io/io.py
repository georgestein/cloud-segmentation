import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_path as path
from pathlib import Path
from PIL import Image


def load_pil_as_nparray(filepath, dtype=np.float32):
    """load pil.Image into numpy array"""
    im_arr = np.array(Image.open(filepath)).astype(dtype)
    return im_arr

def load_image(chip_id,
               data_dir, data_dir_new=None,
               bands=["B02", "B03", "B04", "B08"],
               dtype_out=np.float32):
    """Given the path to the directory of Sentinel-2 chip feature images,
    gets desired images"""
    
    original_bands=["B02", "B03", "B04", "B08"]
    
    # chip_image = np.zeros((len(want_bands), npix[0], npix[1]), dtype=np.uint16)
    chip_image = {}

    for i, band in enumerate(bands):
        if band in original_bands:
            chip_dir = data_dir / chip_id
        else:
            chip_dir = data_dir_new / chip_id

        chip_image[band] = load_pil_as_nparray(chip_dir / f"{band}.tif", dtype=dtype_out)
  
    return chip_image

def load_label(chip_id, data_dir, dtype_out=np.int32):
    """Given the path to the directory of Sentinel-2 chip feature labels"""
        
    chip_label = load_pil_as_nparray(data_dir / f"{chip_id}.tif", dtype=dtype_out)
  
    return chip_label
