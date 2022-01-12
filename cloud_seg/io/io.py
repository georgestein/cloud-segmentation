import os
import numpy as np
import pandas as pd
from PIL import Image
from cloud_seg.utils.chip_vis import ImageChip

NPIX = 512
ORIGINAL_BANDS = ["B02", "B03", "B04", "B08"]

def load_pil_as_nparray(filepath, dtype=np.float32):
    """load pil.Image into numpy array"""
    im_arr = np.array(Image.open(filepath)).astype(dtype)
    return im_arr

def load_image(chip_id,
               data_dir, data_dir_new=None,
               bands=None,
               dtype_out=np.float32):
    """Given the path to the directory of Sentinel-2 chip feature images,
    gets desired images"""
    if bands is None:
        bands = ORIGINAL_BANDS

    # chip_image = np.zeros((len(want_bands), npix[0], npix[1]), dtype=np.uint16)
    chip_image = {}

    for i, band in enumerate(bands):
        if band in ORIGINAL_BANDS:
            chip_dir = data_dir / chip_id
        else:
            chip_dir = data_dir_new / chip_id

        chip_image[band] = load_pil_as_nparray(chip_dir / f"{band}.tif", dtype=dtype_out)

    return chip_image

def load_label(chip_id, data_dir, dtype_out=np.int32):
    """Given the path to the directory of Sentinel-2 chip feature labels"""

    chip_label = load_pil_as_nparray(data_dir / f"{chip_id}.tif", dtype=dtype_out)

    return chip_label

def compile_images(df: pd.DataFrame, name: str, data_dir: os.PathLike):
    """Compile images from the competition bands for chips."""
    # TODO: Generalize to include any number of bands.
    bands = ["B02", "B03", "B04", "B08"]
    nfeatures = len(bands)

    compiled_images = np.zeros((len(df), NPIX, NPIX, nfeatures), np.uint16)
    compiled_labels = np.zeros((len(df), NPIX, NPIX), np.uint8)
    for i, chip_id in enumerate(df['chip_id']):
        chip = ImageChip(chip_id)
        compiled_images[i] = chip.image_array
        compiled_labels[i] = chip.labels
    np.save(data_dir/f'compiled_images_{name}', compiled_images)
    np.save(data_dir/f'compiled_labels_{name}', compiled_labels)
