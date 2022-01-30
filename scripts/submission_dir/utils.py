import numpy as np
import rioxarray
import xrspatial.multispectral as ms
import xarray
import rasterio.warp
import pandas as pd
import os
import skimage.transform as st

from typing import Optional, List

BANDS = ["B02", "B03", "B04", "B08"]

# contains various functions for data loading and plotting
def to_xarray(im_arr):
    """Put images in xarray.DataArray format"""

    return xarray.DataArray(im_arr, dims=["y", "x"])

def resize_image(image_in, outsize=[512,512], interpolation_order=1):
    return st.resize(image_in, outsize, order=interpolation_order)

def true_color_img(chip_id: str, data_dir: os.PathLike):
    """Given the path to the directory of Sentinel-2 chip feature images
    and a chip id, plots the true color image of the chip"""
    chip_dir = data_dir / chip_id
    red = rioxarray.open_rasterio(chip_dir / "B04.tif").squeeze()
    green = rioxarray.open_rasterio(chip_dir / "B03.tif").squeeze()
    blue = rioxarray.open_rasterio(chip_dir / "B02.tif").squeeze()

    return ms.true_color(r=red, g=green, b=blue)

def lat_lon_bounds(filepath: os.PathLike):
    """Given the path to a GeoTIFF, returns the image bounds in latitude and
    longitude coordinates.

    Returns points as a tuple of (left, bottom, right, top)
    """
    with rasterio.open(filepath) as im:
        bounds = im.bounds
        meta = im.meta
    # create a converter starting with the current projection
    return rasterio.warp.transform_bounds(
        meta["crs"],
        4326,  # code for the lat-lon coordinate system
        *bounds,
    )

def add_paths(
    df: pd.DataFrame,
    feature_dir: os.PathLike,
    label_dir: Optional[os.PathLike] = None,
    bands: List[str] = BANDS,
):
    """
    Given dataframe with a column for chip_id, returns a dataframe with a column for
    each of the bands provided as "{band}_path", eg "B02_path". Each band column is
    the path to that band saved as a TIF image. If the path to the labels directory
    is provided, a column is also added to the dataframe with paths to the label TIF.
    """
    for band in bands:
        df[f"{band}_path"] = feature_dir / df["chip_id"] / f"{band}.tif"
        # make sure a random sample of paths exist
        #assert df.sample(n=40, random_state=5)[f"{band}_path"].path.exists().all()
    if label_dir is not None:
        df["label_path"] = label_dir / (df["chip_id"] + ".tif")
        # make sure a random sample of paths exist
        #assert df.sample(n=40, random_state=5)["label_path"].path.exists().all()

    return df
