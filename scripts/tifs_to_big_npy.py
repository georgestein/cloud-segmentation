"""
Loop over chunks of size <chunksize> and save individual .tifs in various folders to
single large numpy arrays.

In order to facilitate faster data/label/prediction investigation
"""
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_path as path
from pathlib import Path
from PIL import Image
import torch
import pytorch_lightning as pl

import xarray
import xrspatial.multispectral as ms
import rasterio
import pyproj
import rasterio.warp

import math
import time
import argparse
import os 

import multiprocessing

from cloud_seg.utils import chip_vis, utils
from cloud_seg.io import io

DATA_DIR = Path.cwd().parent.resolve() / "data/"
DATA_DIR_CLOUDS = DATA_DIR / 'clouds/'
DATA_DIR_CLOUDLESS = DATA_DIR / 'cloudless/'
DATA_DIR_CLOUDLESS_MOST_SIMILAR = DATA_DIR / 'cloudless_most_similar/'
DATA_DIR_CLOUDLESS_TIF = DATA_DIR / 'cloudless_tif/'

# DATA_DIR_OUT = DATA_DIR / "big_numpy_arrays/"
DATA_DIR_OUT = DATA_DIR / "big_numpy_arrays/nchips_100/" #sorted/"

# PREDICTION_DIR = Path.cwd().parent.resolve() / "trained_models/unet/4band_originaldata_resnet18_bce_vfrc_customfeats_None_2022-01-17/predictions/"
# PREDICTION_DIR = Path.cwd().parent.resolve() / "trained_models/unet/4band_originaldata_cloudaugment_resnet18_bce_vfrc_customfeats_None_2022-01-24/predictions/"
PREDICTION_DIR = Path.cwd().parent.resolve() / "../trained_models/submission_dir/predictions"

TRAIN_FEATURES = DATA_DIR / "train_features"
TRAIN_FEATURES_NEW = DATA_DIR / "train_features_new"

TRAIN_LABELS = DATA_DIR / "train_labels"

BANDS = ["B02", "B03", "B04", "B08"]
BANDS_NEW = []
# BANDS_NEW = ["B01", "B11"]

assert TRAIN_FEATURES.exists(), TRAIN_LABELS.exists()

parser = argparse.ArgumentParser(description='runtime parameters')
parser.add_argument("--bands", nargs='+' , default=["B02", "B03", "B04", "B08"],
                    help="bands desired")

parser.add_argument("--bands_new", nargs='+', default=None,
                    help="additional bands to use beyond original four")

parser.add_argument("--sort_by_bad", action='store_true',
                    help="Sort by bad predictions")

parser.add_argument("--chunksize", type=int, default=100,
                    help="Chunksize for output arrays") 

parser.add_argument("--max_pool_size", type=int, default=64,
                    help="Chunksize for output arrays") 

parser.add_argument("--add_predictions", action='store_true',
                    help="Add unet predictions") 
                                               
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")

params = vars(parser.parse_args())
params['bands_use'] = sorted(params['bands'] + params['bands_new']) if params['bands_new'] is not None else params['bands']

params['outsize'] = [512, 512]

if params['verbose']: print("Parameters are: ", params)
    
df_meta = pd.read_csv(DATA_DIR / "train_metadata.csv")

# Shuffle 
# df_meta = df_meta.sample(frac=1, random_state=42).reset_index(drop=True)
    
# how many different chip ids, locations, and datetimes are there?
print(df_meta[["chip_id", "location", "datetime"]].nunique())

df_meta.head()

df_meta = utils.add_paths(df_meta, TRAIN_FEATURES, TRAIN_LABELS, bands=params['bands'])

if params['bands_new'] is not None:

    # ensure that data exists for any desired new bands beyond the 4 originally provided
    for iband, band in enumerate(params['bands_new']):
        band_has_data = (TRAIN_FEATURES_NEW / df_meta["chip_id"] / f"{band}.tif").map(os.path.isfile)
        if iband==0: 
            has_banddata_on_disk = band_has_data
        else:
            has_banddata_on_disk = band_has_data & has_banddata_on_disk
        
        if np.sum(band_has_data) == 0:
            print(f"Band {band} has no data")
    print('Fraction of chips that have new bands on disk = ', has_banddata_on_disk.sum()/has_banddata_on_disk.shape[0])

    # Keep only files that have new bands on disk
    df_meta = df_meta[has_banddata_on_disk]

    df_meta = utils.add_paths(df_meta, TRAIN_FEATURES_NEW, bands=params['bands_new'])
        
# Total number of chunks. Set as global variable
params['nchunks'] = math.ceil(len(df_meta)/params['chunksize'])
params['max_pool_size'] = min(params['nchunks'], params['max_pool_size'])

if params['sort_by_bad']:
    # Sort by bad predictions:
    worst_chip_ids = np.loadtxt("../data/BAD_CHIP_DATA/worst_preds_gbm_chip_ids.txt", dtype=str)
    worst_int_union = np.loadtxt("../data/BAD_CHIP_DATA/worst_preds_gbm_int-union.txt", dtype=float)

    print(len(worst_chip_ids), len(df_meta))
    print(df_meta.head(), worst_chip_ids[:5])
    df_meta['pred_score'] = worst_int_union

    df_meta = df_meta.sort_values(by='pred_score', ascending=False)
    
print(df_meta.head())

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

    # Intersection and union totals                                                                             
    pred_flattened = pred.flatten()
    true_flattened = true.flatten()

    intersection = np.logical_and(true_flattened, pred_flattened)/pred_flattened.shape[0]
    union = np.logical_or(true_flattened, pred_flattened)/pred_flattened.shape[0]

    return float(np.sum(intersection)), float(np.sum(union))




def lat_lon_bounds(filepath: os.PathLike):
    """Given the path to a GeoTIFF, returns the image bounds in latitude and
    longitude coordinates.

    Returns points as a tuple of (left, bottom, right, top)
    """
    with rasterio.open(filepath) as im:
        bounds = im.bounds
        meta = im.meta
    # create a converter starting with the current projection
    
    left, bottom, right, top = rasterio.warp.transform_bounds(
        meta["crs"],
        4326,  # code for the lat-lon coordinate system
        *bounds,
    )
    
    lon = (right+left)/2
    dlon = abs(right-left)
    
    lat = (top+bottom)/2
    dlat = abs(top-bottom)
     
    return lat, lon, dlat, dlon

# def lat_long_bounds(chip_path):
#     """Given the path to a GeoTIFF, returns the image bounds in latitude and
#     longitude coordinates.

#     Returns points as a tuple of (left, bottom, right, top)
#     """

#     with rasterio.open(chip_path) as chip:

#         # create a converter starting with the current projection
#         current_crs = pyproj.CRS(chip.meta["crs"])
#         crs_transform = pyproj.Transformer.from_crs(current_crs, current_crs.geodetic_crs)

#         # returns left, bottom, right, top
#         left, bottom, right, top = crs_transform.transform_bounds(*chip.bounds)
        
#     lon = (right+left)/2
#     dlon = abs(right-left)
    
#     lat = (top+bottom)/2
#     dlat = abs(top-bottom)
    
#     return lat, lon, dlat, dlon

def load_image_to_array(chip_id, bands=["B02", "B03", "B04", "B08"],
               data_dir=TRAIN_FEATURES, data_dir_new=TRAIN_FEATURES_NEW):
    """Given the path to the directory of Sentinel-2 chip feature images,
    plots the true color image"""
    
    original_bands=["B02", "B03", "B04", "B08"]
    
    npixx = 512
    npixy = 512
    
    # chip_image = np.zeros((len(want_bands), npix[0], npix[1]), dtype=np.uint16)
    chip_image = {}

    image_array = np.zeros((len(bands), npixx, npixy), dtype=np.float32)
    for i, band in enumerate(bands):
        if band in original_bands:
            chip_dir = data_dir / chip_id
        else:
            chip_dir = data_dir_new / chip_id

        image_array[i] = np.array(io.load_pil_as_nparray(chip_dir / f"{band}.tif")).astype(np.float32)
  
    return image_array


# def tif_to_numpy(random_state, nplt=1, figsize=None):
    
def get_chips_in_npy(ichip_start, ichip_end, bands=["B02", "B03", "B04", "B08"]):

    npixx = 512
    npixy = 512
    nchips = ichip_end-ichip_start
    images = np.zeros((nchips, len(bands), npixx, npixy), dtype=np.uint16)
    labels = np.zeros((nchips, npixx, npixy), dtype=np.uint8)
    
    labels_mean = np.zeros(nchips, dtype=np.float32)
    
    if params['add_predictions']:
        preds = np.zeros((nchips, npixx, npixy), dtype=np.uint8)
        preds_mean  = np.zeros(nchips, dtype=np.float32)
        intersection = np.zeros(nchips, dtype=np.float32)
        union = np.zeros(nchips, dtype=np.float32)

    chip_ids    = []
    chip_lat   = []
    chip_lon = []
    chip_dlat  = []
    chip_dlon    = []
    
    for ichip, ichip_meta_loc in enumerate(range(ichip_start, ichip_end)):

        chip = df_meta.iloc[ichip_meta_loc]
        
        chip_ids.append(chip.chip_id)
        
        # get lat lon
        lat, lon, dlat, dlon = lat_lon_bounds(chip.B04_path)
        chip_lat.append(lat)
        chip_lon.append(lon)
        chip_dlat.append(dlat)
        chip_dlon.append(dlon)
        
        images[ichip] = load_image_to_array(chip.chip_id, bands=bands)  
        labels[ichip] = np.array(Image.open(chip.label_path))
        labels_mean[ichip] = np.mean(labels[ichip])

        if params['add_predictions']:

            preds_i = np.array(Image.open(PREDICTION_DIR/f"{chip.chip_id}.tif"))
            preds_i = (preds_i > 0.5)*1
            preds[ichip] = preds_i.astype(np.int8)

            intersection[ichip], union[ichip] = intersection_and_union(preds[ichip], labels[ichip])
            preds_mean[ichip] = np.mean(preds[ichip])

    d = {}
    
    d['bands_use'] = params['bands_use']
    d['chip_ids'] = chip_ids
    d['lat'] =  chip_lat 
    d['lon'] =  chip_lon
    d['dlat'] = chip_dlat 
    d['dlon'] = chip_dlon

    d['images'] = images
    d['labels'] = labels
    d['labels_mean'] = labels_mean
       
    # Save bands seperately as well
    for iband, band in enumerate(params['bands_use']):
        d[f"{band}"] = images[:, iband]

    if params['add_predictions']:

        d['preds'] = preds
        d['preds_mean'] = preds_mean

        d['intersection'] = intersection
        d['union'] = union

        d['IoU'] = intersection/union

    return d

def run_on_chunk(ichunk):
    
    tstart = time.time()
    ichip_start = ichunk*params['chunksize']
    ichip_end = min(len(df_meta), (ichunk+1)*params['chunksize'])

    print(f"\nRunning on chunk {ichunk} out of {params['nchunks']}. Index start:{ichip_start}, index end: {ichip_end}")

    data = get_chips_in_npy(ichip_start, ichip_end, bands=params['bands_use'])

    print("data, label shape: ", data['images'].shape, data['labels'].shape)

    for k, v in data.items():
        np.save(DATA_DIR_OUT / f"{k}_{ichip_start:06d}_{ichip_end:06d}.npy", v)

    print("Time elapsed = ", time.time()-tstart)
    
def main():
    """
    Loop over chunks of size <chunksize> and save individual .tifs in various folders to
    single large numpy arrays.
    
    In order to facilitate faster data/label/prediction investigation
    """
    
    if params['max_pool_size'] <= 1:
        for i in range(params['nchunks']):
            run_on_chunk(i)
            
    else:
        # Simple threading with pool and .map
        cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpus if cpus < params['max_pool_size'] else params['max_pool_size'])
        print(f"Number of available cpus = {cpus}")

        pool.map(run_on_chunk, range(params['nchunks']))

        pool.close()
        pool.join()
    # for ichunk in range(params['nchunks']):
    #      run_on_chunk(ichunk)
        
if __name__=='__main__':
    main()
