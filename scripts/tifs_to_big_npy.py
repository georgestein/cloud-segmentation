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

import math
import time
import argparse

import multiprocessing

from cloud_seg.utils import chip_vis, utils
from cloud_seg.io import io

DATA_DIR = Path.cwd().parent.resolve() / "data/"
DATA_DIR_CLOUDS = DATA_DIR / 'clouds/'
DATA_DIR_CLOUDLESS = DATA_DIR / 'cloudless/'
DATA_DIR_CLOUDLESS_MOST_SIMILAR = DATA_DIR / 'cloudless_most_similar/'
DATA_DIR_CLOUDLESS_TIF = DATA_DIR / 'cloudless_tif/'

DATA_DIR_OUT = Path.cwd().parent.resolve() / "notebooks/"

PREDICTION_DIR = Path.cwd().parent.resolve() / "trained_models/unet/test/predictions/"

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

parser.add_argument("--chunksize", type=int, default=1000,
                    help="Chunksize for output arrays") 

parser.add_argument("--max_pool_size", type=int, default=32,
                    help="Chunksize for output arrays") 
                          
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")

params = vars(parser.parse_args())
params['bands_use'] = sorted(params['bands'] + params['bands_new']) if params['bands_new'] is not None else params['bands']

params['outsize'] = [512, 512]

if params['verbose']: print("Parameters are: ", params)
    
train_meta = pd.read_csv(DATA_DIR / "train_metadata.csv")

# how many different chip ids, locations, and datetimes are there?
print(train_meta[["chip_id", "location", "datetime"]].nunique())

train_meta.head()

train_meta = utils.add_paths(train_meta, TRAIN_FEATURES, TRAIN_LABELS, bands=params['bands'])

if params['bands_new'] is not None:
    # has_B01  = (TRAIN_FEATURES_NEW / train_meta["chip_id"] / f"B01.tif").map(os.path.isfile)
    # has_B11 = (TRAIN_FEATURES_NEW / train_meta["chip_id"] / f"B11.tif").map(os.path.isfile)

    # print('Fraction of chips that have B01, B11 = ', has_B01.sum()/has_B01.shape[0], has_B11.sum()/has_B11.shape[0])

    # dm = has_B01 & has_B11
    # train_meta = train_meta[dm]

    train_meta = utils.add_paths(train_meta, TRAIN_FEATURES_NEW, bands=params['bands_new'])

# Total number of chunks. Set as global variable
params['nchunks'] = math.ceil(len(train_meta)/params['chunksize'])
params['max_pool_size'] = min(params['nchunks'], params['max_pool_size'])

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
    preds = np.zeros((nchips, npixx, npixy), dtype=np.uint8)
    
    labels_mean = np.zeros(nchips, dtype=np.float32)
    preds_mean  = np.zeros(nchips, dtype=np.float32)
    intersection = np.zeros(nchips, dtype=np.float32)
    union = np.zeros(nchips, dtype=np.float32)

    chip_ids = []
    for ichip, ichip_meta_loc in enumerate(range(ichip_start, ichip_end)):

        chip = train_meta.iloc[ichip_meta_loc]
        
        chip_ids.append(chip.chip_id)
        images[ichip] = load_image_to_array(chip.chip_id, bands=bands)  
        labels[ichip] = np.array(Image.open(chip.label_path))
        preds_i = np.array(Image.open(PREDICTION_DIR/f"{chip.chip_id}.tif"))
        preds_i = (preds_i > 0.5)*1
        preds[ichip] = preds_i.astype(np.int8)
        
        intersection[ichip], union[ichip] = intersection_and_union(preds[ichip], labels[ichip])
        labels_mean[ichip] = np.mean(labels[ichip])
        preds_mean[ichip] = np.mean(preds[ichip])

    d = {}
    d['chip_ids'] = chip_ids
    d['images'] = images
    d['labels'] = labels
    d['preds'] = preds
    d['labels_mean'] = labels_mean
    d['preds_mean'] = preds_mean
        
    d['intersection'] = intersection
    d['union'] = union

    d['IoU'] = intersection/union

    return d

def run_on_chunk(ichunk):
    
    tstart = time.time()
    ichip_start = ichunk*params['chunksize']
    ichip_end = min(len(train_meta), (ichunk+1)*params['chunksize'])

    print(f"\nRunning on chunk {ichunk} out of {params['nchunks']}. Index start:{ichip_start}, index end: {ichip_end}")

    data = get_chips_in_npy(ichip_start, ichip_end, bands=params['bands_use'])

    print("data, label, prediction shape: ", data['images'].shape, data['labels'].shape, data['preds'].shape)

    for k, v in data.items():
        np.save(DATA_DIR_OUT / f"data/all_chunks/{k}_{ichip_start:06d}_{ichip_end:06d}.npy", v)

    print("Time elapsed = ", time.time()-tstart)
    
def main():
    """
    Loop over chunks of size <chunksize> and save individual .tifs in various folders to
    single large numpy arrays.
    
    In order to facilitate faster data/label/prediction investigation
    """
    
#     # Simple threading with pool and .map
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpus if cpus < params['max_pool_size'] else params['max_pool_size'])
    print(f"Number of available cpus = {cpus}")
    
    pool.map(run_on_chunk, range(params['nchunks']))
        
    # pool.close()
    # pool.join()
    # for ichunk in range(params['nchunks']):
    #      run_on_chunk(ichunk)
        
if __name__=='__main__':
    main()
