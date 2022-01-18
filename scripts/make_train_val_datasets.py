
import numpy as np
import pandas as pd
import pandas_path as path
from pathlib import Path

import argparse

from cloud_seg.utils import utils

import glob
import os

DATA_DIR = Path.cwd().parent.resolve() / "data/"
DATA_DIR_OUT = DATA_DIR / "model_training/"
DATA_DIR_CLOUDLESS = DATA_DIR / "cloudless_tif/"

TRAIN_FEATURES = DATA_DIR / "train_features"
TRAIN_FEATURES_NEW = DATA_DIR / "train_features_new"

TRAIN_LABELS = DATA_DIR / "train_labels"

BAD_CHIPS_FILE = DATA_DIR / "BAD_CHIP_DATA/BAD_CHIP_LABEL_IDS.txt"
EASY_CHIPS_FILE = DATA_DIR / "BAD_CHIP_DATA/EASY_CHIPS_IDS.txt"

BAD_CHIP_IDS = list(np.loadtxt(BAD_CHIPS_FILE, dtype=str))
EASY_CHIP_IDS = list(np.loadtxt(EASY_CHIPS_FILE, dtype=str))

assert TRAIN_FEATURES.exists(), TRAIN_LABELS.exists()

Path(DATA_DIR_OUT).mkdir(parents=True, exist_ok=True)

def construct_dataframe(params: dict):

    df_meta = pd.read_csv(DATA_DIR / "train_metadata.csv")

    df_meta["datetime"] = pd.to_datetime(df_meta["datetime"])
    df_meta["year"] = df_meta.datetime.dt.year

    df_meta = utils.add_paths(df_meta, TRAIN_FEATURES, TRAIN_LABELS, bands=params['bands'])

    if params['bands_new'] is not None:

        # ensure that data exists for any desired new bands beyond the 4 originally provided
        for iband, band in enumerate(params['bands_new']):
            band_has_data = (TRAIN_FEATURES_NEW / df_meta["chip_id"] / f"{band}.tif").map(os.path.isfile)
            if iband==0: 
                has_banddata_on_disk = band_has_data
            else:
                has_banddata_on_disk = band_has_data & has_banddata_on_disk

            print('Fraction of chips that have new bands on disk = ', has_banddata_on_disk.sum()/has_banddata_on_disk.shape[0])

        df_meta = df_meta[has_banddata_on_disk]

        df_meta = utils.add_paths(df_meta, TRAIN_FEATURES_NEW, bands=bands_new)
        
    # Remove chips with incorrect labels
    df_meta = df_meta[~df_meta["chip_id"].isin(BAD_CHIP_IDS)].reset_index(drop=True)
    
    print(f"\nNumber of chips in dataset is {len(df_meta)}")
    return df_meta

def construct_cloudless_datafame(df_val, params: dict):
    """
    construct a dataframe full of cloudless chips
    
    These images will have train_y = None
    
    Dataloader will then draw random index to load in cloud files and cloud labels from cloud chips, and add clouds to images
    """
    all_chips = sorted(glob.glob(str(DATA_DIR_CLOUDLESS) + '/*'))

    # remove cloudless chips that have cloudy versions in validation sample
    in_val = [os.path.basename(i) in df_val['chip_id'].to_numpy() for i in all_chips]
    all_chips = [chip for ichip, chip in enumerate(all_chips) if not in_val[ichip]] 

    num_cloudless_chips = params['num_cloudless_chips']
    if num_cloudless_chips < 0:
        num_cloudless_chips = len(all_chips)
        
    # choose chip (locations) to use
    chips_use = np.random.choice(all_chips, size=num_cloudless_chips, replace=False)

    train_x_cloudless = []
    for ichip, chip in enumerate(chips_use):
        if params['verbose'] and ichip % 1000==0: print(ichip)
        
        all_observations = sorted(os.listdir(str(chip)))

        # for each location choose an image 
        chip_use_paths = sorted(glob.glob(chip + '/*'))
        chip_use_path = np.random.choice(chip_use_paths)

        chip_id = '{:s}_nc_{:s}'.format(os.path.basename(chip), os.path.basename(chip_use_path))
        feature_cols = [chip_use_path + f"/{band}.tif" for band in params['bands_use']]
        train_x_cloudless.append([chip_id]+feature_cols)

    train_x_cloudless = pd.DataFrame(train_x_cloudless, columns=df_val.columns)

    # add new cloudless images to train_y_new
    data = np.c_[np.array(train_x_cloudless['chip_id']),
                 np.array(['None']*len(train_x_cloudless['chip_id']))]
    train_y_cloudless = pd.DataFrame(data, columns=['chip_id', 'label_path'])

    print(f"Number of cloudless chips is not overlapping with validation set is {len(train_x_cloudless)}")

    if params['verbose']: print(train_y_cloudless.head(), train_y_cloudless.tail())
    
    return train_x_cloudless, train_y_cloudless

def split_train_val(df, params):
    np.random.seed(params['seed'])  # set a seed for reproducibility

    # put 1/3 of chips into the validation set
    # chip_ids = train_meta.chip_id.unique().tolist()
    # val_chip_ids = random.sample(chip_ids, round(len(chip_ids) * 0.33))

    # split by location, not by chip
    # else validation set is not a metric of true inference
    location_ids = df.location.unique().tolist()
    print(f"\nNumber of locations in dataset is {len(location_ids)}")

    # val_location_ids = np.random.choice(location_ids,
    #                                     size=round(len(location_ids) * params['val_fraction']),
    #                                     replace=False)
    np.random.shuffle(location_ids)
    if params['verbose']:
        print(location_ids)
    
    num_locations_each = round(len(location_ids) * params['val_fraction'])
    
    print(f"\nMaking {params['num_cross_validation_splits']} cross validation splits")
    print(f"with {num_locations_each} locations in each validation set")

    # make and save each train/val split to disk
    for isplit in range(params['num_cross_validation_splits']):
        ind_start = num_locations_each * isplit
        ind_end   = num_locations_each * (isplit+1)
        if isplit == params['num_cross_validation_splits'] - 1:
            ind_end = len(location_ids)
        
        num_locations_isplit = ind_end-ind_start
        val_location_ids = location_ids[ind_start:ind_end]
        
        print(f"\nCross validation set {isplit} starts on index {ind_start} and ends on {ind_end}. Contains {num_locations_isplit} locations")
        if params['verbose']:
            print(val_location_ids)
    
        val_mask = df.location.isin(val_location_ids)
        val = df[val_mask].copy().reset_index(drop=True)
        train = df[~val_mask].copy().reset_index(drop=True)

        # REMOVE EASY CHIPS FROM TRAIN SET            
        print("Train, val, total shape = ", train.shape, val.shape, train.shape[0]+val.shape[0])
        train = train[~train["chip_id"].isin(EASY_CHIP_IDS)].reset_index(drop=True)
        print("After easy chip removal: Train, val, total shape = ", train.shape, val.shape, train.shape[0]+val.shape[0])

        # separate features from labels
        feature_cols = ["chip_id"] + [f"{band}_path" for band in params['bands_use']]

        val_x = val[feature_cols].copy()
        val_y = val[["chip_id", "label_path"]].copy()

        train_x = train[feature_cols].copy()
        train_y = train[["chip_id", "label_path"]].copy()

        train_x_cloudless, train_y_cloudless = None, None
        if params['construct_cloudless']:
            train_x_cloudless, train_y_cloudless = construct_cloudless_datafame(val_x, params)
            
        if not params['dont_save_to_disk']:
            save_train_val_to_disk(train_x, train_y, val_x, val_y, train_x_cloudless, train_y_cloudless, params, isplit)

    return train_x, train_y, val_x, val_y, train_x_cloudless, train_y_cloudless


def save_train_val_to_disk(train_x, train_y, val_x, val_y, train_x_cloudless, train_y_cloudless, params, isplit):
    
    print(f"Saving training and validation sets from split {isplit} to disk")
    
    # f"train_features_meta_seed{params['seed']}_cv{isplit}.csv"
    train_x.to_csv(DATA_DIR_OUT / f"train_features_meta_cv{isplit}.csv", index=False)
    train_y.to_csv(DATA_DIR_OUT / f"train_labels_meta_cv{isplit}.csv", index=False)

    val_x.to_csv(DATA_DIR_OUT / f"validate_features_meta_cv{isplit}.csv", index=False)
    val_y.to_csv(DATA_DIR_OUT / f"validate_labels_meta_cv{isplit}.csv", index=False)
  
    if train_x_cloudless is not None:
        train_x_cloudless.to_csv(DATA_DIR_OUT / f"train_features_cloudless_meta_cv{isplit}.csv", index=False)
        train_y_cloudless.to_csv(DATA_DIR_OUT / f"train_labels_cloudless_meta_cv{isplit}.csv", index=False)

def main():
    
    parser = argparse.ArgumentParser(description='runtime parameters')
    
    parser.add_argument("--bands", nargs='+' , default=["B02", "B03", "B04", "B08"],
                        help="bands desired")
    
    parser.add_argument("--bands_new", nargs='+', default=None,
                        help="additional bands to use beyond original four")
    
    parser.add_argument("-ncv", "--num_cross_validation_splits", type=int, default=4,
                        help="Number of cross validation splits")    
    
    parser.add_argument("--seed", type=int , default=13579,
                        help="random seed for train test split")
    
    parser.add_argument("--construct_cloudless", action="store_true",
                        help="Construct an additional dataframe of cloudless") 
    
    parser.add_argument("--num_cloudless_chips", type=int, default=-1,
                        help="Number of cloudless samples to include") 
    
    parser.add_argument("--dont_save_to_disk", action="store_true",
                        help="save training and validation sets to disk") 
    
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
   
    params = vars(parser.parse_args())
    params['bands_use'] = sorted(params['bands'] + params['bands_new']) if params['bands_new'] is not None else params['bands']
    
    params['val_fraction'] = float(1./params['num_cross_validation_splits'])

    if params['verbose']: print("Parameters are: ", params)
    
    print(f"Outputs will be saved to:\n{str(DATA_DIR_OUT)}")
    df_meta = construct_dataframe(params)
    
    # split_train_val(df_meta, params)    
    train_x, train_y, val_x, val_y, train_x_cloudless, train_y_cloudless = split_train_val(df_meta, params)
             
if __name__=="__main__":
    main()
