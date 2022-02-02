import shutil
import numpy as np
import pandas as pd
import pandas_path as path
from PIL import Image
import torch
import pytorch_lightning as pl
import glob
from pathlib import Path
from loguru import logger
from predict_unet import make_unet_predictions
from predict_gbm import make_gbm_predictions

from typing import List
import typer
# import logging

import sys
import os
import argparse
import glob

import albumentations as A

try:
    from cloud_seg.models.unet.cloud_model import CloudModel
    from cloud_seg.models.unet.cloud_dataset import CloudDataset
    from cloud_seg.utils.augmentations import CloudAugmentations
    import pull_additional_chip_data
except ImportError:
    from cloud_model import CloudModel
    from cloud_dataset import CloudDataset
    from augmentations import CloudAugmentations
    import pull_additional_chip_data

if os.environ['CONDA_DEFAULT_ENV'] == 'cloud-seg':
    # running in local conda environment (hopefully theirs isnt the same name...)
    ROOT_DIR = Path("./")
    PREDICTIONS_DIR = ROOT_DIR / "predictions"
    ASSETS_DIR = ROOT_DIR / "assets"
    DATA_DIR = ROOT_DIR / "data"
    DATA_DIR_NEW = ROOT_DIR / "data_new"
    # INPUT_IMAGES_DIR = DATA_DIR / "test_features/"

else:
    ROOT_DIR = Path("/codeexecution")
    PREDICTIONS_DIR = ROOT_DIR / "predictions"
    ASSETS_DIR = ROOT_DIR / "assets"
    DATA_DIR = ROOT_DIR / "data"
    DATA_DIR_NEW = ROOT_DIR / "data_new"

    # INPUT_IMAGES_DIR = DATA_DIR / "test_features/"

    # Set the pytorch cache directory and include cached models in your submission.zip
    os.environ["TORCH_HOME"] = str(ASSETS_DIR / "assets/torch")


def get_metadata(features_dir: os.PathLike,
                 features_dir_new: os.PathLike,
                 params):
    """
    Given a folder of feature data, return a dataframe where the index is the chip id
    and there is a column for the path to each band's TIF image.

    Args:
        features_dir (os.PathLike): path to the directory of feature data, which should have
            a folder for each chip
    """
    chip_metadata = pd.DataFrame(index=[f"{band}_path" for band in params['bands_use']])
    chip_ids = (
        pth.name for pth in features_dir.iterdir() if not pth.name.startswith(".")
    )

    for chip_id in sorted(chip_ids):
        # chip_bands = [INPUT_IMAGES_DIR / chip_id / f"{band}.tif" for band in bands]
        if params['bands_new'] is not None:
            chip_bands = [features_dir / chip_id / f"{band}.tif" if band not in params['bands_new'] else features_dir_new / chip_id / f"{band}.tif" for band in params['bands_use']]
        else:
            chip_bands = [features_dir / chip_id / f"{band}.tif" for band in params['bands_use']]

        chip_metadata[chip_id] = chip_bands

    return chip_metadata.transpose().reset_index().rename(columns={"index": "chip_id"})

def compile_predictions(metadata, unet_predictions_dir, gbm_predictions_dir, predictions_dir, hparams):
    """
    load in predictions for Unet and/or Boosting model(s),
    and save to final prediction dir
    """
    for chip_id in metadata['chip_id']:
        preds_unet = []
        for icross_validation in range(hparams['num_cross_validation_splits']):
            #pred_unet = np.load(unet_predictions_dir / f"{chip_id}.npy")
            preds_unet.append(np.load(unet_predictions_dir / f"{chip_id}_cv{icross_validation}.npy"))
        preds_unet = np.stack(preds_unet, axis=-1)

        # Mean of models equals final prediction
        pred_unet = np.mean(preds_unet, axis=-1) 

        if hparams['use_features']:
            pred_gbm  = np.load(gbm_predictions_dir / f"{chip_id}.npy")
            pred_gbm = (pred_gbm>0.1)*1
        
        #pred_final = ( (pred_unet + pred_gbm)/2 >= 0.5)*1
        #pred_final = pred_final.astype("uint8")
        #pred_final = ((pred_unet >= 0.5)*1).astype("uint8")

        pred_final = ((pred_unet >= 0.5) * 1).astype("uint8")

        chip_pred_path = predictions_dir / f"{chip_id}.tif"
        chip_pred_im = Image.fromarray(pred_final)
        chip_pred_im.save(chip_pred_path)

def main(
    model_weights_path = ASSETS_DIR / "cloud_model.pt",
    hparams_unet_path = ASSETS_DIR / "hparams.npy",
    test_features_dir: Path = DATA_DIR / "test_features",
    test_features_dir_new: Path = DATA_DIR_NEW / "test_features_new",
    predictions_dir: Path = PREDICTIONS_DIR,
    fast_dev_run: bool = False,
):
    """
    Generate predictions for the chips in features_dir using the model saved at
    model_path.

    Predictions are saved in predictions_dir. The default paths to all three files are based on
    the structure of the code execution runtime.

    Args:
        model_weights_path (os.PathLike): Path to the weights of a trained CloudModel.
        features_dir (os.PathLike, optional): Path to the features for the data. Defaults
            to 'data/test_features' in the same directory as main.py
        predictions_dir (os.PathLike, optional): Destination directory to save the predicted TIF masks
            Defaults to 'predictions' in the same directory as main.py
    """
    if not test_features_dir.exists():
        raise ValueError(
            f"The directory for feature images must exist and {test_features_dir} does not exist"
        )
    predictions_dir.mkdir(exist_ok=True, parents=True)

    unet_predictions_dir = Path(f"data_new/predictions_unet")
    unet_predictions_dir.mkdir(exist_ok=True, parents=True)

    gbm_predictions_dir = Path("data_new/predictions_gbm")
    gbm_predictions_dir.mkdir(exist_ok=True, parents=True)

    # Load parameters for Unet
    hparams = np.load(hparams_unet_path, allow_pickle=True).item()
    logger.info(hparams)
    
    hparams['batch_size'] = 8
    hparams['num_workers'] = 4
    hparams['weights'] = None
    # Load with gpu=False, then put on GPU
    hparams['gpu'] = False


    download_new_bands = False
    missing_bands = list(set(hparams['bands_use']) - set(['B02', 'B03', 'B04', 'B08']))
    if missing_bands != []:
        download_new_bands = True
    if hparams['use_features']:
        download_new_bands = True
                
    # Load metadata
    params_metadata = {}
    params_metadata['bands_use'] = ['B01', 'B02', 'B03', 'B04', 'B08', 'B11']
    params_metadata['bands_new'] = ['B01', 'B11']

    logger.info("Loading metadata")
    print(test_features_dir, test_features_dir_new)
    metadata = get_metadata(
        test_features_dir,
        test_features_dir_new,
        params_metadata,
    )

    print(metadata.head())
    
    if fast_dev_run:
        metadata = metadata.head()
    logger.info(f"Found {len(metadata)} chips")

    if download_new_bands:
        logger.info("Pulling additional data")
        # Pull additional bands B01 and B11
        # By default saved to data_new/test_features_new
        for ithreaded_pull in range(10):
            # run threaded pull a few times
            try:
                pull_additional_chip_data.main(max_pool_size=8)
            except:
                logger.info(f"Pulling threaded data {ithreaded_pull} to best of ability")

        # pull data one final time with no threading
        # this will skip already downloaded files and catch missing data from any threads that crashed
        pull_additional_chip_data.main(max_pool_size=1)

        # Check everything downloaded properly
        new_chip_dirs = sorted(glob.glob("data_new/test_features_new/*"))
        logger.info(f"New bands downloaded properly? {len(new_chip_dirs)} dirs in data_new/test_features_new")
        num_missing = 0
        for chip_id in metadata['chip_id']:
            exists_on_disk = True
            for band in params_metadata['bands_new']:
                if not os.path.isfile(f"data_new/test_features_new/{chip_id}/{band}.tif"):
                    exists_on_disk = False

            if not exists_on_disk:
                num_missing+=1
            
        logger.info(f"{num_missing} new bands missing")
    
    logger.info("Loading and running Unet model")

    for icross_validation in range(hparams['num_cross_validation_splits']):
        logger.info(f"Running on U-Net cross validation split {icross_validation}")
        # Load unet model
        model = CloudModel(
            bands=hparams['bands_use'],
            hparams=hparams,
        )

        model_weights_path_icv = os.path.splitext(model_weights_path)[0] + f"_cv{icross_validation}.pt"
        print(model_weights_path_icv)
        model.load_state_dict(torch.load(model_weights_path_icv))

        hparams['gpu'] = True
        if hparams['gpu']:
            model = model.cuda()
            model.gpu = True

        # Make predictions and save to disk
        logger.info("Generating U-Net predictions in batches")
        # predict image-based
        make_unet_predictions(model, metadata, hparams, unet_predictions_dir, icross_validation)

    # predict feature based
    if hparams['use_features']:

        logger.info("Generating GBM predictions in batches")

        make_gbm_predictions(metadata, gbm_predictions_dir)

    # compile predictions from each model into final prediction
    compile_predictions(metadata, unet_predictions_dir, gbm_predictions_dir, predictions_dir, hparams)

    logger.info(f"""Saved {len(list(predictions_dir.glob("*.tif")))} predictions""")


if __name__ == "__main__":
    typer.run(main)
