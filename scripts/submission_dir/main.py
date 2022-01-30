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

import os
import argparse

import albumentations as A

try:
    from cloud_seg.models.unet.cloud_model import CloudModel
    from cloud_seg.models.unet.cloud_dataset import CloudDataset
    from cloud_seg.utils.augmentations import CloudAugmentations
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
    # INPUT_IMAGES_DIR = DATA_DIR / "test_features/"

else:
    ROOT_DIR = Path("/codeexecution")
    PREDICTIONS_DIR = ROOT_DIR / "predictions"
    ASSETS_DIR = ROOT_DIR / "assets"
    DATA_DIR = ROOT_DIR / "data"
    # INPUT_IMAGES_DIR = DATA_DIR / "test_features/"

    # Set the pytorch cache directory and include cached models in your submission.zip
    os.environ["TORCH_HOME"] = str(ASSETS_DIR / "assets/torch")


def get_metadata(features_dir: os.PathLike, hparams):
    """
    Given a folder of feature data, return a dataframe where the index is the chip id
    and there is a column for the path to each band's TIF image.

    Args:
        features_dir (os.PathLike): path to the directory of feature data, which should have
            a folder for each chip
    """
    chip_metadata = pd.DataFrame(index=[f"{band}_path" for band in hparams['bands_use']])
    chip_ids = (
        pth.name for pth in features_dir.iterdir() if not pth.name.startswith(".")
    )

    for chip_id in sorted(chip_ids):
        # chip_bands = [INPUT_IMAGES_DIR / chip_id / f"{band}.tif" for band in bands]
        if hparams['bands_new'] is not None:
            chip_bands = [features_dir / chip_id / f"{band}.tif" if band not in hparams['bands_new'] else INPUT_IMAGES_DIR_NEW / chip_id / f"{band}.tif" for band in hparams['bands_use']]
        else:
            chip_bands = [features_dir / chip_id / f"{band}.tif" for band in hparams['bands_use']]

        chip_metadata[chip_id] = chip_bands

    return chip_metadata.transpose().reset_index().rename(columns={"index": "chip_id"})


def main(
    model_weights_path = ASSETS_DIR / "cloud_model.pt",
    hparams_path = ASSETS_DIR / "hparams.npy",
    test_features_dir: Path = DATA_DIR / "test_features",
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


    # Pull additional bands B01 and B11
    # By default saved to data_new/test_features_new
    pull_additional_chip_data.main()

    logger.info("Loading model")

    # Load parameters for Unet
    hparams = np.load(hparams_path, allow_pickle=True).item()
    hparams['batch_size'] = 8
    hparams['weights'] = None
    # Load with gpu=False, then put on GPU
    hparams['gpu'] = False


    # Load metadata
    logger.info("Loading metadata")
    metadata = get_metadata(test_features_dir, hparams)
    if fast_dev_run:
        metadata = metadata.head()
    logger.info(f"Found {len(metadata)} chips")


    # Load unet model
    model = CloudModel(
        bands=hparams['bands_use'],
        hparams=hparams
    )

    model.load_state_dict(torch.load(model_weights_path))

    hparams['gpu'] = True
    if hparams['gpu']:
        model = model.cuda()
        model.gpu = True

    # Make predictions and save to disk
    logger.info("Generating predictions in batches")
    # predict image-based
    make_unet_predictions(model, metadata, hparams, predictions_dir)
    # predict feature based
    make_gbm_predictions(metadata, predictions_dir)

    # compile predictions from each model into final prediction
    logger.info(f"""Saved {len(list(predictions_dir.glob("*.tif")))} predictions""")


if __name__ == "__main__":
    typer.run(main)
