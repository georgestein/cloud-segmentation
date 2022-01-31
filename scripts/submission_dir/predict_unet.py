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

def make_unet_predictions(
    model: CloudModel,
    x_paths: pd.DataFrame,
    hparams,
    predictions_dir: os.PathLike,
):
    """Predicts cloud cover and saves results to the predictions directory.

    Args:
        model (CloudModel): an instantiated CloudModel based on pl.LightningModule
        x_paths (pd.DataFrame): a dataframe with a row for each chip. There must be a column for chip_id,
                and a column with the path to the TIF for each of bands provided
        predictions_dir (os.PathLike): Destination directory to save the predicted TIF masks
    """
    # Set up transforms using Albumentations library
    Augs = CloudAugmentations(hparams)
    test_transforms, _transforms_names = Augs.add_augmentations()
    test_transforms = A.Compose(test_transforms)

    test_dataset = CloudDataset(
        x_paths=x_paths,
        bands=hparams['bands_use'],
        transforms=test_transforms,
        scale_feature_channels=model.scale_feature_channels,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=model.batch_size,
        num_workers=model.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,

    )
    torch.set_grad_enabled(False)
    model.eval()
    for batch_index, batch in enumerate(test_dataloader):

        if batch_index % 100 == 0:
            logger.debug(f"Predicting batch {batch_index} of {len(test_dataloader)}")

        x = batch["chip"]
        if model.gpu:
            x = x.cuda(non_blocking=True)

        preds = model.forward(x)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5) * 1
        preds = preds.detach()

        if model.gpu:
            preds = preds.to("cpu")

        preds = preds.numpy()

        for chip_id, pred in zip(batch["chip_id"], preds):
            chip_pred_path = predictions_dir / f"{chip_id}.npy"
            np.save(chip_pred_path, pred.astype(np.float32))

        '''
        preds = preds.astype("uint8")

        for chip_id, pred in zip(batch["chip_id"], preds):
            chip_pred_path = predictions_dir / f"{chip_id}.tif"
            chip_pred_im = Image.fromarray(pred)
            chip_pred_im.save(chip_pred_path)
        '''
