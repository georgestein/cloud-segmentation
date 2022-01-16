
import shutil
import numpy as np
import pandas as pd
import pandas_path as path
from PIL import Image
import torch
import pytorch_lightning as pl
import glob
from pathlib import Path

from typing import List
# import typer
import logging

import os
import argparse

from cloud_seg.models.unet.cloud_model import CloudModel
from cloud_seg.models.unet.cloud_model import CloudDataset


parser = argparse.ArgumentParser(description='runtime parameters')
parser.add_argument("--bands", nargs='+' , default=["B02", "B03", "B04", "B08"],
                    help="bands desired")
parser.add_argument("--bands_new", nargs='+', default=None,
                    help="additional bands to use beyond original four")

parser.add_argument("-cv", "--cross_validation_split", type=int, default=0,
                    help="cross validation split to use for training") 

parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for model inference")

parser.add_argument("--INPUT_DIR", type=str, default='../trained_models/unet/4band_originaldata_efficientnet-b0_dice__Normalize_VerticalFlip_HorizontalFlip_RandomRotate90/',
                    help="Directory to save logs and trained models model")

parser.add_argument("--LOG_DIR", type=str, default='logs/',
                    help="Sub-directory of OUTPUT_DIR to save logs")

parser.add_argument("--MODEL_DIR", type=str, default='model/',
                    help="Sub-directory of OUTPUT_DIR to save logs")

parser.add_argument("--OUTPUT_DIR", type=str, default='predictions/',
                    help="Directory to save logs and trained models model")

parser.add_argument("--gpu", action="store_true",
                    help="Use GPU")  
                    
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
                    
parser.add_argument("--local_run", action="store_true",
                    help="Whether running locally or on planetary computer")
                    
parser.add_argument("--model_name", type=str, default='cloud_model.pt',
                    help="directory to save trained model")

parser.add_argument("--segmentation_model", type=str, default='unet',
                    help="Encocoder architecture to use", choices=['unet', 'DeepLabV3Plus'])
  
parser.add_argument("--encoder_name", type=str, default='efficientnet-b0',
                    help="Architecture to use", choices=['efficientnet-b0', 'resnet34'])

parser.add_argument("--load_checkpoint", action="store_true",
                    help="Whether loading weights from checkpoint (.ckpt) or just from saved weights state_dict (.pt)")

hparams = vars(parser.parse_args())
hparams['weights'] = None
hparams['bands_use'] = sorted(hparams['bands'] + hparams['bands_new']) if hparams['bands_new'] is not None else hparams['bands']
         
# hparams['INPUT_DIR'] = os.path.join(hparams['INPUT_DIR'], hparams['segmentation_model'], hparams['model_training_name'])
# hparams['MODEL_DIR'] = os.path.join(hparams['INPUT_DIR'], hparams['model_training_name'], hparams['MODEL_DIR'])
# hparams['LOG_DIR'] = os.path.join(hparams['INPUT_DIR'], hparams['model_training_name'], hparams['OUTPUT_DIR'], hparams['LOG_DIR'])
# hparams['OUTPUT_DIR'] = os.path.join(hparams['INPUT_DIR'], hparams['model_training_name'], hparams['OUTPUT_DIR'])

# Path(hparams['LOG_DIR']).mkdir(parents=True, exist_ok=True)
# Path(hparams['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)

if hparams['local_run']:      
    
    ROOT_DIR = Path.cwd().parent.resolve()
    ASSETS_DIR = Path(hparams['INPUT_DIR'])
    MODEL_PATH = ASSETS_DIR / hparams['MODEL_DIR'] / "last.ckpt"
                    
    PREDICTIONS_DIR = ASSETS_DIR / hparams['OUTPUT_DIR']
     
    DATA_DIR = ROOT_DIR / "data/"
    DATA_DIR_MODEL_TRAINING = DATA_DIR / "model_training/"
    DATA_DIR_CLOUDLESS = DATA_DIR / 'cloudless/tif/'
    DATA_DIR_CLOUDS = DATA_DIR / 'clouds/'

    INPUT_IMAGES_DIR = DATA_DIR / "train_features"
    INPUT_IMAGES_DIR_NEW = DATA_DIR / "train_features_new"

    TRAIN_LABELS = DATA_DIR / "train_labels"

    band_mean_std = np.load(DATA_DIR / 'measured_band_stats.npy', allow_pickle=True).item()
       
    logger = logging.getLogger("test_logger")
    fh = logging.FileHandler('test_logger.log')
    ch = logging.StreamHandler()
    
    logger.addHandler(fh)
    logger.addHandler(ch)

    Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
  
else:
    ROOT_DIR = Path("/codeexecution")
    PREDICTIONS_DIR = ROOT_DIR / "predictions"
    ASSETS_DIR = ROOT_DIR / "assets"
                    
    DATA_DIR = ROOT_DIR / "data"
    INPUT_IMAGES_DIR = DATA_DIR / "test_features"
                    
    # Set the pytorch cache directory and include cached models in your submission.zip
    os.environ["TORCH_HOME"] = str(ASSETS_DIRECTORY / "assets/torch")

    MODEL_PATH = ASSETS_DIR / hparams['model_name']            

logger = logging.getLogger()

def get_metadata(features_dir: os.PathLike, bands: List[str]):
    """
    Given a folder of feature data, return a dataframe where the index is the chip id
    and there is a column for the path to each band's TIF image.

    Args:
        features_dir (os.PathLike): path to the directory of feature data, which should have
            a folder for each chip
        bands (list[str]): list of bands provided for each chip
    """
    chip_metadata = pd.DataFrame(index=[f"{band}_path" for band in bands])
    chip_ids = (
        pth.name for pth in features_dir.iterdir() if not pth.name.startswith(".")
    )

    for chip_id in sorted(chip_ids):
        chip_bands = [features_dir / chip_id / f"{band}.tif" for band in bands]
        chip_metadata[chip_id] = chip_bands

    return chip_metadata.transpose().reset_index().rename(columns={"index": "chip_id"})


def make_predictions(
    model: CloudModel,
    x_paths: pd.DataFrame,
    bands: List[str],
    predictions_dir: os.PathLike,
):
    """Predicts cloud cover and saves results to the predictions directory.

    Args:
        model (CloudModel): an instantiated CloudModel based on pl.LightningModule
        x_paths (pd.DataFrame): a dataframe with a row for each chip. There must be a column for chip_id,
                and a column with the path to the TIF for each of bands provided
        bands (list[str]): list of bands provided for each chip
        predictions_dir (os.PathLike): Destination directory to save the predicted TIF masks
    """
    predict_dataset = CloudDataset(
        x_paths=x_paths,
        bands=bands,
    )
    predict_dataloader = torch.utils.data.DataLoader(
        predict_dataset,
        batch_size=model.batch_size,
        num_workers=model.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    
    for batch_index, batch in enumerate(predict_dataloader):
        print("Running on batch: ", batch_index)
        logger.debug(f"Predicting batch {batch_index} of {len(predict_dataloader)}")
       
        x = batch["chip"]
        if model.gpu:
            x = x.cuda(non_blocking=True)
        
        preds = model.forward(x)
        if not hparams['local_run']:
            preds = (preds > 0.5)
            
        preds = preds.detach()
        
        if model.gpu:
            preds = preds.to("cpu").numpy()
        if not hparams['local_run']:
            preds = preds.astype("uint8")

        for chip_id, pred in zip(batch["chip_id"], preds):
            chip_pred_path = predictions_dir / f"{chip_id}.tif"
            chip_pred_im = Image.fromarray(pred)
            chip_pred_im.save(chip_pred_path)


def main(
    bands: List[str] = ["B02", "B03", "B04", "B08"],
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
        bands (List[str], optional): List of bands provided for each chip
    """
    if not INPUT_IMAGES_DIR.exists():
        raise ValueError(
            f"The directory for feature images must exist and {INPUT_IMAGES_DIR} does not exist"
        )
    PREDICTIONS_DIR.mkdir(exist_ok=True, parents=True)

    print('RUNNING')
    logger.info("Loading model")
    print('RUNNING')
    
    # Load with gpu=False, then put on GPU
    hparams['gpu'] = False
    model = CloudModel(
        bands=hparams['bands_use'],
        hparams=hparams
    )
   
    print('Constructed base model')
    # load model from disk
    if not hparams['load_checkpoint']:
        # directly load weights
        model.load_state_dict(torch.load(MODEL_PATH))
        
    if hparams['load_checkpoint']:
        # load weights from checkpoint
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model weights')
    
    hparams['gpu'] = True
    if hparams['gpu']:
        model = model.cuda()
        model.gpu = True

             
    # Load metadata
    logger.info("Loading metadata")
    metadata = get_metadata(INPUT_IMAGES_DIR, bands=bands)
    if fast_dev_run:
        metadata = metadata.head()
    logger.info(f"Found {len(metadata)} chips")
    
    
    print('Loaded metadata')
    # Make predictions and save to disk
    logger.info("Generating predictions in batches")
    make_predictions(model, metadata, bands, PREDICTIONS_DIR)

    logger.info(f"""Saved {len(list(PREDICTIONS_DIR.glob("*.tif")))} predictions""")


if __name__ == "__main__":
    # if hparams['local_run']:              
    #     main()
    # else:
        #typer.run(main)
    main()
