
import shutil
import numpy as np
import pandas as pd
import pandas_path as path
from PIL import Image
import torch
import pytorch_lightning as pl
import glob
from pathlib import Path
import argparse

from pytorch_lightning import loggers as pl_loggers
import albumentations as A

from cloud_seg.models.unet.cloud_model import CloudModel

DATA_DIR = Path.cwd().parent.resolve() / "data/"
DATA_DIR_MODEL_TRAINING = DATA_DIR / "model_training/"
DATA_DIR_CLOUDLESS = DATA_DIR / 'cloudless/tif/'
DATA_DIR_CLOUDS = DATA_DIR / 'clouds/'

TRAIN_FEATURES = DATA_DIR / "train_features"
TRAIN_FEATURES_NEW = DATA_DIR / "train_features_new"

TRAIN_LABELS = DATA_DIR / "train_labels"

band_mean_std = np.load(DATA_DIR / 'measured_band_stats.npy', allow_pickle=True).item()

def main():
    
    parser = argparse.ArgumentParser(description='runtime parameters')
    parser.add_argument("--bands", nargs='+' , default=["B02", "B03", "B04", "B08"],
                        help="bands desired")
    parser.add_argument("--bands_new", nargs='+', default=None,
                        help="additional bands to use beyond original four")
    parser.add_argument("-cv", "--cross_validation_split", type=int, default=0,
                        help="cross validation split to use for training") 

    parser.add_argument("--seed", type=int , default=13579,
                        help="random seed for train test split")
   
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
   
    parser.add_argument("--augmentations_string", type=str, default='nrvfhfrr',
                        help="training augmentations to use")
    parser.add_argument("--log_dir", type=str, default='logs/',
                        help="directory to put logfiles")
    
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU")
    
    parser.add_argument("--test_run", action="store_true",
                        help="Subsample training and validation data")
    parser.add_argument("--test_run_nchips", type=int, default=512,
                        help="Subsample training and validation data to this size")
                
    parser.add_argument("--cloud_augment", action="store_true",
                        help="Use cloud augmentation")
              
    parser.add_argument("--num_workers", type=int, default=3,
                        help="number of data loader workers")
  
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for model training")
  
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for model optimization")
  
    parser.add_argument("--optimizer", type=str, default='ADAM',
                        help="Optimizer to use", choices=['ADAM', 'SGD'])
  
    parser.add_argument("--backbone", type=str, default='efficientnet-b0',
                        help="Optimizer to use", choices=['efficientnet-b0', 'resnet34'])
  
    parser.add_argument("--loss_function", type=str, default='dice',
                        help="loss_function to use", choices=['SGD', 'Dice'])


    hparams = vars(parser.parse_args())
    hparams['bands_use'] = sorted(hparams['bands'] + hparams['bands_new']) if hparams['bands_new'] is not None else hparams['bands']
        
    hparams['persistent_workers'] = True
    hparams['precision'] = 32
    Path(hparams['log_dir']).mkdir(parents=True, exist_ok=True)
    
    pl.seed_everything(hparams['seed'], workers=True)

    augs = [hparams['augmentations_string'][i:i+2] for i in range(0, len(hparams['augmentations_string']), 2)]
    print(augs) 
    hparams['augmentations'] = {}
    if 'nr' in augs:
        hparams['augmentations']['Normalize'] = True
    if 'vf' in augs:
        hparams['augmentations']['VerticalFlip'] = True
    if 'hf' in augs:
        hparams['augmentations']['HorizontalFlip'] = True
    if 'rr' in augs:
        hparams['augmentations']['RandomRotate90'] = True

    print(hparams['augmentations'])
    
    if hparams['verbose']: print("Parameters are: ", hparams)

    dataset_str = 'originaldata'
    if hparams['cloud_augment']:
        dataset_str += '_cloudaugment'
    
    
    val_x = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"validate_features_meta_cv{hparams['cross_validation_split']}.csv")
    val_y = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"validate_labels_meta_cv{hparams['cross_validation_split']}.csv")
    
    # shuffle validation, such that each batch will have samples from different locations,
    # as validation_dataloader has shuffle=False
    val_x = val_x.sample(frac=1, random_state=42).reset_index(drop=True)
    val_y = val_y.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if hparams['verbose']: print(val_y.head())
    
    train_x = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"train_features_meta_cv{hparams['cross_validation_split']}.csv")
    train_y = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"train_labels_meta_cv{hparams['cross_validation_split']}.csv")

    if not hparams['cloud_augment']:
        
        df_cloudbank = None
    
    if hparams['cloud_augment']:

        train_x_cloudless = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"train_features_cloudless_meta_cv{hparams['cross_validation_split']}.csv", index=False)
        train_y_cloudless = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"train_labels_cloudless_meta_cv{hparams['cross_validation_split']}.csv", index=False)

        train_y = train_y.append(train_y_cloudless, ignore_index=True)
        train_x = train_x.append(train_x_cloudless, ignore_index=True)

        df_cloudbank = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"cloudbank_meta_cv{hparams['cross_validation_split']}.csv")


    if hparams['test_run']:
        nuse = hparams['test_run_nchips']

        train_x = train_x.iloc[:nuse]
        train_y = train_y.iloc[:nuse]

        val_x = val_x.iloc[:nuse]
        val_x = val_x.iloc[:nuse]

        df_cloudbank = df_cloudbank.iloc[:nuse] if df_cloudbank is not None else None
        
    band_means = [band_mean_std[i]['mean'] for i in hparams['bands_use']]
    band_stds = [band_mean_std[i]['std'] for i in hparams['bands_use']]

    # set up transforms in albumentations 
    train_transforms = []
    if hparams['augmentations']['Normalize']:
        train_transforms.append(A.Normalize(mean=band_means,
                                            std=band_stds,
                                            max_pixel_value=1.0,
                                            p=1.0))
    if hparams['augmentations']['VerticalFlip']:
        train_transforms.append(A.VerticalFlip(p=0.5))
    if hparams['augmentations']['HorizontalFlip']:
        train_transforms.append(A.HorizontalFlip(p=0.5))
    if hparams['augmentations']['RandomRotate90']:
        train_transforms.append(A.RandomRotate90(p=0.5))


    val_transforms = []
    if hparams['augmentations']['Normalize']:
        val_transforms.append(A.Normalize(mean=band_means,
                                            std=band_stds,
                                            max_pixel_value=1.0,
                                            p=1.0))

    if train_transforms==[]:
        train_transforms = None
    else:
        train_transforms = A.Compose(train_transforms)

    if val_transforms==[]:
        val_transforms = None
    else:
        val_transforms = A.Compose(val_transforms)

    # set up logger to have meaningful name
    augmentations_used_string = ''
    for k, v in hparams['augmentations'].items():
        if v:
            augmentations_used_string += '_'+k

    hparams['model_training_name'] = f"{len(hparams['bands_use'])}band_{dataset_str}_{hparams['backbone']}_{hparams['loss_function']}_{augmentations_used_string}"
    if hparams['test_run']:
        model_training_name = 'test'
    
    cloud_model = CloudModel(
        bands=hparams['bands_use'],
        x_train=train_x,
        y_train=train_y,
        x_val=val_x,
        y_val=val_y,
        cloudbank=df_cloudbank,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        hparams=hparams,
    )


    tb_logger = pl_loggers.TensorBoardLogger(save_dir=hparams['log_dir'], name=hparams['model_training_name']),

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=hparams['log_dir'],
                                                       monitor="val_iou_epoch",
                                                       mode="max",
                                                       verbose=True,
    )

    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_iou_epoch",
        patience=(cloud_model.patience * 3),
        mode="max",
        verbose=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # "ddp_spawn" needed for interactive jupyter, but best to use "ddp" if not
    trainer = pl.Trainer(
        gpus=-1,
        # deterministic=True,
        fast_dev_run=False,
        # profiler="simple",
        # max_epochs=2,
        # overfit_batches=1,
        # auto_scale_batch_size=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        precision=hparams['precision'],
        strategy="ddp",
        # plugins=DDPSpawnPlugin(find_unused_parameters=False),
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=tb_logger,
    )

    # Fit the model
    trainer.fit(model=cloud_model)
    
if __name__=='__main__':
    main()