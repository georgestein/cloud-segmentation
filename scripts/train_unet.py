
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
import os
import datetime

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
import albumentations as A

from cloud_seg.models.unet.cloud_model import CloudModel
from cloud_seg.utils.augmentations import CloudAugmentations

import matplotlib.pyplot as plt

DATA_DIR = Path.cwd().parent.resolve() / "data/"
DATA_DIR_MODEL_TRAINING = DATA_DIR / "model_training/"
DATA_DIR_CLOUDLESS = DATA_DIR / 'cloudless/tif/'
DATA_DIR_CLOUDS = DATA_DIR / 'clouds/'

TRAIN_FEATURES = DATA_DIR / "train_features"
TRAIN_FEATURES_NEW = DATA_DIR / "train_features_new"

TRAIN_LABELS = DATA_DIR / "train_labels"

band_mean_std = np.load(DATA_DIR / 'measured_band_stats.npy', allow_pickle=True).item()

def none_or_str(value):
    if value == 'None':
        return None
    return value

def main(args):
    # train_data_string = ""   
    train_data_string = "_new"
    
    hparams = vars(args)
    if hparams['verbose']: print("Parameters are: ", hparams)

    pl.seed_everything(hparams['seed'], workers=True)
    hparams['precision'] = 32
    hparams['bands_use'] = sorted(hparams['bands'] + hparams['bands_new']) if hparams['bands_new'] is not None else hparams['bands']
    
    # hparams['band_means'] = [band_mean_std[i]['mean'] for i in hparams['bands_use']]
    # hparams['band_stds'] = [band_mean_std[i]['std'] for i in hparams['bands_use']]
    # hparams['max_pixel_value'] = 1.0
    # if hparams['custom_feature_channels'] == "true_color":
    #     hparams['bands_use'] = ['B04', 'B03', 'B02']
    #     hparams['band_means'] = [0.485, 0.456, 0.406] # imagenet for now
    #     hparams['band_stds'] = [0.229, 0.224, 0.225]      
    #     hparams['max_pixel_value'] = 255


    hparams['OUTPUT_DIR'] = os.path.join(hparams['OUTPUT_DIR'], hparams['segmentation_model'])
    Path(hparams['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)

    # Set up transforms using Albumentations library
    Augs = CloudAugmentations(hparams)
    train_transforms, train_transforms_names = Augs.add_augmentations()
    train_transforms = A.Compose(train_transforms)

    augs_val = 'nr' if 'Normalize' in train_transforms_names else ''
    val_transforms, val_transforms_names = Augs.add_augmentations(augs_val)
    val_transforms = A.Compose(val_transforms)

    print("Train, val transforms: ", train_transforms_names, val_transforms_names)
    
    cloud_transforms = None
    if hparams['cloud_augment']:
        # Set up transforms using Albumentations library
        hparams_cloud = hparams.copy()
        hparams_cloud['sigma_brightness'] = hparams['cloud_sigma_brightness']
        hparams_cloud['mean_brightness'] = hparams['cloud_mean_brightness']
        hparams_cloud['uniform_brightness'] = hparams['cloud_uniform_brightness']
        
        Augs = CloudAugmentations(hparams_cloud)
        
        cloud_transforms, cloud_transforms_names = Augs.add_augmentations(hparams['cloud_augmentations'])
        cloud_transforms = A.Compose(cloud_transforms)
        print("cloud transforms: ", cloud_transforms_names)
    
    # set up logger and model outputs to have meaningful name

    dataset_str = 'originaldata'
    if hparams['cloud_augment']:
        dataset_str += '_cloudaugment'
    curent_time = datetime.datetime.now().strftime("%Y-%m-%d")#-%H:%M:%S")
 
    model_out_name = f"{len(hparams['bands_use'])}band"
    model_out_name += f"_{dataset_str}"  
    model_out_name += f"_{hparams['encoder_name']}"
    model_out_name += f"_{hparams['loss_function']}"
    model_out_name += f"_{hparams['augmentations']}"
    model_out_name += f"_customfeats_{hparams['scale_feature_channels']}"
    # model_out_name += f"_depth{hparams['encoder_depth']}"
    model_out_name += f"_{curent_time}"
    model_out_name += f"_cv{hparams['cross_validation_split']}"

    hparams['model_training_name'] = model_out_name
    if hparams['test_run']:
        model_training_name = 'test'
    
    hparams['LOG_DIR'] = os.path.join(hparams['OUTPUT_DIR'], hparams['model_training_name'], hparams['LOG_DIR'])
    hparams['MODEL_DIR'] = os.path.join(hparams['OUTPUT_DIR'], hparams['model_training_name'], hparams['MODEL_DIR'])
                                      
    Path(hparams['LOG_DIR']).mkdir(parents=True, exist_ok=True)
    Path(hparams['MODEL_DIR']).mkdir(parents=True, exist_ok=True)
    
    # Load Data
    val_x = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"validate_features_meta_cv{hparams['cross_validation_split']}{train_data_string}.csv")
    val_y = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"validate_labels_meta_cv{hparams['cross_validation_split']}{train_data_string}.csv")
    
    # shuffle validation, such that each batch will have samples from different locations,
    # as validation_dataloader has shuffle=False
    val_x = val_x.sample(frac=1, random_state=42).reset_index(drop=True)
    val_y = val_y.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if hparams['verbose']: print(val_y.head())
    
    train_x = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"train_features_meta_cv{hparams['cross_validation_split']}{train_data_string}.csv")
    train_y = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"train_labels_meta_cv{hparams['cross_validation_split']}{train_data_string}.csv")

    if not hparams['cloud_augment']:
        
        df_cloudbank = None
    
    if hparams['cloud_augment']:

        train_x_cloudless = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"train_features_cloudless_meta_cv{hparams['cross_validation_split']}{train_data_string}.csv")
        train_y_cloudless = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"train_labels_cloudless_meta_cv{hparams['cross_validation_split']}{train_data_string}.csv")

        # duplicate cloudless chips
        for i in range(1):
            train_y = train_y.append(train_y_cloudless, ignore_index=True)
            train_x = train_x.append(train_x_cloudless, ignore_index=True)

        df_cloudbank = pd.read_csv(DATA_DIR_MODEL_TRAINING / f"cloudbank_meta_cv{hparams['cross_validation_split']}{train_data_string}.csv")

    if hparams['test_run']:
        nuse = hparams['test_run_nchips']

        train_x = train_x.iloc[:nuse]
        train_y = train_y.iloc[:nuse]

        val_x = val_x.iloc[:nuse]
        val_x = val_x.iloc[:nuse]

        df_cloudbank = df_cloudbank.iloc[:nuse] if df_cloudbank is not None else None
        

    # Set up models and callbacks
    cloud_model = CloudModel(
        bands=hparams['bands_use'],
        x_train=train_x,
        y_train=train_y,
        x_val=val_x,
        y_val=val_y,
        cloudbank=df_cloudbank,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        cloud_transforms=cloud_transforms,
        hparams=hparams,
    )
        
    if hparams['load_checkpoint_path'] is not None:
        cloud_model = cloud_model.load_from_checkpoint(hparams['load_checkpoint_path'])
  
        
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=hparams['LOG_DIR'],
        name='log',
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=hparams['MODEL_DIR'],
        filename="best_{epoch}-{val_iou:.4f}",
        monitor="val_iou",
        mode="max",
        verbose=True,
        save_last=True,
    )

    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_iou",
        patience=(cloud_model.patience * 3),
        mode="max",
        verbose=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='epoch'
    )

    # if hparams['strategy'] == 'ddp':
    #     strategy = DDPPlugin(find_unused_parameters=False)
    # else:
    #     strategy = hparams['strategy']

    strategy = hparams['strategy']

    # Train model
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
        max_epochs=hparams['max_epochs'],
        precision=hparams['precision'],
        strategy=strategy,
        # plugins=DDPSpawnPlugin(find_unused_parameters=False),
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=tb_logger,
        auto_lr_find=True,
    )

    # Fit the model
    trainer.fit(model=cloud_model)
    
#     # tune learning rate
#     # trainer.fit(model=cloud_model)
#     # Run learning rate finder
#     lr_finder = trainer.tuner.lr_find(cloud_model)

#     # Results can be found in
#     print(lr_finder.results)

#     # Plot with
    
#     fig = lr_finder.plot(suggest=True)
#     plt.savefig("lr_tune.png")
#     fig.show()

#     # Pick point based on plot, or get suggestion
#     new_lr = lr_finder.suggestion() 
#     print(new_lr)
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='runtime parameters')
    
    # Data and IO
    parser.add_argument("--bands", nargs='+' , default=["B02", "B03", "B04", "B08"],
                        help="bands desired")
    
    parser.add_argument("--bands_new", nargs='+', default=None,
                        help="additional bands to use beyond original four")
    
    parser.add_argument("-cv", "--cross_validation_split", type=int, default=0,
                        help="cross validation split to use for training") 

    parser.add_argument("--OUTPUT_DIR", type=str, default='../trained_models/',
                        help="Directory to save logs and trained models model")
                                      
    parser.add_argument("--LOG_DIR", type=str, default='logs/',
                        help="Sub-directory of OUTPUT_DIR to save logs")
    
    parser.add_argument("--MODEL_DIR", type=str, default='model/',
                        help="Sub-directory of OUTPUT_DIR to save logs")

    parser.add_argument("--seed", type=int , default=13579,
                        help="random seed for train test split")
   
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
   

    # Training (gpus, optimization, etc...)
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU")
    
    parser.add_argument("--strategy", type=str, default='ddp',
                        help="Distributed training strategy")
        
    parser.add_argument("--test_run", action="store_true",
                        help="Subsample training and validation data")
    
    parser.add_argument("--test_run_nchips", type=int, default=512,
                        help="Subsample training and validation data to this size")

    parser.add_argument("--num_workers", type=int, default=12,
                        help="number of data loader workers")
    
    parser.add_argument("--persistent_workers", action="store_false",
                        help="Persistent data loader workers")
    
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for model training")
    
    parser.add_argument("--loss_function", type=str, default='bce',
                        help="loss_function to use", choices=['bce', 'dice', 'jaccard', 'focal'])
      
    parser.add_argument("-lr", "--learning_rate", type=float, default=2e-3,
                        help="Learning rate for model optimization")
    
    parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4,
                        help="Learning rate for model optimization")
    
    parser.add_argument("--optimizer", type=str, default='ADAM',
                        help="Optimizer to use", choices=['ADAM', 'SGD', 'ADAMW'])
    
    parser.add_argument("--scheduler", type=str, default='plateau',
                        help="Learning rate scheduler to use", choices=['plateau', 'exponential', 'cosine'])
    
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of warmup epochs, linear from lr=0 to lr=lr")
    
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Max number of epochs for cosine scheduler")
                              
    parser.add_argument("--plot_validation_images", action="store_false",
                        help="Plot final batch to tensorboard")
                      
    # Models and Augmentations
    parser.add_argument("--segmentation_model", type=str, default='unet',
                        help="Encocoder architecture to use", choices=['unet', 'DeepLabV3Plus'])
  
    parser.add_argument("--encoder_name", type=str, default='resnet18',
                        help="Encocoder architecture to use", choices=['efficientnet-b0','efficientnet-b3','efficientnet-b5',
                                                                       'resnet18', 'resnet34', 'resnet50',
                                                                       'vgg19_bn', 'tu-xception65',
                                                                      'tu-efficientnetv2_m'])
    
    parser.add_argument("--encoder_depth", type=int, default=5,
                        help="Encoder depth")
    
    parser.add_argument("--weights", type=none_or_str, default=None,
                        help="Pretrained_weights architecture to use")
    
    parser.add_argument("--decoder_attention_type", type=none_or_str, default=None,
                        help="Attention in decoder")
    
    parser.add_argument("--augmentations", type=str, default='vfhfrrtrrcgdbr',
                        help="training augmentations to use")
    
    parser.add_argument("--cloud_augmentations", type=str, default='vfhfrrtrrcgdbr',
                        help="training augmentations to use for cloudmix")
        
    parser.add_argument("--cloud_augment", action="store_true",
                        help="Use cloud augmentation")

    parser.add_argument("--aug_prob_soft", type=float, default=0.5,
                        help="Probability of performing simple rotation augmentations")
    
    parser.add_argument("--aug_prob_medium", type=float, default=0.8,
                        help="Probability of performing medium augmentations")
    
    parser.add_argument("--aug_prob_hard", type=float, default=0.5,
                        help="Probability of performing hard augmentations")

    parser.add_argument("--grid_distort_limit", type=float, default=0.3,
                        help="gd")
    
    parser.add_argument("--sigma_brightness", type=float, default=0.1,
                        help="gd")
    
    parser.add_argument("--mean_brightness", type=float, default=1.0,
                        help="gd")
    
    parser.add_argument("--uniform_brightness", action='store_true',
                        help="Uniform draw rather than gaussian")
    
    parser.add_argument("--cloud_sigma_brightness", type=float, default=0.1,
                        help="gd")
    
    parser.add_argument("--cloud_mean_brightness", type=float, default=1.0,
                        help="gd")
    
    parser.add_argument("--cloud_uniform_brightness", action='store_true',
                        help="Uniform draw rather than gaussian")

    
    parser.add_argument("--scale_feature_channels", type=str, default=None,
                        help="Transform from band values to others", choices=['feder_scale', 'true_color', 'log_bands', 'custom'])
    
    parser.add_argument("--custom_features", nargs='+' , default=["luminosity", "B02/B03", "B02/B04", "B02/B08", "B02/B11"],
                        help="bands desired")
    
    parser.add_argument("--load_checkpoint_path", type=str, default=None,
                        help="checkpoint path to initialize training from")

  
    args = parser.parse_args()
    
    main(args)
