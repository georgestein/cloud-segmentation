import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .cloud_mlp import LitMLP

def train_mlp(image_cloudless, image_cloudy, cloud_label, params, cloudfrac_max=0.9, sigma_smooth=10):
    """
    Trains an MLP on non-cloud pixels to map cloudless_image to cloudy_image
    """
    
    val_frac = 0.25
    max_epochs = 10
    learning_rate = 5e-2
    batch_size = 4096

    cloud_use = gaussian_filter(cloud_label.astype(np.float32), sigma=sigma_smooth)
    cloud_use = ((cloud_use > 0.05)*1).astype(np.uint8)
    
    if np.mean(cloud_use) > cloudfrac_max:
        return image_cloudless
    
    # images come in as list of arrays, so convert to array and flatten
    x = np.stack([v for k,v in sorted(image_cloudless.items())], -1)
    y = np.stack([v for k,v in sorted(image_cloudy.items())], -1)

    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    dm = cloud_use.reshape(-1)


    x_train = torch.tensor(x[dm==0])
    y_train = torch.tensor(y[dm==0])

    model = LitMLP(
        x_train,
        y_train,
        batch_size=batch_size,
        val_frac=val_frac,
        learning_rate=learning_rate,
    )

    pl.seed_everything(13579)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="min")
    
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=max_epochs,
        callbacks=[early_stop_callback],
    )
    trainer.fit(model)

    x = torch.tensor(x)
    matched_cloudless_image = model(x).detach().numpy().reshape(512, 512, -1)
    
    # convert from array back to list
    cloudless_image = {}
    for iband, band in enumerate(params['bands_use']):
        cloudless_image[band] = matched_cloudless_image[..., iband]
            
    return cloudless_image

def find_and_return_most_similar_image(params, image, label, images_cloudless, brightness_correct_model=None):
    """Given a cloudy chip and a number of cloudless versions of the same area, choose or create 
    the most similar one to the cloudy chip.
    
    The simple approximation is to just calculate which set of images best matches in regions where labels==0. 
    This does not accound for shadows.
    """
    
     # determine which new cloudless image is most similar to the old
     # by calculating agreement in non-cloudy regions
    diffs = np.zeros( (len(params['bands_use']), images_cloudless['B02'].shape[0]) )
    for i, band in enumerate(params['bands_use']):

        diff = (image[band]-images_cloudless[band]) * label
        diffs[i] = np.sum(diff, axis=(1,2))

        if diffs[i].max() > 0.:
            # if totally cloud covered label==0 everywhere, and max will be 0.
            diffs[i] /= diffs[i].max()

    total_diffs = np.mean(diffs, axis=0)

    ind_min_band_diff = np.argmin(total_diffs)  
    
    image_cloudless = {}
    for band in params['bands_use']:
        image_cloudless[band] = images_cloudless[band][ind_min_band_diff]
    
        if brightness_correct_model=='median':
            # try to match the average intensity in non cloudy regions
            dm = label == 0

            if np.sum(dm) > 0:
                mean_diff = np.median(image[band][dm] - image_cloudless[band][dm])
            else:
                mean_diff = 1.

            # print('mean_diff', mean_diff)
            images_cloudless[band] += mean_diff

    if brightness_correct_model=='mlp':
            
        if np.mean(label) > 0.:
            # Train MLP on non-cloudy portions of images, to better match cloudless to cloudy
            image_cloudless = train_mlp(image_cloudless, image, label, params, cloudfrac_max=0.9)  
        
     
    return image_cloudless

def extract_clouds(params, image, label, images_cloudless, cloud_extract_model='opacity', brightness_correct_model=None):
    """Given cloudy image/label pair, and 'cloudless' images of the same area pulled from the planetary computer,
    extract brightness changes due to clouds.
    
    First find "most similar" cloudless image to the cloudy one
    
    The simplest model is to assume clouds simply add brightness to each pixel that they cover. 
    If true, assuming that the land does not change between when the cloudy and cloudless images were taken,
    clouds = (images - images_cloudless)*labels.
    
    Unfortunately, both of these assumptions are incorrect
    
    1.) The cloudy and cloudless images are of the same location, but are often seperated by months or years.
        Over this timeframe plants change color, water levels change, and human infrastructure near cities changes.
        Additionally, the images might not be taken from the same angle, causing mis-alignments between each image set.
        
    2.) Clouds are sometimes transparent, sometimes not. An additive model does not correcely account for this
    
    3.) Cloud shadows... We know what angle the sun makes for each chip (in chip properties) can we come up with a way to project these?
    
    """
    cloud_extract_models = ['additive', 'opacity'] # Add transparency later
    
    if cloud_extract_model not in cloud_extract_models:
        print(f"WARNING: cloud model {cloud_extract_model} is not a possible value to use. Using {cloud_extract_models[0]} instead \
            Possible choices are:", cloud_extract_models)
        
    image_cloudless = find_and_return_most_similar_image(params, image, label, images_cloudless, brightness_correct_model=brightness_correct_model)   
    
    # and save to disk as .tif
    clouds = {}

    if cloud_extract_model=='additive':
        
        for band in params['bands_use']:
            clouds[band] = (image[band] - image_cloudless[band]) 

        opacity_mask = np.zeros_like(clouds[band], dtype=np.uint8)
        
    if cloud_extract_model=='opacity':
        # calculate per pixel luminence
        min_opacity_luminence = 5000.
        
        luminence = np.mean(
            np.stack(
                [image['B02'],image['B03'],image['B04'], image['B08']],
                axis=-1,
            ),
            axis=-1,
        )
        
        opacity_mask = luminence > min_opacity_luminence
        for band in params['bands_use']:
            clouds_in_band = np.zeros_like(image['B02'], dtype=image['B02'].dtype)

            clouds_in_band[opacity_mask] = image[band][opacity_mask] 
            clouds_in_band[~opacity_mask] = (image[band] - image_cloudless[band])[~opacity_mask] 

            clouds[band] = clouds_in_band
        
    return image_cloudless, clouds, ((opacity_mask > 0.5)*1).astype(np.uint8)


    
