"""Useful utilities for feature-based classification."""

import os
import numpy as np
import random
from pathlib import Path

NPIX = 512
NFEATURES = 4
IMAGE_BANDS = ['B02', 'B03', 'B04', 'B08']
ALL_BANDS = ['B02', 'B03', 'B04', 'B08',
             'B05', 'B06', 'B07','B09',
             'B8A', 'B11', 'B12', 'B01',
             'SCL', 'AOT', 'LC']
NBANDS_PER_FILE = 4
PIX_PER_IMAGE = NPIX*NPIX
DATA_DIR = Path('.')

def load_dataset(name: str, data_dir: os.PathLike=DATA_DIR, max_images: int=None):
    """Load compiled datasets of images and labels.

    Flattens the features to have shape (-1, nfeatures), and
    flattens the labels completely.
    """
    features = np.load(data_dir/f'compiled_images_{name}.npy')
    labels = np.load(data_dir/f'compiled_labels_{name}.npy')

    if max_images is not None:
        features = features[:max_images, ...]
        labels = labels[:max_images]

    nfeatures = features.shape[-1]
    assert nfeatures == NFEATURES
    assert features.shape[1] == NPIX
    assert features.shape[2] == NPIX
    features = features.reshape(-1, nfeatures)

    assert labels.shape[1] == NPIX
    assert labels.shape[2] == NPIX
    labels = labels.flatten()

    feature_names = IMAGE_BANDS.copy()

    return features, labels, feature_names

def add_intensity(features: np.ndarray, feature_names: list) -> tuple:
    """Add intensity to features."""
    newfeature = np.zeros((features.shape[0]), dtype=features.dtype)
    for band in IMAGE_BANDS:
        idx = feature_names.index(band)
        newfeature += features[..., idx]

    newfeature = newfeature/len(IMAGE_BANDS)

    features = np.concatenate((features, newfeature[..., np.newaxis]), axis=-1)
    feature_names += ['I']

    return features, feature_names

def add_logintensity(features: np.ndarray, feature_names: list) -> tuple:
    """Add intensity to features."""
    newfeature = np.zeros((features.shape[0]), dtype=features.dtype)
    for band in IMAGE_BANDS:
        idx = feature_names.index(band)
        newfeature += features[..., idx]

    newfeature = newfeature/len(IMAGE_BANDS)
    newfeature = np.log10(newfeature)

    features = np.concatenate((features, newfeature[..., np.newaxis]), axis=-1)
    feature_names += ['logI']

    return features, feature_names

def add_logbands(features: np.ndarray, feature_names: list) -> tuple:
    """Add intensity to features."""
    for band in IMAGE_BANDS:
        idx = feature_names.index(band)
        newfeature = np.log10(features[..., idx])

        features = np.concatenate((features, newfeature[..., np.newaxis]), axis=-1)
        feature_names += [f'log{band}']

    return features, feature_names

def sample_compiled_images(image_paths, label_path, npix):
    """Sample `npix` pixels from each image in a compiled dataset."""
    pixels_to_sample = get_pixels_to_sample(npix)

    nfeatures = len(image_paths)

    labels = np.load(label_path)
    
    images = None
    for i, image_path in enumerate(image_paths):
        image = np.load(image_path)
        if images is None:
            nimages, npixx, npixy = image.shape
            images = np.zeros((nimages, npixx, npixy, nfeatures), dtype=image.dtype)
        images[..., i] = image
    
    sampled_features = np.zeros((nimages, npix, nfeatures), dtype=images.dtype)
    sampled_labels = np.zeros((nimages, npix), dtype=labels.dtype)
    for i, pixidx in enumerate(pixels_to_sample):
        idxx, idxy = np.unravel_index(pixidx, (NPIX, NPIX))
        sampled_features[:, i, :] = images[:, idxx, idxy, :]
        sampled_labels[:, i] = labels[:, idxx, idxy]

    return sampled_features.reshape(-1, nfeatures), sampled_labels.reshape(-1)

def get_pixels_to_sample(npix_to_sample: int) -> list:
    pixels_to_sample = []
    npix_sampled = 0
    while npix_sampled < npix_to_sample:
        next_pix = random.randrange(PIX_PER_IMAGE)
        if next_pix not in pixels_to_sample:
            pixels_to_sample += [next_pix]
            npix_sampled += 1
    return pixels_to_sample

def get_band(band: str, validation: bool=False, name: str=None, data_dir: os.PathLike=DATA_DIR) -> np.ndarray:
    if not validation:
        return get_train_band(band, data_dir=data_dir)
    else:
        return get_validation_band(band, name, data_dir=data_dir)

def get_train_band(band: str, data_dir: os.PathLike=DATA_DIR) -> np.ndarray:
    band_idx = ALL_BANDS.index(band)
    file_idx = band_idx//NBANDS_PER_FILE
    file_bands = ALL_BANDS[file_idx*NBANDS_PER_FILE:(file_idx+1)*NBANDS_PER_FILE]
    
    feature = np.load(data_dir/f"train_features_{'_'.join(file_bands)}_seed0.npy")[:, file_bands.index(band)]

    return feature

def get_validation_band(band: str, name: str, data_dir: os.PathLike=DATA_DIR) -> np.ndarray:
    feature = np.load(data_dir/f"{band}_{name}.npy").reshape(-1)
    return feature

def generate_colour_difference(band1: str, band2: str, validation: bool=False, name: str=None, data_dir: os.PathLike=DATA_DIR) -> np.ndarray:
    feature1 = get_band(band1, validation, name, data_dir)
    feature2 = get_band(band2, validation, name, data_dir)

    colour = (feature1-feature2) / (feature1+feature2)

    # Protect against nan's
    colour[colour==np.inf] = 0.
    colour[colour==-1*np.inf] = 0.
    colour[colour==np.nan] = 0.
    colour[colour!=colour] = 0.

    return colour

def generate_colour_ratio(band1: str, band2: str, validation: bool=False, name: str=None, data_dir: os.PathLike=DATA_DIR) -> np.ndarray:
    feature1 = get_band(band1, validation, name, data_dir)
    feature2 = get_band(band2, validation, name, data_dir)

    colour = feature1/feature2

    colour[np.abs(feature1) < 1e-5] = 0.
    colour[(np.abs(feature2) < 1e-5) & (feature1 > 0)] = 100
    colour[(np.abs(feature2) < 1e-5) & (feature1 < 0)] = -100

    # Protect against nan's
    colour[colour==np.inf] = 0.
    colour[colour==-1*np.inf] = 0.
    colour[colour==np.nan] = 0.
    colour[colour!=colour] = 0.

    return colour

class Features():
    def __init__(self, set_type: str='train', file_name: str=None, data_dir=DATA_DIR):
        assert set_type in ['train', 'val']
        if set_type == 'train':
            self.npixels = 23756800
        else:
            self.npixels = 26214400
            assert file_name is not None
        
        self.value = np.zeros((self.npixels, 0))
        self.names = []
        self.set_type = set_type
        self.validation = set_type == 'val'
        self.file_name = file_name
        self.nfeatures = 0
        self.data_dir = data_dir
        
    def add(self, feature: str):
        if '-' in feature:
            new_feature = generate_colour_difference(*feature.split('-'), self.validation, self.file_name, self.data_dir)
        elif '/' in feature:
            new_feature = generate_colour_ratio(*feature.split('/'), self.validation, self.file_name, self.data_dir)
        else:
            new_feature = get_band(feature, self.validation, self.file_name, self.data_dir)
        
        self.value = np.concatenate(
            (self.value, new_feature[:, np.newaxis]), axis=-1)
        self.names += [feature]
        self.nfeatures += 1
    
    def get_value_for(self, feature: str) -> np.ndarray:
        return self.value[:, self.names.index(feature)]

    def get_values(self) -> np.ndarray:
        return self.value
