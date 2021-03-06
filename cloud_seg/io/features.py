"""Useful utilities for feature-based classification."""

import os
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

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

PIX_PER_IMAGE_INDS = np.arange(PIX_PER_IMAGE)

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

def sample_compiled_images(image_paths: list, label_path: str,
                           num_pixels_per_image: list, smooth_sigma: float) -> np.ndarray:
    """Sample `npix` pixels from each image in a compiled dataset."""
    nfeatures = len(image_paths)
    labels = np.load(label_path)

    images = None
    for i, image_path in enumerate(image_paths):
        image = np.load(image_path)
        if smooth_sigma is not None:
            image = gaussian_filter(image, smooth_sigma)
        if images is None:
            nimages, npixx, npixy = image.shape
            images = np.zeros((nimages, npixx, npixy, nfeatures), dtype=image.dtype)
        images[..., i] = image

    sampled_features = np.zeros((0, nfeatures), dtype=images.dtype)
    sampled_labels = np.zeros((0), dtype=labels.dtype)

    for image_idx, npix in enumerate(num_pixels_per_image):
        pixels_to_sample = get_pixels_to_sample(npix)
        tmp_features = np.zeros((npix, nfeatures), dtype=images.dtype)
        tmp_labels = np.zeros((npix), dtype=labels.dtype)

        for i, pixidx in enumerate(pixels_to_sample):
            idxx, idxy = np.unravel_index(pixidx, (NPIX, NPIX))

            tmp_features[i, :] = images[image_idx, idxx, idxy, :]
            tmp_labels[i] = labels[image_idx, idxx, idxy]

        sampled_features = np.concatenate((sampled_features, tmp_features), axis=0)
        sampled_labels = np.concatenate((sampled_labels, tmp_labels))

    return sampled_features, sampled_labels

def get_pixels_to_sample(npix_to_sample: int) -> list:

    pixels_to_sample = np.random.choice(PIX_PER_IMAGE_INDS, size=npix_to_sample, replace=False)

    return list(pixels_to_sample)

def get_band(band: str, validation: bool=False, name: str=None,
             data_dir: os.PathLike=DATA_DIR) -> np.ndarray:
    if not validation:
        return get_train_band(band, data_dir=data_dir)
    return get_validation_band(band, name, data_dir=data_dir)

def get_train_band(band: str, data_dir: os.PathLike=DATA_DIR) -> np.ndarray:
    band_idx = ALL_BANDS.index(band)
    file_idx = band_idx//NBANDS_PER_FILE
    file_bands = ALL_BANDS[file_idx*NBANDS_PER_FILE:(file_idx+1)*NBANDS_PER_FILE]

    feature = np.load(
        data_dir/f"train_features_{'_'.join(file_bands)}_seed0.npy"
        )[:, file_bands.index(band)]

    return feature

def get_validation_band(band: str, name: str,
                        data_dir: os.PathLike=DATA_DIR) -> np.ndarray:
    feature = np.load(data_dir/f"{band}_{name}.npy").reshape(-1)
    return feature

def generate_colour_difference(band1: str, band2: str, validation: bool=False, name: str=None,
                               data_dir: os.PathLike=DATA_DIR) -> np.ndarray:
    feature1 = get_band(band1, validation, name, data_dir)
    feature2 = get_band(band2, validation, name, data_dir)

    colour = (feature1-feature2) / (feature1+feature2)

    # Protect against nan's
    colour[colour==np.inf] = 0.
    colour[colour==-1*np.inf] = 0.
    colour[colour==np.nan] = 0.
    colour[colour!=colour] = 0.

    return colour

def generate_colour_ratio(band1: str, band2: str, validation: bool=False, name: str=None,
                          data_dir: os.PathLike=DATA_DIR) -> np.ndarray:
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
    def __init__(self, set_type: str='train', file_name: str=None, data_dir: os.PathLike=DATA_DIR):
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
            new_feature = generate_colour_difference(
                *feature.split('-'), self.validation, self.file_name, self.data_dir)
        elif '/' in feature:
            new_feature = generate_colour_ratio(
                *feature.split('/'), self.validation, self.file_name, self.data_dir)
        else:
            new_feature = get_band(feature, self.validation, self.file_name, self.data_dir)

        try:
            self.value = np.concatenate(
                (self.value, new_feature[:, np.newaxis]), axis=-1)

        except ValueError as e:
            if self.value.size > 0:
                raise e

            self.npixels = new_feature.shape[0]
            self.value = new_feature[:, np.newaxis]

        self.names += [feature]
        self.nfeatures += 1

    def get_value_for(self, feature: str) -> np.ndarray:
        return self.value[:, self.names.index(feature)]

    def get_values(self) -> np.ndarray:
        return self.value
