"""Useful utilities for feature-based classification."""

import os
import numpy as np

NPIX = 512
NFEATURES = 4
BANDS = ['B02', 'B03', 'B04', 'B08']

PIX_PER_IMAGE = NPIX*NPIX

def load_dataset(name: str, data_dir: os.PathLike, max_images: int=None):
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

    feature_names = BANDS.copy()

    return features, labels, feature_names

def add_spatial_features(features, feature_names):
    raise NotImplementedError

def add_colour_differences(features, feature_names):
    """Add colours (normalized differences) to features."""
    for i, band1 in enumerate(BANDS[:-1]):
        for band2 in BANDS[i+1:]:
            idx1 = feature_names.index(band1)
            idx2 = feature_names.index(band2)
            newfeature = (
                (features[..., idx1]-features[..., idx2]) /
                (features[..., idx1]+features[..., idx2]))
            features = np.concatenate((features, newfeature[:, np.newaxis]), axis=-1)
            feature_names += [f'{band1}-{band2}']

    return features, feature_names

def add_colour_ratios(features, feature_names):
    """Add colours (ratios) to features."""
    for i, band1 in enumerate(BANDS[:-1]):
        for band2 in BANDS[i+1:]:
            idx1 = feature_names.index(band1)
            idx2 = feature_names.index(band2)
            newfeature = (features[..., idx1]/features[..., idx2])
            features = np.concatenate((features, newfeature[:, np.newaxis]), axis=-1)
            feature_names += [f'{band1}/{band2}']

    return features, feature_names

def add_intensity(features, feature_names):
    """Add intensity to features."""
    newfeature = np.zeros((features.shape[0]), dtype=features.dtype)
    for band in BANDS:
        idx = feature_names.index(band)
        newfeature += features[..., idx]

    newfeature = newfeature/len(BANDS)

    features = np.concatenate((features, newfeature[..., np.newaxis]), axis=-1)
    feature_names += ['I']

    return features, feature_names

def add_logintensity(features, feature_names):
    """Add intensity to features."""
    newfeature = np.zeros((features.shape[0]), dtype=features.dtype)
    for band in BANDS:
        idx = feature_names.index(band)
        newfeature += features[..., idx]

    newfeature = newfeature/len(BANDS)
    newfeature = np.log10(newfeature)

    features = np.concatenate((features, newfeature[..., np.newaxis]), axis=-1)
    feature_names += ['logI']

    return features, feature_names

def add_logbands(features, feature_names):
    """Add intensity to features."""
    for band in BANDS:
        idx = feature_names.index(band)
        newfeature = np.log10(features[..., idx])

        features = np.concatenate((features, newfeature[..., np.newaxis]), axis=-1)
        feature_names += [f'log{band}']

    return features, feature_names

def sample_compiled_images(image_path, label_path, npix):
    """Sample `npix` pixels from each image in a compiled dataset."""
    pixels_to_sample = get_pixels_to_sample(npix)
    images = np.load(image_path)
    labels = np.load(label_path)
    nimages, nfeatures, npixx, npixy = images.shape
    sampled_features = np.zeros((nimages, npix, nfeatures), dtype=images.dtype)
    sampled_labels = np.zeros((nimages, npix), dtype=labels.dtype)
    for i, pixidx in enumerate(pixels_to_sample):
        idxx, idxy = np.unravel_index(pixidx, (NPIX, NPIX))
        sampled_features[:, i, :] = images[:, :, idxx, idxy]
        sampled_labels[:, i] = labels[:, idxx, idxy]
    return sampled_features.reshape(-1, nfeatures), sampled_labels

def get_pixels_to_sample(npix_to_sample):
    pixels_to_sample = []
    npix_sampled = 0
    while npix_sampled < npix_to_sample:
        next_pix = random.randrange(PIX_PER_IMAGE)
        if next_pix not in pixels_to_sample:
            pixels_to_sample += [next_pix]
            npix_sampled += 1
    return pixels_to_sample
