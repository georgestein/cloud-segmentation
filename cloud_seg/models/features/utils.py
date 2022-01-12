"""Useful utilities for feature-based classification."""

import os
import numpy as np

NPIX = 512
NFEATURES = 4
BANDS = ['B02', 'B03', 'B04', 'B08']

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

    return features, labels

def intersection_over_union(predictions: np.array, labels: np.array) -> float:
    """Calculate IOU over valid pixels."""
    valid_pixel_mask = labels != 255
    labels = labels.copy()[valid_pixel_mask]
    predictions = predictions.copy()[valid_pixel_mask]

    # Intersection and union totals
    intersection = np.logical_and(labels, predictions)
    union = np.logical_or(labels, predictions)

    return intersection.sum() / union.sum()
