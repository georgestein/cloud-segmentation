"""Useful utilities for feature-based classification."""

import numpy as np

def intersection_over_union(predictions: np.array, labels: np.array) -> float:
    """Calculate IOU over valid pixels."""
    valid_pixel_mask = labels != 255
    labels = labels.copy()[valid_pixel_mask]
    predictions = predictions.copy()[valid_pixel_mask]

    # Intersection and union totals
    intersection = np.logical_and(labels, predictions)
    union = np.logical_or(labels, predictions)

    return intersection.sum() / union.sum()

def TPR(predictions, labels):
    valid_pixel_mask = labels != 255
    labels = labels.copy()[valid_pixel_mask]
    predictions = predictions.copy()[valid_pixel_mask]
    return (predictions & labels).sum()/labels.sum()
