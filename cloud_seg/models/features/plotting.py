"""Plot results of feature-based learning."""

import random
import numpy as np
import matplotlib.pyplot as plt
import xrspatial.multispectral as ms
from cloud_seg.utils.utils import to_xarray

NPIX = 512
NFEATURES = 4
BANDS = ['B02', 'B03', 'B04', 'B08']

def plot_validation(
    features: np.ndarray, labels: np.array, predictions: np.array,
    nplot: int=4) -> plt.figure:
    """Plot a random selection from the validation set.

    Features and labels can be in (-1, nfeatures) and (-1) shape,
    and will be reshaped before plotting.
    """

    assert len(predictions) == len(labels)

    predictions = predictions.reshape(-1, NPIX, NPIX)
    labels = labels.reshape(-1, NPIX, NPIX)
    features = features.reshape(-1, NPIX, NPIX, NFEATURES)
    nimages = predictions.shape[0]

    images_to_plot = random.sample(range(nimages), nplot)

    fig, ax = plt.subplots(nplot, 3, figsize=(12, 4*nplot))
    for pidx, imidx in enumerate(images_to_plot):
        full_colour_image = get_full_colour(features[imidx])
        for a in ax[pidx, :2]:
            a.imshow(full_colour_image)
        ax[pidx, 0].imshow(
            predictions[imidx], vmin=0, vmax=1, cmap=plt.get_cmap('Reds'),
            alpha=0.5*predictions[imidx])
        ax[pidx, 1].imshow(
            labels[imidx], vmin=0, vmax=1, cmap=plt.get_cmap('Reds'),
             alpha=0.5*labels[imidx])
        ax[pidx, 2].imshow(predictions[imidx], vmin=0, vmax=1, cmap=plt.get_cmap('Greys_r'))
        ax[pidx, 2].contour(labels[imidx], vmin=0, vmax=1, cmap=plt.get_cmap('Reds'))
        ax[pidx, 0].set_title('Prediction')
        ax[pidx, 1].set_title('Label')

    return fig

def get_full_colour(features: np.ndarray) -> ms.true_color:
    """Get the true colour image from features."""
    red = to_xarray(features[..., BANDS.index('B04')])
    green = to_xarray(features[..., BANDS.index('B03')])
    blue = to_xarray(features[..., BANDS.index('B02')])
    return ms.true_color(r=red, g=green, b=blue)
