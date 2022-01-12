"""Read and visualize a single chip."""

import os
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
import rioxarray
import xrspatial.multispectral as ms
import matplotlib.pyplot as plt

from . import metrics

BANDS = ['B02', 'B03', 'B04', 'B08']
NPIX = 512


class ImageChip():
    """Image data from multiple bands on a single chip."""

    def __init__(self, chip: str, data_dir: os.PathLike, prediction_dir: None or os.PathLike,
                 load_features: bool=True, load_labels: bool=True, load_prediction: bool=False):
        """Read in the features and labels for a single chip."""
        self.chip = chip
        self.feature_dir = data_dir/'train_features'
        self.label_dir = data_dir/'train_labels'
        self.prediction_dir = prediction_dir
        
        band_to_ind = {k: v for v, k in enumerate(BANDS)}
            
        self.meta = dict({})
        self.bounds = dict({})

        self.image_array = None
        self.labels = None
        self.preds = None
        if load_features:
            self.image_array = np.zeros((NPIX, NPIX, len(BANDS)), dtype=np.uint16)
            
            for iband, band in enumerate(BANDS):
                try:
                    image, meta, bounds = self.load_features(band)
                except RasterioIOError:
                    self.image_array[..., iband] = None
                    self.meta[band] = None
                    self.bounds[band] = None
                else:
                    self.image_array[..., iband] = image
                    self.meta[band] = meta
                    self.bounds[band] = bounds

        if load_labels:
            try:
                image, meta, bounds = self.load_labels()
            except RasterioIOError:
                self.labels = None
            else:
                self.labels = image

        if load_prediction and prediction_dir is not None:
            try:
                image, meta, bounds = self.load_prediction()
            except RasterioIOError:
                self.preds = None
            else:
                self.preds = image
            
    def load_features(self, band: str) -> tuple:
        """Read in features."""
        feature_path = self.feature_dir/self.chip/f'{band}.tif'
        return read_geotiff(feature_path)

    def load_labels(self) -> tuple:
        """Read in labels."""
        label_path = self.label_dir/f'{self.chip}.tif'
        return read_geotiff(label_path)

    def load_prediction(self) -> tuple:
        """Read in predictions."""
        label_path = self.prediction_dir/f'{self.chip}.tif'
        return read_geotiff(label_path)

    def plot_true_colour(self) -> plt.figure:
        """Make a true-colour plot of the features."""
        nx = 1
        if self.labels is not None:
            nx += 1
        if self.preds is not None:
            nx += 1
            
        fig, ax = plt.subplots(1, nx, figsize=(nx*4, 4),
                               sharex=True, sharey=True, squeeze=False)

        # True color
        ax[0, 0].imshow(self.true_colour())
        ax[0, 0].set_title(f"chip: {self.chip}")
        
        # Label panel
        if self.labels is not None:
            ax[0, 1].imshow(self.labels, vmin=0, vmax=1, interpolation='none')
            ax[0, 1].set_title(f"True label")
            
        # Prediction panel
        if self.preds is not None:
            pred_title_string = "Pred"
            if self.labels is not None:
                self.intersection, self.union = metrics.intersection_and_union(self.preds, self.labels)
                self.IoU = self.intersection/self.union
                pred_title_string += f": Int.={self.intersection:.2f}, Un.={self.union:.2f}, IoU={self.IoU:.3f}"
                
            ax[0, 2].imshow(self.preds, vmin=0, vmax=1, interpolation='none')
            ax[0, 2].set_title(pred_title_string)
            

        fig.subplots_adjust(hspace=0, wspace=0)
        return fig

    def plot_all_bands(self) -> plt.figure:
        """Plot the features."""
        fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
        ax[0, 0].imshow(self.image_array[..., 0])
        ax[0, 1].imshow(self.image_array[..., 1])
        ax[1, 0].imshow(self.image_array[..., 2])
        ax[1, 1].imshow(self.image_array[..., 3])
        ax[0, 2].imshow(self.labels)
        ax[1, 2].imshow(self.true_colour())
        fig.subplots_adjust(hspace=0, wspace=0)
        return fig

    def true_colour(self) -> np.array:
        """Get the true-colour representation of the chip."""
        chipdir = self.feature_dir/self.chip
        red = rioxarray.open_rasterio(chipdir/"B04.tif").squeeze()
        green = rioxarray.open_rasterio(chipdir/"B03.tif").squeeze()
        blue = rioxarray.open_rasterio(chipdir/"B02.tif").squeeze()

        return ms.true_color(r=red, g=green, b=blue)

def read_geotiff(filepath: str) -> tuple:
    """Read an image, along with metadata and bounds, from geotiff."""
    with rasterio.open(filepath) as f:
        meta = f.meta
        bounds = f.bounds
        image = f.read(1)
    return image, meta, bounds
