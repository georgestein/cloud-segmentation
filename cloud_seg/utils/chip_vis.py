"""Read and visualize a single chip."""

import os
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
import rioxarray
import xrspatial.multispectral as ms
import matplotlib.pyplot as plt

BANDS = ['B02', 'B03', 'B04', 'B08']
NPIX = 512

class ImageChip():
    """Image data from multiple bands on a single chip."""

    def __init__(self, chip: str, data_dir: os.PathLike):
        """Read in the features and labels for a single chip."""
        self.chip = chip
        self.feature_dir = data_dir/'train_features'
        self.label_dir = data_dir/'train_labels'

        self.image_array = np.zeros((NPIX, NPIX, len(BANDS)), dtype=np.uint16)
        self.meta = dict({})
        self.bounds = dict({})

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

        try:
            image, meta, bounds = self.load_labels()
        except RasterioIOError:
            self.labels = None
        else:
            self.labels = image

    def load_features(self, band: str) -> tuple:
        """Read in features."""
        feature_path = self.feature_dir/self.chip/f'{band}.tif'
        return read_geotiff(feature_path)

    def load_labels(self) -> tuple:
        """Read in labels."""
        label_path = self.label_dir/f'{self.chip}.tif'
        return read_geotiff(label_path)

    def plot_true_colour(self) -> plt.figure:
        """Make a true-colour plot of the features."""
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax[0].imshow(self.true_colour())
        ax[0].imshow(self.labels)
        fig.subplots_adjust(hspace=0, wspace=0)
        return fig

    def plot_all_bands(self) -> plt.figure:
        """Plot the four features."""
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
