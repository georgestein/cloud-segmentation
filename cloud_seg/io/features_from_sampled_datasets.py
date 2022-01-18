from .features import sample_compiled_images
import random
import os
import numpy as np
import random
import subprocess

NCHIPS_TRAIN = 11600
NCHIPS_VAL = 178
PIX_SAMPLED_PER_IMAGE = 2048
IMAGE_URL = 'https://portal.nersc.gov/project/cusp/ssl_galaxy_surveys/random/'
RANDOM_SEED = 0

def create_compiled_dataset(bands, download=False):
    nfeatures = len(bands)
    nfiles = NCHIPS_TRAIN//100

    train_features = np.zeros((nfiles, 100*PIX_SAMPLED_PER_IMAGE, nfeatures))
    train_labels = np.zeros((nfiles, 100*PIX_SAMPLED_PER_IMAGE))

    random.seed(RANDOM_SEED)

    for i in range(nfiles):

        label_name = f'labels_{i*100:06d}_{(i+1)*100:06d}.npy'

        if download:
            download_file(label_name)

        image_names = []
        for band in bands:
            image_name = f'{band}_{i*100:06d}_{(i+1)*100:06d}.npy'
            if download:
                download_file(image_name)
            image_names += [image_name]

        features_tmp, labels_tmp = sample_compiled_images(image_names, label_name, PIX_SAMPLED_PER_IMAGE)
        train_features[i, ...] = features_tmp
        train_labels[i, ...] = labels_tmp

        if download:
            for image_name in image_names:
                os.remove(image_name)
            os.remove(label_name)

    train_features = train_features.reshape(-1, nfeatures)
    train_labels = train_labels.reshape(-1)

    filename = f"{'_'.join(bands)}_seed{RANDOM_SEED}"
    np.save(f'train_features_{filename}.npy', train_features)
    np.save(f'train_labels_{filename}.npy', train_labels)

def download_file(file_name: str) -> None:
    """Download file to local directory."""
    p = subprocess.Popen(
        ['wget', f'{IMAGE_URL}{file_name}'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()
    if stderr is not None:
        print(stderr)
