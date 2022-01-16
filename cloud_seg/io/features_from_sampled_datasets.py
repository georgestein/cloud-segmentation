from .features import sample_compiled_images
import wget
import random
import os
import numpy as np
import random
import subprocess

NCHIPS_TRAIN = 11600
NCHIPS_VAL = 178
PIX_SAMPLED_PER_IMAGE = 2048
NFEATURES = 4
IMAGE_URL = 'https://portal.nersc.gov/project/cusp/ssl_galaxy_surveys/random/'
RANDOM_SEED = 0

def create_compiled_dataset():
    nfiles = NCHIPS_TRAIN//100

    train_features = np.zeros((nfiles, 100*PIX_SAMPLED_PER_IMAGE, 4))
    train_labels = np.zeros((nfiles, 100*PIX_SAMPLED_PER_IMAGE))

    random.seed(RANDOM_SEED)

    for i in range(nfiles):
        image_name = f'images_{i*100:06d}_{(i+1)*100:06d}.npy'
        label_name = f'labels_{i*100:06d}_{(i+1)*100:06d}.npy'
        p = subprocess.Popen(
            ['wget', f'{IMAGE_URL}{image_name}'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = p.communicate()
        if stderr is not None:
            print(stderr)
        p = subprocess.Popen(
            ['wget', f'{IMAGE_URL}{label_name}'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = p.communicate()
        if stderr is not None:
            print(stderr)
        features_tmp, labels_tmp = sample_compiled_images(image_name, label_name, PIX_SAMPLED_PER_IMAGE)
        train_features[i, ...] = features_tmp
        train_labels[i, ...] = labels_tmp
        os.remove(image_name)
        os.remove(label_name)

    train_features = train_features.reshape(-1, NFEATURES)
    np.save(f'train_features_seed{RANDOM_SEED}.npy', train_features)
    np.save(f'train_labels_seed{RANDOM_SEED}.npy', train_labels)
