from .features import sample_compiled_images
import wget
import random
import os

NCHIPS_TRAIN = 11600
NCHIPS_VAL = 178
PIX_SAMPLED_PER_IMAGE = 2048
NFEATURES = 4
IMAGE_URL = 'https://portal.nersc.gov/project/cusp/ssl_galaxy_surveys/random/'
RANDOM_SEED = 0

def create_compiled_dataset():
    nfiles = NCHIPS_TRAIN//100

    train_features = np.zeros((nfiles, PIX_SAMPLED_PER_IMAGE, 4))
    train_labels = np.zeros((nfiles, PIX_SAMPLED_PER_IMAGE, 4))

    random.seed(RANDOM_SEED)

    for i in range(nfiles):
        image_name = 'images_{i:06d}_{i+1:06d}.npy'
        label_name = 'labels_{i:06d}_{i+1:06d}.npy'
        wget(f'{IMAGE_URL}{image_name}')
        wget(f'{IMAGE_URL}{image_name}')
        features_tmp, labels_tmp = sample_compiled_images(image_name, label_name, PIX_SAMPLED_PER_IMAGE)
        train_features[i, ...] = features_tmp
        train_labels[i, ...] = labels_tmp
        os.remove(image_name)
        os.remove(label_name)

    train_features = train_features.reshape(-1, NFEATURES)
    np.save(f'train_features_seed{RANDOM_SEED}.npy', train_features)
    np.save(f'train_labels_seed{RANDOM_SEED}.npy', train_labels)
