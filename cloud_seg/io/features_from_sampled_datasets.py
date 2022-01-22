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
BAD_CHIP_PATH = './cloud-segmentation/data/BAD_CHIP_DATA/BAD_CHIP_LABEL_IDS.txt'
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

def create_chip_mask(bad_chip_path: str=BAD_CHIP_PATH, download: bool=False):
    """Create a msk of bad chips."""
    bad_chips = get_bad_chips(bad_chip_path)

    is_bad_chip = np.zeros((NCHIPS_TRAIN//100, 100), dtype=np.uint8)

    for i in range(NCHIPS_TRAIN//100):

        chip_id_name = f'chip_ids_{i*100:06d}_{(i+1)*100:06d}.npy'
        if download:
            download_file(chip_id_name)
        else:
            chip_id_name = DATA_DIR/chip_id_name

        chip_ids = np.load(chip_id_name)
        for j, chip_id in enumerate(chip_ids):
            if chip_id in bad_chips:
                is_bad_chip[i, j] = 1

        if download:
            os.remove(chip_id_name)

    is_bad_chip = np.tile(is_bad_chip, (1, 1, PIX_SAMPLED_PER_IMAGE))
    is_bad_chip = is_bad_chip.flatten()

    np.save(f'train_features_bad_chip_mask.npy', is_bad_chip)

def get_bad_chips(bad_chip_path):
    """Get the list of bad chips."""
    with open(bad_chip_path) as f:
        bad_chips = f.readlines()
    bad_chips = [bad_chip.strip('\n') for bad_chip in bad_chips]
    return bad_chips
