from collections import namedtuple
import random
import os
import subprocess
import numpy as np
from .features import sample_compiled_images


NCHIPS_TRAIN = 11600
NCHIPS_VAL = 178
PIX_SAMPLED_PER_IMAGE = 2048
IMAGE_URL = 'https://portal.nersc.gov/project/cusp/ssl_galaxy_surveys/random/'
BAD_CHIP_PATH = './cloud-segmentation/data/BAD_CHIP_DATA/BAD_CHIP_LABEL_IDS.txt'
RANDOM_SEED = 0
NUM_LC_CLASSES = 11

def get_lc_classes():
    LCClass = namedtuple('LCClass', field_names='type label num_pixels_per_image')
    lc_classes = [
        LCClass(type='nodata', label=0, num_pixels_per_image=555),
        LCClass(type='water', label=1, num_pixels_per_image=3954),
        LCClass(type='trees', label=2, num_pixels_per_image=1021),
        LCClass(type='grass', label=3, num_pixels_per_image=11278),
        LCClass(type='flooded veg', label=4, num_pixels_per_image=38419),
        LCClass(type='crops', label=5, num_pixels_per_image=3106),
        LCClass(type='scrub', label=6, num_pixels_per_image=767),
        LCClass(type='built area', label=7, num_pixels_per_image=11764),
        LCClass(type='bare', label=8, num_pixels_per_image=25780),
        LCClass(type='snow/ice', label=9, num_pixels_per_image=262144),
        LCClass(type='clouds', label=10, num_pixels_per_image=247126)]
    return lc_classes

def create_compiled_dataset(bands, output_str, sample_by_LC=True, download=False, smooth_sigma=None):
    nfeatures = len(bands)
    nfiles = NCHIPS_TRAIN//100

    train_features = np.zeros((nfiles*100*PIX_SAMPLED_PER_IMAGE, nfeatures))
    train_labels = np.ones((nfiles*100*PIX_SAMPLED_PER_IMAGE))*255

    lc_classes = get_lc_classes()

    random.seed(RANDOM_SEED)

    current_idx = 0
    max_idx = len(train_labels)
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

        if sample_by_LC:
            LC_name = f'LC_{i*100:06d}_{(i+1)*100:06d}.npy'
            if download:
                download_file(LC_name)

            dominant_LCs = get_dominant_LC(LC_name)

            num_pixels_per_image = [lc_classes[dominant_LC].num_pixels_per_image
                                    for dominant_LC in dominant_LCs]

        else:
            num_pixels_per_image = [PIX_SAMPLED_PER_IMAGE]*100

        features_tmp, labels_tmp = sample_compiled_images(
            image_names, label_name, num_pixels_per_image, smooth_sigma)
        npixels = len(labels_tmp)

        if current_idx + npixels > max_idx:
            npixels_keep = max_idx-current_idx
            train_features[current_idx: :] = features_tmp[:npixels_keep, :]
            train_labels[current_idx:] = labels_tmp[:npixels_keep]
            current_idx += npixels_keep

            if download:
                for image_name in image_names:
                    os.remove(image_name)
                os.remove(label_name)
            break

        train_features[current_idx:current_idx+npixels :] = features_tmp
        train_labels[current_idx:current_idx+npixels] = labels_tmp
        current_idx += npixels

        if download:
            for image_name in image_names:
                os.remove(image_name)
            os.remove(label_name)
            os.remove(LC_name)

        print(f'{i}/{nfiles}: {current_idx} pixels set out of {max_idx}')

    train_features = train_features.reshape(-1, nfeatures)
    train_labels = train_labels.reshape(-1)

    filename = f"{output_str}_{'_'.join(bands)}_seed{RANDOM_SEED}"
    np.save(f'train_features_{filename}.npy', train_features)
    np.save(f'train_labels_{filename}.npy', train_labels)

def download_file(file_name: str) -> None:
    """Download file to local directory."""
    p = subprocess.Popen(
        ['wget', f'{IMAGE_URL}{file_name}'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate(timeout=60)
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

        chip_ids = np.load(chip_id_name)
        for j, chip_id in enumerate(chip_ids):
            if chip_id in bad_chips:
                is_bad_chip[i, j] = 1

        if download:
            os.remove(chip_id_name)

    is_bad_chip = np.tile(is_bad_chip, (1, 1, PIX_SAMPLED_PER_IMAGE))
    is_bad_chip = is_bad_chip.flatten()

    np.save('train_features_bad_chip_mask.npy', is_bad_chip)

def get_bad_chips(bad_chip_path):
    """Get the list of bad chips."""
    with open(bad_chip_path) as f:
        bad_chips = f.readlines()
    bad_chips = [bad_chip.strip('\n') for bad_chip in bad_chips]
    return bad_chips

def get_dominant_LC(LC_name):
    LC_per_pixel = np.load(LC_name)
    LC_per_pixel = LC_per_pixel.reshape((100, 512*512))
    pixels_per_class = np.zeros((100, NUM_LC_CLASSES), dtype=np.int)
    for i in range(NUM_LC_CLASSES):
        pixels_per_class[:, i] = (LC_per_pixel == i).sum(-1)
    return np.argmax(pixels_per_class, axis=-1)
