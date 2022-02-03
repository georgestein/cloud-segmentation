import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pandas
import argparse

def load_labels(data_dir, image_id, bad_chip_label_path):
    labels = np.load(data_dir/f'labels_{image_id}.npy')
    chip_ids = np.load(data_dir/f'chip_ids_{image_id}.npy')
    with open(bad_chip_label_path) as f:
        bad_chips = f.readlines()
    bad_chips = [bad_chip.replace('\n', '') for bad_chip in bad_chips]
    for i, chip_id in enumerate(chip_ids):
        if chip_id in bad_chips:
            labels[i, ...] = 255
    return labels

def load_gbm(data_dir, image_id):
    gbm = np.load(data_dir/f'preds_gbm_{image_id}.npy')
    gbm_smoothed = gbm.copy()
    for i in range(gbm.shape[0]):
        gbm_smoothed[i, ...] = gaussian_filter(gbm[i, ...], 8)
    gbm_smoothed = gbm_smoothed > 0.1
    gbm = gbm.astype('uint8')
    gbm_smoothed = gbm_smoothed.astype('uint8')
    return gbm, gbm_smoothed

def calculate_combined_ious(data_dir, bad_chip_label_path):
    intersection_unet_and_gbm = 0
    intersection_unet_or_gbm = 0
    intersection_unet = 0
    intersection_gbm = 0
    intersection_unet_and_gbm_smoothed = 0
    intersection_unet_or_gbm_smoothed = 0
    intersection_gbm_smoothed = 0

    union_unet_and_gbm = 0
    union_unet_or_gbm = 0
    union_unet = 0
    union_gbm = 0
    union_unet_and_gbm_smoothed = 0
    union_unet_or_gbm_smoothed = 0
    union_gbm_smoothed = 0

    for start_img in range(0, 11748, 100):

        print(f"Running on {start_img}")
        if start_img == 11700:
            end_img = 11748
        else:
            end_img = start_img + 100
        image_id = f'{start_img:06d}_{end_img:06d}'

        unet = np.load(data_dir/f'preds_unet_{image_id}.npy')
        labels = load_labels(data_dir, image_id, bad_chip_label_path)
        gbm, gbm_smoothed = load_gbm(data_dir, image_id)

        gbm = gbm.flatten()
        gbm_smoothed = gbm_smoothed.flatten()
        unet = unet.flatten()
        labels = labels.flatten()

        mask = labels<2

        gbm = gbm[mask]
        gbm_smoothed = gbm_smoothed[mask]
        unet = unet[mask]
        labels = labels[mask]

        unet_and_gbm = unet & gbm
        unet_or_gbm = unet | gbm
        unet_and_gbm_smoothed = unet & gbm_smoothed
        unet_or_gbm_smoothed = unet | gbm_smoothed

        intersection_unet_and_gbm += np.sum(unet_and_gbm & labels)
        intersection_unet_or_gbm += np.sum(unet_or_gbm & labels)
        intersection_unet += np.sum(unet & labels)
        intersection_gbm += np.sum(gbm & labels)
        intersection_unet_and_gbm_smoothed += np.sum(unet_and_gbm_smoothed & labels)
        intersection_unet_or_gbm_smoothed += np.sum(unet_or_gbm_smoothed & labels)
        intersection_gbm_smoothed += np.sum(gbm_smoothed & labels)

        union_unet_and_gbm += np.sum(unet_and_gbm | labels)
        union_unet_or_gbm += np.sum(unet_or_gbm | labels)
        union_unet += np.sum(unet | labels)
        union_gbm += np.sum(gbm | labels)
        union_unet_and_gbm_smoothed += np.sum(unet_and_gbm_smoothed | labels)
        union_unet_or_gbm_smoothed += np.sum(unet_or_gbm_smoothed | labels)
        union_gbm_smoothed += np.sum(gbm_smoothed | labels)

    print(f'unet | gbm: {intersection_unet_or_gbm/union_unet_or_gbm}')
    print(f'unet & gbm: {intersection_unet_and_gbm/union_unet_and_gbm}')
    print(f'unet | gbm_smoothed: {intersection_unet_or_gbm_smoothed/union_unet_or_gbm_smoothed}')
    print(f'unet & gbm_smoothed: {intersection_unet_and_gbm_smoothed/union_unet_and_gbm_smoothed}')
    print(f'unet: {intersection_unet/union_unet}')
    print(f'gbm: {intersection_gbm/union_gbm}')
    print(f'gbm_smoothed: {intersection_gbm_smoothed/union_gbm_smoothed}')

def parse_commandline_arguments() -> "argparse.Namespace":
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate iou in different feature combinations.')
    parser.add_argument(
        '--data_dir',
        type=str,
        help='path to the directory containing the predictions',
        default='./predictions')
    parser.add_argument(
        '--path_to_badchips',
        type=str,
        help='path to the text file containing the list of bad chips',
        default='./cloud-segmentation/data/BAD_CHIP_DATA/BAD_CHIP_LABEL_IDS.txt')

    args = parser.parse_args()
    return args


if __name__=='__main__':
    ARGS = parse_commandline_arguments()
    calculate_combined_ious(Path(ARGS.data_dir), ARGS.path_to_badchips)
