from pathlib import Path
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import pandas
import argparse

UNET_CUTOFF = 0.3

def load_labels(data_dir, image_id, bad_chip_label_path):
    labels = np.load(data_dir/f'labels_{image_id}.npy')
    chip_ids = np.load(data_dir/f'chip_ids_{image_id}.npy')
    with open(bad_chip_label_path, encoding='utf8') as f:
        bad_chips = f.readlines()
    bad_chips = [bad_chip.replace('\n', '') for bad_chip in bad_chips]
    for i, chip_id in enumerate(chip_ids):
        if chip_id in bad_chips:
            labels[i, ...] = 255
    return labels

def load_feature(data_dir, image_id, feature_str):
    feature = np.load(data_dir/f'preds_{feature_str}_{image_id}.npy')
    feature_smoothed = feature.copy()
    for i in range(feature.shape[0]):
        feature_smoothed[i, ...] = gaussian_filter(feature[i, ...], 8)
    feature_smoothed = feature_smoothed > 0.1
    feature = feature.astype('uint8')
    feature_smoothed = feature_smoothed.astype('uint8')
    return feature, feature_smoothed

def load_unet(data_dir, image_id, feature_str):
    unet = np.load(data_dir/f'preds_{feature_str}_{image_id}.npy')
    unet_smoothed = unet.copy()
    for i in range(unet.shape[0]):
        unet_smoothed[i, ...] = gaussian_filter( (unet[i, ...] > UNET_CUTOFF)*1., 4)
    unet = (unet > UNET_CUTOFF) * 1
    unet_smoothed = (unet_smoothed > 0.1) * 1.
    unet = unet.astype('uint8')
    unet_smoothed = unet_smoothed.astype('uint8')
    return unet, unet_smoothed

def calculate_combined_ious(data_dir, bad_chip_label_path, unet_str, feature_str):
    intersection_unet_and_feature = [0]*11
    intersection_unet_or_feature = [0]*11
    intersection_unet = [0]*11
    intersection_unet_smoothed = [0]*11
    intersection_feature = [0]*11
    intersection_unet_and_feature_smoothed = [0]*11
    intersection_unet_or_feature_smoothed = [0]*11
    intersection_feature_smoothed = [0]*11

    union_unet_and_feature = [0]*11
    union_unet_or_feature = [0]*11
    union_unet = [0]*11
    union_unet_smoothed = [0]*11
    union_feature = [0]*11
    union_unet_and_feature_smoothed = [0]*11
    union_unet_or_feature_smoothed = [0]*11
    union_feature_smoothed = [0]*11

    for start_img in range(0,11748, 100):

        print(f"Running on {start_img}")
        if start_img == 11700:
            end_img = 11748
        else:
            end_img = start_img + 100
        image_id = f'{start_img:06d}_{end_img:06d}'

        unet, unet_smoothed = load_unet(data_dir, image_id, unet_str)

        labels = load_labels(data_dir, image_id, bad_chip_label_path)
        pixelLC = np.load(data_dir/f'LC_{image_id}.npy')
        feature, feature_smoothed = load_feature(data_dir, image_id, feature_str)

        feature_all = feature.flatten()
        feature_smoothed_all = feature_smoothed.flatten()
        unet_all = unet.flatten()
        unet_smoothed_all = unet_smoothed.flatten()
        labels_all = labels.flatten()
        pixelLC_all = pixelLC.flatten()

        for LC in range(11):

            mask = (labels_all < 2) & (pixelLC_all == LC)

            feature = feature_all[mask]
            feature_smoothed = feature_smoothed_all[mask]
            unet = unet_all[mask]
            unet_smoothed = unet_smoothed_all[mask]
            labels = labels_all[mask]

            unet_and_feature = unet & feature
            unet_or_feature = unet | feature
            unet_and_feature_smoothed = unet & feature_smoothed
            unet_or_feature_smoothed = unet | feature_smoothed

            intersection_unet_and_feature[LC] += np.sum(unet_and_feature & labels)
            intersection_unet_or_feature[LC] += np.sum(unet_or_feature & labels)
            intersection_unet[LC] += np.sum(unet & labels)
            intersection_unet_smoothed[LC] += np.sum(unet_smoothed & labels)
            intersection_feature[LC] += np.sum(feature & labels)
            intersection_unet_and_feature_smoothed[LC] += np.sum(unet_and_feature_smoothed & labels)
            intersection_unet_or_feature_smoothed[LC] += np.sum(unet_or_feature_smoothed & labels)
            intersection_feature_smoothed[LC] += np.sum(feature_smoothed & labels)

            union_unet_and_feature[LC] += np.sum(unet_and_feature | labels)
            union_unet_or_feature[LC] += np.sum(unet_or_feature | labels)
            union_unet[LC] += np.sum(unet | labels)
            union_unet_smoothed[LC] += np.sum(unet_smoothed | labels)
            union_feature[LC] += np.sum(feature | labels)
            union_unet_and_feature_smoothed[LC] += np.sum(unet_and_feature_smoothed | labels)
            union_unet_or_feature_smoothed[LC] += np.sum(unet_or_feature_smoothed | labels)
            union_feature_smoothed[LC] += np.sum(feature_smoothed | labels)

    for LC in range(11):
        print(f'LC {LC}: unet | feature: {intersection_unet_or_feature[LC]/union_unet_or_feature[LC]}')
        print(f'LC {LC}: unet & feature: {intersection_unet_and_feature[LC]/union_unet_and_feature[LC]}')
        print(f'LC {LC}: unet | feature_smoothed: {intersection_unet_or_feature_smoothed[LC]/union_unet_or_feature_smoothed[LC]}')
        print(f'LC {LC}: unet & feature_smoothed: {intersection_unet_and_feature_smoothed[LC]/union_unet_and_feature_smoothed[LC]}')
        print(f'LC {LC}: unet: {intersection_unet[LC]/union_unet[LC]}')
        print(f'LC {LC}: unet_cmoothed: {intersection_unet_smoothed[LC]/union_unet_smoothed[LC]}')
        print(f'LC {LC}: feature: {intersection_feature[LC]/union_feature[LC]}')
        print(f'LC {LC}: feature_smoothed: {intersection_feature_smoothed[LC]/union_feature_smoothed[LC]}')
        print('')

    print(f'unet | feature: {sum(intersection_unet_or_feature)/sum(union_unet_or_feature)}')
    print(f'unet & feature: {sum(intersection_unet_and_feature)/sum(union_unet_and_feature)}')
    print(f'unet | feature_smoothed: {sum(intersection_unet_or_feature_smoothed)/sum(union_unet_or_feature_smoothed)}')
    print(f'unet & feature_smoothed: {sum(intersection_unet_and_feature_smoothed)/sum(union_unet_and_feature_smoothed)}')
    print(f'unet: {sum(intersection_unet)/sum(union_unet)}')
    print(f'unet_smoothed: {sum(intersection_unet_smoothed)/sum(union_unet_smoothed)}')
    print(f'feature: {sum(intersection_feature)/sum(union_feature)}')
    print(f'feature_smoothed: {sum(intersection_feature_smoothed)/sum(union_feature_smoothed)}')

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
    parser.add_argument(
        '--unet_str',
        type=str,
        help='string identifying predictions from unet',
        default='')
    parser.add_argument(
        '--feature_str',
        type=str,
        help='string identifying predictions from feature based classifier',
        default='ctb')

    args = parser.parse_args()
    return args


if __name__=='__main__':
    ARGS = parse_commandline_arguments()
    calculate_combined_ious(
        Path(ARGS.data_dir), ARGS.path_to_badchips, ARGS.unet_str, ARGS.feature_str)
