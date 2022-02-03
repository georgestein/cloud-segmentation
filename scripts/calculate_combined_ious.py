import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pandas

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

def load_feature(data_dir, image_id, feature_str):
    feature = np.load(data_dir/f'preds_{feature_str}_{image_id}.npy')
    feature_smoothed = feature.copy()
    for i in range(feature.shape[0]):
        feature_smoothed[i, ...] = gaussian_filter(feature[i, ...], 8)
    feature_smoothed = feature_smoothed > 0.1
    feature = feature.astype('uint8')
    feature_smoothed = feature_smoothed.astype('uint8')
    return feature, feature_smoothed

def calculate_combined_ious(data_dir, bad_chip_label_path, unet_str, feature_str):
    intersection_unet_and_feature = 0
    intersection_unet_or_feature = 0
    intersection_unet = 0
    intersection_feature = 0
    intersection_unet_and_feature_smoothed = 0
    intersection_unet_or_feature_smoothed = 0
    intersection_feature_smoothed = 0

    union_unet_and_feature = 0
    union_unet_or_feature = 0
    union_unet = 0
    union_feature = 0
    union_unet_and_feature_smoothed = 0
    union_unet_or_feature_smoothed = 0
    union_feature_smoothed = 0

    for start_img in range(0, 11748, 100):

        if start_img == 11700:
            end_img = 11748
        else:
            end_img = start_img + 100
        image_id = f'{start_img:06d}_{end_img:06d}'

        unet = np.load(data_dir/f'preds_{unet_str}_{image_id}.npy')
        labels = load_labels(data_dir, image_id, bad_chip_label_path)
        feature, feature_smoothed = load_feature(data_dir, image_id, feature_str)

        feature = feature.flatten()
        feature_smoothed = feature_smoothed.flatten()
        unet = unet.flatten()
        labels = labels.flatten()

        mask = labels<2

        feature = feature[mask]
        feature_smoothed = feature_smoothed[mask]
        unet = unet[mask]
        labels = labels[mask]

        unet_and_feature = unet & feature
        unet_or_feature = unet | feature
        unet_and_feature_smoothed = unet & feature_smoothed
        unet_or_feature_smoothed = unet | feature_smoothed

        intersection_unet_and_feature += np.sum(unet_and_feature & labels)
        intersection_unet_or_feature += np.sum(unet_or_feature & labels)
        intersection_unet += np.sum(unet & labels)
        intersection_feature += np.sum(feature & labels)
        intersection_unet_and_feature_smoothed += np.sum(unet_and_feature_smoothed & labels)
        intersection_unet_or_feature_smoothed += np.sum(unet_or_feature_smoothed & labels)
        intersection_feature_smoothed += np.sum(feature_smoothed & labels)

        union_unet_and_feature += np.sum(unet_and_feature | labels)
        union_unet_or_feature += np.sum(unet_or_feature | labels)
        union_unet += np.sum(unet | labels)
        union_feature += np.sum(feature | labels)
        union_unet_and_feature_smoothed += np.sum(unet_and_feature_smoothed | labels)
        union_unet_or_feature_smoothed += np.sum(unet_or_feature_smoothed | labels)
        union_feature_smoothed += np.sum(feature_smoothed | labels)

    print(f'unet | feature: {intersection_unet_or_feature/union_unet_or_feature}')
    print(f'unet & feature: {intersection_unet_and_feature/union_unet_and_feature}')
    print(f'unet | feature_smoothed: {intersection_unet_or_feature_smoothed/union_unet_or_feature_smoothed}')
    print(f'unet & feature_smoothed: {intersection_unet_and_feature_smoothed/union_unet_and_feature_smoothed}')
    print(f'unet: {intersection_unet/union_unet}')
    print(f'feature: {intersection_feature/union_feature}')
    print(f'feature_smoothed: {intersection_feature_smoothed/union_feature_smoothed}')

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
