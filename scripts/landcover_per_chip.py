import os
from pathlib import Path
import argparse
import pandas
import numpy as np


def landcover_per_chip(data_dir: os.PathLike):
    lc_classes = ['nodata', 'water', 'trees', 'grass', 'flooded veg', 'crops',
                  'scrub', 'built area', 'bare', 'snow/ice',  'clouds']

    df = pandas.DataFrame()
    for start_idx in range(0, 11800, 100):
        if start_idx == 11700:
            end_idx = 11748
        else:
            end_idx = start_idx + 100

        chip_ids = np.load(data_dir/f'chip_ids_{start_idx:06d}_{end_idx:06d}.npy')
        landcover = np.load(data_dir/f'LC_{start_idx:06d}_{end_idx:06d}.npy')
        landcover = landcover.reshape(chip_ids.shape[0], 512*512)

        pixels_per_class = np.zeros((landcover.shape[0], len(lc_classes)), dtype='uint8')
        for i, lc_class in enumerate(lc_classes):
            pixels_per_class[:, i] = (landcover == i).sum(-1)

        for i, chip_id in chip_ids:
            for j, lc_class in lc_classes:
                df.loc[chip_id, lc_class] = pixels_per_class[i, j]

    df.to_csv('landcover_class_distribution.csv')

def parse_commandline_arguments() -> "argparse.Namespace":
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate landcover distribution per chip.')
    parser.add_argument(
        '--data_dir',
        type=str,
        help='path to the directory containing the band data',
        default='./')

    args = parser.parse_args()
    return args


if __name__=='__main__':
    ARGS = parse_commandline_arguments()
    landcover_per_chip(Path(ARGS.data_dir))
