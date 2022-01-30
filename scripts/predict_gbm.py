import re
import shutil
import numpy as np
from pathlib import Path
import tqdm
from joblib import load
import argparse
import multiprocessing

from cloud_seg.io.features import *

NPIX_SIDE = 512
REFBAND = 'B04'

def feature_classification(image_id):
    image_features = Features(set_type='val', file_name=image_id, data_dir=data_dir)
    for feature in feature_list:
        image_features.add(feature)

    predictions = clf.predict(image_features.get_values())
            
    predictions = predictions.reshape(-1, 512, 512)
    np.save(prediction_dir/f'preds_gbm_{image_id}.npy', predictions)
    

def run_feature_classification(args=None):

    if args.max_pool_size <= 1:
        for image_id in tqdm.tqdm(image_ids):
            feature_classification(image_id)

    else:
        # Simple threading with pool and .map                                                    
        cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpus if cpus < args.max_pool_size else args.max_pool_size)
        print(f"Number of available cpus = {cpus}")

        pool.map(feature_classification, image_ids)

        pool.close()
        pool.join()


def get_image_id(image: str):
    return re.findall(
        '\d\d\d\d\d\d_\d\d\d\d\d\d',
        image)[-1]

def input_image_ids(data_dir, name):
    images = sorted(data_dir.glob(f'{name}_*.npy'))
    image_ids = [get_image_id(str(image)) for image in images]
    return image_ids

def load_data_parameters(args):
    if args is None:
        data_dir = Path('./')
        prediction_dir = Path('./feature_predictions/')
    else:
        data_dir = Path(args.data_dir)
        prediction_dir = Path(args.prediction_dir)

    return data_dir, prediction_dir

def load_model_parameters(args):
    if args is None:
        feature_list = ['B04', 'B03-B11', 'B08-B04', 'B08/B03',
                        'B02/B11', 'B08/B11', 'B02/B04']
        model_dir = Path('./')
        model_name = 'gbm_defaultfeatures_20220119.joblib'
    else:
        feature_list = args.model_features
        model_dir = Path(args.model_dir)
        model_name = args.model_name

    model_path = model_dir/model_name

    return model_path, model_name, feature_list


def parse_commandline_arguments() -> "argparse.Namespace":
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(
        description='Predict cloud pixels using gradient boost model.')
    parser.add_argument(
        '--data_dir',
        type=str,
        help='path to the directory containing the band data',
        default='./')
    parser.add_argument(
        '--prediction_dir',
        type=str,
        help='path to the directory to which predictions are written',
        default='./feature_predictions')
    parser.add_argument(
        '--model_dir',
        type=str,
        help='directory containing the model as a joblib pickle',
        default='./')
    parser.add_argument(
        '--model_name',
        type=str,
        help='name of the pickle file containing the model',
        default='gbm_defaultfeatures_20220119.joblib')
    parser.add_argument(
        '--model_features',
        type=str,
        nargs='*',
        help='the names of the features used in the model, in order',
        default = ['B04', 'B03-B11', 'B08-B04', 'B08/B03', 'B02/B11', 'B08/B11', 'B02/B04'])
    parser.add_argument(
        '--max_pool_size',
        type=int,
        default=1,
        )

    args = parser.parse_args()
    return args

ARGS = parse_commandline_arguments()
args = ARGS
data_dir, prediction_dir = load_data_parameters(args)
model_path, model_name, feature_list = load_model_parameters(args)

if not prediction_dir.exists():
    prediction_dir.mkdir()
if not (prediction_dir/model_name).exists():
    shutil.copyfile(model_path, prediction_dir/model_name)

image_ids = input_image_ids(data_dir, REFBAND)
clf = load(model_path)

if __name__ == '__main__':
    run_feature_classification(ARGS)
