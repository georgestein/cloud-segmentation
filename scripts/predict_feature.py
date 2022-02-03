import re
import shutil
from pathlib import Path
import argparse
import multiprocessing
import numpy as np
import tqdm
from joblib import load

from cloud_seg.io.features import *

NPIX_SIDE = 512
REFBAND = 'B04'

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
        '--output_str',
        type=str,
        help='a string to label the output predictions',
        default='')
    parser.add_argument(
        '--max_pool_size',
        type=int,
        default=1,
        )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    ARGS = parse_commandline_arguments()

    MAX_POOL_SIZE = ARGS.max_pool_size
    DATA_DIR, PREDICTION_DIR = load_data_parameters(ARGS)
    MODEL_PATH, MODEL_NAME, FEATURE_LIST, OUTPUT_STR = load_model_parameters(ARGS)

    if not PREDICTION_DIR.exists():
        PREDICTION_DIR.mkdir()
    if not (PREDICTION_DIR/MODEL_NAME).exists():
        shutil.copyfile(MODEL_PATH, PREDICTION_DIR/MODEL_NAME)

    IMAGE_IDS = input_image_ids(DATA_DIR, REFBAND)
    CLF = load(MODEL_PATH)

def feature_classification(image_id):
    """Classify a single image."""
    image_features = Features(set_type='val', file_name=image_id, data_dir=DATA_DIR)
    for feature in FEATURE_LIST:
        image_features.add(feature)

    predictions = CLF.predict(image_features.get_values())
    predictions = predictions.reshape(-1, 512, 512)

    np.save(PREDICTION_DIR/f'preds_{OUTPUT_STR}_{image_id}.npy', predictions)

def run_feature_classification():
    """Run feature classification using pools."""
    if MAX_POOL_SIZE <= 1:
        for image_id in tqdm.tqdm(IMAGE_IDS):
            feature_classification(image_id)

    else:
        # Simple threading with pool and .map
        cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(min(cpus, MAX_POOL_SIZE))
        print(f"Number of available cpus = {cpus}")

        pool.map(feature_classification, IMAGE_IDS)

        pool.close()
        pool.join()


def get_image_id(image: str):
    """Get the image id from an image path."""
    return re.findall(
        '\d\d\d\d\d\d_\d\d\d\d\d\d',
        image)[-1]

def input_image_ids(data_dir, name):
    """Get the list of images to classify."""
    images = sorted(data_dir.glob(f'{name}_*.npy'))
    image_ids = [get_image_id(str(image)) for image in images]
    return image_ids

def load_data_parameters(args):
    """Define directories for chips and predictions."""
    if args is None:
        data_dir = Path('./')
        prediction_dir = Path('./feature_predictions/')
    else:
        data_dir = Path(args.data_dir)
        prediction_dir = Path(args.prediction_dir)

    return data_dir, prediction_dir

def load_model_parameters(args):
    """Define location of model and feature list."""
    if args is None:
        feature_list = ['B04', 'B03-B11', 'B08-B04', 'B08/B03',
                        'B02/B11', 'B08/B11', 'B02/B04']
        model_dir = Path('./')
        model_name = 'gbm_defaultfeatures_20220119.joblib'
        output_str = ''
    else:
        feature_list = args.model_features
        model_dir = Path(args.model_dir)
        model_name = args.model_name
        output_str = args.output_str

    if not output_str:
        output_str = 'ftr'

    model_path = model_dir/model_name

    return model_path, model_name, feature_list, output_str

if __name__ == '__main__':
    run_feature_classification()
