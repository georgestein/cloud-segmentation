import shutil
import numpy as np
from pathlib import Path
import tqdm
from joblib import load
from cloud_seg.io.features import *

NPIX_SIDE = 512

def load_data_parameters():
    bands = ['B02', 'B03', 'B04', 'B08',
             'B05', 'B06', 'B07','B09',
             'B8A', 'B11', 'B12', 'B01']
    nbands_per_file = 4
    data_dir = Path('./')
    prediction_dir = Path('./feature_predictions/')
    return data_dir, bands, nbands_per_file, prediction_dir

def load_model_parameters():
    feature_list = ['B04', 'B03-B11', 'B08-B04', 'B08/B03',
                    'B02/B11', 'B08/B11', 'B02/B04']
    model_dir = Path('./')
    model_name = 'gbm_defaultfeatures_20220119.joblib'
    model_path = model_dir/model_name
    return model_path, model_name, feature_list

def input_image_ids(data_dir, name):
    images = sorted(data_dir.glob('{name}_*.npy'))
    image_ids = ['_'.join(str(image).split('/')[-1].split('.')[-1].split('_')[1:3]) for image in images]
    return image_ids

def feature_classification():
    data_dir, bands, nbands_per_file, prediction_dir = load_data_parameters()
    model_path, model_name, feature_list = load_model_parameters()

    if not prediction_dir.exists():
        prediction_dir.mkdir()
    if not (prediction_dir/model_name).exists():
        shutil.copyfile(model_path, prediction_dir/model_name)

    image_ids = input_image_ids(data_dir, bands[0])
    clf = load(model_path)

    for image_id in tqdm.tqdm(image_ids):
        image_features = Features(set_type='val', file_name=image_id)
        for feature in feature_list:
            image_features.add(feature)

        predictions = clf.predict(image_features.get_values())

        predictions = predictions.reshape(-1, 512, 512)
        np.save(prediction_dir/f'preds_{image_id}.npy', predictions)

if __name__=='__main__':
    feature_classification()
