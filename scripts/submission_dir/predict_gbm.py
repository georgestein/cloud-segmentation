from joblib import load
from loguru import logger
import torch
from scipy.ndimage import gaussian_filter
import numpy as np

try:
    from cloud_seg.models.unet.cloud_dataset import CloudDataset
except ImportError:
    from cloud_dataset import CloudDataset

torch.set_grad_enabled(False)

def set_gbm_params() -> dict:
    """Set parameters for the gbm model."""
    gbm_params = {
        'npix_side': 512,
        'model_path': './assets/gbm_modelI.joblib',
        'bands_use': ['B04', 'B03', 'B08', 'B02', 'B11', 'B01'],
        'model_features': ['B04', 'B03-B11', 'B08-B04', 'B08/B03', 'B02/B11', 'B08/B11',
                          'B02/B04', 'B02/B03', 'B03/B01'],
        'batch_size': 8,
        'num_workers': 1}
    gbm_params['nfeatures'] = len(gbm_params['model_features'])

    return gbm_params

def make_gbm_predictions(x_paths: "pandas.DataFrame", predictions_dir: "os.pathlike"):
    """Generate and save predictions for each chip using the gbm model."""
    params = set_gbm_params()

    clf = load(params['model_path'])

    dataloader = get_dataloader(x_paths, params)

    for batch_index, batch in enumerate(dataloader):
        if batch_index % 100 == 0:
            logger.debug(f"Predicting batch {batch_index} of {len(dataloader)} with "
                         f"{params['model_path']}")

        x = batch["chip"].numpy() # float32 array, NCHW, is this numpy?

        preds = feature_classification(x, clf, params)

        for chip_id, pred in zip(batch["chip_id"], preds):
            chip_pred_path = predictions_dir/f"{chip_id}.npy"
            np.save(chip_pred_path, pred)

def feature_classification(batch: np.ndarray, clf: "sklearn.GradientBoostingClassifier",
                           params: dict):
    batch_size = batch.shape[0]
    nfeatures = params['nfeatures']
    npix_side = params['npix_side']

    batch = batch.reshape(batch_size, nfeatures, npix_side*npix_side)
    batch = np.swapaxes(batch, 1, 2)
    batch = batch.reshape(batch_size*npix_side*npix_side, nfeatures)

    predictions = clf.predict(batch)

    predictions = predictions.reshape(batch_size, npix_side, npix_side)

    predictions = smooth_predictions(predictions)

    return predictions

def smooth_predictions(predictions: np.ndarray) -> np.ndarray:
    predictions_smoothed = np.zeros(predictions.shape, np.float32)
    for i in range(predictions.shape[0]):
        predictions_smoothed[i, ...] = gaussian_filter(predictions[i, ...], 8)
    return predictions_smoothed

def get_dataloader(x_paths: "pandas.DataFrame", params: dict) -> "torch.utils.data.DataLoader":
    dataset = CloudDataset(
        x_paths=x_paths,
        bands=params['bands_use'],
        custom_features=params['model_features'],
        scale_feature_channels='custom')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
        shuffle=False,
        drop_last=False,
        pin_memory=True)
    return dataloader
