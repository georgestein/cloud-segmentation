# from submission_dir.main import main 
from main import main

from pathlib import Path

submission_dir = Path('./')
TRAIN_FEATURES = Path('../../data/train_features/')

main(
    model_weights_path=submission_dir / "assets/cloud_model.pt",
    hparams_path = submission_dir / "assets/hparams.npy",
    test_features_dir=TRAIN_FEATURES,
    predictions_dir=submission_dir / "predictions",
#    fast_dev_run=True,
)
