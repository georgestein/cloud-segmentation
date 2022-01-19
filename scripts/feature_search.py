from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from joblib import dump


DATA_DIR = Path('/content/drive/MyDrive/Clouds/')

def train_gbc(iteration_name):

    features = np.load(DATA_DIR/f'train_features_{iteration_name}.npy')
    labels = np.load(DATA_DIR/f'train_labels_{iteration_name}.npy')

    clf = GradientBoostingClassifier(random_state=0, max_depth=2)
    clf.fit(train_features, train_labels)
    dump(clf, DATA_DIR/f'gbm_{iteration_name}.joblib')
