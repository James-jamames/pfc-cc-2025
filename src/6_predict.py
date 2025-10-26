from pathlib import Path

MODELS_DIR = Path("../models")
SAMPLES_DIR = Path("../data")
OUTPUT_DIR = Path("../results")
OUTPUT_DIR.mkdir(exist_ok=True)

import time
import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

def load_samples(itter: int, class_name: str) -> pd.DataFrame:
    target_value = {'oxc': 1, 'oxn': 2, 'oxs': 3}[class_name]

    dataset = ds.dataset(Path('../data') / f"{itter:02}_samples_test.pq", format="parquet")
    table = dataset.to_table(filter=ds.field('class').isin(range(0, target_value + 1, target_value)))

    samples = table.to_pandas()

    samples['class'] = samples['class'].apply(lambda x: x // target_value)
    
    return samples

def get_optimal_threshold(y, y_hat_prob):
    precision, recall, threshold = precision_recall_curve(y, y_hat_prob)
    
    nonzero_mask = np.logical_and((precision != 0.0), (recall != 0.0))
    
    optimal_idx = np.argmax(1 - np.abs(precision[nonzero_mask] - recall[nonzero_mask]))
    
    return threshold[optimal_idx]

CLASS_NAMES = ['oxc', 'oxn']

COVARIATES = [f'B{i:02}' for i in range(1, 65)]

for class_name in CLASS_NAMES:
    for metric in ['minkowski', 'euclidean', 'manhattan', 'cosine']:
        filename = f"knn.m_{metric}.{class_name}.lz4"

        if (OUTPUT_DIR / filename).exists():
            continue

        model_path = MODELS_DIR / filename
        model_dict = joblib.load(model_path)

        y = []
        y_hat = []
        time_records = []

        for itter in range(5):
            model = model_dict[itter]['model']

            samples = load_samples(itter+1, class_name).sample(frac=0.1)

            s_time = time.time()

            print("Começou ", itter)

            model.predict(samples[COVARIATES])

            time_records.append(time.time() - s_time)
            y.extend(samples['class'].to_list())
            y_hat.extend(model.predict(samples[COVARIATES]))

            print(time.time() - s_time, "Terminado - ", itter)

        joblib.dump({
            'threshold': None,
            'f1_score': f1_score(y, y_hat),
            'recall_score': recall_score(y, y_hat),
            'precision_score': precision_score(y, y_hat),
            'time_records': time_records
        }, OUTPUT_DIR / filename)

        break