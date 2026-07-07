import os
import json
import yaml
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate():

    with open("configs/pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = joblib.load(config["output"]["model"])

    X_test = pd.read_csv(config["output"]["x_test"])
    y_test = pd.read_csv(config["output"]["y_test"]).squeeze()

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    report = {
        "accuracy": float(round(accuracy_score(y_test, predictions), 4)),
        "precision": float(round(precision_score(y_test, predictions), 4)),
        "recall": float(round(recall_score(y_test, predictions), 4)),
        "f1": float(round(f1_score(y_test, predictions), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, probabilities), 4)),
    }

    os.makedirs(os.path.dirname(config["output"]["report"]), exist_ok=True)

    with open(config["output"]["report"], "w") as f:
        json.dump(report, f, indent=4)

    print("Evaluation completed.")
