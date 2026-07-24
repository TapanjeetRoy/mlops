import os
import yaml
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train():

    with open("configs/pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config["data"]["features"])

    X = df.drop(columns=["patient_id", "readmitted"])
    y = df["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"],
    )

    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(config["output"]["model"]), exist_ok=True)

    joblib.dump(model, config["output"]["model"])

    os.makedirs(os.path.dirname(config["output"]["x_test"]), exist_ok=True)

    X_test.to_csv(config["output"]["x_test"], index=False)
    y_test.to_frame().to_csv(config["output"]["y_test"], index=False)

    print(f"Model saved to {config['output']['model']}")
    print("Test dataset saved.")
if __name__ == "__main__":
    train()
