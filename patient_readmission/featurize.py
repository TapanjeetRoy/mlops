import os
import yaml
import pandas as pd


def featurize():

    with open("configs/pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    processed_path = config["data"]["processed"]
    features_path = config["data"]["features"]

    df = pd.read_csv(processed_path)

    df["gender"] = df["gender"].map(
        {
            "M": 1,
            "F": 0
        }
    )

    df["smoker"] = df["smoker"].map(
        {
            "Yes": 1,
            "No": 0
        }
    )

    os.makedirs(os.path.dirname(features_path), exist_ok=True)

    df.to_csv(features_path, index=False)

    print(f"Features saved to {features_path}")


if __name__ == "__main__":
    featurize()
