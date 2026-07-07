import os
import yaml
import pandas as pd


def preprocess():

    with open("configs/pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_path = config["data"]["raw"]
    processed_path = config["data"]["processed"]

    df = pd.read_csv(raw_path)

    df = df.drop_duplicates()

    df = df.dropna()

    df = df[df["age"] >= 18]

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    df.to_csv(processed_path, index=False)

    print(f"Processed data saved to {processed_path}")


if __name__ == "__main__":
    preprocess()
