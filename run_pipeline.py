import json
import yaml
import mlflow

from patient_readmission.preprocess import preprocess
from patient_readmission.featurize import featurize
from patient_readmission.train import train
from patient_readmission.evaluate import evaluate


def main():

    with open("configs/pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():

        mlflow.log_param(
            "model_type",
            config["model"]["model_type"],
        )

        mlflow.log_param(
            "n_estimators",
            config["model"]["n_estimators"],
        )

        mlflow.log_param(
            "max_depth",
            config["model"]["max_depth"],
        )

        preprocess()

        featurize()

        train()

        evaluate()

        with open(config["output"]["report"], "r") as f:
            metrics = json.load(f)

        mlflow.log_metrics(metrics)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
