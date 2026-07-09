# 🏥 Patient Readmission Prediction Pipeline

## 📌 Project Overview

This project demonstrates an end-to-end MLOps pipeline for predicting whether a patient is likely to be readmitted based on clinical and demographic information. The pipeline is designed as a sequence of independent stages that perform data preprocessing, feature engineering, model training, evaluation, and experiment tracking using MLflow.
The project was developed as part of an MLOps R&D exercise inspired by an industry-style pipeline orchestration scenario.

# 🎯 Problem Statement

Healthcare organizations often aim to identify patients who are at a higher risk of readmission after discharge. Early identification enables hospitals to improve patient care, optimize resource allocation, and reduce healthcare costs.
The objective of this project is to build a modular machine learning pipeline capable of processing patient data, training a classification model, evaluating its performance, and logging the complete experiment into MLflow.

# 🏗️ Project Architecture

                           +----------------------+
                           |  Raw Patient Dataset |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |   preprocess.py      |
                           | Remove duplicates    |
                           | Remove missing data  |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |    featurize.py      |
                           | Encode features      |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |      train.py        |
                           | Train Random Forest  |
                           | Save model.pkl       |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |    evaluate.py       |
                           | Accuracy             |
                           | F1 Score             |
                           | ROC-AUC              |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |      MLflow          |
                           | Params              |
                           | Metrics             |
                           | Experiment Tracking |
                           +----------------------+


# 📁 Project Structure

01-patient-readmission-pipeline/
│
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── test/
│
├── models/
├── reports/
├── patient_readmission/
│
├── screenshots/
│
├── requirements.txt
├── run_pipeline.py
├── README.md
└── .gitignore


# 🛠️ Technology Stack

* Python 3
* Pandas
* NumPy
* Scikit-learn
* MLflow
* PyYAML
* Joblib
* Git
* GitHub

# 📊 Dataset

The project uses a small synthetic healthcare dataset containing patient demographic and clinical information such as:

* Patient ID
* Age
* Gender
* BMI
* Blood Glucose
* Blood Pressure
* Smoking Status
* Days in Hospital
* Readmission Status

The dataset is intentionally small to keep the project lightweight while demonstrating the complete MLOps workflow.

# ⚙️ Pipeline Workflow

## Stage 1 – Data Preprocessing

* Load raw patient data
* Remove duplicate records
* Remove missing values
* Filter invalid records
* Save cleaned dataset

Output:

data/processed/patients_clean.csv

## Stage 2 – Feature Engineering

* Encode categorical features
* Prepare model-ready dataset

Output:

data/features/features.csv

## Stage 3 – Model Training

* Split dataset into training and testing sets
* Train a Random Forest classifier
* Save trained model
* Save test dataset

Outputs:

models/model.pkl
data/test/X_test.csv
data/test/y_test.csv

## Stage 4 – Model Evaluation

* Load trained model
* Predict on unseen test data
* Calculate evaluation metrics
* Save evaluation report

Output:
reports/evaluation.json

## Stage 5 – MLflow Tracking

The complete pipeline execution is tracked as a single MLflow experiment.
Logged Parameters:

* Model Type
* Number of Estimators
* Maximum Tree Depth

Logged Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

# ▶️ Running the Project

Execute the complete pipeline using:

python3 run_pipeline.py

# 📂 Project Outputs

Successful execution generates:

models/model.pkl
reports/evaluation.json
data/processed/patients_clean.csv
data/features/features.csv
data/test/X_test.csv
data/test/y_test.csv

# 📸 Screenshots

The repository includes screenshots demonstrating:

* Project Directory Structure
* Successful Pipeline Execution
* MLflow Experiment Dashboard
* MLflow Run Details
* Evaluation Metrics
* Generated Model

# 📚 Key Learnings

This project demonstrates the following MLOps concepts:

* Modular pipeline development
* Configuration-driven execution
* Data preprocessing
* Feature engineering
* Train/Test split
* Model serialization
* Performance evaluation
* MLflow experiment tracking
* Git version control
* GitHub project management

# 🚀 Future Improvements

Potential enhancements include:

* Hyperparameter tuning
* Model versioning
* Docker containerization
* CI/CD integration
* Unit testing
* Data validation
* Logging framework
* Kubernetes deployment
* DVC integration
* Automated pipeline scheduling

# 👨‍💻 Author

**Tapanjeet Roy**

MLOps • DevOps • Kubernetes • Cloud Engineering


**⭐ This project is part of my hands-on MLOps learning journey, where each project is built independently to understand real-world engineering concepts rather than simply reproducing lab solutions.**
