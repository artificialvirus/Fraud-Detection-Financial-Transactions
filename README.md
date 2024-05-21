# Fraud Detection in Financial Transactions Using Machine Learning

## Overview
This project presents a comprehensive machine learning pipeline for detecting fraudulent financial transactions. The pipeline involves data preprocessing, model training, evaluation, and deployment of machine learning models to serve real-time predictions via a RESTful API. The objective is to address the challenge of identifying fraudulent transactions in a highly imbalanced dataset using advanced techniques and tools within an MLOps framework. The implementation includes the use of XGBoost and deep learning models, which are evaluated and deployed for practical use.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Deployment](#deployment)
6. [Results](#results)
7. [Paper](#paper)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction
Fraud detection in financial transactions has emerged as a critical challenge in the modern financial landscape. This project aims to develop a robust, scalable, and interpretable fraud detection system using both XGBoost and deep learning models within an MLOps framework.

## Data Preprocessing
The data preprocessing steps include:
- Loading the dataset.
- Checking for missing values.
- Standardizing the features.
- Handling imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique).

The preprocessing code can be found in `/data_preprocessing.py`.

## Model Training
We implemented two models:
1. **XGBoost Model**: A powerful gradient boosting algorithm.
2. **Deep Learning Model**: A neural network using Keras.

The training code for these models is located in `/model_training.py`.

### XGBoost Model
The XGBoost model is trained with hyperparameter tuning using GridSearchCV.

### Deep Learning Model
The deep learning model is built using Keras and hyperparameters are optimized using Keras Tuner.

## Model Evaluation
Model evaluation includes:
- Generating classification reports.
- Plotting confusion matrices.
- Calculating ROC AUC scores.
- Using SHAP values for model interpretability.

The evaluation code is in `/model_evaluation.py`.

## Deployment
The trained models are deployed using a Flask web application, with Docker used to containerize the application for easy deployment across different environments. 

### API Endpoints
- `/predict_xgb`: Returns the prediction made by the XGBoost model.
- `/predict_dl`: Returns the prediction made by the deep learning model.

The deployment code is located in `/app.py`.

## Results
The results of the model evaluation, including performance metrics and SHAP analysis, are detailed in the [paper](#paper).

## Paper
You can read the full paper, "Fraud Detection in Financial Transactions Using Machine Learning: A Comprehensive MLOps Approach," [here](paper.pdf).

## Usage
To use this project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-repository-link.git
    cd your-repository
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the data preprocessing**:
    ```sh
    python scripts/data_preprocessing.py
    ```

4. **Train the models**:
    ```sh
    python scripts/train.py
    ```

5. **Run the Flask app**:
    ```sh
    python app.py
    ```

6. **Make predictions**:
    Use an API client like Postman to send POST requests to the `/predict_xgb` or `/predict_dl` endpoints with transaction data in JSON format.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.