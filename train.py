# File: /train.py
# This file contains the training code for the model.
# It is responsible for training the model and saving the model weights.
# It also saves the model weights after every epoch.
import mlflow
import mlflow.sklearn
import mlflow.keras
from scripts.data_preprocessing import load_data, preprocess_data
from scripts.model_training import train_xgboost, train_dl
from scripts.model_evaluation import evaluate_model, plot_shap_summary, plot_training_history
import joblib
import os

# Set a temporary artifact location
mlflow.set_tracking_uri("file:///tmp/mlruns")

# Check if file exists
file_path = "data/creditcard.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load and preprocess the data
data = load_data(file_path)

# Print data columns and sample for debugging
print("Data Columns:", data.columns)
print("Data Sample:\n", data.head())

X_train_res, X_test, y_train_res, y_test = preprocess_data(data)

# Train XGBoost model
best_xgb_model, best_xgb_params = train_xgboost(X_train_res, y_train_res)

# Evaluate XGBoost model
xgb_roc_auc = evaluate_model(best_xgb_model, X_test, y_test, model_type='xgb')
plot_shap_summary(best_xgb_model, X_test, model_type='xgb')

# Train Deep Learning model
best_dl_model, best_dl_hp, history = train_dl(X_train_res, y_train_res)

# Evaluate Deep Learning model
dl_roc_auc = evaluate_model(best_dl_model, X_test, y_test, model_type='dl')
plot_shap_summary(best_dl_model, X_test, model_type='dl')
plot_training_history(history)

# Save XGBoost model
xgb_model_path = 'best_xgb_model.pkl'
joblib.dump(best_xgb_model, xgb_model_path)
print(f"XGBoost model saved at {xgb_model_path}")

# Save Deep Learning model
dl_model_path = 'best_dl_model.keras'
best_dl_model.save(dl_model_path)
print(f"Deep Learning model saved at {dl_model_path}")

# Log XGBoost model with MLflow
with mlflow.start_run(run_name="XGBoost Hyperparameter Tuning"):
    mlflow.sklearn.log_model(best_xgb_model, "model")
    mlflow.log_params(best_xgb_params)
    mlflow.log_metric("roc_auc", xgb_roc_auc)

# Log Deep Learning model with MLflow
with mlflow.start_run(run_name="Deep Learning Hyperparameter Tuning"):
    mlflow.keras.log_model(best_dl_model, "model")
    mlflow.log_params(best_dl_hp.values)
    mlflow.log_metric("roc_auc", dl_roc_auc)

# Print ROC AUC scores
print(f'XGBoost ROC AUC Score: {xgb_roc_auc}')
print(f'Deep Learning ROC AUC Score: {dl_roc_auc}')
