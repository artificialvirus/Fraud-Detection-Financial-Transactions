# File: /model_evaluation.py
# This file contains the model evaluation code.
# It is responsible for evaluating the model performance using various metrics.
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap
import matplotlib.pyplot as plt
import json

def evaluate_model(model, X_test, y_test, model_type='xgb'):
    if model_type == 'dl':
        predictions = (model.predict(X_test) > 0.5).astype(int)
    else:
        predictions = model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, predictions)

    # Log metrics to a file
    metrics = classification_report(y_test, predictions, output_dict=True)
    with open(f"{model_type}_classification_report.json", 'w') as f:
        json.dump(metrics, f)

    print(f"{model_type.upper()} Classification Report")
    print(classification_report(y_test, predictions))
    print(f"{model_type.upper()} Confusion Matrix")
    print(confusion_matrix(y_test, predictions))
    print(f"{model_type.upper()} ROC AUC Score: {roc_auc}")

    return roc_auc

def plot_shap_summary(model, X_test, model_type='xgb'):
    if model_type == 'dl':
        explainer = shap.Explainer(model, X_test)
    else:
        explainer = shap.Explainer(model)

    shap_values = explainer(X_test)

    # Plot SHAP summary
    shap.summary_plot(shap_values, X_test)

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
