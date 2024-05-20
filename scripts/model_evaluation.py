# File: /model_evaluation.py
# This file contains the model evaluation code.
# It is responsible for evaluating the model performance using various metrics.
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, model_type='xgb'):
    predictions = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, predictions)

    print(f"{model_type.upper()} Classification Report")
    print(classification_report(y_test, predictions))
    print(f"{model_type.upper()} Confusion Matrix")
    print(confusion_matrix(y_test, predictions))
    print(f"{model_type.upper()} ROC AUC Score: {roc_auc}")

    return roc_auc

def plot_shap_summary(model, X_test, model_type='xgb'):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)

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
    plt.show()
