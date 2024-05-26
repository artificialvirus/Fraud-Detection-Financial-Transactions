# File: /app.py
# This file contains the main application code.
# It is responsible for orchestrating the model training and evaluation process.
from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained models
xgb_model = joblib.load("best_xgb_model.pkl")
dl_model = load_model("best_dl_model.keras")  # Change to load the model with .keras extension

app = Flask(__name__)

@app.route('/predict_xgb', methods=['POST'])
def predict_xgb():
    try:
        data = request.get_json(force=True)
        if 'features' not in data:
            return jsonify({'error': 'Invalid input'}), 400
        prediction = xgb_model.predict(np.array(data['features']).reshape(1, -1))
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        logger.error(f"Error in /predict_xgb: {e}")
        return jsonify({'error': 'An error occurred'}), 500

@app.route('/predict_dl', methods=['POST'])
def predict_dl():
    try:
        data = request.get_json(force=True)
        if 'features' not in data:
            return jsonify({'error': 'Invalid input'}), 400
        prediction = dl_model.predict(np.array(data['features']).reshape(1, -1))
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        logger.error(f"Error in /predict_dl: {e}")
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == "__main__":
    app.run(debug=True)
