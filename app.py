# File: /app.py
# This file contains the main application code.
# It is responsible for orchestrating the model training and evaluation process.
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("best_model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(np.array(data['features']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "main":
    app.run(debug=True)
