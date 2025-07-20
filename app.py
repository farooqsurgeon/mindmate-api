print("✅ app.py is running")

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ✅ Load trained ML model with fallback path support
try:
    model = pickle.load(open("stress_predictor.pkl", "rb"))
except FileNotFoundError:
    # Fallback if file is inside a subfolder like 'ml_model/'
    model_path = os.path.join(os.path.dirname(__file__), "ml_model", "stress_predictor.pkl")
    model = pickle.load(open(model_path, "rb"))

# ✅ Homepage route
@app.route("/")
def home():
    return "Hello from MindMate API!"

# ✅ Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        mood = float(data.get("mood", 0))
        sentiment = float(data.get("sentiment", 0))
        usage = float(data.get("usage", 0))

        features = np.array([[mood, sentiment, usage]])
        prediction = model.predict(features)[0]

        stress_map = {
            0: "Low Stress",
            1: "Moderate Stress",
            2: "High Stress"
        }

        print("✅ Prediction made:", stress_map.get(prediction, "Unknown"))
        return jsonify({"stress_level": stress_map.get(prediction, "Unknown")})

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500
