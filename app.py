print("✅ app.py is running")

from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ✅ Load your trained ML model
model = pickle.load(open("stress_predictor.pkl", "rb"))

# ✅ Add a simple homepage route
@app.route("/")
def home():
    return "Hello from MindMate API!"

# ✅ Your prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([[data["mood"], data["sentiment"], data["usage"]]])
    prediction = model.predict(features)[0]
    
    stress_map = {0: "Low Stress 🟢", 1: "Moderate Stress 🟡", 2: "High Stress 🔴"}
    return jsonify({"stress_level": stress_map[prediction]})

def home():
    return "Hello from MindMate API!"
# ✅ Start the Flask server
if __name__ == "__main__":
    print("✅ app.py is running")
    print("🚀 Flask server starting...")
    app.run(host="0.0.0.0", port=5000, debug=True)

