"""
app.py
------
Flask web app that serves the dashboard frontend and a /predict API
that uses the trained model in model/model.pkl.

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_DIR = "model"
model = None
scaler = None
type_encoder = None
metrics = {}


def load_artifacts():
    global model, scaler, type_encoder, metrics
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    encoder_path = os.path.join(MODEL_DIR, "type_encoder.pkl")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        type_encoder = joblib.load(encoder_path)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)


load_artifacts()


@app.route("/")
def index():
    return render_template("index.html", metrics=metrics, model_ready=model is not None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained yet. Run train_model.py first."}), 400

    data = request.get_json()

    try:
        machine_type = data["type"]  # "L", "M", or "H"
        air_temp = float(data["air_temp"])
        process_temp = float(data["process_temp"])
        rpm = float(data["rpm"])
        torque = float(data["torque"])
        tool_wear = float(data["tool_wear"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    type_encoded = type_encoder.transform([machine_type])[0]
    temp_diff = process_temp - air_temp
    # Power in Watts = Torque [Nm] * angular velocity [rad/s]
    power = torque * rpm * (2 * np.pi / 60)

    feature_cols = metrics.get("feature_cols", [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        "Temp_diff", "Power",
    ])
    row = [type_encoded, air_temp, process_temp, rpm, torque, tool_wear, temp_diff, power]
    features_df = pd.DataFrame([row], columns=feature_cols)
    features_scaled = scaler.transform(features_df)

    proba = model.predict_proba(features_scaled)[0][1]
    prediction = int(proba >= 0.5)

    # Simple heuristic risk breakdown for the frontend, based on known
    # AI4I failure-mode thresholds (illustrative, not a substitute for
    # per-failure-mode models)
    risk_breakdown = {
        "Tool Wear Failure": min(100, max(0, (tool_wear - 200) / 40 * 100)) if tool_wear > 200 else 0,
        "Heat Dissipation Failure": min(100, max(0, (100 - abs(temp_diff - 8.5) * 20))) if temp_diff < 8.6 and rpm < 1380 else 0,
        "Power Failure": min(100, max(0, (3500 - power) / 15)) if power < 3500 else (min(100, max(0, (power - 9000) / 20)) if power > 9000 else 0),
        "Overstrain Failure": min(100, max(0, (torque * tool_wear - 11000) / 100)) if torque * tool_wear > 11000 else 0,
    }

    return jsonify({
        "prediction": prediction,
        "probability": round(float(proba) * 100, 2),
        "status": "FAILURE RISK" if prediction == 1 else "NORMAL",
        "risk_breakdown": {k: round(v, 1) for k, v in risk_breakdown.items()},
    })


@app.route("/metrics")
def get_metrics():
    return jsonify(metrics)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
