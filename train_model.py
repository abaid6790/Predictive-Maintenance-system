"""
train_model.py
----------------
Trains a machine-failure prediction model on the AI4I 2020 Predictive
Maintenance dataset and saves the trained pipeline to model/model.pkl.

Dataset columns expected (AI4I 2020 Predictive Maintenance Dataset):
UDI, Product ID, Type, Air temperature [K], Process temperature [K],
Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure,
TWF, HDF, PWF, OSF, RNF

Download: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
Place the CSV file at: data/predictive_maintenance.csv
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
)
import joblib

DATA_PATH = os.path.join("data", "predictive_maintenance.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COL = "Machine failure"


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nCould not find dataset at '{path}'.\n"
            "Download it from:\n"
            "https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020\n"
            "and place the CSV inside the 'data' folder as 'predictive_maintenance.csv'."
        )
    df = pd.read_csv(path)
    return df


def build_features(df):
    df = df.copy()
    # Extra engineered features that tend to help tree models here
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    # Power in Watts = Torque [Nm] * angular velocity [rad/s]
    df["Power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    return df


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    df = build_features(df)

    feature_cols = FEATURE_COLS + ["Temp_diff", "Power"]

    # Encode categorical "Type" column (L/M/H)
    le = LabelEncoder()
    df["Type"] = le.fit_transform(df["Type"])

    X = df[feature_cols]
    y = df[TARGET_COL]

    print(f"Dataset shape: {df.shape}")
    print(f"Failure rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",  # dataset is imbalanced (~3% failures)
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n--- Evaluation ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    importances = dict(zip(feature_cols, model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
    print("\nFeature importances:")
    for k, v in importances.items():
        print(f"  {k}: {v:.4f}")

    # Save model, scaler, encoder, and metadata together
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "type_encoder.pkl"))

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": auc,
        "failure_rate": float(y.mean()),
        "n_samples": int(df.shape[0]),
        "feature_importances": importances,
        "feature_cols": feature_cols,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model, scaler, encoder, and metrics.json to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
