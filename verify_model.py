"""
verify_model.py
----------------
Sanity-check the trained model against real rows from your dataset.
Run this from inside the project folder AFTER training:

    python verify_model.py
"""
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("data/predictive_maintenance.csv")
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
le = joblib.load("model/type_encoder.pkl")

df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["Power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
df["TypeEnc"] = le.transform(df["Type"])

feature_cols = ["TypeEnc", "Air temperature [K]", "Process temperature [K]",
                 "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
                 "Temp_diff", "Power"]

failures = df[df["Machine failure"] == 1].sample(min(5, (df["Machine failure"] == 1).sum()), random_state=1)
normals = df[df["Machine failure"] == 0].sample(5, random_state=1)
sample = pd.concat([failures, normals])

X = sample[feature_cols].copy()
X.columns = ["Type", "Air temperature [K]", "Process temperature [K]",
             "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
             "Temp_diff", "Power"]
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)[:, 1]

sample = sample.assign(Predicted=preds, Probability=probs.round(3))
print(sample[["Machine failure", "Predicted", "Probability",
              "Tool wear [min]", "Torque [Nm]", "Rotational speed [rpm]"]]
      .to_string(index=False))

correct = (sample["Machine failure"] == sample["Predicted"]).sum()
print(f"\n{correct}/{len(sample)} matched the real label.")
