# Predictive Maintenance Console

A machine failure prediction project: a trained Random Forest model
(AI4I 2020 dataset) served through a Flask API, with a dashboard
frontend styled like an industrial control panel.

## 1. Where to put this folder

Unzip the project anywhere you like, e.g.:

```
C:\Users\<you>\Documents\predictive-maintenance-project
```

Then in **VS Code**: `File → Open Folder...` and select
`predictive-maintenance-project`. That folder becomes your workspace —
you do not need to move it into any special VS Code directory.

## 2. Folder structure

```
predictive-maintenance-project/
│
├── data/
│   └── predictive_maintenance.csv   <-- PUT YOUR DOWNLOADED CSV HERE
│
├── model/                            <-- created automatically after training
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── type_encoder.pkl
│   └── metrics.json
│
├── static/
│   ├── css/style.css
│   └── js/main.js
│
├── templates/
│   └── index.html
│
├── app.py            <-- Flask server (frontend + prediction API)
├── train_model.py     <-- trains the model from the CSV
├── requirements.txt
└── README.md
```

## 3. Put the dataset in place

Take the CSV file you downloaded from Kaggle
(`predictive-maintenance-dataset-ai4i-2020`) and rename/place it as:

```
data/predictive_maintenance.csv
```

The script expects these columns (this is exactly what the AI4I 2020
CSV contains): `Type`, `Air temperature [K]`, `Process temperature [K]`,
`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`,
`Machine failure`.

## 4. Set up the environment (in VS Code's terminal)

Open a terminal in VS Code (`` Ctrl+` ``) and run:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## 5. Train the model

```bash
python train_model.py
```

This prints accuracy/F1/ROC-AUC, a classification report, and feature
importances, then saves the trained model into `model/`.

## 6. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Move the sliders to
set sensor readings, click **Run Diagnostic**, and the gauge + risk
breakdown update from a live call to the model.

## Notes

- The dataset is imbalanced (~3% failures), so the model uses
  `class_weight="balanced"` and is evaluated with F1/ROC-AUC rather
  than accuracy alone.
- The "risk breakdown" bars (tool wear / heat dissipation / power /
  overstrain) are simple rule-of-thumb indicators based on the known
  AI4I failure-mode thresholds — they're illustrative, not separate
  trained models. You could extend the project by training one
  classifier per failure mode using the `TWF`, `HDF`, `PWF`, `OSF`
  columns for a stronger version of this.
- To retrain after changing features, just rerun `train_model.py` —
  it overwrites the files in `model/`.
