import os
import json
from pathlib import Path

import dvc.api
import joblib
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
from sklearn.metrics import (
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)

params = dvc.api.params_show()

preprocess_path = Path(params["dirs"]["outputs"]["preprocess"])
train_path = Path(params["dirs"]["outputs"]["train"])

evaluate_path = Path(params["dirs"]["outputs"]["evaluate"])
os.makedirs(evaluate_path, exist_ok=True)

encoder = joblib.load(preprocess_path / "encoder.joblib")
model = load_model(train_path / "model.keras")

x_test = joblib.load(preprocess_path / "x_test.joblib")[:params["evaluate"]["slice"]]
y_test = joblib.load(preprocess_path / "y_test.joblib")[:params["evaluate"]["slice"]]

y_pred = model.predict(x_test, batch_size=params["evaluate"]["batch_size"]).flatten()
y_pred_binary = y_pred.round().astype(int)

RocCurveDisplay.from_predictions(y_test, y_pred).plot()
plt.savefig(evaluate_path / "roc_curve.png")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_binary, display_labels=encoder.classes_).plot()
plt.savefig(evaluate_path / "conf_matrix.png")

metrics = {
  **classification_report(y_test, y_pred_binary, target_names=encoder.classes_, output_dict=True),
  "auc": roc_auc_score(y_test, y_pred)
}

with open(evaluate_path / "metrics.json", "w") as fp:
  json.dump(metrics, fp)
