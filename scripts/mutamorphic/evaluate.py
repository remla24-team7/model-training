from pathlib import Path

import dvc.api
import joblib

from scripts.evaluate import evaluate_model, save_metrics

if __name__ == "__main__":
    params = dvc.api.params_show()

    preprocess_path = Path(params["dirs"]["outputs"]["preprocess"])
    mutamorphic_path = Path(params["dirs"]["outputs"]["mutamorphic"])

    x_test = joblib.load(mutamorphic_path / "x_test.joblib")
    y_test = joblib.load(preprocess_path / "y_test.joblib")

    roc_curve, confusion_matrix, metrics = evaluate_model(x_test, y_test, params)
    save_metrics(roc_curve, confusion_matrix, metrics, evaluate_path=mutamorphic_path)
