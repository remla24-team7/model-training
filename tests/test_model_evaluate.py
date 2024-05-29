# test_evaluate_model.py
import pytest
from src.models.model_evaluate import evaluate_model

def test_evaluate_model():
    model_path = 'outputs/model.h5'
    x_test_path = 'outputs/x_test.joblib'
    y_test_path = 'outputs/y_test.joblib'
    report, confusion_mat, accuracy, auc = evaluate_model(model_path, x_test_path, y_test_path, True)

    assert confusion_mat.size > 0, "Confusion matrix should not be empty"
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    assert 0 <= auc <= 1, "AUC should be between 0 and 1"
