import dvc.api
from scripts.evaluate import evaluate_model


def test_evaluate_model():
    params = dvc.api.params_show()
    roc_curve, confusion_matrix, metrics = evaluate_model(params)
    assert roc_curve is not None, "ROC curve should not be None"
    assert confusion_matrix is not None, "Confusion matrix should not be None"
    assert 0 <= metrics["accuracy"] <= 1, "Accuracy should be between 0 and 1"
    assert 0 <= metrics["auc"] <= 1, "AUC should be between 0 and 1"
