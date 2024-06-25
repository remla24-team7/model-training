import pytest
import joblib
from pathlib import Path
from keras.models import load_model
from numpy.testing import assert_almost_equal
import dvc.api

# Assuming build_model, train_model are from your model training script
from scripts.train import build_model, train_model, load_data

# Path to the parameters and model if needed
model_path = 'outputs/train/model.keras'
params = dvc.api.params_show()

@pytest.fixture(scope="module")
def data():
    params = dvc.api.params_show()
    return load_data(params)

@pytest.fixture(scope="module")
def trained_model():
    return load_model(model_path)

def test_model_slices(data):
    x_train, y_train, x_val, y_val = data

    results = []

    for i in range(2):
        low = i * 100
        end = low * 2
        model = build_model(params)
        train_model(model, params, x_train[low:end], y_train[low:end], validation_data=(x_val[low:end], y_val[low:end]))
        loss, accuracy = model.evaluate(x_val, y_val)
        results.append(accuracy)

    mean_accuracy = sum(results) / len(results)

    for acc in results:
        assert len(results) == len(set(results)), "Not all accuracies are unique."


def test_model_non_determinism(data):
    x_train, y_train, x_val, y_val = data
    x_train = x_train[:500]
    y_train = y_train[:500]
    x_val = x_val[:100]
    y_val = y_val[:100]

    results = []

    for _ in range(2):
        model = build_model(params)
        train_model(model, params, x_train, y_train, validation_data=(x_val, y_val))
        loss, accuracy = model.evaluate(x_val, y_val)
        results.append(accuracy)

    mean_accuracy = sum(results) / len(results)

    for acc in results:
        assert_almost_equal(acc, mean_accuracy, decimal=2, err_msg="Model outputs vary too much")

