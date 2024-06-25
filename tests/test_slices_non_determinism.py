# pylint: disable=import-error
import pytest
from keras.models import load_model
from numpy.testing import assert_almost_equal
import dvc.api
import joblib

from scripts.train import build_model, train_model

model_path = 'outputs/train/model.keras'
params = dvc.api.params_show()


def load_data(params):
    preprocess_path = "outputs/preprocess/"

    return (
        joblib.load(preprocess_path + "x_train.joblib"),
        joblib.load(preprocess_path + "y_train.joblib"),
        joblib.load(preprocess_path + "x_val.joblib"),
        joblib.load(preprocess_path + "y_val.joblib"),
    )


@pytest.fixture(scope="module")
def data():
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
        _, accuracy = model.evaluate(x_val, y_val)
        results.append(accuracy)

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
        _, accuracy = model.evaluate(x_val, y_val)
        results.append(accuracy)

    mean_accuracy = sum(results) / len(results)

    for acc in results:
        assert_almost_equal(acc, mean_accuracy, decimal=2, err_msg="Model outputs vary too much")
