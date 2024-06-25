# pylint: disable=import-error
import pytest
from numpy.testing import assert_almost_equal
import dvc.api
import joblib
from keras.models import load_model
from keras.src.models import Sequential
from keras.src.layers import Embedding, Conv1D, MaxPooling1D, Dropout, Flatten, Dense

model_path = 'outputs/train/model.keras'
params = dvc.api.params_show()


# Copied code for testing
def load_data():
    preprocess_path = "outputs/preprocess/"

    return (
        joblib.load(preprocess_path + "x_train.joblib"),
        joblib.load(preprocess_path + "y_train.joblib"),
        joblib.load(preprocess_path + "x_val.joblib"),
        joblib.load(preprocess_path + "y_val.joblib"),
    )


# Copied code for testing
def build_model(params):
    preprocess_path = "outputs/preprocess/"

    tokenizer = joblib.load(preprocess_path + "tokenizer.joblib")
    encoder = joblib.load(preprocess_path + "encoder.joblib")

    input_dim = len(tokenizer.word_index.keys()) + 1
    output_dim = len(encoder.classes_) - 1

    return Sequential([
        Embedding(input_dim, params["train"]["embedding_dim"]),

        Conv1D(128, 3, activation="tanh"),
        MaxPooling1D(3),
        Dropout(0.2),

        Conv1D(128, 7, activation="tanh", padding="same"),
        Dropout(0.2),

        Conv1D(128, 5, activation="tanh", padding="same"),
        Dropout(0.2),

        Conv1D(128, 3, activation="tanh", padding="same"),
        MaxPooling1D(3),
        Dropout(0.2),

        Conv1D(128, 5, activation="tanh", padding="same"),
        Dropout(0.2),

        Conv1D(128, 3, activation="tanh", padding="same"),
        MaxPooling1D(3),
        Dropout(0.2),

        Conv1D(128, 3, activation="tanh", padding="same"),
        MaxPooling1D(3),
        Dropout(0.2),

        Flatten(),
        Dense(output_dim, activation="sigmoid"),
    ])


# Copied code for testing
def train_model(model, x_train, y_train, validation_data=None):
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=['accuracy'],
    )

    return model.fit(
        x_train, y_train,
        validation_data=validation_data,
        batch_size=50,
        epochs=4,
    )


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def trained_model():
    return load_model(model_path)


def test_model_slices(data):
    x_train, y_train, x_val, y_val = data

    results = []

    for i in range(2):
        low = i * 100 + 100
        end = low * 2
        model = build_model(params)
        train_model(model, x_train[low:end], y_train[low:end], validation_data=(x_val[low:end], y_val[low:end]))
        _, accuracy = model.evaluate(x_val, y_val)
        results.append(accuracy)

    assert len(results) == len(set(results)), "Not all accuracies are unique."


def test_model_non_determinism(data):
    x_train, y_train, x_val, y_val = data
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    x_val = x_val[:100]
    y_val = y_val[:100]

    results = []

    for _ in range(2):
        model = build_model(params)
        train_model(model, x_train, y_train, validation_data=(x_val, y_val))
        _, accuracy = model.evaluate(x_val, y_val)
        results.append(accuracy)

    difference = abs(results[1] - results[0])
    print(difference)
    assert difference < 0.06
