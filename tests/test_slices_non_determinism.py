# pylint: disable=import-error
import pytest
import dvc.api
import joblib
import numpy as np
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


def process_urls(filename):
    phishing_indices = []
    legitimate_indices = []
    index = 0

    with open(filename, 'r') as file: # pylint: disable=unspecified-encoding
        for line in file:
            if len(line.strip().split('\t')) == 2:
                label, _ = line.strip().split('\t')
                if label == "phishing":
                    phishing_indices.append(index)
                if label == "legitimate":
                    legitimate_indices.append(index)
            index += 1

    return phishing_indices, legitimate_indices


def create_subset(data_array, phishing_indices, legitimate_indices):
    subset = []
    combined_indices = set(phishing_indices + legitimate_indices)
    for idx in combined_indices:
        subset.append(data_array[idx])
    return np.array(subset)


def test_slices(data):
    phishing_indices, legitimate_indices = process_urls('dataset/train.txt')

    x_train, y_train, x_val, y_val = data

    # First subset 50-50
    x_train_50 = create_subset(x_train, phishing_indices[:500], legitimate_indices[:500])
    y_train_50 = create_subset(y_train, phishing_indices[:500], legitimate_indices[:500])
    assert len(x_train_50) == len(y_train_50) == 1000

    # Second subset 70-30
    x_train_70 = create_subset(x_train, phishing_indices[:300], legitimate_indices[:700])
    y_train_70 = create_subset(y_train, phishing_indices[:300], legitimate_indices[:700])
    assert len(x_train_70) == len(y_train_70) == 1000

    model = build_model(params)
    train_model(model, x_train_50, y_train_50, validation_data=(x_val[:1000], y_val[:1000]))
    _, accuracy_50 = model.evaluate(x_val, y_val)
    model = build_model(params)
    train_model(model, x_train_70, y_train_70, validation_data=(x_val[:1000], y_val[:1000]))
    _, accuracy_70 = model.evaluate(x_val, y_val)

    assert accuracy_50 > 0.6
    assert accuracy_70 > 0.6
    assert abs(accuracy_70 - accuracy_50) < 0.1

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
