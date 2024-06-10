import os
import json
from pathlib import Path

import dvc.api
import joblib

from keras.src.models import Sequential
from keras.src.layers import Embedding, Conv1D, MaxPooling1D, Dropout, Flatten, Dense


def load_data(params):
    preprocess_path = Path(params["dirs"]["outputs"]["preprocess"])

    return (
        joblib.load(preprocess_path / "x_train.joblib"),
        joblib.load(preprocess_path / "y_train.joblib"),
        joblib.load(preprocess_path / "x_val.joblib"),
        joblib.load(preprocess_path / "y_val.joblib"),
    )


def build_model(params):
    preprocess_path = Path(params["dirs"]["outputs"]["preprocess"])

    tokenizer = joblib.load(preprocess_path / "tokenizer.joblib")
    encoder = joblib.load(preprocess_path / "encoder.joblib")

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


def train_model(model, params, x_train, y_train, validation_data=None):
    model.compile(
        loss=params["train"]["loss_function"],
        optimizer=params["train"]["optimizer"],
        metrics=params["train"]["metrics"],
    )

    return model.fit(
        x_train, y_train,
        validation_data=validation_data,
        batch_size=params["train"]["batch_size"],
        epochs=params["train"]["epochs"],
    )


if __name__ == "__main__":
    params = dvc.api.params_show()

    train_path = Path(params["dirs"]["outputs"]["train"])
    os.makedirs(train_path, exist_ok=True)

    x_train, y_train, x_val, y_val = load_data(params)
    model = build_model(params)
    history = train_model(model, params, x_train, y_train, validation_data=(x_val, y_val))

    model.save(train_path / "model.keras")
    with open(train_path / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history.history, fp)
