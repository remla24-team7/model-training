import os
import json
from pathlib import Path

import dvc.api
import joblib

from keras.src.models import Sequential
from keras.src.layers import Embedding, Conv1D, MaxPooling1D, Dropout, Flatten, Dense

params = dvc.api.params_show()

preprocess_path = Path(params["dirs"]["outputs"]["preprocess"])

train_path = Path(params["dirs"]["outputs"]["train"])
os.makedirs(train_path, exist_ok=True)

tokenizer = joblib.load(preprocess_path / "tokenizer.joblib")
encoder = joblib.load(preprocess_path / "encoder.joblib")

input_dim = len(tokenizer.word_index.keys()) + 1
output_dim = len(encoder.classes_) - 1

model = Sequential([
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

model.compile(
    loss=params["train"]["loss_function"],
    optimizer=params["train"]["optimizer"],
    metrics=params["train"]["metrics"],
)

x_train = joblib.load(preprocess_path / "x_train.joblib")
y_train = joblib.load(preprocess_path / "y_train.joblib")

x_val = joblib.load(preprocess_path / "x_val.joblib")
y_val = joblib.load(preprocess_path / "y_val.joblib")

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=params["train"]["batch_size"],
    epochs=params["train"]["epochs"],
)

model.save(train_path / "model.keras")
with open(train_path / "history.json", "w") as fp:
    json.dump(history.history, fp)
