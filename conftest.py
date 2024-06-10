# conftest.py
import pytest


@pytest.fixture
def params():
    return {
        "dirs": {
            "dataset": "dataset",
            "outputs": {
                "preprocess": "outputs/preprocess",
                "train": "outputs/train",
                "evaluate": "outputs/evaluate",
                "mutamorphic": "outputs/mutamorphic",
            },
        },
        "preprocess": {"sequence_length": 200},
        "train": {
            "embedding_dim": 50,
            "loss_function": "binary_crossentropy",
            "optimizer": "adam",
            "metrics": ["accuracy"],
            "batch_size": 5000,
            "epochs": 2,
        },
        "evaluate": {"slice": 500, "batch_size": 50},
    }
