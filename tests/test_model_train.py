"""
Test model train fucntion
"""
import pytest
from joblib import load
from src.models.model_train import load_data, train_model, build_model


def test_load_data():
    x_train, y_train, x_val, y_val, char_index = load_data()
    assert x_train is not None, "x_train should not be None"
    assert y_train is not None, "y_train should not be None"
    assert x_val is not None, "x_val should not be None"
    assert y_val is not None, "y_val should not be None"
    assert char_index is not None, "char_index should not be None"


def test_build_model():
    char_index = load('outputs/char_index.joblib')
    params = {'loss_function': 'binary_crossentropy',
              'optimizer': 'adam',
              'sequence_length': 200,
              'batch_train': 5000,
              'batch_test': 5000,
              'categories': ['phishing', 'legitimate'],
              'char_index': None,
              'epoch': 30,
              'embedding_dimension': 50,
              'dataset_dir': "dataset/small_dataset/"
              }
    voc_size = len(char_index.keys())
    model = build_model(voc_size, params['categories'], params['embedding_dimension'])
    assert model is not None, "Model should be initialized"


def test_train_model():
    x_train, y_train, x_val, y_val, char_index = load_data()
    params = {
        'loss_function': 'binary_crossentropy',
        'optimizer': 'adam',
        'batch_train': 10,
        'batch_test': 10,
        'categories': ['phishing', 'legitimate'],
        'epoch': 1,
        'embedding_dimension': 50
    }
    model, history = train_model(x_train[:10], y_train[:10], x_val[:10], y_val[:10], params, char_index)
    assert history.history['loss'][0] is not None, "Training did not produce a loss"
    assert len(history.history['accuracy']) > 0, "Training did not produce accuracy metrics"
