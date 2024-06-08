from scripts.train import load_data, train_model, build_model
import dvc.api


def test_load_data():
    params = dvc.api.params_show()
    x_train, y_train, x_val, y_val = load_data(params)
    assert x_train is not None, "x_train should not be None"
    assert y_train is not None, "y_train should not be None"
    assert x_val is not None, "x_val should not be None"
    assert y_val is not None, "y_val should not be None"


def test_build_model():
    params = dvc.api.params_show()
    model = build_model(params)
    assert model is not None, "Model should be initialized"


def test_train_model():
    params = dvc.api.params_show()
    x_train, y_train, x_val, y_val = load_data(params)
    model = build_model(params)
    history = train_model(model, params, x_train[:10], y_train[:10], validation_data=(x_val[:10], y_val[:10]))
    assert history.history["loss"][0] is not None, "Training did not produce a loss"
    assert history.history["accuracy"][0] is not None, "Training did not produce accuracy metrics"
