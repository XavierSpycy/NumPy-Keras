"""End-to-end tests for Sequential: training, prediction, and API behaviour."""
import numpy as np
import pytest

from numpy_keras import Sequential
from numpy_keras import layers
from numpy_keras.callbacks import EarlyStopping


def build_regression_model(hidden_units=16, lr=0.01):
    model = Sequential()
    model.add(layers.Input(2))
    model.add(layers.Dense(hidden_units, activation="relu"))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="adam")
    model.optimizer.learning_rate = lr
    return model


def test_toy_regression_fit_reduces_loss():
    np.random.seed(0)
    rng = np.random.RandomState(0)
    X = rng.randn(200, 2)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 + 0.1 * rng.randn(200)
    y = y.reshape(-1, 1)

    model = build_regression_model(hidden_units=8, lr=0.1)
    history = model.fit(X, y, batch_size=32, epochs=150, shuffle=False)
    assert len(history.loss) == 150
    assert history.loss[-1] < history.loss[0] * 0.5


def test_regression_predict_shape_and_fit_quality():
    np.random.seed(1)
    rng = np.random.RandomState(1)
    X = rng.randn(100, 2)
    y = (2 * X[:, 0] - X[:, 1]).reshape(-1, 1)

    model = build_regression_model(hidden_units=8, lr=0.1)
    model.fit(X, y, batch_size=32, epochs=200, shuffle=False)
    pred = model.predict(X, batch_size=32)
    assert pred.shape == (100,)
    # simple linear target: a small net should get close
    assert np.mean((pred - y.ravel()) ** 2) < 0.5


def test_xor_classification_reaches_high_accuracy():
    np.random.seed(2)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])

    model = Sequential()
    model.add(layers.Input(2))
    model.add(layers.Dense(8, activation="tanh"))
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.optimizer.learning_rate = 0.2
    history = model.fit(X, y, batch_size=4, epochs=400, shuffle=False)
    assert history.metrics["train_accuracy"][-1] > 0.9
    pred = model.predict(X)
    assert pred.shape == (4,)
    assert np.all(np.isin(pred, [0, 1]))
    assert np.mean(pred == y) > 0.9


def test_model_gradient_check_against_finite_differences():
    """Backprop grads must equal finite differences of the reported loss."""
    np.random.seed(3)
    rng = np.random.RandomState(3)
    model = Sequential()
    model.add(layers.Input(2))
    model.add(layers.Dense(3, activation="tanh"))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="sgd")
    for layer in model.layers.values():
        if hasattr(layer, "params"):
            for k in layer.params:
                layer.params[k] = rng.randn(*layer.params[k].shape) * 0.3

    X = rng.randn(5, 2)
    y = rng.randn(5, 1) * 0.5

    y_hat = model._Sequential__forward(X, is_training=True)
    _, grad = model._Sequential__criterion(y, y_hat)
    model._Sequential__backward(grad)

    eps = 1e-6
    for name, layer in model.layers.items():
        if not hasattr(layer, "params"):
            continue
        for k, p in layer.params.items():
            numerical = np.zeros_like(p.ravel())
            for i in range(p.size):
                orig = p.ravel()[i]
                p.ravel()[i] = orig + eps
                l1 = model._Sequential__loss_func(y, model._Sequential__forward(X, is_training=False))
                p.ravel()[i] = orig - eps
                l2 = model._Sequential__loss_func(y, model._Sequential__forward(X, is_training=False))
                p.ravel()[i] = orig
                numerical[i] = (l1 - l2) / (2 * eps)
            np.testing.assert_allclose(layer.grads[k].ravel(), numerical, rtol=1e-3, atol=1e-6, err_msg=f"{name}.{k}")


def test_predict_multiclass_one_hot_returns_probabilities():
    np.random.seed(4)
    rng = np.random.RandomState(4)
    model = Sequential()
    model.add(layers.Input(2))
    model.add(layers.Dense(4, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    X = rng.randn(6, 2)
    y = np.eye(3)[rng.randint(0, 3, 6)]
    model.fit(X, y, batch_size=3, epochs=1)
    pred = model.predict(X)
    assert pred.shape == (6, 3)
    np.testing.assert_allclose(pred.sum(axis=1), 1.0, atol=1e-6)


def test_evaluate_returns_loss_close_to_final_training_loss():
    np.random.seed(5)
    rng = np.random.RandomState(5)
    X = rng.randn(60, 2)
    y = (X[:, 0] - X[:, 1]).reshape(-1, 1)

    model = build_regression_model(hidden_units=8, lr=0.1)
    history = model.fit(X, y, batch_size=12, epochs=30, shuffle=False)
    loss = model.evaluate(X, y)
    assert np.isscalar(loss)
    # history.loss is the mean of per-batch losses (each computed mid-update),
    # so it matches the full-data recomputation only approximately
    assert np.isclose(loss, history.loss[-1], atol=1e-3)


def test_history_structure_and_getitem():
    np.random.seed(6)
    rng = np.random.RandomState(6)
    X = rng.randn(30, 2)
    y = rng.randn(30, 1)
    model = build_regression_model(hidden_units=4, lr=0.01)
    history = model.fit(X, y, batch_size=8, epochs=5)
    assert len(history["loss"]) == 5
    assert history["loss"] is history.loss


def test_validation_split_populates_val_loss():
    np.random.seed(7)
    rng = np.random.RandomState(7)
    X = rng.randn(50, 2)
    y = rng.randn(50, 1)
    model = build_regression_model(hidden_units=4, lr=0.01)
    history = model.fit(X, y, batch_size=10, epochs=3, validation_split=0.2)
    assert "val_loss" in history.metrics
    assert len(history.metrics["val_loss"]) == 3


def test_steps_per_epoch_limits_batches():
    np.random.seed(8)
    rng = np.random.RandomState(8)
    X = rng.randn(30, 2)
    y = rng.randn(30, 1)
    model = build_regression_model(hidden_units=4, lr=0.01)
    history = model.fit(X, y, batch_size=10, epochs=2, steps_per_epoch=1, shuffle=False)
    # each epoch sees only the first batch of 10 samples
    assert len(history.loss) == 2


def test_add_and_pop_layers():
    model = Sequential()
    model.add(layers.Input(2))
    model.add(layers.Dense(3, activation="relu"))
    model.add(layers.Dense(1, activation="linear"))
    assert list(model.layers.keys())[-1] == "dense_2"
    model.pop()
    assert list(model.layers.keys())[-1] == "dense_1"


def test_summary_prints_table(capsys):
    model = build_regression_model(hidden_units=4, lr=0.01)
    model.summary()
    out = capsys.readouterr().out
    assert "Model: Sequential" in out
    assert "Total params" in out
    assert "input_1" in out


def test_fit_without_compile_raises():
    model = Sequential()
    model.add(layers.Input(2))
    model.add(layers.Dense(1, activation="linear"))
    with pytest.raises(AttributeError):
        model.fit(np.random.randn(8, 2), np.random.randn(8, 1), epochs=1)


def test_sparse_labels_accept_column_vector():
    """Labels shaped (n, 1) -- a very common data layout -- must be accepted."""
    np.random.seed(10)
    rng = np.random.RandomState(10)
    X = rng.randn(40, 2)
    y = rng.randint(0, 2, 40).reshape(-1, 1)

    model = Sequential()
    model.add(layers.Input(2))
    model.add(layers.Dense(8, activation="tanh"))
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    model.optimizer.learning_rate = 0.1
    history = model.fit(X, y, batch_size=8, epochs=30)
    assert len(history.loss) == 30
    pred = model.predict(X)
    assert pred.shape == (40,)


def test_single_dense_model_can_fit():
    """A model with a single Dense layer must train (no previous-layer
    activation deriv to fall back on in the criterion)."""
    np.random.seed(9)
    rng = np.random.RandomState(9)
    X = rng.randn(80, 2)
    y = (1.5 * X[:, 0] - 0.5 * X[:, 1]).reshape(-1, 1)

    model = Sequential()
    model.add(layers.Input(2))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="sgd")
    model.optimizer.learning_rate = 0.1
    history = model.fit(X, y, batch_size=16, epochs=200, shuffle=False)
    assert history.loss[-1] < history.loss[0] * 0.5
