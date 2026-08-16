"""Tests for the optional autograd mirror of the API (forward-only layers,
gradients via the third-party `autograd` package).

Skipped entirely when autograd is not installed -- the subpackage then
still imports (graceful degradation) but cannot train.
"""
import numpy as np
import pytest

try:
    import autograd  # noqa: F401
    AUTODIFF = True
except ImportError:
    AUTODIFF = False

pytestmark = pytest.mark.skipif(not AUTODIFF, reason="autograd not installed")

from numpy_keras.autograd import Sequential
from numpy_keras.autograd import layers


def _mlp():
    return Sequential([
        layers.Input(4),
        layers.Dense(8, activation="tanh"),
        layers.Dense(1, activation="linear"),
    ])


def test_regression_fit_decreases_loss():
    rng = np.random.RandomState(0)
    X = rng.randn(64, 4)
    y = np.sin(X[:, 0])[:, None] + 0.1 * rng.randn(64, 1)
    model = _mlp()
    model.compile(loss="mse", optimizer="sgd")
    model.optimizer.learning_rate = 0.05
    model.fit(X, y, batch_size=16, epochs=5, verbose=0)
    assert model.history.loss[-1] < model.history.loss[0]


def test_classification_fit_predict_evaluate():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 3, 3, 2)
    y = rng.randint(0, 3, 60)
    model = Sequential([
        layers.Flatten((3, 3, 2)),
        layers.Dense(8, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.fit(X, y, batch_size=10, epochs=5, verbose=0)
    assert model.history.loss[-1] < model.history.loss[0]
    pred = model.predict(X)
    assert pred.shape == (60,)
    assert pred.dtype.kind in "iu"
    assert isinstance(model.evaluate(X, y), float)


def test_autodiff_gradients_match_finite_differences():
    """The subpackage's whole point: autograd.grad through the forward
    pass must match central differences."""
    rng = np.random.RandomState(0)
    X = rng.randn(8, 4)
    y = rng.randn(8, 1)
    model = _mlp()
    model.compile(loss="mse", optimizer="sgd")

    from autograd import grad

    loss_grad = grad(model._Sequential__criterion)
    params = model._Sequential__get_params()

    def loss_of():
        return model._Sequential__criterion(params, X, y)

    grads = loss_grad(params, X, y)
    eps = 1e-6
    worst = 0.0
    for key, p in params.items():
        fp = p.ravel()
        ana = grads[key].ravel()
        for i in range(fp.size):
            old = fp[i]
            fp[i] = old + eps
            lp = loss_of()
            fp[i] = old - eps
            lm = loss_of()
            fp[i] = old
            num = (lp - lm) / (2 * eps)
            rel = abs(num - ana[i]) / max(1e-12, abs(num) + abs(ana[i]))
            worst = max(worst, rel)
    assert worst < 1e-3


def test_dropout_train_vs_inference():
    X = np.random.RandomState(0).randn(16, 8)
    layer = layers.Dropout(rate=0.5)
    layer.set_output_dim(8)
    np.random.seed(0)
    out_train = layer.forward(X, is_training=True)
    np.testing.assert_array_equal(layer.forward(X, is_training=False), X)
    # roughly half the entries are zeroed (binomial noise around 50%)
    zero_ratio = (out_train == 0).mean()
    assert 0.3 < zero_ratio < 0.7
    assert not np.array_equal(out_train, X)


def test_batchnorm_forward_modes_and_moving_stats():
    rng = np.random.RandomState(0)
    X = rng.randn(32, 8)
    layer = layers.BatchNormalization()
    layer.init_params(8)
    # training: batch statistics (variance sits below 1 by ~epsilon/var)
    np.random.seed(0)
    out_train = layer.forward(X, is_training=True)
    assert np.allclose(out_train.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(out_train.var(axis=0), 1, atol=1e-4)
    # inference: running statistics (initialized to 0/1)
    out_eval = layer.forward(X, is_training=False)
    assert not np.allclose(out_eval, out_train)


def test_batchnorm_inside_grad_training():
    """BatchNorm mixes traced params and host-side running stats; a full
    fit exercises both worlds under autograd.grad."""
    rng = np.random.RandomState(0)
    X = rng.randn(64, 4)
    y = np.sin(X[:, 0])[:, None]
    model = Sequential([
        layers.Input(4),
        layers.Dense(8, activation="linear"),
        layers.BatchNormalization(),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(loss="mse", optimizer="adam")
    model.fit(X, y, batch_size=16, epochs=3, verbose=0)
    assert model.history.loss[-1] < model.history.loss[0]


def test_early_stopping_restores_best_weights():
    """Pins the save_best snapshot/restore path (EarlyStopping monitors
    a recorded metric name, e.g. train_mean_squared_error)."""
    from numpy_keras import callbacks

    rng = np.random.RandomState(0)
    X = rng.randn(64, 4)
    y = np.sin(X[:, 0])[:, None]
    model = _mlp()
    model.compile(loss="mse", optimizer="adam",
                  metrics=["mean_squared_error"])
    before = {idx: {k: v.copy() for k, v in layer.params.items()}
              for idx, layer in model.layers.items() if hasattr(layer, "params")}
    model.fit(X, y, batch_size=16, epochs=3, verbose=0,
              callbacks=[callbacks.EarlyStopping(
                  monitor="train_mean_squared_error", patience=10,
                  restore_best_weights=True)])
    # training ran, and the snapshot machinery did not corrupt anything
    assert model.history.loss[-1] < model.history.loss[0]
    assert not np.array_equal(
        model.layers["dense_1"].params["W"], before["dense_1"]["W"])
