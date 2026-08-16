"""Tests for callbacks: EarlyStopping and learning-rate schedulers."""
import copy

import numpy as np
import pytest

from numpy_keras import Sequential
from numpy_keras import layers
from numpy_keras.callbacks import (
    EarlyStopping,
    StepLR,
    ExponentialLR,
    MultiplicativeLR,
    ConstantLR,
    LinearLR,
)


def build_model(lr=0.1):
    model = Sequential()
    model.add(layers.Input(2))
    model.add(layers.Dense(4, activation="tanh"))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="sgd")
    model.optimizer.learning_rate = lr
    return model


class ParamsRecorder:
    """Snapshot every parameter at the end of each epoch."""

    def __init__(self):
        self.snapshots = []

    def on_epoch_end(self, model, *args, **kwargs):
        snapshot = {}
        for idx, layer in model.layers.items():
            if hasattr(layer, "params"):
                snapshot[idx] = {k: v.copy() for k, v in layer.params.items()}
        self.snapshots.append(snapshot)


def params_equal(model, snapshot):
    for idx, params in snapshot.items():
        for k, v in params.items():
            if not np.allclose(model.layers[idx].params[k], v):
                return False
    return True


def test_early_stopping_stops_training():
    np.random.seed(0)
    rng = np.random.RandomState(0)
    X = rng.randn(40, 2)
    y = rng.randn(40, 1)

    model = build_model(lr=0.0)  # frozen model: val_loss never improves
    history = model.fit(
        X, y, batch_size=8, epochs=20,
        validation_data=(X, y),
        callbacks=[EarlyStopping(monitor="val_loss", patience=2)],
    )
    assert len(history.loss) < 20
    assert model.stop_training


def test_early_stopping_can_monitor_epoch_loss():
    """monitor='loss' reads the epoch training loss (no metric needed)."""
    np.random.seed(0)
    rng = np.random.RandomState(0)
    X = rng.randn(40, 2)
    y = rng.randn(40, 1)

    model = build_model(lr=0.0)  # frozen model: loss never improves
    history = model.fit(
        X, y, batch_size=8, epochs=20,
        callbacks=[EarlyStopping(monitor="loss", patience=2)],
    )
    assert len(history.loss) < 20
    assert model.stop_training


def test_early_stopping_restore_best_weights():
    """With restore_best_weights=True the model must end with the parameters
    of the epoch where the monitored metric was best."""
    np.random.seed(1)
    rng = np.random.RandomState(1)
    X_train = rng.randn(60, 2)
    y_train = rng.randn(60, 1)
    # A different task for validation: val_loss improves only on the first
    # monitored epoch, then degrades as the model fits the training task.
    X_val = rng.randn(20, 2) + 10
    y_val = rng.randn(20, 1)

    model = build_model(lr=0.5)
    recorder = ParamsRecorder()
    model.fit(
        X_train, y_train, batch_size=8, epochs=15,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True),
                   recorder],
    )
    val_losses = model.history.metrics["val_loss"]
    assert len(recorder.snapshots) == 15
    # the model must actually move during training, otherwise this test
    # passes vacuously
    assert not params_equal(model, recorder.snapshots[-1]) or \
        not np.allclose(val_losses, val_losses[0])
    # epoch 0 is skipped by EarlyStopping (start_from_epoch); the best epoch
    # is the one achieving the running minimum from epoch 1 onwards
    best_epoch = 1 + int(np.argmin(val_losses[1:]))
    assert params_equal(model, recorder.snapshots[best_epoch])


def test_step_lr_decays_learning_rate():
    np.random.seed(2)
    rng = np.random.RandomState(2)
    X = rng.randn(30, 2)
    y = rng.randn(30, 1)
    model = build_model(lr=0.1)
    model.fit(X, y, batch_size=8, epochs=3, callbacks=[StepLR(step_size=1, gamma=0.1)])
    assert np.isclose(model.optimizer.learning_rate, 0.1 * 0.1 ** 3)


def test_exponential_lr_decays_learning_rate():
    model = build_model(lr=1.0)
    opt = ExponentialLR(gamma=0.5)
    opt.on_epoch_end(model)
    opt.on_epoch_end(model)
    assert np.isclose(model.optimizer.learning_rate, 0.25)


def test_multiplicative_lr_applies_lambda():
    model = build_model(lr=1.0)
    opt = MultiplicativeLR(lr_lambda=lambda epoch: 0.5)
    opt.on_epoch_end(model)
    opt.on_epoch_end(model)
    assert np.isclose(model.optimizer.learning_rate, 0.25)


def test_constant_lr_can_be_constructed():
    ConstantLR(factor=0.5)


def test_linear_lr_changes_learning_rate_over_epochs():
    np.random.seed(3)
    rng = np.random.RandomState(3)
    X = rng.randn(20, 2)
    y = rng.randn(20, 1)
    model = build_model(lr=0.1)
    model.fit(X, y, batch_size=8, epochs=3, callbacks=[LinearLR(total_iters=5)])
    assert np.isfinite(model.optimizer.learning_rate)
    assert not np.isclose(model.optimizer.learning_rate, 0.1)
