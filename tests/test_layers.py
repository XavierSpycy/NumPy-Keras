"""Unit tests for individual layers: shapes, forward math, and backward gradients."""
import numpy as np
import pytest

from numpy_keras import layers


def make_dense(units, activation="tanh", input_dim=3, use_bias=True):
    layer = layers.Dense(units, activation=activation, use_bias=use_bias)
    layer.init_params(input_dim)
    return layer


def test_dense_param_shapes():
    d = make_dense(4, input_dim=3)
    assert d.params["W"].shape == (3, 4)
    assert d.params["b"].shape == (4,)
    assert d.grads["W"].shape == (3, 4)
    assert d.output_dim == 4

    d_no_bias = make_dense(4, input_dim=3, use_bias=False)
    assert "b" not in d_no_bias.params


def test_dense_forward_matches_manual_computation():
    rng = np.random.RandomState(0)
    d = make_dense(2, activation="tanh", input_dim=3)
    d.params["W"] = rng.randn(3, 2)
    d.params["b"] = rng.randn(2)
    X = rng.randn(5, 3)
    expected = np.tanh(X @ d.params["W"] + d.params["b"])
    np.testing.assert_allclose(d.forward(X, is_training=True), expected)


def test_dense_forward_linear_activation():
    d = make_dense(2, activation="linear", input_dim=3)
    d.params["W"] = np.ones((3, 2))
    d.params["b"] = np.zeros(2)
    X = np.ones((4, 3))
    np.testing.assert_allclose(d.forward(X, is_training=True), np.full((4, 2), 3.0))


def test_dense_backward_weight_gradients():
    """grads["W"] = inputs.T @ grad, grads["b"] = sum(grad) -- plain chain rule."""
    rng = np.random.RandomState(2)
    d = make_dense(2, activation="linear", input_dim=3)
    d.params["W"] = rng.randn(3, 2)
    d.params["b"] = rng.randn(2)
    X = rng.randn(5, 3)
    d.forward(X, is_training=True)
    grad = rng.randn(5, 2)
    d.backward(grad)
    np.testing.assert_allclose(d.grads["W"], X.T @ grad)
    np.testing.assert_allclose(d.grads["b"], grad.sum(axis=0))


def test_dense_backward_applies_previous_layer_activation_deriv():
    """The returned gradient is (grad @ W.T) elementwise-multiplied by the
    previous layer's activation derivative, evaluated at its output (= this
    layer's input). This is the mechanism Sequential relies on."""
    rng = np.random.RandomState(3)
    d = make_dense(2, activation="linear", input_dim=3)
    d.set_activation_deriv("tanh", {})
    d.params["W"] = rng.randn(3, 2)
    X = rng.randn(5, 3)
    d.forward(X, is_training=True)
    grad = rng.randn(5, 2)
    returned = d.backward(grad)
    expected = (grad @ d.params["W"].T) * (1 - X ** 2)  # tanh_deriv
    np.testing.assert_allclose(returned, expected)


def test_dropout_rate_validation():
    with pytest.raises(ValueError):
        layers.Dropout(rate=1.0)
    with pytest.raises(ValueError):
        layers.Dropout(rate=-0.1)


def test_dropout_inference_is_identity():
    rng = np.random.RandomState(4)
    d = layers.Dropout(rate=0.5)
    X = rng.randn(100, 10)
    np.testing.assert_allclose(d.forward(X, is_training=False), X)


def test_dropout_training_mask_is_scaled_and_sparse():
    np.random.seed(5)
    d = layers.Dropout(rate=0.5)
    X = np.random.randn(1000, 10)
    out = d.forward(X, is_training=True)
    # Inverted dropout: kept values scaled by 1/(1-rate), mean preserved
    assert np.isclose(out.mean(), X.mean(), atol=0.05)
    mask = out / X
    kept = np.isclose(mask, 1 / 0.5)
    dropped = np.isclose(mask, 0.0)
    assert np.isclose(dropped.mean(), 0.5, atol=0.1)
    assert np.isclose((kept | dropped).mean(), 1.0)


def test_batchnorm_training_normalizes():
    rng = np.random.RandomState(6)
    bn = layers.BatchNormalization()
    bn.init_params(4)
    X = rng.randn(64, 4) * 5 + 10
    out = bn.forward(X, is_training=True)
    assert np.allclose(out.mean(axis=0), bn.params["beta"], atol=1e-6)
    assert np.allclose(out.var(axis=0), bn.params["gamma"] ** 2, atol=1e-6)


def test_batchnorm_updates_moving_statistics():
    rng = np.random.RandomState(7)
    bn = layers.BatchNormalization(momentum=0.9)
    bn.init_params(3)
    X = rng.randn(32, 3) * 2 + 3
    bn.forward(X, is_training=True)
    np.testing.assert_allclose(bn.moving_mean, 0.9 * 0 + 0.1 * X.mean(axis=0))
    np.testing.assert_allclose(bn.moving_variance, 0.9 * 1 + 0.1 * X.var(axis=0))


def test_batchnorm_inference_uses_moving_statistics():
    rng = np.random.RandomState(8)
    bn = layers.BatchNormalization(momentum=1.0)  # freeze the running stats
    bn.init_params(3)
    bn.params["gamma"] = np.array([1.0, 2.0, 0.5])
    bn.params["beta"] = np.array([0.0, 1.0, -1.0])
    X = rng.randn(16, 3) * 3 + 5
    bn.forward(X, is_training=True)  # seed moving stats with momentum=1
    assert np.allclose(bn.moving_mean, 0.0)
    assert np.allclose(bn.moving_variance, 1.0)
    X2 = rng.randn(16, 3) * 7 - 2
    out = bn.forward(X2, is_training=False)
    expected = bn.params["gamma"] * (X2 / np.sqrt(1.0 + bn.epsilon)) + bn.params["beta"]
    np.testing.assert_allclose(out, expected)


def test_batchnorm_backward_gradient_check():
    """Finite-difference check of the gamma/beta gradients."""
    rng = np.random.RandomState(9)
    bn = layers.BatchNormalization()
    bn.init_params(3)
    X = rng.randn(8, 3)
    delta = np.ones((8, 3))  # makes backward compute grad of sum(output)

    def total_output():
        return np.sum(bn.forward(X, is_training=True))

    eps = 1e-6
    for key in ["gamma", "beta"]:
        p = bn.params[key]
        numerical = np.zeros_like(p)
        for i in range(p.size):
            orig = p[i]
            p[i] = orig + eps
            l1 = total_output()
            p[i] = orig - eps
            l2 = total_output()
            p[i] = orig
            numerical[i] = (l1 - l2) / (2 * eps)
        bn.forward(X, is_training=True)  # refresh cached batch statistics
        bn.backward(delta)
        np.testing.assert_allclose(bn.grads[key], numerical, rtol=1e-3, atol=1e-6)


def test_flatten_reshape_and_output_dim():
    f = layers.Flatten(input_shape=(28, 28))
    X = np.random.randn(4, 28, 28)
    np.testing.assert_allclose(f.forward(X, is_training=True), X.reshape(4, -1))
    assert f.output_dim == 28 * 28


def test_input_output_dim():
    assert layers.Input(10).output_dim == 10


def test_activation_layer_forward_and_backward():
    act = layers.Activation("tanh")
    act.set_activation_deriv("sigmoid", {})
    rng = np.random.RandomState(10)
    X = rng.randn(5, 3)
    np.testing.assert_allclose(act.forward(X, is_training=True), np.tanh(X))
    grad = rng.randn(5, 3)
    # the stored deriv (of the previous layer's activation) is applied
    # directly to this layer's input: sigmoid_deriv(x) = x * (1 - x)
    expected = grad * (X * (1 - X))
    np.testing.assert_allclose(act.backward(grad), expected)
