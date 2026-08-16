"""Value and gradient checks for the loss functions."""
import numpy as np
import pytest

from numpy_keras.losses import MSE, CategoricalCrossEntropy, _LossMapper

EPS = 1e-6


def test_mse_value():
    y = np.array([[1.0], [2.0], [3.0]])
    p = np.array([[0.5], [2.5], [2.0]])
    assert MSE()(y, p) == np.mean((y - p) ** 2)


def test_mse_grad_matches_finite_difference():
    """d(loss)/d(pred) must equal the finite difference of the reported loss."""
    rng = np.random.RandomState(0)
    y = rng.randn(6, 3)
    p = rng.randn(6, 3)
    loss = MSE()
    numerical = np.zeros_like(p)
    for i in np.ndindex(p.shape):
        p[i] += EPS
        l1 = loss(y, p)
        p[i] -= 2 * EPS
        l2 = loss(y, p)
        p[i] += EPS
        numerical[i] = (l1 - l2) / (2 * EPS)
    np.testing.assert_allclose(loss.grad(y, p), numerical, rtol=1e-4, atol=1e-6)


def test_categorical_crossentropy_value():
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    p = np.array([[0.9, 0.1], [0.2, 0.8]])
    expected = -(np.log(0.9) + np.log(0.8)) / 2
    assert np.isclose(CategoricalCrossEntropy()(y, p), expected)


def test_categorical_crossentropy_zero_loss_for_perfect_prediction():
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.isclose(CategoricalCrossEntropy()(y, p), 0.0)


def test_categorical_crossentropy_grad_matches_finite_difference():
    """grad returns d(loss)/d(y_pred): verify against finite differences.
    The softmax chain is handled by the softmax layer itself, so here the
    prediction is just a probability vector."""
    rng = np.random.RandomState(1)
    y = np.eye(3)[rng.randint(0, 3, 6)]
    p = rng.uniform(0.05, 0.9, (6, 3))
    p = p / p.sum(axis=1, keepdims=True)
    loss = CategoricalCrossEntropy()

    numerical = np.zeros_like(p)
    for i in np.ndindex(p.shape):
        p[i] += EPS
        l1 = loss(y, p)
        p[i] -= 2 * EPS
        l2 = loss(y, p)
        p[i] += EPS
        numerical[i] = (l1 - l2) / (2 * EPS)

    np.testing.assert_allclose(loss.grad(y, p), numerical, rtol=1e-4, atol=1e-6)


def test_softmax_chain_matches_finite_difference():
    """The softmax layer's backward (Jacobian contraction) applied to the
    raw CE grad must reproduce d(loss∘softmax)/d(logits) -- the classic
    softmax+CE combination, now owned by the softmax layer."""
    from numpy_keras.activations._mapper import _ActivationMapper
    from numpy_keras.activations.functional import softmax

    rng = np.random.RandomState(1)
    y = np.eye(3)[rng.randint(0, 3, 6)]
    logits = rng.randn(6, 3)
    loss = CategoricalCrossEntropy()

    def composed(z):
        return loss(y, softmax(z))

    numerical = np.zeros_like(logits)
    for i in np.ndindex(logits.shape):
        logits[i] += EPS
        l1 = composed(logits)
        logits[i] -= 2 * EPS
        l2 = composed(logits)
        logits[i] += EPS
        numerical[i] = (l1 - l2) / (2 * EPS)

    y_hat = softmax(logits)
    chained = _ActivationMapper().backward(
        "softmax", y_hat, loss.grad(y, y_hat), {})
    np.testing.assert_allclose(chained, numerical, rtol=1e-4, atol=1e-6)


def test_categorical_crossentropy_grad_is_clipped_for_extreme_predictions():
    loss = CategoricalCrossEntropy()
    y = np.array([[1.0, 0.0]])
    grad = loss.grad(y, np.array([[0.0, 1.0]]))
    assert np.isfinite(grad).all()
    # 0 must be clipped to the epsilon lower bound instead of producing inf
    assert grad[0, 0] == -1 / 1e-10
    assert grad[0, 1] == 0.0


def test_loss_mapper():
    assert isinstance(_LossMapper()["mse"], MSE)
    assert isinstance(_LossMapper()["categorical_crossentropy"], CategoricalCrossEntropy)
    sparse = _LossMapper()["sparse_categorical_crossentropy"]
    assert sparse.name == "sparse_categorical_crossentropy"
    with pytest.raises(ValueError):
        _LossMapper()["not_a_loss"]
