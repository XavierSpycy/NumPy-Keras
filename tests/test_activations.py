"""Gradient checks for every activation function.

The library convention is that ``*_deriv(a)`` takes the *post-activation*
value ``a = f(x)`` and returns ``f'(x)``. These tests verify that convention
against finite differences, sampling points away from kinks where the
subgradient is ambiguous.
"""
import numpy as np
import pytest

from numpy_keras.activations import functional as F

EPS = 1e-6


def finite_diff(f, x):
    return (f(x + EPS) - f(x - EPS)) / (2 * EPS)


# (name, x values, kwargs for both f and deriv)
SMOOTH_CASES = [
    ("sigmoid", np.linspace(-3.0, 3.0, 31), {}),
    ("tanh", np.linspace(-3.0, 3.0, 31), {}),
    ("softsign", np.linspace(-3.0, 3.0, 31), {}),
    ("log_sigmoid", np.linspace(-3.0, 3.0, 31), {}),
    ("softplus", np.linspace(-3.0, 3.0, 31), {}),
    ("linear", np.linspace(-3.0, 3.0, 31), {}),
]

KINKED_CASES = [
    ("relu", np.concatenate([np.linspace(-3.0, -0.2, 15), np.linspace(0.2, 3.0, 15)]), {}),
    ("leaky_relu", np.concatenate([np.linspace(-3.0, -0.2, 15), np.linspace(0.2, 3.0, 15)]), {}),
    ("elu", np.concatenate([np.linspace(-3.0, -0.2, 15), np.linspace(0.2, 3.0, 15)]), {}),
    ("celu", np.concatenate([np.linspace(-3.0, -0.2, 15), np.linspace(0.2, 3.0, 15)]), {}),
    ("selu", np.concatenate([np.linspace(-3.0, -0.2, 15), np.linspace(0.2, 3.0, 15)]), {}),
    ("hardsigmoid", np.linspace(-2.9, 2.9, 31), {}),
    ("relu6", np.concatenate([np.linspace(-3.0, -0.2, 15), np.linspace(0.2, 5.9, 15)]), {}),
    ("hardtanh", np.concatenate([np.linspace(-0.9, 0.9, 15), np.linspace(1.2, 3.0, 10), np.linspace(-3.0, -1.2, 10)]), {}),
    ("softshrink", np.concatenate([np.linspace(-3.0, -0.8, 12), np.linspace(-0.3, 0.3, 10), np.linspace(0.8, 3.0, 12)]), {}),
    ("hardshrink", np.concatenate([np.linspace(-3.0, -0.8, 12), np.linspace(-0.3, 0.3, 10), np.linspace(0.8, 3.0, 12)]), {}),
]


@pytest.mark.parametrize("name,x,kwargs", SMOOTH_CASES)
def test_smooth_activation_deriv_matches_finite_difference(name, x, kwargs):
    f = getattr(F, name)
    deriv = getattr(F, name + "_deriv")
    numerical = finite_diff(lambda v: f(v, **kwargs), x)
    analytical = deriv(f(x, **kwargs), **kwargs)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("name,x,kwargs", KINKED_CASES)
def test_piecewise_activation_deriv_matches_finite_difference(name, x, kwargs):
    f = getattr(F, name)
    deriv = getattr(F, name + "_deriv")
    numerical = finite_diff(lambda v: f(v, **kwargs), x)
    analytical = deriv(f(x, **kwargs), **kwargs)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-4)


def test_elu_deriv_with_custom_alpha():
    x = np.concatenate([np.linspace(-3.0, -0.2, 15), np.linspace(0.2, 3.0, 15)])
    kwargs = {"alpha": 2.0}
    numerical = finite_diff(lambda v: F.elu(v, **kwargs), x)
    np.testing.assert_allclose(F.elu_deriv(F.elu(x, **kwargs), **kwargs), numerical, rtol=1e-3, atol=1e-4)


def test_celu_deriv_with_custom_alpha():
    x = np.concatenate([np.linspace(-3.0, -0.2, 15), np.linspace(0.2, 3.0, 15)])
    kwargs = {"alpha": 2.0}
    numerical = finite_diff(lambda v: F.celu(v, **kwargs), x)
    np.testing.assert_allclose(F.celu_deriv(F.celu(x, **kwargs), **kwargs), numerical, rtol=1e-3, atol=1e-4)


def test_softmax_output_is_probability_distribution():
    x = np.random.randn(7, 5)
    p = F.softmax(x)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0)
    assert np.all(p > 0)
    np.testing.assert_allclose(F.softmax(x + 100.0), F.softmax(x))  # shift-invariant


def test_sigmoid_is_numerically_stable():
    assert F.sigmoid(np.array([1000.0]))[0] == 1.0
    assert F.sigmoid(np.array([-1000.0]))[0] == 0.0


def test_activation_mapper_rejects_unknown_name():
    from numpy_keras.activations._mapper import _ActivationMapper

    with pytest.raises(ValueError):
        _ActivationMapper()["not_an_activation"]
