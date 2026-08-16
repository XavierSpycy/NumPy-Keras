"""Regression tests for the initializer scales.

The kaiming_* initializers must carry He initialization's sqrt(2) factor:
without it, a stack of ReLU layers loses variance layer after layer (the
01_activations tutorial found this bug by measuring activation stds
through a 10-layer network).
"""

import numpy as np

from numpy_keras.initializers import functional as F


def test_kaiming_normal_fan_in_scale():
    np.random.seed(0)
    w = F.kaiming_normal((64, 128))
    assert w.shape == (64, 128)
    assert np.isclose(w.std(), np.sqrt(2 / 64), rtol=0.05)


def test_kaiming_normal_fan_out_scale():
    np.random.seed(0)
    w = F.kaiming_normal((64, 128), mode="fan_out")
    assert np.isclose(w.std(), np.sqrt(2 / 128), rtol=0.05)


def test_kaiming_uniform_fills_he_bound():
    np.random.seed(0)
    w = F.kaiming_uniform((64, 128))
    bound = np.sqrt(6 / 64)
    assert np.max(np.abs(w)) <= bound + 1e-12
    assert np.max(np.abs(w)) > 0.9 * bound


def test_glorot_normal_scale():
    np.random.seed(0)
    w = F.xavier_normal((64, 128))
    assert np.isclose(w.std(), np.sqrt(2 / (64 + 128)), rtol=0.05)


def test_glorot_uniform_fills_xavier_bound():
    np.random.seed(0)
    w = F.xavier_uniform((64, 128))
    bound = np.sqrt(6 / (64 + 128))
    assert np.max(np.abs(w)) <= bound + 1e-12
    assert np.max(np.abs(w)) > 0.9 * bound


def test_xaiver_aliases_stay_available():
    """The historical misspelled names keep working (backward compat)."""
    np.random.seed(0)
    w1 = F.xavier_normal((32, 16))
    np.random.seed(0)
    w2 = F.xaiver_normal((32, 16))
    np.testing.assert_array_equal(w1, w2)
    np.random.seed(0)
    w3 = F.xavier_uniform((32, 16))
    np.random.seed(0)
    w4 = F.xaiver_uniform((32, 16))
    np.testing.assert_array_equal(w3, w4)
