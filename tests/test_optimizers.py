"""Update-rule tests for SGD, Adam, Adagrad, and Adadelta."""
import numpy as np
import pytest

from numpy_keras import optimizers


class FakeLayer:
    def __init__(self, **params):
        self.params = {k: np.array(v, dtype=float) for k, v in params.items()}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}


def set_grads(layer, **grads):
    for k, v in grads.items():
        layer.grads[k] = np.array(v, dtype=float)
    return layer


def test_sgd_updates_params_along_negative_gradient():
    layer = set_grads(FakeLayer(w=np.array([1.0, 2.0])), w=np.array([0.5, -0.5]))
    optimizers.SGD(learning_rate=0.1).update([layer])
    np.testing.assert_allclose(layer.params["w"], [0.95, 2.05])


def test_sgd_momentum_accumulates_velocity():
    layer = set_grads(FakeLayer(w=np.array([0.0])), w=np.array([1.0]))
    opt = optimizers.SGD(learning_rate=0.1, momentum=0.9)
    opt.update([layer])
    np.testing.assert_allclose(layer.params["w"], [-0.1])
    opt.update([layer])  # velocity carries over
    np.testing.assert_allclose(layer.params["w"], [-0.29])  # -0.1 - (0.9*0.1 + 0.1)


def test_sgd_weight_decay_changes_update():
    g = np.array([1.0])
    plain = set_grads(FakeLayer(w=np.array([2.0])), w=g.copy())
    decayed = set_grads(FakeLayer(w=np.array([2.0])), w=g.copy())
    optimizers.SGD(learning_rate=0.1, weight_decay=0.0).update([plain])
    optimizers.SGD(learning_rate=0.1, weight_decay=0.5).update([decayed])
    assert not np.allclose(plain.params["w"], decayed.params["w"])
    np.testing.assert_allclose(decayed.params["w"], [2.0 - 0.1 * (1.0 + 0.5 * 2.0)])


def test_adagrad_scales_by_accumulated_square():
    layer = set_grads(FakeLayer(w=np.array([0.0])), w=np.array([1.0]))
    opt = optimizers.Adagrad(learning_rate=1.0, epsilon=0.0)
    opt.update([layer])
    np.testing.assert_allclose(layer.params["w"], [-1.0])
    opt.update([layer])  # second step divided by sqrt(2)
    np.testing.assert_allclose(layer.params["w"], [-1.0 - 1.0 / np.sqrt(2.0)])


def test_adagrad_weight_decay_changes_update():
    """Adagrad is invariant to constant gradient scaling on the first step,
    so compare several steps where the decayed path sees growing gradients."""
    plain = set_grads(FakeLayer(w=np.array([2.0])), w=np.array([1.0]))
    decayed = set_grads(FakeLayer(w=np.array([2.0])), w=np.array([1.0]))
    plain_opt = optimizers.Adagrad(learning_rate=0.1, weight_decay=0.0)
    decayed_opt = optimizers.Adagrad(learning_rate=0.1, weight_decay=0.5)
    for _ in range(5):
        plain_opt.update([plain])
        decayed_opt.update([decayed])
    assert not np.allclose(plain.params["w"], decayed.params["w"])


def test_adam_weight_decay_changes_update():
    """Adam must actually apply weight decay; it is a no-op if ignored.

    Note: the first Adam step is scale-invariant (lr * g/|g|), so a single
    step cannot distinguish the paths -- compare several steps."""
    plain = set_grads(FakeLayer(w=np.array([2.0])), w=np.array([1.0]))
    decayed = set_grads(FakeLayer(w=np.array([2.0])), w=np.array([1.0]))
    plain_opt = optimizers.Adam(learning_rate=0.1, weight_decay=0.0)
    decayed_opt = optimizers.Adam(learning_rate=0.1, weight_decay=0.5)
    for _ in range(5):
        set_grads(plain, w=np.array([1.0]))
        set_grads(decayed, w=np.array([1.0]))
        plain_opt.update([plain])
        decayed_opt.update([decayed])
    assert not np.array_equal(plain.params["w"], decayed.params["w"])


def test_adadelta_weight_decay_changes_update():
    """Adadelta's first step only depends on the sign of the gradient, so
    compare several steps where the decayed path sees growing gradients."""
    plain = set_grads(FakeLayer(w=np.array([2.0])), w=np.array([1.0]))
    decayed = set_grads(FakeLayer(w=np.array([2.0])), w=np.array([1.0]))
    plain_opt = optimizers.Adadelta(learning_rate=1.0, weight_decay=0.0)
    decayed_opt = optimizers.Adadelta(learning_rate=1.0, weight_decay=0.5)
    for _ in range(5):
        set_grads(plain, w=np.array([1.0]))
        set_grads(decayed, w=np.array([1.0]))
        plain_opt.update([plain])
        decayed_opt.update([decayed])
    # Adadelta's step size is bounded by its accumulated deltas, so the two
    # paths diverge slowly -- require exact inequality rather than a margin
    assert not np.array_equal(plain.params["w"], decayed.params["w"])


@pytest.mark.parametrize("name", ["sgd", "adam", "adagrad", "adadelta"])
def test_optimizer_reduces_convex_quadratic(name):
    """min_w (w - 2)^2 with gradient 2(w - 2): every optimizer must descend."""
    np.random.seed(11)
    w = np.array([0.0])
    layer = FakeLayer(w=w)
    opt = optimizers._OptimMapper()[name]
    if name == "sgd":
        opt.learning_rate = 0.5
    elif name in ("adam", "adagrad"):
        opt.learning_rate = 0.5
    for _ in range(3000):
        grad = 2 * (layer.params["w"] - 2.0)
        set_grads(layer, w=grad)
        opt.update([layer])
    assert abs(layer.params["w"][0] - 2.0) < 1e-3


def test_optimizer_mapper():
    assert isinstance(optimizers._OptimMapper()["sgd"], optimizers.SGD)
    assert isinstance(optimizers._OptimMapper()["adam"], optimizers.Adam)
    assert isinstance(optimizers._OptimMapper()["adagrad"], optimizers.Adagrad)
    assert isinstance(optimizers._OptimMapper()["adadelta"], optimizers.Adadelta)
