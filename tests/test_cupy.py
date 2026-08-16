"""GPU parity tests: the CuPy backend must reproduce the NumPy reference.

Skipped entirely when CuPy is unavailable (probed at import time).  The
autouse fixture switches to the cupy backend for each test and restores
numpy afterwards, so the rest of the suite is unaffected even when this
module ran in the same process.  All comparisons convert device results
back to the host first; host-side random number generation keeps seeded
weights and dropout masks bit-identical across backends, which is what
makes these parity tests strong.
"""
import numpy as np
import pytest

from numpy_keras import backend as B

B.set_backend("cupy")
CUPY = B.is_cupy_mode()
B.set_backend("numpy")

pytestmark = pytest.mark.skipif(not CUPY, reason="cupy not available")

from numpy_keras import Sequential
from numpy_keras import layers
from numpy_keras.activations import functional as F
from numpy_keras.activations._mapper import _ActivationMapper

_ALL_ACTIVATIONS = [
    name for name in _ActivationMapper.activations
    if not name.endswith("_deriv")
]


@pytest.fixture(autouse=True)
def _cupy_backend():
    B.set_backend("cupy")
    yield
    B.set_backend("numpy")


def _build(backend, builder, seed):
    """Build under the given backend with np.random re-seeded."""
    B.set_backend(backend)
    np.random.seed(seed)
    return builder()


# ---------------------------------------------------------------------------
# 1. backend helpers on the device
# ---------------------------------------------------------------------------

def test_scatter_add_with_slice_on_device():
    a_cpu = np.zeros((2, 3))
    np.add.at(a_cpu, (np.array([0, 1]), slice(None)), np.ones((2, 3)))
    a_gpu = B.asarray(np.zeros((2, 3)))
    B.scatter_add(a_gpu, (B.asarray([0, 1]), slice(None)), B.asarray(np.ones((2, 3))))
    np.testing.assert_array_equal(B.asnumpy(a_gpu), a_cpu)


def test_scatter_add_overlapping_indices_on_device():
    idx = np.array([0, 1, 1, 3])
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    a_cpu = np.zeros(5)
    np.add.at(a_cpu, (idx,), vals)
    a_gpu = B.asarray(np.zeros(5))
    B.scatter_add(a_gpu, (B.asarray(idx),), B.asarray(vals))
    np.testing.assert_array_equal(B.asnumpy(a_gpu), a_cpu)


def test_sliding_window_view_parity():
    x = np.random.RandomState(0).randn(4, 6, 2)
    cpu = np.lib.stride_tricks.sliding_window_view(x, (2, 3), axis=(0, 1))
    gpu = B.sliding_window_view(B.asarray(x), (2, 3), axis=(0, 1))
    np.testing.assert_array_equal(B.asnumpy(gpu), cpu)


# ---------------------------------------------------------------------------
# 2. activations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", _ALL_ACTIVATIONS)
def test_activation_parity(name):
    rng = np.random.RandomState(0)
    x = rng.randn(64, 64)
    mapper = _ActivationMapper()
    B.set_backend("numpy")
    cpu = mapper[name](x)
    B.set_backend("cupy")
    gpu = B.asnumpy(mapper[name](B.asarray(x)))
    np.testing.assert_allclose(gpu, cpu, rtol=1e-10, atol=1e-12)


def test_softmax_backward_parity():
    B.set_backend("numpy")
    rng = np.random.RandomState(1)
    y = F.softmax(rng.randn(32, 10))
    g = rng.randn(32, 10)
    mapper = _ActivationMapper()
    cpu = mapper.backward("softmax", y, g, {})
    B.set_backend("cupy")
    gpu = B.asnumpy(mapper.backward("softmax", B.asarray(y), B.asarray(g), {}))
    np.testing.assert_allclose(gpu, cpu, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Dense / Conv2D / MaxPool2D / Dropout / BatchNormalization
# ---------------------------------------------------------------------------

def _dense_builder():
    layer = layers.Dense(16, activation="tanh")
    layer.set_input_shape((32,))
    layer.init_params(32)
    return layer


def test_dense_parity():
    rng = np.random.RandomState(0)
    X = rng.randn(16, 32)
    grad = rng.randn(16, 16)
    cpu_layer = _build("numpy", _dense_builder, 7)
    out_cpu = cpu_layer.forward(X, is_training=True)
    grad_cpu = cpu_layer.backward(grad)

    gpu_layer = _build("cupy", _dense_builder, 7)  # same seed -> same weights
    out_gpu = B.asnumpy(gpu_layer.forward(B.asarray(X), is_training=True))
    grad_gpu = B.asnumpy(gpu_layer.backward(B.asarray(grad)))
    np.testing.assert_allclose(out_gpu, out_cpu, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(grad_gpu, grad_cpu, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(B.asnumpy(gpu_layer.grads["W"]), cpu_layer.grads["W"], rtol=1e-9)
    np.testing.assert_allclose(B.asnumpy(gpu_layer.grads["b"]), cpu_layer.grads["b"], rtol=1e-9)


def _conv_builder():
    layer = layers.Conv2D(4, kernel_size=3, stride=2, activation="relu")
    layer.set_input_shape((8, 8, 2))
    return layer


def test_conv2d_parity():
    rng = np.random.RandomState(0)
    X = rng.randn(4, 8, 8, 2)
    grad = rng.randn(4, 3, 3, 4)
    cpu_layer = _build("numpy", _conv_builder, 11)
    out_cpu = cpu_layer.forward(X, is_training=True)
    grad_cpu = cpu_layer.backward(grad)

    gpu_layer = _build("cupy", _conv_builder, 11)
    out_gpu = B.asnumpy(gpu_layer.forward(B.asarray(X), is_training=True))
    grad_gpu = B.asnumpy(gpu_layer.backward(B.asarray(grad)))
    np.testing.assert_allclose(out_gpu, out_cpu, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(grad_gpu, grad_cpu, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(B.asnumpy(gpu_layer.grads["W"]), cpu_layer.grads["W"], rtol=1e-9)
    np.testing.assert_allclose(B.asnumpy(gpu_layer.grads["b"]), cpu_layer.grads["b"], rtol=1e-9)


def _maxpool_builder():
    layer = layers.MaxPool2D(pool_size=2, stride=1)
    layer.set_input_shape((6, 6, 3))
    return layer


def test_maxpool2d_parity():
    rng = np.random.RandomState(0)
    X = rng.randn(4, 6, 6, 3)
    grad = rng.randn(4, 5, 5, 3)
    cpu_layer = _build("numpy", _maxpool_builder, 0)
    out_cpu = cpu_layer.forward(X, is_training=True)
    grad_cpu = cpu_layer.backward(grad)

    gpu_layer = _build("cupy", _maxpool_builder, 0)
    out_gpu = B.asnumpy(gpu_layer.forward(B.asarray(X), is_training=True))
    grad_gpu = B.asnumpy(gpu_layer.backward(B.asarray(grad)))
    np.testing.assert_array_equal(out_gpu, out_cpu)
    np.testing.assert_array_equal(grad_gpu, grad_cpu)


def test_dropout_seeded_parity_exact():
    X = np.random.RandomState(3).randn(32, 64)

    def run(backend):
        B.set_backend(backend)
        np.random.seed(42)  # host RNG -> identical masks on both backends
        layer = layers.Dropout(rate=0.4)
        return layer.forward(B.asarray(X) if backend == "cupy" else X, is_training=True)

    cpu = run("numpy")
    gpu = B.asnumpy(run("cupy"))
    # elementwise multiply is correctly rounded on both devices -> exact
    np.testing.assert_array_equal(gpu, cpu)


def _bn_builder():
    layer = layers.BatchNormalization()
    layer.set_input_shape((8,))
    layer.init_params(8)
    return layer


def test_batchnorm_parity():
    rng = np.random.RandomState(0)
    X = rng.randn(32, 8)
    grad = rng.randn(32, 8)
    cpu_layer = _build("numpy", _bn_builder, 5)
    out_cpu = cpu_layer.forward(X, is_training=True)
    grad_cpu = cpu_layer.backward(grad)

    gpu_layer = _build("cupy", _bn_builder, 5)
    out_gpu = B.asnumpy(gpu_layer.forward(B.asarray(X), is_training=True))
    grad_gpu = B.asnumpy(gpu_layer.backward(B.asarray(grad)))
    np.testing.assert_allclose(out_gpu, out_cpu, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(grad_gpu, grad_cpu, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(B.asnumpy(gpu_layer.moving_mean), cpu_layer.moving_mean, rtol=1e-10)
    np.testing.assert_allclose(B.asnumpy(gpu_layer.moving_variance), cpu_layer.moving_variance, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. RNNs (GPU path uses the batched input projection + tensordot grads)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rnn_cls", [layers.SimpleRNN, layers.LSTM, layers.GRU])
@pytest.mark.parametrize("return_sequences", [False, True])
def test_rnn_parity(rnn_cls, return_sequences):
    rng = np.random.RandomState(0)
    X = rng.randn(6, 5, 4)
    grad = rng.randn(6, 5, 8) if return_sequences else rng.randn(6, 8)

    def builder():
        layer = rnn_cls(units=8, return_sequences=return_sequences)
        layer.set_input_shape((5, 4))
        return layer

    cpu_layer = _build("numpy", builder, 13)
    out_cpu = cpu_layer.forward(X, is_training=True)
    grad_cpu = cpu_layer.backward(grad)

    gpu_layer = _build("cupy", builder, 13)
    out_gpu = B.asnumpy(gpu_layer.forward(B.asarray(X), is_training=True))
    grad_gpu = B.asnumpy(gpu_layer.backward(B.asarray(grad)))
    np.testing.assert_allclose(out_gpu, out_cpu, rtol=1e-10, atol=1e-12)
    # tensordot accumulates in one reduction vs the CPU's per-step +=
    np.testing.assert_allclose(grad_gpu, grad_cpu, rtol=1e-9, atol=1e-12)
    for key in cpu_layer.grads:
        np.testing.assert_allclose(
            B.asnumpy(gpu_layer.grads[key]), cpu_layer.grads[key], rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. optimizers (in-place grad write-back must hold on the device)
# ---------------------------------------------------------------------------

_OPT_BUILDERS = {
    "sgd": lambda: __import__("numpy_keras.optimizers", fromlist=["SGD"]).SGD(learning_rate=0.1, momentum=0.9, weight_decay=1e-3),
    "adam": lambda: __import__("numpy_keras.optimizers", fromlist=["Adam"]).Adam(learning_rate=1e-3, weight_decay=1e-3),
    "adagrad": lambda: __import__("numpy_keras.optimizers", fromlist=["Adagrad"]).Adagrad(learning_rate=0.1, weight_decay=1e-3),
    "adadelta": lambda: __import__("numpy_keras.optimizers", fromlist=["Adadelta"]).Adadelta(learning_rate=1.0, weight_decay=1e-3),
}


@pytest.mark.parametrize("opt_name", ["sgd", "adam", "adagrad", "adadelta"])
def test_optimizer_parity(opt_name):
    rng = np.random.RandomState(0)
    gW = rng.randn(8, 8)
    gb = rng.randn(8)

    def build_pair(backend):
        B.set_backend(backend)
        np.random.seed(3)
        layer = layers.Dense(8, activation="tanh")
        layer.set_input_shape((8,))
        layer.init_params(8)
        # identical host-drawn grads on both devices
        layer.grads["W"] = B.asarray(gW) if backend == "cupy" else gW.copy()
        layer.grads["b"] = B.asarray(gb) if backend == "cupy" else gb.copy()
        return layer

    # the backend is a global switch: run each side entirely under its own
    cpu_layer = build_pair("numpy")
    cpu_opt = _OPT_BUILDERS[opt_name]()
    for _ in range(5):
        cpu_opt.update([cpu_layer])

    gpu_layer = build_pair("cupy")
    assert B.on_gpu(gpu_layer.params["W"])
    gpu_opt = _OPT_BUILDERS[opt_name]()
    for _ in range(5):
        gpu_opt.update([gpu_layer])

    np.testing.assert_allclose(B.asnumpy(gpu_layer.params["W"]), cpu_layer.params["W"], rtol=1e-10)
    np.testing.assert_allclose(B.asnumpy(gpu_layer.params["b"]), cpu_layer.params["b"], rtol=1e-10)
    # the in-place weight-decay write-back happened on the device
    assert not np.array_equal(B.asnumpy(gpu_layer.grads["W"]), gW)
    np.testing.assert_allclose(B.asnumpy(gpu_layer.grads["W"]), cpu_layer.grads["W"], rtol=1e-10)


def test_sgd_nesterov_parity():
    """The fused GPU kernel's Nesterov branch vs the CPU pure path."""
    rng = np.random.RandomState(0)
    gW = rng.randn(8, 8)
    gb = rng.randn(8)

    def build_pair(backend):
        B.set_backend(backend)
        np.random.seed(3)
        layer = layers.Dense(8, activation="tanh")
        layer.set_input_shape((8,))
        layer.init_params(8)
        layer.grads["W"] = B.asarray(gW) if backend == "cupy" else gW.copy()
        layer.grads["b"] = B.asarray(gb) if backend == "cupy" else gb.copy()
        return layer

    cpu_layer = build_pair("numpy")
    cpu_opt = __import__("numpy_keras.optimizers", fromlist=["SGD"]).SGD(
        learning_rate=0.1, momentum=0.9, nesterov=True, weight_decay=1e-3)
    for _ in range(5):
        cpu_opt.update([cpu_layer])

    gpu_layer = build_pair("cupy")
    gpu_opt = __import__("numpy_keras.optimizers", fromlist=["SGD"]).SGD(
        learning_rate=0.1, momentum=0.9, nesterov=True, weight_decay=1e-3)
    for _ in range(5):
        gpu_opt.update([gpu_layer])

    np.testing.assert_allclose(B.asnumpy(gpu_layer.params["W"]), cpu_layer.params["W"], rtol=1e-10)
    np.testing.assert_allclose(B.asnumpy(gpu_layer.params["b"]), cpu_layer.params["b"], rtol=1e-10)
    np.testing.assert_allclose(B.asnumpy(gpu_opt.velocity[0]["W"]), cpu_opt.velocity[0]["W"], rtol=1e-10)
    np.testing.assert_allclose(B.asnumpy(gpu_layer.grads["W"]), cpu_layer.grads["W"], rtol=1e-10)


def test_fused_gpu_kernel_is_actually_used(monkeypatch):
    """The dispatch guard must really route device arrays to the fused
    kernel -- a regression here would silently fall back to the slow path
    and every other test would still pass."""
    from numpy_keras import optimizers as opt_mod
    from numpy_keras.optimizers import _gpu_kernels as _gk

    B.set_backend("cupy")
    np.random.seed(3)
    layer = layers.Dense(8, activation="tanh")
    layer.set_input_shape((8,))
    layer.init_params(8)
    layer.grads["W"] = B.asarray(np.random.randn(8, 8))
    layer.grads["b"] = B.asarray(np.random.randn(8))

    calls = []
    original = _gk.adam_update

    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(_gk, "adam_update", spy)
    opt_mod.Adam(weight_decay=1e-3).update([layer])
    assert calls, "the fused GPU kernel was not dispatched"


def test_conv_batchnorm_parity():
    """BatchNorm's 4D reduce path (Conv2D output) on the device."""
    rng = np.random.RandomState(0)
    X = rng.randn(4, 8, 8, 2)
    grad = rng.randn(4, 6, 6, 4)

    def build(backend):
        B.set_backend(backend)
        np.random.seed(11)
        conv = layers.Conv2D(4, kernel_size=3, activation=None)
        conv.set_input_shape((8, 8, 2))
        bn = layers.BatchNormalization()
        bn.set_input_shape((6, 6, 4))
        bn.init_params(4)
        return conv, bn

    conv_cpu, bn_cpu = build("numpy")
    out_cpu = bn_cpu.forward(conv_cpu.forward(X, is_training=True), is_training=True)
    grad_cpu = conv_cpu.backward(bn_cpu.backward(grad))

    conv_gpu, bn_gpu = build("cupy")
    out_gpu = B.asnumpy(bn_gpu.forward(conv_gpu.forward(B.asarray(X), is_training=True), is_training=True))
    grad_gpu = B.asnumpy(conv_gpu.backward(bn_gpu.backward(B.asarray(grad))))
    np.testing.assert_allclose(out_gpu, out_cpu, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(grad_gpu, grad_cpu, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(B.asnumpy(bn_gpu.moving_mean), bn_cpu.moving_mean, rtol=1e-10)
    np.testing.assert_allclose(B.asnumpy(bn_gpu.moving_variance), bn_cpu.moving_variance, rtol=1e-10)


def test_dropout_3d_seeded_parity_exact():
    X = np.random.RandomState(3).randn(4, 5, 6)

    def run(backend):
        B.set_backend(backend)
        np.random.seed(42)
        layer = layers.Dropout(rate=0.4)
        return layer.forward(B.asarray(X) if backend == "cupy" else X, is_training=True)

    cpu = run("numpy")
    gpu = B.asnumpy(run("cupy"))
    np.testing.assert_array_equal(gpu, cpu)


def test_backend_switch_with_sgd_nesterov_state():
    """Mid-training backend switch with SGD(Nesterov): the flat
    velocity_prev dict is synced by __sync_backend."""
    B.set_backend("numpy")
    np.random.seed(1)
    model = _toy_model()
    model.compile(loss="mse", optimizer=__import__(
        "numpy_keras.optimizers", fromlist=["SGD"]).SGD(
        learning_rate=0.1, momentum=0.9, nesterov=True))
    X = np.random.RandomState(0).randn(32, 8)
    y = np.random.RandomState(0).randn(32, 2)
    model.fit(X, y, batch_size=8, epochs=1, verbose=0)

    B.set_backend("cupy")
    model.fit(X, y, batch_size=8, epochs=1, verbose=0)   # velocity on GPU now
    B.set_backend("numpy")
    model.fit(X, y, batch_size=8, epochs=1, verbose=0)   # and back, with nesterov
    for per_layer in model.optimizer.velocity.values():
        for arr in per_layer.values():
            assert not B.on_gpu(arr)


# ---------------------------------------------------------------------------
# 6. end-to-end fit / finite differences / backend switching
# ---------------------------------------------------------------------------

def _toy_model():
    return Sequential([
        layers.Input((8,)),
        layers.Dense(16, activation="tanh"),
        layers.Dense(2, activation="softmax"),
    ])


def test_end_to_end_fit_parity():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 8)
    y = np.eye(2)[rng.randint(0, 2, size=200)]

    def run(backend):
        B.set_backend(backend)
        np.random.seed(9)
        model = _toy_model()
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        model.fit(X, y, batch_size=32, epochs=3, verbose=0)
        return model

    cpu_model = run("numpy")
    gpu_model = run("cupy")
    np.testing.assert_allclose(
        gpu_model.history.loss, cpu_model.history.loss, rtol=1e-5, atol=1e-8)
    pred = gpu_model.predict(X[:16])
    assert isinstance(pred, np.ndarray)
    assert isinstance(gpu_model.evaluate(X, y), float)


def test_finite_difference_gradcheck_on_gpu():
    B.set_backend("cupy")
    np.random.seed(0)
    X = np.random.randn(8, 4)
    y = np.eye(3)[np.random.randint(0, 3, size=8)]
    model = Sequential([
        layers.Input((4,)),
        layers.Dense(6, activation="tanh"),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    np.random.seed(0)
    y_hat = model._Sequential__forward(B.asarray(X), is_training=True)
    _, g = model._Sequential__criterion(B.asarray(y), y_hat)
    model._Sequential__backward(g)

    def loss_of():
        np.random.seed(0)
        y_hat = model._Sequential__forward(B.asarray(X), is_training=True)
        loss, _ = model._Sequential__criterion(B.asarray(y), y_hat)
        return loss

    eps = 1e-6
    worst = 0.0
    for name, params in model.parameters.items():
        for key, p in params.items():
            ana = model.layers[name].grads[key]
            fp = p.ravel()
            fa = B.asnumpy(ana.ravel())  # host snapshot: cupy element reads are views
            for i in range(fp.size):
                old = fp[i].item()       # Python float snapshot of the device value
                fp[i] = old + eps
                lp = loss_of()
                fp[i] = old - eps
                lm = loss_of()
                fp[i] = old
                num = (lp - lm) / (2 * eps)
                worst = max(worst, abs(num - fa[i]) / max(1e-12, abs(num) + abs(fa[i])))
    assert worst < 1e-3


def test_backend_switch_after_build():
    B.set_backend("numpy")  # build on the host first (the fixture starts on cupy)
    np.random.seed(1)
    model = _toy_model()
    model.compile(loss="mse", optimizer="sgd")
    assert not B.on_gpu()
    X = np.random.RandomState(0).randn(32, 8)
    y = np.random.RandomState(0).randn(32, 2)

    # built under numpy, then switched: __sync_backend reconciles on fit
    B.set_backend("cupy")
    model.fit(X, y, batch_size=8, epochs=1, verbose=0)
    assert B.on_gpu(model.layers["dense_1"].params["W"])

    # and switching back moves everything (optimizer state included)
    B.set_backend("numpy")
    model.fit(X, y, batch_size=8, epochs=1, verbose=0)
    assert not B.on_gpu(model.layers["dense_1"].params["W"])
    for per_layer in model.optimizer.velocity.values():
        for arr in per_layer.values():
            assert not B.on_gpu(arr)
