"""Float32 support: the compute dtype of a Sequential is a first-class
choice (``Sequential(..., dtype="float32")``), not an accident of the
input data.  The GPU cases run when CuPy is available; the rest of the
suite stays float64, so its rtol=1e-12 pins are untouched.
"""
import numpy as np
import pytest

from numpy_keras import backend as B

B.set_backend("cupy")
CUPY = B.is_cupy_mode()
B.set_backend("numpy")

from numpy_keras import Sequential
from numpy_keras import layers


@pytest.fixture(autouse=True)
def _numpy_backend():
    B.set_backend("numpy")
    yield
    B.set_backend("numpy")


def _build(dtype="float32"):
    return Sequential([
        layers.Input(8),
        layers.Dense(16, activation="tanh"),
        layers.Dropout(rate=0.3),
        layers.Dense(2, activation="softmax"),
    ], dtype=dtype)


def test_dtype_propagates_to_params_and_grads():
    model = _build()
    for layer in model.layers.values():
        for attr in ("params", "grads"):
            for value in getattr(layer, attr, {}).values():
                assert value.dtype == np.float32


def test_fit_runs_and_decreases_loss_in_float32():
    rng = np.random.RandomState(0)
    X = rng.randn(128, 8).astype(np.float32)
    y = np.eye(2)[rng.randint(0, 2, 128)].astype(np.float32)
    np.random.seed(0)                # fixed initial weights: deterministic
    model = Sequential([
        layers.Input(8),
        layers.Dense(16, activation="tanh"),
        layers.Dense(2, activation="softmax"),
    ], dtype="float32")              # no dropout: keep the trajectory clean
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.fit(X, y, batch_size=16, epochs=4, verbose=0)
    assert model.history.loss[-1] < model.history.loss[0]
    pred = model.predict(X[:8])
    assert pred.shape == (8, 2)      # categorical loss -> softmax outputs
    assert pred.dtype == np.float32  # predictions keep the model dtype


def test_float64_input_is_cast_to_model_dtype():
    """A float64 dataset must not silently promote a float32 model."""
    rng = np.random.RandomState(0)
    X = rng.randn(32, 8)             # float64 on purpose
    y = np.eye(2)[rng.randint(0, 2, 32)]
    np.random.seed(0)                # fixed initial weights: deterministic
    model = _build()
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.fit(X, y, batch_size=8, epochs=2, verbose=0)
    assert model.layers["dense_1"].params["W"].dtype == np.float32
    assert model.history.loss[-1] < model.history.loss[0]


def test_dropout_keeps_float32_outputs():
    model = Sequential([
        layers.Input(8),
        layers.Dropout(rate=0.4),
    ], dtype="float32")
    X = np.random.RandomState(0).randn(16, 8).astype(np.float32)
    out = model._Sequential__forward(X, is_training=True)
    assert out.dtype == np.float32


def test_default_dtype_stays_float64():
    model = Sequential([
        layers.Input(8),
        layers.Dense(4, activation="tanh"),
    ])
    assert model.layers["dense_1"].params["W"].dtype == np.float64


# ---------------------------------------------------------------------------
# GPU cases (skip without CuPy)
# ---------------------------------------------------------------------------

pytestmark_gpu = pytest.mark.skipif(not CUPY, reason="cupy not available")


@pytest.fixture
def _cupy_backend():
    B.set_backend("cupy")
    yield
    B.set_backend("numpy")


@pytestmark_gpu
def test_float32_dense_parity(_cupy_backend):
    rng = np.random.RandomState(0)
    X = rng.randn(16, 32).astype(np.float32)
    grad = rng.randn(16, 16).astype(np.float32)

    def build(backend):
        B.set_backend(backend)
        np.random.seed(7)
        layer = layers.Dense(16, activation="tanh")
        layer.set_input_shape((32,))
        layer.init_params(32)
        layer.params["W"] = B.asarray(layer.params["W"], dtype=np.float32)
        layer.params["b"] = B.asarray(layer.params["b"], dtype=np.float32)
        layer.grads["W"] = B.asarray(layer.grads["W"], dtype=np.float32)
        layer.grads["b"] = B.asarray(layer.grads["b"], dtype=np.float32)
        return layer

    cpu_layer = build("numpy")
    out_cpu = cpu_layer.forward(X, is_training=True)
    grad_cpu = cpu_layer.backward(grad)

    gpu_layer = build("cupy")
    out_gpu = B.asnumpy(gpu_layer.forward(B.asarray(X), is_training=True))
    grad_gpu = B.asnumpy(gpu_layer.backward(B.asarray(grad)))
    np.testing.assert_allclose(out_gpu, out_cpu, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(grad_gpu, grad_cpu, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(
        B.asnumpy(gpu_layer.grads["W"]), cpu_layer.grads["W"], rtol=1e-3, atol=1e-5)


@pytestmark_gpu
def test_float32_fused_optimizer_parity(_cupy_backend):
    """The dtype-generic fused GPU kernels must match the CPU pure path
    in float32 too."""
    from numpy_keras import optimizers

    rng = np.random.RandomState(0)
    gW = rng.randn(8, 8).astype(np.float32)
    gb = rng.randn(8).astype(np.float32)

    def build_pair(backend):
        B.set_backend(backend)
        np.random.seed(3)
        layer = layers.Dense(8, activation="tanh")
        layer.set_input_shape((8,))
        layer.init_params(8)
        for key in layer.params:
            layer.params[key] = B.asarray(layer.params[key], dtype=np.float32)
            layer.grads[key] = B.asarray(layer.grads[key], dtype=np.float32)
        layer.grads["W"] = B.asarray(gW) if backend == "cupy" else gW.copy()
        layer.grads["b"] = B.asarray(gb) if backend == "cupy" else gb.copy()
        return layer

    cpu_layer = build_pair("numpy")
    cpu_opt = optimizers.Adam(learning_rate=1e-3, weight_decay=1e-3)
    for _ in range(5):
        cpu_opt.update([cpu_layer])

    gpu_layer = build_pair("cupy")
    gpu_opt = optimizers.Adam(learning_rate=1e-3, weight_decay=1e-3)
    for _ in range(5):
        gpu_opt.update([gpu_layer])

    np.testing.assert_allclose(
        B.asnumpy(gpu_layer.params["W"]), cpu_layer.params["W"], rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(
        B.asnumpy(gpu_layer.params["b"]), cpu_layer.params["b"], rtol=1e-4, atol=1e-7)


@pytestmark_gpu
def test_float32_end_to_end_trajectory_parity(_cupy_backend):
    rng = np.random.RandomState(0)
    X = rng.randn(200, 8).astype(np.float32)
    y = np.eye(2)[rng.randint(0, 2, 200)].astype(np.float32)

    def run(backend):
        B.set_backend(backend)
        np.random.seed(9)
        model = _build()
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        model.fit(X, y, batch_size=32, epochs=3, verbose=0)
        return model

    cpu_model = run("numpy")
    gpu_model = run("cupy")
    np.testing.assert_allclose(
        gpu_model.history.loss, cpu_model.history.loss, rtol=1e-3, atol=1e-5)


@pytestmark_gpu
def test_float32_rnn_parity(_cupy_backend):
    """The RNN GPU branch (batched projection + tensordot grads) in f32."""
    rng = np.random.RandomState(0)
    X = rng.randn(6, 5, 4).astype(np.float32)
    grad = rng.randn(6, 8).astype(np.float32)

    def build(backend):
        B.set_backend(backend)
        np.random.seed(13)
        layer = layers.LSTM(units=8)
        layer.set_input_shape((5, 4))
        for key in layer.params:
            layer.params[key] = B.asarray(layer.params[key], dtype=np.float32)
            layer.grads[key] = B.asarray(layer.grads[key], dtype=np.float32)
        return layer

    cpu_layer = build("numpy")
    out_cpu = cpu_layer.forward(X, is_training=True)
    grad_cpu = cpu_layer.backward(grad)

    gpu_layer = build("cupy")
    out_gpu = B.asnumpy(gpu_layer.forward(B.asarray(X), is_training=True))
    grad_gpu = B.asnumpy(gpu_layer.backward(B.asarray(grad)))
    np.testing.assert_allclose(out_gpu, out_cpu, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(grad_gpu, grad_cpu, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(
        B.asnumpy(gpu_layer.grads["W_xh"]), cpu_layer.grads["W_xh"], rtol=1e-3, atol=1e-5)
