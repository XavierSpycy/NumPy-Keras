"""Unit tests for the convolutional layers: Conv2D and MaxPool2D.

The layers are pinned against nested-loop reference implementations (the
plain definition of convolution and max pooling), plus a whole-model
finite-difference gradient check and an end-to-end training test.
"""
import numpy as np
import pytest

from numpy_keras import Sequential
from numpy_keras import layers


# ---------------------------------------------------------------------------
# Reference implementations (plain nested loops — the teaching definition)
# ---------------------------------------------------------------------------

def same_pad(H, kh, sh):
    """'same' padding: output = ceil(H / stride); odd total pad goes bottom/right."""
    OH = -(-H // sh)
    pad = max((OH - 1) * sh + kh - H, 0)
    return OH, (pad // 2, pad - pad // 2)


def pad4d(X, pad):
    (ph_b, ph_a), (pw_b, pw_a) = pad
    return np.pad(X, ((0, 0), (ph_b, ph_a), (pw_b, pw_a), (0, 0)))


def conv_forward_ref(X, W, b, stride, pad):
    """X: (N, H, W, C), W: (kh, kw, C, F), b: (F,) or None."""
    N, H, Wd, C = X.shape
    kh, kw, _, F = W.shape
    sh = sw = stride
    Xp = pad4d(X, pad)
    OH = (Xp.shape[1] - kh) // sh + 1
    OW = (Xp.shape[2] - kw) // sw + 1
    out = np.zeros((N, OH, OW, F))
    for n in range(N):
        for i in range(OH):
            for j in range(OW):
                for f in range(F):
                    s = 0.0 if b is None else b[f]
                    for ii in range(kh):
                        for jj in range(kw):
                            for c in range(C):
                                s += Xp[n, i * sh + ii, j * sw + jj, c] * W[ii, jj, c, f]
                    out[n, i, j, f] = s
    return out


def conv_backward_ref(X, W, grad, stride, pad):
    """dW, db and dX by the chain rule, summed over every window."""
    N, H, Wd, C = X.shape
    kh, kw, _, F = W.shape
    sh = sw = stride
    Xp = pad4d(X, pad)
    _, OH, OW, _ = grad.shape
    dW = np.zeros_like(W)
    db = np.zeros(F)
    dXp = np.zeros_like(Xp)
    for n in range(N):
        for i in range(OH):
            for j in range(OW):
                for f in range(F):
                    g = grad[n, i, j, f]
                    db[f] += g
                    for ii in range(kh):
                        for jj in range(kw):
                            for c in range(C):
                                dW[ii, jj, c, f] += Xp[n, i * sh + ii, j * sw + jj, c] * g
                                dXp[n, i * sh + ii, j * sw + jj, c] += W[ii, jj, c, f] * g
    (ph_b, ph_a), (pw_b, pw_a) = pad
    dX = dXp[:, ph_b:dXp.shape[1] - ph_a, pw_b:dXp.shape[2] - pw_a, :]
    return dW, db, dX


def maxpool_forward_ref(X, ph, pw, sh, sw, pad):
    N, H, Wd, C = X.shape
    Xp = pad4d(X, pad)
    OH = (Xp.shape[1] - ph) // sh + 1
    OW = (Xp.shape[2] - pw) // sw + 1
    out = np.zeros((N, OH, OW, C))
    amax = np.zeros((N, OH, OW, C), dtype=int)
    for n in range(N):
        for i in range(OH):
            for j in range(OW):
                for c in range(C):
                    window = [Xp[n, i * sh + ii, j * sw + jj, c]
                              for ii in range(ph) for jj in range(pw)]
                    out[n, i, j, c] = max(window)
                    amax[n, i, j, c] = window.index(out[n, i, j, c])
    return out, amax


def maxpool_backward_ref(X, grad, amax, ph, pw, sh, sw, pad):
    N, H, Wd, C = X.shape
    Xp = pad4d(X, pad)
    _, OH, OW, _ = grad.shape
    dXp = np.zeros_like(Xp)
    for n in range(N):
        for i in range(OH):
            for j in range(OW):
                for c in range(C):
                    dXp[n, i * sh + amax[n, i, j, c] // pw,
                        j * sw + amax[n, i, j, c] % pw, c] += grad[n, i, j, c]
    (ph_b, ph_a), (pw_b, pw_a) = pad
    return dXp[:, ph_b:dXp.shape[1] - ph_a, pw_b:dXp.shape[2] - pw_a, :]


# ---------------------------------------------------------------------------
# Conv2D
# ---------------------------------------------------------------------------

def make_conv(filters=3, kernel_size=3, stride=1, padding=0, use_bias=True,
              input_shape=(6, 7, 2), **kwargs):
    layer = layers.Conv2D(filters, kernel_size=kernel_size, stride=stride,
                          padding=padding, use_bias=use_bias, **kwargs)
    layer.set_input_shape(input_shape)
    return layer


CONV_CONFIGS = [
    # stride, padding
    (1, 0),
    (2, 0),
    (1, "same"),
    (2, "same"),
]


@pytest.mark.parametrize("stride,padding", CONV_CONFIGS)
def test_conv_forward_matches_reference(stride, padding):
    rng = np.random.RandomState(0)
    layer = make_conv(filters=3, kernel_size=3, stride=stride, padding=padding)
    layer.params["W"] = rng.randn(*layer.params["W"].shape)
    layer.params["b"] = rng.randn(*layer.params["b"].shape)
    X = rng.randn(4, 6, 7, 2)

    if padding == "same":
        OH, pad_h = same_pad(6, 3, stride)
        OW, pad_w = same_pad(7, 3, stride)
        pad = (pad_h, pad_w)
    else:
        pad = ((padding, padding), (padding, padding))

    got = layer.forward(X, is_training=True)
    want = conv_forward_ref(X, layer.params["W"], layer.params["b"], stride, pad)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_conv_forward_without_bias():
    rng = np.random.RandomState(1)
    layer = make_conv(use_bias=False)
    layer.params["W"] = rng.randn(*layer.params["W"].shape)
    X = rng.randn(4, 6, 7, 2)
    got = layer.forward(X, is_training=True)
    want = conv_forward_ref(X, layer.params["W"], None, 1, ((0, 0), (0, 0)))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_conv_forward_with_rectangular_kernel():
    rng = np.random.RandomState(2)
    layer = make_conv(kernel_size=(2, 3), stride=(1, 2))
    layer.params["W"] = rng.randn(*layer.params["W"].shape)
    layer.params["b"] = rng.randn(*layer.params["b"].shape)
    X = rng.randn(4, 6, 7, 2)
    got = layer.forward(X, is_training=True)
    # the loop reference handles rectangular kernels / strides directly
    kh, kw = 2, 3
    sh, sw = 1, 2
    Xp = pad4d(X, ((0, 0), (0, 0)))
    OH = (Xp.shape[1] - kh) // sh + 1
    OW = (Xp.shape[2] - kw) // sw + 1
    N = X.shape[0]
    F = layer.params["W"].shape[3]
    want = np.zeros((N, OH, OW, F))
    for n in range(N):
        for i in range(OH):
            for j in range(OW):
                for f in range(F):
                    s = layer.params["b"][f]
                    for ii in range(kh):
                        for jj in range(kw):
                            for c in range(X.shape[3]):
                                s += Xp[n, i * sh + ii, j * sw + jj, c] * layer.params["W"][ii, jj, c, f]
                    want[n, i, j, f] = s
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_conv_forward_applies_activation():
    rng = np.random.RandomState(3)
    layer = make_conv(activation="relu")
    layer.params["W"] = rng.randn(*layer.params["W"].shape)
    layer.params["b"] = rng.randn(*layer.params["b"].shape)
    X = rng.randn(4, 6, 7, 2)
    got = layer.forward(X, is_training=True)
    want = conv_forward_ref(X, layer.params["W"], layer.params["b"], 1, ((0, 0), (0, 0)))
    want = np.maximum(want, 0) * (want > 0)  # F.relu: x * (x > 0)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("stride,padding", CONV_CONFIGS)
@pytest.mark.parametrize("use_bias", [True, False])
def test_conv_backward_matches_reference(stride, padding, use_bias):
    rng = np.random.RandomState(4)
    layer = make_conv(filters=2, kernel_size=3, stride=stride, padding=padding,
                      use_bias=use_bias)
    layer.params["W"] = rng.randn(*layer.params["W"].shape)
    if use_bias:
        layer.params["b"] = rng.randn(*layer.params["b"].shape)
    X = rng.randn(4, 6, 7, 2)

    if padding == "same":
        OH, pad_h = same_pad(6, 3, stride)
        OW, pad_w = same_pad(7, 3, stride)
        pad = (pad_h, pad_w)
    else:
        pad = ((padding, padding), (padding, padding))

    layer.forward(X, is_training=True)
    grad = rng.randn(4, layer.output_shape[0], layer.output_shape[1], 2)
    grad_next = layer.backward(grad)

    dW, db, dX = conv_backward_ref(X, layer.params["W"], grad, stride, pad)
    np.testing.assert_allclose(layer.grads["W"], dW, rtol=1e-12, atol=1e-12)
    if use_bias:
        np.testing.assert_allclose(layer.grads["b"], db, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(grad_next, dX, rtol=1e-12, atol=1e-12)


def test_conv_backward_applies_previous_activation_deriv():
    """The returned gradient is multiplied by the previous layer's
    activation derivative, evaluated at this layer's input -- the same
    mechanism Dense relies on."""
    rng = np.random.RandomState(5)
    layer = make_conv(use_bias=False)
    layer.set_activation_deriv("tanh", {})
    layer.params["W"] = rng.randn(*layer.params["W"].shape)
    X = rng.randn(4, 6, 7, 2)
    layer.forward(X, is_training=True)
    grad = rng.randn(4, 4, 5, 3)
    grad_next = layer.backward(grad)
    _, _, dX = conv_backward_ref(X, layer.params["W"], grad, 1, ((0, 0), (0, 0)))
    np.testing.assert_allclose(grad_next, dX * (1 - X ** 2), rtol=1e-12, atol=1e-12)


def test_conv_output_shape_with_same_padding():
    layer = make_conv(kernel_size=3, stride=2, padding="same", input_shape=(5, 5, 1))
    assert layer.output_shape == (3, 3, 3)
    assert layer.output_dim == 27
    assert layer.params["W"].shape == (3, 3, 1, 3)
    assert layer.params["b"].shape == (3,)


def test_conv_raises_when_kernel_too_large():
    layer = layers.Conv2D(2, kernel_size=3)
    with pytest.raises(ValueError):
        layer.set_input_shape((2, 2, 1))


def test_conv_raises_without_input_shape():
    layer = layers.Conv2D(2, kernel_size=3)
    with pytest.raises(ValueError):
        layer.forward(np.random.randn(4, 6, 6, 1), is_training=True)


def test_conv_rejects_wrong_ndim_input():
    layer = make_conv(input_shape=(6, 6, 2))
    with pytest.raises(ValueError):
        layer.forward(np.random.randn(4, 6), is_training=True)
    with pytest.raises(ValueError):
        layer.forward(np.random.randn(4, 6, 6, 3), is_training=True)  # wrong channels


# ---------------------------------------------------------------------------
# MaxPool2D
# ---------------------------------------------------------------------------

def make_pool(pool_size=2, stride=None, padding=0, input_shape=(6, 6, 2)):
    layer = layers.MaxPool2D(pool_size=pool_size, stride=stride, padding=padding)
    layer.set_input_shape(input_shape)
    return layer


POOL_CONFIGS = [
    # pool_size, stride, padding
    (2, None, 0),
    (2, 1, 0),          # overlapping windows: backward must accumulate
    (3, 2, "same"),
]


@pytest.mark.parametrize("pool_size,stride,padding", POOL_CONFIGS)
def test_maxpool_forward_matches_reference(pool_size, stride, padding):
    rng = np.random.RandomState(6)
    layer = make_pool(pool_size=pool_size, stride=stride, padding=padding)
    X = rng.randn(4, 6, 6, 2)
    ph = pw = pool_size
    sh = sw = pool_size if stride is None else stride

    if padding == "same":
        OH, pad_h = same_pad(6, ph, sh)
        OW, pad_w = same_pad(6, pw, sw)
        pad = (pad_h, pad_w)
    else:
        pad = ((padding, padding), (padding, padding))

    got = layer.forward(X, is_training=True)
    want, amax = maxpool_forward_ref(X, ph, pw, sh, sw, pad)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(layer._MaxPool2D__amax, amax)


@pytest.mark.parametrize("pool_size,stride,padding", POOL_CONFIGS)
def test_maxpool_backward_matches_reference(pool_size, stride, padding):
    rng = np.random.RandomState(7)
    layer = make_pool(pool_size=pool_size, stride=stride, padding=padding)
    X = rng.randn(4, 6, 6, 2)
    ph = pw = pool_size
    sh = sw = pool_size if stride is None else stride

    if padding == "same":
        OH, pad_h = same_pad(6, ph, sh)
        OW, pad_w = same_pad(6, pw, sw)
        pad = (pad_h, pad_w)
    else:
        pad = ((padding, padding), (padding, padding))

    layer.forward(X, is_training=True)
    grad = rng.randn(4, *layer.output_shape)
    got = layer.backward(grad)
    want = maxpool_backward_ref(X, grad, layer._MaxPool2D__amax, ph, pw, sh, sw, pad)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_maxpool_output_shape():
    layer = make_pool(pool_size=2, input_shape=(7, 9, 3))
    assert layer.output_shape == (3, 4, 3)
    assert layer.output_dim == 36


# ---------------------------------------------------------------------------
# Shape chain + Flatten + Input
# ---------------------------------------------------------------------------

def test_input_accepts_tuple_shape():
    inp = layers.Input((28, 28, 1))
    assert inp.output_shape == (28, 28, 1)


def test_flatten_auto_infers_input_shape_from_previous_layer():
    model = Sequential()
    model.add(layers.Input((4, 4, 2)))
    model.add(layers.Conv2D(3, kernel_size=3))
    model.add(layers.Flatten())          # no manual input_shape
    model.add(layers.Dense(1, activation="linear"))
    conv_out = model.layers["conv2_d_1"].output_shape   # (2, 2, 3)
    flatten = model.layers["flatten_1"]
    assert flatten.output_dim == 2 * 2 * 3
    assert model.layers["dense_1"].params["W"].shape == (conv_out[0] * conv_out[1] * conv_out[2], 1)


def test_flatten_manual_input_shape_wins_over_inference():
    f = layers.Flatten(input_shape=(28, 28))
    f.set_input_shape((4, 4, 2))
    assert f.output_dim == 28 * 28


def test_flatten_backward_reshapes_gradient():
    f = layers.Flatten(input_shape=(2, 2, 3))
    X = np.random.randn(5, 2, 2, 3)
    np.testing.assert_allclose(f.forward(X, is_training=True), X.reshape(5, -1))
    grad = np.random.randn(5, 12)
    np.testing.assert_allclose(f.backward(grad), grad.reshape(5, 2, 2, 3))


def test_shape_chain_passes_through_activation_and_dropout():
    model = Sequential()
    model.add(layers.Input((4, 4, 1)))
    model.add(layers.Conv2D(2, kernel_size=3))
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="linear"))
    assert model.layers["flatten_1"].output_dim == 2 * 2 * 2


def test_shape_chain_passes_through_batch_norm():
    model = Sequential()
    model.add(layers.Input((4, 4, 1)))
    model.add(layers.Conv2D(2, kernel_size=3))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="linear"))
    assert model.layers["flatten_1"].output_dim == 2 * 2 * 2
    assert model.layers["batch_normalization_1"].params["gamma"].shape == (2,)


# ---------------------------------------------------------------------------
# Whole-model checks
# ---------------------------------------------------------------------------

def test_cnn_gradient_check_against_finite_differences():
    """Backprop grads of a Conv-Pool-Flatten-Dense model must equal finite
    differences of the reported loss."""
    np.random.seed(8)
    rng = np.random.RandomState(8)
    model = Sequential()
    model.add(layers.Input((6, 6, 1)))
    model.add(layers.Conv2D(2, kernel_size=3, activation="tanh"))
    model.add(layers.MaxPool2D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="sgd")
    for layer in model.layers.values():
        if hasattr(layer, "params"):
            for k in layer.params:
                layer.params[k] = rng.randn(*layer.params[k].shape) * 0.3

    X = rng.randn(4, 6, 6, 1)
    y = rng.randn(4, 1) * 0.5

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
            np.testing.assert_allclose(
                layer.grads[k].ravel(), numerical, rtol=1e-3, atol=1e-6,
                err_msg=f"{name}.{k}")


def test_cnn_end_to_end_classifies_striped_images():
    """A tiny CNN must learn to separate 4x4 images with a vertical white
    stripe (class 0) from those with a horizontal one (class 1)."""
    np.random.seed(9)
    rng = np.random.RandomState(9)
    n = 120
    X = np.zeros((n, 4, 4, 1))
    y = np.arange(n) % 2
    X[y == 0, :, :2, 0] = 1.0    # vertical stripe: left two columns
    X[y == 1, :2, :, 0] = 1.0    # horizontal stripe: top two rows
    X += 0.05 * rng.randn(n, 4, 4, 1)

    model = Sequential()
    model.add(layers.Input((4, 4, 1)))
    model.add(layers.Conv2D(2, kernel_size=3, activation="relu"))
    model.add(layers.MaxPool2D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.optimizer.learning_rate = 0.05

    history = model.fit(X, y, batch_size=16, epochs=40, shuffle=True)
    assert history.metrics["train_accuracy"][-1] > 0.9
    pred = model.predict(X)
    assert pred.shape == (n,)
    assert np.mean(pred == y) > 0.9


def test_cnn_supports_validation_and_summary(capsys):
    np.random.seed(10)
    rng = np.random.RandomState(10)
    X = rng.randn(24, 4, 4, 1)
    y = rng.randint(0, 2, 24)
    model = Sequential()
    model.add(layers.Input((4, 4, 1)))
    model.add(layers.Conv2D(2, kernel_size=3, activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    history = model.fit(X, y, batch_size=8, epochs=2, validation_split=0.25)
    assert "val_loss" in history.metrics
    model.summary()
    out = capsys.readouterr().out
    assert "(4, 4, 1)" in out
    assert "(2, 2, 2)" in out
