"""Parity tests: every Cython kernel must match its pure NumPy counterpart.

These tests are skipped when the compiled module is not built
(``python build_cython.py build_ext --inplace``).  Tolerance notes:

- Activation kernels use libm exp/tanh, which may differ from NumPy's SIMD
  loops by ~1 ulp: rtol=1e-12.
- relu/relu_deriv contain no transcendentals and must be bit-exact,
  including the NaN semantics (relu propagates NaN, relu_deriv does not).
- Optimizer kernels contain only sqrt/multiply/add (all correctly rounded
  in both NumPy and libm) and mirror NumPy's operation order statement by
  statement: rtol=1e-15.
- dense_backward's db uses a sequential reduction where NumPy sums
  pairwise: rtol=1e-12.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from numpy_keras.cython import _kernels

pytestmark = pytest.mark.skipif(_kernels is None, reason="compiled kernels not available")

from numpy_keras import optimizers
from numpy_keras.activations import functional as F


# ---------------------------------------------------------------------------
# Elementwise activations
# ---------------------------------------------------------------------------

ELEMENTWISE = ["relu", "relu_deriv", "sigmoid", "sigmoid_deriv", "tanh", "tanh_deriv"]


@pytest.mark.parametrize("name", ELEMENTWISE)
def test_activation_1d_matches_numpy(name):
    rng = np.random.RandomState(0)
    x = rng.randn(1000)
    got = getattr(_kernels, name)(x)
    want = getattr(F, name)(x)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", ELEMENTWISE)
@pytest.mark.parametrize("shape", [(64, 32), (3, 5)])
def test_activation_2d_matches_numpy(name, shape):
    rng = np.random.RandomState(1)
    x = rng.randn(*shape)
    np.testing.assert_allclose(
        getattr(_kernels, name)(x), getattr(F, name)(x), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("name", ELEMENTWISE)
def test_activation_strided_inputs(name):
    rng = np.random.RandomState(2)
    base = rng.randn(64, 64)
    strided = base[:, ::2]          # non-contiguous view
    transposed = base.T
    want_strided = getattr(F, name)(strided)
    want_transposed = getattr(F, name)(transposed)
    np.testing.assert_allclose(
        getattr(_kernels, name)(strided), want_strided, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        getattr(_kernels, name)(transposed), want_transposed, rtol=1e-12, atol=1e-12
    )


def test_relu_nan_semantics_bit_exact():
    """relu is ``x * (x > 0)`` (NaN propagates); relu_deriv is ``1. * (a > 0)``
    (the comparison consumes NaN, so NaN -> 0.0).  Both must be bit-exact."""
    x = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0])
    np.testing.assert_array_equal(_kernels.relu(x), F.relu(x))
    np.testing.assert_array_equal(_kernels.relu_deriv(x), F.relu_deriv(x))


def test_sigmoid_extreme_values():
    x = np.array([-1000.0, -700.0, -1.0, -0.0, 0.0, 1.0, 700.0, 1000.0])
    np.testing.assert_allclose(_kernels.sigmoid(x), F.sigmoid(x), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def test_softmax_matches_numpy():
    rng = np.random.RandomState(3)
    x = rng.randn(5, 4)
    np.testing.assert_allclose(_kernels.softmax(x), F.softmax(x), rtol=1e-12, atol=1e-12)
    # max-subtraction must be shift-invariant
    np.testing.assert_allclose(
        _kernels.softmax(x + 100.0), F.softmax(x + 100.0), rtol=1e-12, atol=1e-12
    )


def test_softmax_nan_row_propagates_like_numpy():
    rng = np.random.RandomState(4)
    x = rng.randn(5, 4)
    x[2, 1] = np.nan
    np.testing.assert_allclose(
        _kernels.softmax(x), F.softmax(x), rtol=1e-12, atol=1e-12, equal_nan=True
    )


# ---------------------------------------------------------------------------
# Fused Dense forward / backward
# ---------------------------------------------------------------------------

ACT_CODES = [(0, "linear"), (1, "relu"), (2, "sigmoid"), (3, "tanh"), (4, "softmax")]


def _pure_forward(X, W, b, act):
    lin = X @ W + b
    if act == "linear":
        return lin
    return getattr(F, act)(lin)


@pytest.mark.parametrize("code,act", ACT_CODES)
def test_dense_forward_with_bias(code, act):
    rng = np.random.RandomState(5)
    X = rng.randn(7, 5)
    W = rng.randn(5, 4)
    b = rng.randn(4)
    got = _kernels.dense_forward(X, W, b, code)
    want = _pure_forward(X, W, b, act)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("code,act", ACT_CODES)
def test_dense_forward_without_bias(code, act):
    rng = np.random.RandomState(6)
    X = rng.randn(7, 5)
    W = rng.randn(5, 4)
    got = _kernels.dense_forward(X, W, None, code)
    want = getattr(F, act)(X @ W) if act != "linear" else X @ W
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_dense_forward_strided_inputs():
    rng = np.random.RandomState(7)
    base = rng.randn(14, 10)
    X = base[:, ::2]              # strided view
    W = rng.randn(5, 4)
    b = rng.randn(4)
    np.testing.assert_allclose(
        _kernels.dense_forward(X, W, b, 3),
        _pure_forward(X, W, b, "tanh"),
        rtol=1e-12, atol=1e-12,
    )


DERIV_CODES = [(1, F.relu_deriv), (2, F.sigmoid_deriv), (3, F.tanh_deriv)]


@pytest.mark.parametrize("code,deriv", DERIV_CODES)
def test_dense_backward_matches_numpy(code, deriv):
    rng = np.random.RandomState(8)
    X = rng.randn(64, 5)
    W = rng.randn(5, 4)
    g = rng.randn(64, 4)
    out = rng.randn(64, 4)        # the layer's cached post-activation output
    dz = g * deriv(out)           # the kernel scales first: dz = grad ⊙ f'(out)
    dW, db, grad_next = _kernels.dense_backward(X, out, g, W, code)
    np.testing.assert_allclose(dW, X.T @ dz, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(db, dz.sum(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(grad_next, dz @ W.T, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("code", [-1, 0])
def test_dense_backward_no_deriv_multiply(code):
    """Codes -1 (none) and 0 (linear): grad_next must skip the deriv pass."""
    rng = np.random.RandomState(9)
    X = rng.randn(10, 5)
    W = rng.randn(5, 4)
    g = rng.randn(10, 4)
    out = rng.randn(10, 4)
    dW, db, grad_next = _kernels.dense_backward(X, out, g, W, code)
    np.testing.assert_allclose(grad_next, g @ W.T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dW, X.T @ g, rtol=1e-12, atol=1e-12)


def test_dense_backward_strided_inputs():
    rng = np.random.RandomState(10)
    base = rng.randn(20, 10)
    X = base[:, ::2]              # strided view
    W = rng.randn(5, 4)
    g = rng.randn(20, 4)
    out = rng.randn(20, 4)
    _, _, grad_next = _kernels.dense_backward(X, out, g, W, 3)
    np.testing.assert_allclose(
        grad_next, (g * F.tanh_deriv(out)) @ W.T, rtol=1e-12, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Optimizer kernels vs the pure optimizer classes
# ---------------------------------------------------------------------------

class FakeLayer:
    def __init__(self, **params):
        self.params = {k: np.array(v, dtype=float) for k, v in params.items()}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}


def make_layer():
    rng = np.random.RandomState(11)
    return FakeLayer(w=rng.randn(4, 3), b=rng.randn(3))


def set_grads(layer):
    rng = np.random.RandomState(12)
    for k, v in layer.grads.items():
        layer.grads[k] = rng.randn(*v.shape) * 0.1


def _adam_kernel_steps(layer, steps, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.05):
    m = {k: np.zeros_like(v) for k, v in layer.params.items()}
    v = {k: np.zeros_like(v) for k, v in layer.params.items()}
    for t in range(1, steps + 1):
        bc1 = 1 - beta1 ** t
        bc2 = 1 - beta2 ** t
        for k in layer.params:
            _kernels.adam_update(
                layer.params[k], layer.grads[k], m[k], v[k],
                lr, beta1, beta2, eps, wd, bc1, bc2,
            )
    return m, v


def test_adam_kernel_matches_pure_update():
    pure = make_layer()
    fast = make_layer()
    for layer in (pure, fast):
        set_grads(layer)

    opt = optimizers.Adam(learning_rate=0.01, beta1=0.9, beta2=0.999,
                          epsilon=1e-8, weight_decay=0.05)
    for _ in range(5):
        opt.update([pure])
    m_fast, v_fast = _adam_kernel_steps(fast, 5)

    for k in pure.params:
        np.testing.assert_allclose(fast.params[k], pure.params[k], rtol=1e-15, atol=0)
        # the decayed gradient must be written back into layer.grads
        np.testing.assert_allclose(fast.grads[k], pure.grads[k], rtol=1e-15, atol=0)
    for i in opt.first_moment:
        for k in opt.first_moment[i]:
            np.testing.assert_allclose(m_fast[k], opt.first_moment[i][k], rtol=1e-15, atol=0)
            np.testing.assert_allclose(v_fast[k], opt.second_moment[i][k], rtol=1e-15, atol=0)


def _sgd_kernel_steps(layer, steps, lr=0.05, momentum=0.9, wd=0.05, nesterov=False):
    v = {k: np.zeros_like(p) for k, p in layer.params.items()}
    for _ in range(steps):
        for k in layer.params:
            _kernels.sgd_update(layer.params[k], layer.grads[k], v[k],
                                lr, momentum, wd, int(nesterov))
    return v


@pytest.mark.parametrize("nesterov", [False, True])
def test_sgd_kernel_matches_pure_update(nesterov):
    pure = make_layer()
    fast = make_layer()
    for layer in (pure, fast):
        set_grads(layer)

    opt = optimizers.SGD(learning_rate=0.05, momentum=0.9, nesterov=nesterov,
                         weight_decay=0.05)
    for _ in range(5):
        opt.update([pure])
    v_fast = _sgd_kernel_steps(fast, 5, nesterov=nesterov)

    for k in pure.params:
        np.testing.assert_allclose(fast.params[k], pure.params[k], rtol=1e-15, atol=0)
        np.testing.assert_allclose(fast.grads[k], pure.grads[k], rtol=1e-15, atol=0)
    for i in opt.velocity:
        for k in opt.velocity[i]:
            np.testing.assert_allclose(v_fast[k], opt.velocity[i][k], rtol=1e-15, atol=0)


def _adagrad_kernel_steps(layer, steps, lr=0.1, eps=1e-10, wd=0.05):
    gs = {k: np.zeros_like(p) for k, p in layer.params.items()}
    for _ in range(steps):
        for k in layer.params:
            _kernels.adagrad_update(layer.params[k], layer.grads[k], gs[k], lr, eps, wd)
    return gs


def test_adagrad_kernel_matches_pure_update():
    pure = make_layer()
    fast = make_layer()
    for layer in (pure, fast):
        set_grads(layer)

    opt = optimizers.Adagrad(learning_rate=0.1, epsilon=1e-10, weight_decay=0.05)
    for _ in range(5):
        opt.update([pure])
    gs_fast = _adagrad_kernel_steps(fast, 5)

    for k in pure.params:
        np.testing.assert_allclose(fast.params[k], pure.params[k], rtol=1e-15, atol=0)
        np.testing.assert_allclose(fast.grads[k], pure.grads[k], rtol=1e-15, atol=0)
    for i in opt.grad_square:
        for k in opt.grad_square[i]:
            np.testing.assert_allclose(gs_fast[k], opt.grad_square[i][k], rtol=1e-15, atol=0)


def _adadelta_kernel_steps(layer, steps, lr=1.0, rho=0.9, eps=1e-6, wd=0.05):
    ags = {k: np.zeros_like(p) for k, p in layer.params.items()}
    ads = {k: np.zeros_like(p) for k, p in layer.params.items()}
    for _ in range(steps):
        for k in layer.params:
            _kernels.adadelta_update(layer.params[k], layer.grads[k], ags[k], ads[k],
                                     lr, rho, eps, wd)
    return ags, ads


def test_adadelta_kernel_matches_pure_update():
    pure = make_layer()
    fast = make_layer()
    for layer in (pure, fast):
        set_grads(layer)

    opt = optimizers.Adadelta(learning_rate=1.0, rho=0.9, epsilon=1e-6, weight_decay=0.05)
    for _ in range(5):
        opt.update([pure])
    ags_fast, ads_fast = _adadelta_kernel_steps(fast, 5)

    for k in pure.params:
        np.testing.assert_allclose(fast.params[k], pure.params[k], rtol=1e-15, atol=0)
        np.testing.assert_allclose(fast.grads[k], pure.grads[k], rtol=1e-15, atol=0)
    for i in opt.accum_grad_square:
        for k in opt.accum_grad_square[i]:
            np.testing.assert_allclose(ags_fast[k], opt.accum_grad_square[i][k],
                                       rtol=1e-15, atol=0)
            np.testing.assert_allclose(ads_fast[k], opt.accum_delta_square[i][k],
                                       rtol=1e-15, atol=0)


# ---------------------------------------------------------------------------
# argmax_rows
# ---------------------------------------------------------------------------

def test_argmax_rows_matches_numpy():
    rng = np.random.RandomState(13)
    x = rng.randn(6, 5)
    np.testing.assert_array_equal(_kernels.argmax_rows(x), np.argmax(x, axis=1))
    # explicit tie: the first maximum wins, like np.argmax
    x[3, 0] = x[3, 2]
    np.testing.assert_array_equal(_kernels.argmax_rows(x), np.argmax(x, axis=1))


# ---------------------------------------------------------------------------
# Convolution kernels: im2col / col2im / maxpool
# ---------------------------------------------------------------------------

def _im2col_ref(x, kh, kw, sh, sw):
    """The pure NumPy im2col from layers/conv2d.py."""
    cols = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))[:, ::sh, ::sw]
    cols = cols.transpose(0, 1, 2, 4, 5, 3)   # (N, OH, OW, kh, kw, C)
    N, OH, OW, _, _, C = cols.shape
    return cols.reshape(N * OH * OW, kh * kw * C)


def test_col2im_matches_numpy_scatter():
    rng = np.random.RandomState(15)
    x = rng.randn(3, 7, 7, 2)
    kh = kw = 3
    sh = sw = 2
    grad_cols = rng.randn(*_im2col_ref(x, kh, kw, sh, sw).shape)
    OH = (7 - 3) // 2 + 1
    OW = OH
    N, Hp, Wp, C = x.shape

    # the pure np.add.at scatter from Conv2D.__col2im
    want = np.zeros((N, Hp, Wp, C))
    M = OH * OW
    n_idx = np.repeat(np.arange(N), M)
    i_idx = np.tile(np.repeat(np.arange(OH), OW), N)
    j_idx = np.tile(np.tile(np.arange(OW), OH), N)
    flat = grad_cols.reshape(N * M, kh * kw, C)
    for i in range(kh):
        for j in range(kw):
            np.add.at(want, (n_idx, i_idx * sh + i, j_idx * sw + j, slice(None)),
                      flat[:, i * kw + j, :])

    got = np.zeros((N, Hp, Wp, C))
    _kernels.col2im(grad_cols, got, kh, kw, sh, sw)
    # accumulation order differs (rows vs kernel offsets): ~1 ulp
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_maxpool_forward_matches_numpy():
    rng = np.random.RandomState(16)
    x = rng.randn(3, 8, 8, 2)
    ph = pw = 2
    sh = sw = 2
    win = np.lib.stride_tricks.sliding_window_view(x, (ph, pw), axis=(1, 2))[:, ::sh, ::sw]
    win = win.transpose(0, 1, 2, 4, 5, 3)
    N, OH, OW, _, _, C = win.shape
    want = np.max(win, axis=(3, 4))
    want_amax = np.argmax(win.reshape(N, OH, OW, ph * pw, C), axis=3)
    got, got_amax = _kernels.maxpool_forward(x, ph, pw, sh, sw)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(got_amax, want_amax)


def test_maxpool_forward_nan_semantics():
    """np.max propagates NaN and np.argmax picks the first NaN position."""
    x = np.zeros((2, 4, 4, 1))
    x[0, 1, 1, 0] = np.nan
    win = np.lib.stride_tricks.sliding_window_view(x, (2, 2), axis=(1, 2))[:, ::2, ::2]
    win = win.transpose(0, 1, 2, 4, 5, 3)
    N, OH, OW, _, _, C = win.shape
    want = np.max(win, axis=(3, 4))
    want_amax = np.argmax(win.reshape(N, OH, OW, 4, C), axis=3)
    got, got_amax = _kernels.maxpool_forward(x, 2, 2, 2, 2)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(got_amax, want_amax)


def test_maxpool_backward_matches_numpy_scatter():
    rng = np.random.RandomState(17)
    x = rng.randn(2, 5, 5, 2)
    ph = pw = 2
    sh = sw = 1          # overlapping windows: contributions accumulate
    out, amax = _kernels.maxpool_forward(x, ph, pw, sh, sw)
    grad = rng.randn(*out.shape)

    # the pure np.add.at scatter from MaxPool2D.backward
    N, OH, OW, C = grad.shape
    want = np.zeros_like(x)
    n_idx, i_idx, j_idx, c_idx = np.meshgrid(
        np.arange(N), np.arange(OH), np.arange(OW), np.arange(C), indexing='ij')
    h_abs = i_idx * sh + amax // pw
    w_abs = j_idx * sw + amax % pw
    np.add.at(want, (n_idx.ravel(), h_abs.ravel(), w_abs.ravel(), c_idx.ravel()),
              grad.ravel())

    got = np.zeros_like(x)
    _kernels.maxpool_backward(grad, amax, got, ph, pw, sh, sw)
    # same accumulation order here, but keep the ~1 ulp tolerance
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_conv_kernels_reject_float32():
    with pytest.raises(ValueError):
        _kernels.col2im(np.ones((4, 4), dtype=np.float32),
                        np.zeros((1, 4, 4, 1)), 2, 2, 1, 1)
    with pytest.raises(ValueError):
        _kernels.maxpool_forward(np.ones((2, 4, 4, 1), dtype=np.float32), 2, 2, 2, 2)


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------

def test_kernels_reject_float32():
    x = np.ones((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        _kernels.relu(x)
    with pytest.raises(ValueError):
        _kernels.softmax(x)


def test_kill_switch_disables_kernels():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "from numpy_keras.cython import _kernels\n"
        "assert _kernels is None, 'kill switch failed'"
    )
    env = dict(os.environ, NUMPY_KERAS_DISABLE_CYTHON="1")
    subprocess.run([sys.executable, "-c", code], env=env, cwd=repo_root, check=True)
