"""Fused numeric kernels backing the optional Cython acceleration.

Every public function mirrors a NumPy expression in ``numpy_keras``; the
parity tests in ``tests/test_cython_kernels.py`` pin them against the pure
implementations.  All kernels are float64-only (call sites guard the
dispatch), and the optimizer kernels update the parameter/gradient/state
arrays *in place* -- including the write-back of the decayed gradient that
the pure optimizers perform via ``grad += weight_decay * p``.

Rounding notes:
- Statements mirror NumPy's operation order one rounding step at a time,
  split across statements so the compiler cannot contract them into FMA
  (NumPy elementwise ops never fuse).
- ``exp``/``tanh`` come from libm and may differ from NumPy's SIMD loops by
  ~1 ulp; parity tests use rtol=1e-12.  ``sqrt`` is correctly rounded in
  both, so the optimizer kernels match to ~1 ulp at worst (rtol=1e-15).
- ``cdivision=True`` means degenerate 0/0 evaluates to 0 instead of NumPy's
  nan; the optimizers' default epsilons keep every denominator > 0, so this
  cannot fire under normal use.
"""

# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport cython
cimport numpy as cnp
from libc.math cimport sqrt, exp, tanh as c_tanh

# Cython 3 ships no libc.float NAN constant; build one once at import time.
cdef double _NAN = float("nan")


# ---------------------------------------------------------------------------
# Per-element math helpers.
#
# These mirror numpy_keras/activations/functional.py, including the NaN
# semantics: relu is ``x * (x > 0)`` (NaN propagates), while relu_deriv is
# ``1. * (a > 0)`` (the comparison consumes NaN, so NaN -> 0.0).
# ---------------------------------------------------------------------------

cdef inline double _relu(double v) noexcept nogil:
    return v if v > 0.0 else 0.0 * v


cdef inline double _relu_deriv(double a) noexcept nogil:
    return 1.0 if a > 0.0 else 0.0


cdef inline double _sigmoid(double v) noexcept nogil:
    # NumPy stable split: v >= 0 -> 1/(1+exp(-v)); else e/(1+e) with e = exp(v)
    if v >= 0.0:
        return 1.0 / (1.0 + exp(-v))
    cdef double e = exp(v)
    return e / (1.0 + e)


cdef inline double _sigmoid_deriv(double a) noexcept nogil:
    return a * (1.0 - a)


cdef inline double _tanh(double v) noexcept nogil:
    return c_tanh(v)


cdef inline double _tanh_deriv(double a) noexcept nogil:
    return 1.0 - a * a


# ---------------------------------------------------------------------------
# Generic elementwise dispatch: kind selects the helper, loops outside.
# ---------------------------------------------------------------------------

cdef cnp.ndarray _elementwise(cnp.ndarray x, int kind):
    if x.dtype != np.float64:
        raise ValueError("kernels require float64 input")
    cdef cnp.ndarray out
    cdef Py_ssize_t i, j, n, m
    cdef double[:] x1, o1
    cdef double[:, :] x2, o2

    if x.ndim == 1:
        out = np.empty_like(x)
        x1 = x
        o1 = out
        n = x1.shape[0]
        if kind == 0:      # relu
            for i in range(n):
                o1[i] = _relu(x1[i])
        elif kind == 1:    # relu_deriv
            for i in range(n):
                o1[i] = _relu_deriv(x1[i])
        elif kind == 2:    # sigmoid
            for i in range(n):
                o1[i] = _sigmoid(x1[i])
        elif kind == 3:    # sigmoid_deriv
            for i in range(n):
                o1[i] = _sigmoid_deriv(x1[i])
        elif kind == 4:    # tanh
            for i in range(n):
                o1[i] = _tanh(x1[i])
        elif kind == 5:    # tanh_deriv
            for i in range(n):
                o1[i] = _tanh_deriv(x1[i])
        else:
            raise ValueError("unknown elementwise kind")
        return out

    if x.ndim == 2:
        out = np.empty_like(x)
        x2 = x
        o2 = out
        n = x2.shape[0]
        m = x2.shape[1]
        if kind == 0:
            for i in range(n):
                for j in range(m):
                    o2[i, j] = _relu(x2[i, j])
        elif kind == 1:
            for i in range(n):
                for j in range(m):
                    o2[i, j] = _relu_deriv(x2[i, j])
        elif kind == 2:
            for i in range(n):
                for j in range(m):
                    o2[i, j] = _sigmoid(x2[i, j])
        elif kind == 3:
            for i in range(n):
                for j in range(m):
                    o2[i, j] = _sigmoid_deriv(x2[i, j])
        elif kind == 4:
            for i in range(n):
                for j in range(m):
                    o2[i, j] = _tanh(x2[i, j])
        elif kind == 5:
            for i in range(n):
                for j in range(m):
                    o2[i, j] = _tanh_deriv(x2[i, j])
        else:
            raise ValueError("unknown elementwise kind")
        return out

    raise ValueError("kernels expect 1D or 2D input")


# ---------------------------------------------------------------------------
# Softmax (row-wise, max-subtracted) -- mirrors functional.softmax.
# ---------------------------------------------------------------------------

cdef void _softmax_rows(double[:, :] A) noexcept:
    cdef Py_ssize_t i, j, n = A.shape[0], m = A.shape[1]
    cdef double row_max, s, v
    cdef bint has_nan
    for i in range(n):
        # np.max propagates NaN: track it explicitly (NaN compares false
        # against everything, so it would otherwise be silently dropped).
        row_max = A[i, 0]
        has_nan = row_max != row_max
        for j in range(1, m):
            v = A[i, j]
            if v != v:
                has_nan = True
            elif v > row_max:
                row_max = v
        if has_nan:
            row_max = _NAN
        s = 0.0
        for j in range(m):
            A[i, j] = exp(A[i, j] - row_max)
            s += A[i, j]
        for j in range(m):
            A[i, j] = A[i, j] / s


def softmax(cnp.ndarray x):
    """Row-wise softmax of a 2D float64 array (matches functional.softmax)."""
    if x.dtype != np.float64 or x.ndim != 2:
        raise ValueError("softmax kernel requires 2D float64 input")
    # always copy: the algorithm is in-place, and ascontiguousarray would
    # alias (and destroy) a contiguous input
    cdef cnp.ndarray out = np.array(x, dtype=np.float64, copy=True)
    cdef double[:, :] ov = out
    _softmax_rows(ov)
    return out


# ---------------------------------------------------------------------------
# Public elementwise activations (1D or 2D float64).
# ---------------------------------------------------------------------------

def relu(cnp.ndarray x):
    return _elementwise(x, 0)


def relu_deriv(cnp.ndarray x):
    return _elementwise(x, 1)


def sigmoid(cnp.ndarray x):
    return _elementwise(x, 2)


def sigmoid_deriv(cnp.ndarray x):
    return _elementwise(x, 3)


def tanh(cnp.ndarray x):
    return _elementwise(x, 4)


def tanh_deriv(cnp.ndarray x):
    return _elementwise(x, 5)


# ---------------------------------------------------------------------------
# Fused Dense forward/backward.  Matmuls stay in BLAS (np.dot); the bias-add,
# activation, gradient reduction and deriv multiply are fused single passes.
# ---------------------------------------------------------------------------

def dense_forward(cnp.ndarray inputs, cnp.ndarray W, b, int act_code):
    """Fused ``activation(np.dot(inputs, W) + b)``.

    act_code: 0=linear, 1=relu, 2=sigmoid, 3=tanh, 4=softmax.  ``b`` may be
    None; the bias is folded into the activation pass (no intermediate
    ``dot + b`` array, unlike the pure NumPy version).
    """
    if inputs.dtype != np.float64 or W.dtype != np.float64:
        raise ValueError("dense_forward requires float64 inputs and W")
    if inputs.ndim != 2 or W.ndim != 2:
        raise ValueError("dense_forward requires 2D inputs and W")
    cdef cnp.ndarray lin = np.dot(inputs, W)
    cdef double[:, :] lv = lin
    cdef double[:] bv
    cdef bint has_bias = b is not None
    if has_bias:
        if b.dtype != np.float64:
            raise ValueError("dense_forward requires a float64 bias")
        bv = b
    cdef Py_ssize_t i, j, n = lv.shape[0], m = lv.shape[1]

    if act_code == 0:            # linear
        if has_bias:
            for i in range(n):
                for j in range(m):
                    lv[i, j] += bv[j]
    elif act_code == 1:          # relu
        if has_bias:
            for i in range(n):
                for j in range(m):
                    lv[i, j] = _relu(lv[i, j] + bv[j])
        else:
            for i in range(n):
                for j in range(m):
                    lv[i, j] = _relu(lv[i, j])
    elif act_code == 2:          # sigmoid
        if has_bias:
            for i in range(n):
                for j in range(m):
                    lv[i, j] = _sigmoid(lv[i, j] + bv[j])
        else:
            for i in range(n):
                for j in range(m):
                    lv[i, j] = _sigmoid(lv[i, j])
    elif act_code == 3:          # tanh
        if has_bias:
            for i in range(n):
                for j in range(m):
                    lv[i, j] = _tanh(lv[i, j] + bv[j])
        else:
            for i in range(n):
                for j in range(m):
                    lv[i, j] = _tanh(lv[i, j])
    elif act_code == 4:          # softmax
        if has_bias:
            for i in range(n):
                for j in range(m):
                    lv[i, j] += bv[j]
        _softmax_rows(lv)
    else:
        raise ValueError("unknown act_code")
    return lin


def dense_backward(cnp.ndarray inputs, cnp.ndarray grad, cnp.ndarray W, int deriv_code):
    """Mirrors Dense.backward: (inputs.T @ grad, sum(grad, 0), grad @ W.T),
    with the activation-derivative multiply fused into the last pass.

    deriv_code: -1/0 = no multiply (none / linear), 1=relu, 2=sigmoid,
    3=tanh.  ``db`` uses a plain sequential reduction (NumPy sums pairwise),
    so values agree to ~1 ulp -- parity tests use rtol=1e-12.
    """
    if inputs.dtype != np.float64 or grad.dtype != np.float64 or W.dtype != np.float64:
        raise ValueError("dense_backward requires float64 inputs, grad and W")
    if inputs.ndim != 2 or grad.ndim != 2 or W.ndim != 2:
        raise ValueError("dense_backward requires 2D inputs, grad and W")
    cdef cnp.ndarray dW = np.dot(inputs.T, grad)
    cdef cnp.ndarray db = np.zeros(W.shape[1], dtype=np.float64)
    cdef cnp.ndarray grad_next = np.dot(grad, W.T)
    cdef double[:, :] Xv = inputs
    cdef double[:, :] Gv = grad
    cdef double[:, :] gv = grad_next
    cdef double[:] bv = db
    cdef Py_ssize_t i, j, n = Gv.shape[0], k = Gv.shape[1], nx = Xv.shape[1]

    for i in range(n):
        for j in range(k):
            bv[j] += Gv[i, j]

    if deriv_code == 1:          # relu_deriv
        for i in range(n):
            for j in range(nx):
                gv[i, j] *= _relu_deriv(Xv[i, j])
    elif deriv_code == 2:        # sigmoid_deriv
        for i in range(n):
            for j in range(nx):
                gv[i, j] *= _sigmoid_deriv(Xv[i, j])
    elif deriv_code == 3:        # tanh_deriv
        for i in range(n):
            for j in range(nx):
                gv[i, j] *= _tanh_deriv(Xv[i, j])
    return dW, db, grad_next


def argmax_rows(cnp.ndarray x):
    """Per-row index of the first maximum (numpy.argmax tie-break)."""
    if x.dtype != np.float64 or x.ndim != 2:
        raise ValueError("argmax_rows requires 2D float64 input")
    cdef double[:, :] Xv = x
    cdef Py_ssize_t i, j, n = Xv.shape[0], m = Xv.shape[1]
    cdef cnp.ndarray out = np.empty(n, dtype=np.int64)
    cdef long long[:] ov = out
    cdef Py_ssize_t best
    for i in range(n):
        best = 0
        for j in range(1, m):
            if Xv[i, j] > Xv[i, best]:
                best = j
        ov[i] = best
    return out


# ---------------------------------------------------------------------------
# Fused optimizer update kernels (one pass per parameter array, in place).
#
# Each kernel mirrors the pure NumPy update statement by statement.  The
# decayed gradient is written back into ``g`` -- the pure optimizers mutate
# ``layer.grads[key]`` in place via ``grad += weight_decay * p``, and
# Adadelta re-reads it, so dropping the write-back would change results.
# ---------------------------------------------------------------------------

def adam_update(cnp.ndarray p, cnp.ndarray g, cnp.ndarray m, cnp.ndarray v,
                double lr, double beta1, double beta2, double eps, double wd,
                double bc1, double bc2):
    """One Adam step over flat views of p/g/m/v (all float64, C-contiguous).

    bc1/bc2 are the bias corrections 1 - beta1**t and 1 - beta2**t,
    computed once per update() in the caller.
    """
    if not (p.dtype == np.float64 and g.dtype == np.float64
            and m.dtype == np.float64 and v.dtype == np.float64):
        raise ValueError("adam_update requires float64 arrays")
    if not (p.flags.c_contiguous and g.flags.c_contiguous
            and m.flags.c_contiguous and v.flags.c_contiguous):
        raise ValueError("adam_update requires C-contiguous arrays")
    cdef double[:] pv = p.reshape(-1)
    cdef double[:] gv = g.reshape(-1)
    cdef double[:] mv = m.reshape(-1)
    cdef double[:] vv = v.reshape(-1)
    cdef Py_ssize_t i, n = pv.shape[0]
    cdef double gi, t, m_hat, v_hat, u
    for i in range(n):
        t = wd * pv[i]                  # round 1
        gi = gv[i] + t                  # round 2 (no FMA: separate statements)
        gv[i] = gi                      # observable grad write-back
        mv[i] *= beta1                  # round 3
        vv[i] *= beta2                  # round 4
        t = (1.0 - beta1) * gi          # round 5
        mv[i] += t                      # round 6
        t = gi * gi                     # round 7 (np.square)
        t *= (1.0 - beta2)              # round 8
        vv[i] += t                      # round 9
        m_hat = mv[i] / bc1             # round 10
        v_hat = vv[i] / bc2             # round 11
        u = lr * m_hat                  # round 12
        t = sqrt(v_hat)                 # round 13
        t += eps                        # round 14
        u = u / t                       # round 15
        pv[i] -= u                      # round 16


def sgd_update(cnp.ndarray p, cnp.ndarray g, cnp.ndarray v,
               double lr, double momentum, double wd, int nesterov):
    """One SGD step; ``nesterov`` mirrors the NAG branch of SGD.update.

    The nesterov path snapshots the old velocity per element before
    overwriting it, exactly like the pure code's velocity_prev alias.
    """
    if not (p.dtype == np.float64 and g.dtype == np.float64
            and v.dtype == np.float64):
        raise ValueError("sgd_update requires float64 arrays")
    if not (p.flags.c_contiguous and g.flags.c_contiguous
            and v.flags.c_contiguous):
        raise ValueError("sgd_update requires C-contiguous arrays")
    cdef double[:] pv = p.reshape(-1)
    cdef double[:] gv = g.reshape(-1)
    cdef double[:] vv = v.reshape(-1)
    cdef Py_ssize_t i, n = pv.shape[0]
    cdef double gi, t, t1, t2, t3, t4, t5, vp
    if nesterov:
        for i in range(n):
            t = wd * pv[i]
            gi = gv[i] + t
            gv[i] = gi
            vp = vv[i]                      # snapshot of the old velocity
            t1 = momentum * vp              # round 3
            t2 = lr * gi                    # round 4
            vv[i] = t1 - t2                 # round 5
            t3 = momentum * vp              # round 6 (m * v_prev)
            t4 = (1.0 + momentum) * vv[i]   # round 7
            t5 = t3 - t4                    # round 8
            pv[i] -= t5                     # round 9
    else:
        for i in range(n):
            t = wd * pv[i]
            gi = gv[i] + t
            gv[i] = gi
            t1 = momentum * vv[i]           # round 3
            t2 = lr * gi                    # round 4
            vv[i] = t1 + t2                 # round 5
            pv[i] -= vv[i]                  # round 6


def adagrad_update(cnp.ndarray p, cnp.ndarray g, cnp.ndarray gs,
                   double lr, double eps, double wd):
    """One Adagrad step over flat views of p/g/gs."""
    if not (p.dtype == np.float64 and g.dtype == np.float64
            and gs.dtype == np.float64):
        raise ValueError("adagrad_update requires float64 arrays")
    if not (p.flags.c_contiguous and g.flags.c_contiguous
            and gs.flags.c_contiguous):
        raise ValueError("adagrad_update requires C-contiguous arrays")
    cdef double[:] pv = p.reshape(-1)
    cdef double[:] gv = g.reshape(-1)
    cdef double[:] sv = gs.reshape(-1)
    cdef Py_ssize_t i, n = pv.shape[0]
    cdef double gi, t, u
    for i in range(n):
        t = wd * pv[i]
        gi = gv[i] + t
        gv[i] = gi
        t = gi * gi                     # np.square
        sv[i] += t                      # round 4
        u = lr * gi                     # round 5
        t = sqrt(sv[i])                 # round 6
        t += eps                        # round 7
        u = u / t                       # round 8
        pv[i] -= u                      # round 9


def adadelta_update(cnp.ndarray p, cnp.ndarray g, cnp.ndarray ags, cnp.ndarray ads,
                    double lr, double rho, double eps, double wd):
    """One Adadelta step over flat views of p/g/ags/ads."""
    if not (p.dtype == np.float64 and g.dtype == np.float64
            and ags.dtype == np.float64 and ads.dtype == np.float64):
        raise ValueError("adadelta_update requires float64 arrays")
    if not (p.flags.c_contiguous and g.flags.c_contiguous
            and ags.flags.c_contiguous and ads.flags.c_contiguous):
        raise ValueError("adadelta_update requires C-contiguous arrays")
    cdef double[:] pv = p.reshape(-1)
    cdef double[:] gv = g.reshape(-1)
    cdef double[:] sv = ags.reshape(-1)     # accum_grad_square
    cdef double[:] dv = ads.reshape(-1)     # accum_delta_square
    cdef Py_ssize_t i, n = pv.shape[0]
    cdef double gi, t, t1, t2, delta
    for i in range(n):
        t = wd * pv[i]
        gi = gv[i] + t
        gv[i] = gi
        sv[i] *= rho                        # round 3
        t = gi * gi                         # round 4 (np.square of mutated grad)
        t *= (1.0 - rho)                    # round 5
        sv[i] += t                          # round 6
        t1 = dv[i] + eps                    # round 7
        t1 = sqrt(t1)                       # round 8
        t1 = -t1                            # exact
        t2 = sv[i] + eps                    # round 9
        t2 = sqrt(t2)                       # round 10
        t = t1 / t2                         # round 11
        delta = t * gi                      # round 12
        t = lr * delta                      # round 13
        pv[i] += t                          # round 14
        dv[i] *= rho                        # round 15
        t = delta * delta                   # round 16 (np.square(delta))
        t *= (1.0 - rho)                    # round 17
        dv[i] += t                          # round 18


# ---------------------------------------------------------------------------
# Convolution kernels (col2im / max pooling).
#
# im2col itself is NOT compiled: building the column matrix is a strided
# copy that NumPy's sliding_window_view + reshape already perform at memcpy
# speed, and a plain C loop cannot beat it.  The column layout matches the
# pure NumPy im2col in layers/conv2d.py: each row of ``cols`` orders its
# kh * kw * C entries as (kh, kw, C), so ``cols @ W.reshape(kh * kw * C, F)``
# receives bit-identical inputs in both modes.  col2im accumulates with +=
# in row order; the pure implementation scatters with np.add.at in
# kernel-offset order, so the two agree to ~1 ulp (parity tests use
# rtol=1e-12).
# ---------------------------------------------------------------------------

def col2im(cnp.ndarray cols, cnp.ndarray grad_padded, int kh, int kw, int sh, int sw):
    """Scatter-add the column-space gradient back onto the padded image.

    cols: (N * OH * OW, kh * kw * C) float64; grad_padded: (N, Hp, Wp, C)
    float64, allocated as zeros by the caller -- contributions accumulate
    with +=, so it must start at zero.
    """
    if cols.dtype != np.float64 or cols.ndim != 2:
        raise ValueError("col2im requires 2D float64 cols")
    if grad_padded.dtype != np.float64 or grad_padded.ndim != 4:
        raise ValueError("col2im requires 4D float64 grad_padded")
    if not (cols.flags.c_contiguous and grad_padded.flags.c_contiguous):
        raise ValueError("col2im requires C-contiguous arrays")
    cdef double[:, ::1] Cv = cols
    cdef double[:, :, :, ::1] Gv = grad_padded
    cdef Py_ssize_t N = Gv.shape[0], Hp = Gv.shape[1], Wp = Gv.shape[2], C = Gv.shape[3]
    cdef Py_ssize_t OH = (Hp - kh) // sh + 1
    cdef Py_ssize_t OW = (Wp - kw) // sw + 1
    cdef Py_ssize_t K = kh * kw * C
    # flat pointers: both arrays are C-contiguous
    cdef const double* cp = &Cv[0, 0]
    cdef double* gp = &Gv[0, 0, 0, 0]
    cdef const double* col_row
    cdef Py_ssize_t n, i, j, ii, jj, c, m, k, base
    m = 0
    for n in range(N):
        for i in range(OH):
            for j in range(OW):
                col_row = cp + m * K
                k = 0
                for ii in range(kh):
                    base = ((n * Hp + i * sh + ii) * Wp + j * sw) * C
                    for jj in range(kw):
                        for c in range(C):
                            gp[base + jj * C + c] += col_row[k]
                            k += 1
                m += 1


def maxpool_forward(cnp.ndarray x, int ph, int pw, int sh, int sw):
    """Max pooling, caching the winning in-window index for the backward pass.

    x: (N, Hp, Wp, C) float64; returns (out, amax) with out (N, OH, OW, C)
    float64 and amax (N, OH, OW, C) int64 -- the flat row-major index of the
    maximum inside each window.  NaN propagates like np.max: a NaN window
    yields NaN, and its argmax is the first NaN position.
    """
    if x.dtype != np.float64 or x.ndim != 4:
        raise ValueError("maxpool_forward requires 4D float64 input")
    if not x.flags.c_contiguous:
        raise ValueError("maxpool_forward requires a C-contiguous input")
    cdef double[:, :, :, ::1] Xv = x
    cdef Py_ssize_t N = Xv.shape[0], Hp = Xv.shape[1], Wp = Xv.shape[2], C = Xv.shape[3]
    cdef Py_ssize_t OH = (Hp - ph) // sh + 1
    cdef Py_ssize_t OW = (Wp - pw) // sw + 1
    if OH <= 0 or OW <= 0:
        raise ValueError("maxpool_forward: input is smaller than the pool")
    cdef cnp.ndarray out = np.empty((N, OH, OW, C), dtype=np.float64)
    cdef cnp.ndarray amax = np.empty((N, OH, OW, C), dtype=np.int64)
    # flat pointers: x is C-contiguous; out/amax are freshly allocated
    cdef const double* xp = &Xv[0, 0, 0, 0]
    cdef double[:, :, :, ::1] Ov = out
    cdef long long[:, :, :, ::1] Av = amax
    cdef Py_ssize_t n, i, j, c, ii, jj, k, best, base
    cdef double best_val, v
    for n in range(N):
        for i in range(OH):
            for j in range(OW):
                for c in range(C):
                    best = 0
                    best_val = xp[((n * Hp + i * sh) * Wp + j * sw) * C + c]
                    k = 0
                    for ii in range(ph):
                        base = ((n * Hp + i * sh + ii) * Wp + j * sw) * C + c
                        for jj in range(pw):
                            v = xp[base + jj * C]
                            if v != v:              # NaN: wins the first time only
                                if best_val == best_val:
                                    best_val = v
                                    best = k
                            elif v > best_val:
                                best_val = v
                                best = k
                            k += 1
                    Ov[n, i, j, c] = best_val
                    Av[n, i, j, c] = best
    return out, amax


def maxpool_backward(cnp.ndarray grad, cnp.ndarray amax, cnp.ndarray grad_padded,
                     int ph, int pw, int sh, int sw):
    """Scatter each output gradient onto the winning input position.

    grad: (N, OH, OW, C) float64; amax: (N, OH, OW, C) int64 from
    maxpool_forward; grad_padded: (N, Hp, Wp, C) zeros allocated by the
    caller -- overlapping windows accumulate with +=.
    """
    if grad.dtype != np.float64 or grad.ndim != 4:
        raise ValueError("maxpool_backward requires 4D float64 grad")
    if amax.dtype != np.int64 or amax.ndim != 4:
        raise ValueError("maxpool_backward requires 4D int64 amax")
    if grad_padded.dtype != np.float64 or grad_padded.ndim != 4:
        raise ValueError("maxpool_backward requires 4D float64 grad_padded")
    if not (grad.flags.c_contiguous and grad_padded.flags.c_contiguous):
        raise ValueError("maxpool_backward requires C-contiguous arrays")
    cdef double[:, :, :, ::1] Gv = grad
    cdef long long[:, :, :, ::1] Av = amax
    cdef double[:, :, :, ::1] Pv = grad_padded
    cdef Py_ssize_t N = Gv.shape[0], OH = Gv.shape[1], OW = Gv.shape[2], C = Gv.shape[3]
    cdef Py_ssize_t n, i, j, c, k
    for n in range(N):
        for i in range(OH):
            for j in range(OW):
                for c in range(C):
                    k = Av[n, i, j, c]
                    Pv[n, i * sh + k // pw, j * sw + k % pw, c] += Gv[n, i, j, c]
