"""Fused GPU optimizer kernels: one pass per param array on the device.

The pure-Python update paths issue one tiny kernel per arithmetic op;
at teaching-scale shapes the launch overhead dominates (a profiled
MLP fit spends ~40% of its time there).  These kernels mirror the pure
NumPy statements element-by-element -- the same fusion idea as the
optional Cython kernels in ``numpy_keras.cython``, for the other device.

Each array is passed twice, once as a (read-only) input and once as the
writable output, aliasing the same memory: within one thread the output
write happens strictly after the input read of the same index, so the
in-place mutation semantics of the pure path are preserved exactly.
The kernels are compiled lazily on first use (importing this module
never pulls in CuPy by itself).
"""

_KERNEL_CACHE = {}


def _elementwise(name, in_params, out_params, body):
    if name not in _KERNEL_CACHE:
        import cupy
        _KERNEL_CACHE[name] = cupy.ElementwiseKernel(
            in_params, out_params, body, name)
    return _KERNEL_CACHE[name]


def adam_update(
        p, g, m, v,
        learning_rate, beta1, beta2, epsilon, weight_decay,
        bias_corr1, bias_corr2,
    ) -> None:
    """One fused Adam pass; mirrors the pure path statement by statement."""
    kernel = _elementwise(
        "nk_adam_update",
        "T g_in, T p_in, T m_in, T v_in, "
        "T lr, T b1, T b2, T eps, T wd, "
        "T bc1, T bc2",
        "T g_out, T p_out, T m_out, T v_out",
        """
        T gi = g_in + wd * p_in;
        m_out = b1 * m_in + ((T)1.0 - b1) * gi;
        v_out = b2 * v_in + ((T)1.0 - b2) * gi * gi;
        T mh = m_out / bc1;
        T vh = v_out / bc2;
        p_out = p_in - lr * mh / (sqrt(vh) + eps);
        g_out = gi;
        """)
    kernel(g, p, m, v, learning_rate, beta1, beta2, epsilon, weight_decay,
           bias_corr1, bias_corr2, g, p, m, v)


def sgd_update(
        p, g, v,
        learning_rate, momentum, weight_decay, nesterov,
    ) -> None:
    """One fused SGD (momentum / NAG) pass; mirrors the pure path."""
    kernel = _elementwise(
        "nk_sgd_update",
        "T g_in, T p_in, T v_in, "
        "T lr, T momentum, T wd, T nesterov",
        "T g_out, T p_out, T v_out",
        """
        T gi = g_in + wd * p_in;
        if (nesterov) {
            T vp = v_in;
            v_out = momentum * vp - lr * gi;
            p_out = p_in - momentum * vp + ((T)1.0 + momentum) * v_out;
        } else {
            v_out = momentum * v_in + lr * gi;
            p_out = p_in - v_out;
        }
        g_out = gi;
        """)
    kernel(g, p, v, learning_rate, momentum, weight_decay,
           float(nesterov), g, p, v)


def adagrad_update(
        p, g, grad_square,
        learning_rate, epsilon, weight_decay,
    ) -> None:
    """One fused Adagrad pass; mirrors the pure path."""
    kernel = _elementwise(
        "nk_adagrad_update",
        "T g_in, T p_in, T s_in, "
        "T lr, T eps, T wd",
        "T g_out, T p_out, T s_out",
        """
        T gi = g_in + wd * p_in;
        s_out = s_in + gi * gi;
        p_out = p_in - lr * gi / (sqrt(s_out) + eps);
        g_out = gi;
        """)
    kernel(g, p, grad_square, learning_rate, epsilon, weight_decay,
           g, p, grad_square)


def adadelta_update(
        p, g, accum_grad_square, accum_delta_square,
        learning_rate, rho, epsilon, weight_decay,
    ) -> None:
    """One fused Adadelta pass; mirrors the pure path."""
    kernel = _elementwise(
        "nk_adadelta_update",
        "T g_in, T p_in, T ag_in, T ad_in, "
        "T lr, T rho, T eps, T wd",
        "T g_out, T p_out, T ag_out, T ad_out",
        """
        T gi = g_in + wd * p_in;
        ag_out = rho * ag_in + ((T)1.0 - rho) * gi * gi;
        T d = -sqrt(ad_in + eps) / sqrt(ag_out + eps) * gi;
        p_out = p_in + lr * d;
        ad_out = rho * ad_in + ((T)1.0 - rho) * d * d;
        g_out = gi;
        """)
    kernel(g, p, accum_grad_square, accum_delta_square,
           learning_rate, rho, epsilon, weight_decay,
           g, p, accum_grad_square, accum_delta_square)
