"""Optional CuPy GPU backend.

This library is pure NumPy by default; the CuPy backend is opt-in and
follows the same graceful-degradation philosophy as the optional Cython
kernels (see :mod:`numpy_keras.cython`).  Enable it either way:

* ``export NUMPY_KERAS_BACKEND=cupy`` before importing numpy_keras, or
* ``numpy_keras.set_backend("cupy")`` at any time (e.g. from a notebook
  after the package has already been imported).

When ``cupy`` is requested but not installed, a warning is emitted and
the numpy backend stays active.  Models built under one backend are
moved to the active device automatically at the next
``fit``/``predict``/``evaluate`` call.

Every consumer module binds the active array module as ``np`` via
``from ..backend import xp as np``, so its code stays readable NumPy
code.  Module globals are looked up at call time, which is why
``set_backend`` can rebind the ``np`` attribute of the already-imported
consumer modules listed in ``_CONSUMER_MODULES``.

What stays on the host even under the CuPy backend: random number
generation (``numpy.random`` — this keeps seeded runs bit-identical
across backends), data preparation (``np.array`` conversion, shuffling,
one-hot encoding) and label/metric math.
"""
import os
import sys
import warnings

import numpy as _np  # the real numpy: isinstance checks and host-side ops

__all__ = [
    "xp",
    "set_backend",
    "get_backend",
    "is_cupy_mode",
    "on_gpu",
    "is_numpy_array",
    "is_cupy_array",
    "asarray",
    "asnumpy",
    "item",
    "zeros_like",
    "scatter_add",
    "sliding_window_view",
]

# Modules that bind the backend alias as ``np``; rebinding their ``np``
# attribute is what makes runtime ``set_backend`` switches take effect.
_CONSUMER_MODULES = [
    "numpy_keras.activations.functional",
    "numpy_keras.activations._mapper",
    "numpy_keras.initializers.functional",
    "numpy_keras.layers.activation",
    "numpy_keras.layers.batch_norm",
    "numpy_keras.layers.dense",
    "numpy_keras.layers.conv2d",
    "numpy_keras.layers.maxpool2d",
    "numpy_keras.layers.dropout",
    "numpy_keras.layers.simple_rnn",
    "numpy_keras.layers.lstm",
    "numpy_keras.layers.gru",
    "numpy_keras.losses.categorical_crossentropy",
    "numpy_keras.losses.mse",
    "numpy_keras.optimizers.sgd",
    "numpy_keras.optimizers.adam",
    "numpy_keras.optimizers.adagrad",
    "numpy_keras.optimizers.adadelta",
]

_cupy = None  # loaded lazily on first explicit request
xp = _np      # the active array module


def _load_cupy():
    """Import cupy once; return None (with a warning) if unavailable."""
    global _cupy
    if _cupy is None:
        try:
            import cupy
            _cupy = cupy
        except ImportError:
            warnings.warn(
                "NUMPY_KERAS_BACKEND=cupy was requested but CuPy is not "
                "installed; falling back to the numpy backend. "
                "Install it with: pip install cupy-cuda13x>=13.6.0",
                RuntimeWarning)
            _cupy = False
    return _cupy or None


def _patch_consumers():
    """Rebind the backend alias ``np`` in every consumer module."""
    for modname in _CONSUMER_MODULES:
        module = sys.modules.get(modname)
        if module is None:
            try:
                __import__(modname)
                module = sys.modules[modname]
            except ImportError:
                continue
        setattr(module, "np", xp)


def set_backend(name: str) -> None:
    """Switch the active array module to ``"numpy"`` or ``"cupy"``.

    Raises ``ValueError`` for unknown names.  Falls back to numpy with a
    warning when cupy is requested but not installed.
    """
    global xp
    name = name.lower()
    if name == "numpy":
        xp = _np
    elif name == "cupy":
        cp = _load_cupy()
        if cp is None:
            return
        xp = cp
    else:
        raise ValueError(f"Unknown backend {name!r}; expected 'numpy' or 'cupy'.")
    _patch_consumers()


def get_backend() -> str:
    """Return the active backend name: ``"numpy"`` or ``"cupy"``."""
    return "cupy" if is_cupy_mode() else "numpy"


def is_cupy_mode() -> bool:
    """Whether the active backend is cupy."""
    return _cupy is not None and _cupy is not False and xp is _cupy


def is_numpy_array(a) -> bool:
    """Whether ``a`` is a host numpy ndarray."""
    return isinstance(a, _np.ndarray)


def is_cupy_array(a) -> bool:
    """Whether ``a`` is a device cupy ndarray."""
    return _cupy is not None and _cupy is not False and isinstance(a, _cupy.ndarray)


def on_gpu(arr=None) -> bool:
    """Whether ``arr`` lives on the device (or the backend is cupy, if unset)."""
    return is_cupy_array(arr) if arr is not None else is_cupy_mode()


def asarray(a, dtype=None):
    """Move ``a`` to the active backend's device (identity under numpy)."""
    return xp.asarray(a, dtype=dtype) if dtype is not None else xp.asarray(a)


def asnumpy(a):
    """Move ``a`` to the host (identity under numpy)."""
    return _cupy.asnumpy(a) if is_cupy_array(a) else _np.asarray(a)


def item(a):
    """Convert a 0-d array (possibly on device) to a Python scalar."""
    if isinstance(a, (int, float)):
        return a
    return a.item()


def zeros_like(a, dtype=None):
    """Like ``xp.zeros_like`` — allocates on the same device as ``a``."""
    return xp.zeros_like(a, dtype=dtype) if dtype is not None else xp.zeros_like(a)


def scatter_add(a, slices, value) -> None:
    """In-place scatter-add with ``np.add.at`` semantics on the active backend.

    Both numpy and cupy (>= 13.6) provide ``add.at`` with the same
    broadcasting and unbuffered-accumulation semantics, so this is a
    straight pass-through; it exists as a helper so call sites do not
    need to know which module is active.
    """
    xp.add.at(a, slices, value)


def sliding_window_view(x, window_shape, axis=None):
    """Strided sliding-window view of ``x`` on the active backend."""
    if axis is None:
        return xp.lib.stride_tricks.sliding_window_view(x, window_shape)
    return xp.lib.stride_tricks.sliding_window_view(x, window_shape, axis=axis)


# Module-level selection: read once at import time.
_requested = os.environ.get("NUMPY_KERAS_BACKEND", "numpy")
if _requested.lower() not in ("numpy", "cupy"):
    raise ValueError(
        f"Unknown NUMPY_KERAS_BACKEND={_requested!r}; expected 'numpy' or 'cupy'.")
if _requested.lower() == "cupy":
    set_backend("cupy")
