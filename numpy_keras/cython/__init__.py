"""Optional Cython acceleration.

The compiled module ``_kernels`` is only used when (a) it has been built
(``python build_cython.py build_ext --inplace``) and (b) the environment
variable ``NUMPY_KERAS_DISABLE_CYTHON`` is not set.  The library behaves
identically without it; call sites fall back to the pure NumPy code.

Note: the import must run before ``_kernels`` exists as a package attribute,
because ``from . import _kernels`` skips the submodule import when the name
is already bound (Python falls back to the existing attribute).
"""
import os

if os.environ.get("NUMPY_KERAS_DISABLE_CYTHON"):
    _kernels = None
else:
    try:
        from . import _kernels
    except ImportError:
        _kernels = None
