"""Build the optional Cython acceleration kernels in place.

Usage (from the repository root):

    python build_cython.py build_ext --inplace

This drops ``_kernels.cpython-<version>-<platform>.so`` next to the ``.pyx``
file.  The library auto-detects the compiled module at import time and falls
back to the pure NumPy implementations when it is absent (or when
``NUMPY_KERAS_DISABLE_CYTHON`` is set).
"""
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name="numpy_keras",
    packages=["numpy_keras", "numpy_keras.cython"],
    ext_modules=cythonize(
        [
            Extension(
                "numpy_keras.cython._kernels",
                sources=["numpy_keras/cython/_kernels.pyx"],
                include_dirs=[np.get_include()],
            )
        ],
        compiler_directives={"language_level": "3"},
    ),
    zip_safe=False,
)
