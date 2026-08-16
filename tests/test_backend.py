"""Tests for the optional backend abstraction (no GPU required).

The GPU-dependent cases probe cupy availability at import time and skip
when it is missing; they run alongside ``tests/test_cupy.py`` when the
device is present.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from numpy_keras import backend as B

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _restore_numpy_backend():
    """Every test runs under the numpy backend and restores it afterwards."""
    B.set_backend("numpy")
    yield
    B.set_backend("numpy")


def test_default_backend_is_numpy():
    assert B.get_backend() == "numpy"
    assert not B.is_cupy_mode()
    assert B.xp is np


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        B.set_backend("bogus")
    assert B.get_backend() == "numpy"


def test_set_backend_numpy_restores():
    B.set_backend("numpy")
    assert B.get_backend() == "numpy"
    assert not B.on_gpu()


def test_asnumpy_and_item_identity_on_host():
    arr = np.arange(6.0).reshape(2, 3)
    assert B.asnumpy(arr) is arr
    assert B.asarray(arr) is arr
    assert B.item(3.5) == 3.5
    assert B.item(np.float64(2.25)) == 2.25
    assert B.item(np.array(7.0)) == 7.0


def test_scatter_add_matches_np_add_at():
    a = np.zeros(5)
    idx = np.array([0, 1, 1, 3])
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    B.scatter_add(a, (idx,), vals)
    expected = np.zeros(5)
    np.add.at(expected, (idx,), vals)
    np.testing.assert_array_equal(a, expected)


def test_scatter_add_supports_slices_on_host():
    a = np.zeros((2, 3))
    B.scatter_add(a, (np.array([0, 1]), slice(None)), np.ones((2, 3)))
    np.testing.assert_array_equal(a, np.ones((2, 3)))


def test_sliding_window_view_matches_numpy():
    x = np.arange(24.0).reshape(4, 6)
    got = B.sliding_window_view(x, (2, 3), axis=(0, 1))
    expected = np.lib.stride_tricks.sliding_window_view(x, (2, 3), axis=(0, 1))
    np.testing.assert_array_equal(got, expected)


def _run_subprocess(code, extra_env):
    env = dict(os.environ)
    env.update(extra_env)
    env.setdefault("PYTHONPATH", REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=REPO_ROOT, env=env)


def test_env_var_selects_cupy_when_available():
    # Only meaningful when cupy is installed; the subprocess asserts
    # the backend follows the environment variable.
    code = (
        "import numpy_keras\n"
        "print(numpy_keras.get_backend())\n"
        "print(numpy_keras.backend.is_cupy_mode())"
    )
    result = _run_subprocess(code, {"NUMPY_KERAS_BACKEND": "cupy"})
    line1, line2 = result.stdout.strip().splitlines()
    try:
        import cupy  # noqa: F401
        cupy_installed = True
    except ImportError:
        cupy_installed = False
    if cupy_installed:
        assert result.returncode == 0
        assert line1 == "cupy" and line2 == "True"
    else:
        # graceful degradation: numpy backend + a warning, no crash
        assert result.returncode == 0
        assert line1 == "numpy" and line2 == "False"
        assert "falling back" in result.stderr


def test_env_var_default_is_numpy():
    result = _run_subprocess("import numpy_keras; print(numpy_keras.get_backend())", {})
    assert result.returncode == 0
    assert result.stdout.strip() == "numpy"


def test_env_var_invalid_name_raises():
    result = _run_subprocess(
        "import numpy_keras", {"NUMPY_KERAS_BACKEND": "bogus"})
    assert result.returncode != 0
    assert "NUMPY_KERAS_BACKEND" in result.stderr


def test_set_backend_patches_consumer_modules():
    from numpy_keras.activations import functional as F
    B.set_backend("numpy")
    assert F.np is np
    B.set_backend("numpy")  # idempotent
    assert F.np is np


def test_every_backend_consumer_is_in_patch_list():
    """Meta-test for the set_backend patch mechanism: every module that
    binds the backend alias as ``np`` must appear in backend's consumer
    list, or a runtime backend switch would silently leave it stale."""
    import importlib
    import inspect
    import pkgutil
    import re

    import numpy_keras

    for modinfo in pkgutil.walk_packages(numpy_keras.__path__,
                                         prefix="numpy_keras."):
        if modinfo.name.startswith("numpy_keras.autograd"):
            continue  # the autograd mirror stays pure numpy by design
        if modinfo.name == "numpy_keras.backend":
            continue  # its own docstring mentions the alias pattern
        try:
            mod = importlib.import_module(modinfo.name)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue  # compiled extensions (e.g. cython._kernels) have no source
        if re.search(r"from \.+backend import .*\bxp as np\b", src):
            assert modinfo.name in B._CONSUMER_MODULES, (
                f"{modinfo.name} binds the backend alias but is missing "
                f"from backend._CONSUMER_MODULES")
