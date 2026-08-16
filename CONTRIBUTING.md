# Contributing

Thanks for your interest in NumPy-Keras. This library is written to read
like a textbook, and every contribution is expected to preserve that:
the pure NumPy code paths are the reference implementation and must not
change behavior (the CPU reference tests pin them at `rtol=1e-12`), and
every optional acceleration (Cython, CuPy) must be validated against
them by parity tests.

## Getting started

```bash
pip install -e . pytest pytest-cov
```

## Running the test suite

```bash
python -m pytest tests/ -q          # the full suite; optional-dependency
                                    # parts skip when unavailable
python -m pytest tests/ -q --cov=numpy_keras --cov-report=term-missing
                                    # with coverage (CI enforces >= 75%)
```

Optional paths you can exercise on this machine:

```bash
pip install autograd==1.7.0                 # tests/test_autograd.py
pip install "cython>=3.0.10" && python build_cython.py build_ext --inplace
                                            # tests/test_cython_kernels.py
pip install "cupy-cuda12x>=13.6.0"          # tests/test_cupy.py + test_float32.py
                                            # (GPU parity, needs an NVIDIA GPU)
```

## Tutorial snippets

The articles in `tutorials/` quote library source as `# excerpt:` code
blocks that must match the files byte for byte:

```bash
python tutorials/check_snippets.py         # validates all 14 articles
```

When you change library code that an article excerpts, re-sync the
excerpt (the checker tells you which block and which file). Each
article's complete code block must also stay byte-identical to its
`tutorials/code/NN_*.py` counterpart.

## Benchmarks

Benchmarks record environment and methodology; re-run them on your own
machine before drawing conclusions:

```bash
python benchmarks/bench_cython.py          # Cython vs pure NumPy
NUMPY_KERAS_BACKEND=numpy python benchmarks/bench_cupy.py --micro --mnist
NUMPY_KERAS_BACKEND=cupy  python benchmarks/bench_cupy.py --micro --mnist
```

## Conventions

- Default behavior is pure NumPy and must stay bit-identical; optional
  features degrade gracefully when their dependency is missing.
- The Cython kernels are CPU-only; the fused GPU kernels mirror the pure
  update rules statement by statement (see `numpy_keras/optimizers/_gpu_kernels.py`).
- Random numbers are generated on the host so same-seed runs are
  bit-identical across backends — parity tests depend on this.
- New backend-aware modules must bind the backend alias
  (`from ..backend import xp as np`); a meta-test in
  `tests/test_backend.py` asserts they are registered in the patch list.
- Commit messages follow `feat:` / `fix:` / `perf:` / `docs:` / `test:` /
  `ci:` / `chore:` prefixes.
