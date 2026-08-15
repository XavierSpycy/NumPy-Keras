"""Benchmark for the optional Cython acceleration layer.

Run twice and compare the model-level times:

    python benchmarks/bench_cython.py
    NUMPY_KERAS_DISABLE_CYTHON=1 python benchmarks/bench_cython.py

The first run exercises the compiled kernels, the second the pure NumPy
fallback.  Model-level timings include the pure-Python hot-path fixes
(metrics-skip, cached activation lookups), which apply to both modes --
the gap between the two runs is the Cython-only speedup.
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy_keras.cython import _kernels

FAST = _kernels is not None


# ---------------------------------------------------------------------------
# Micro-benchmarks: one kernel vs its pure NumPy equivalent
# ---------------------------------------------------------------------------

def _timeit_pair(fast_fn, pure_fn, steps=200):
    """Median per-call time of fast_fn and pure_fn over `steps` calls."""
    for _ in range(10):            # warmup
        fast_fn()
        pure_fn()
    fast_times, pure_times = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        fast_fn()
        fast_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        pure_fn()
        pure_times.append(time.perf_counter() - t0)
    return statistics.median(fast_times), statistics.median(pure_times)


def _bench_activation(name, pure_fn):
    rng = np.random.RandomState(0)
    x = rng.randn(64, 64)
    fast = getattr(_kernels, name)
    f, p = _timeit_pair(lambda: fast(x), lambda: pure_fn(x))
    print(f"{name:<14} {p * 1e6:>9.2f} us -> {f * 1e6:>9.2f} us   {p / f:>5.2f}x")


def _bench_dense_forward():
    rng = np.random.RandomState(1)
    X = rng.randn(64, 64)
    W = rng.randn(64, 64)
    b = rng.randn(64)
    f, p = _timeit_pair(
        lambda: _kernels.dense_forward(X, W, b, 3),
        lambda: np.tanh(X @ W + b),
    )
    print(f"{'dense_forward':<14} {p * 1e6:>9.2f} us -> {f * 1e6:>9.2f} us   {p / f:>5.2f}x")


def _bench_adam_update():
    rng = np.random.RandomState(2)

    def setup():
        p = rng.randn(64, 64)
        g = rng.randn(64, 64) * 0.1
        m = np.zeros_like(p)
        v = np.zeros_like(p)
        return p, g, m, v

    lr, b1, b2, eps, wd = 0.01, 0.9, 0.999, 1e-8, 0.05
    bc1, bc2 = 1 - b1, 1 - b2

    def fast(state):
        p, g, m, v = state
        _kernels.adam_update(p, g, m, v, lr, b1, b2, eps, wd, bc1, bc2)

    def pure(state):
        p, g, m, v = state
        g += wd * p
        m *= b1
        v *= b2
        m += (1 - b1) * g
        v += (1 - b2) * np.square(g)
        p -= lr * (m / bc1) / (np.sqrt(v / bc2) + eps)

    # state evolves in place, identically in both paths
    fast_state, pure_state = setup(), setup()
    for _ in range(10):
        fast(fast_state)
        pure(pure_state)
    fast_times, pure_times = [], []
    for _ in range(200):
        t0 = time.perf_counter()
        fast(fast_state)
        fast_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        pure(pure_state)
        pure_times.append(time.perf_counter() - t0)
    f, p = statistics.median(fast_times), statistics.median(pure_times)
    print(f"{'adam_update':<14} {p * 1e6:>9.2f} us -> {f * 1e6:>9.2f} us   {p / f:>5.2f}x")


def microbench():
    from numpy_keras.activations import functional as F

    print("\nMicro-benchmarks (64x64 float64, median of 200 calls):")
    print(f"{'kernel':<14} {'pure':>9} -> {'fast':>9}   speedup")
    _bench_activation("relu", F.relu)
    _bench_activation("sigmoid", F.sigmoid)
    _bench_activation("tanh", F.tanh)
    _bench_activation("softmax", F.softmax)
    _bench_dense_forward()
    _bench_adam_update()


# ---------------------------------------------------------------------------
# Model-level benchmarks
# ---------------------------------------------------------------------------

def build_model(mnist=False):
    from numpy_keras import Sequential
    from numpy_keras import layers

    if mnist:
        model = Sequential()
        model.add(layers.Input(784))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(256, activation="tanh"))
        model.add(layers.Dense(10, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        model.optimizer.learning_rate = 0.01
        return model
    model = Sequential()
    model.add(layers.Input(64))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="tanh"))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="adam")
    model.optimizer.learning_rate = 0.01
    return model


def model_bench(mnist=False, reps=5):
    np.random.seed(0)
    rng = np.random.RandomState(0)
    if mnist:
        X = rng.randn(10000, 784)
        y = np.eye(10)[rng.randint(0, 10, 10000)]
        batch_size, epochs, label = 64, 3, "MNIST-like 10000x784, 256/256/10"
    else:
        X = rng.randn(3000, 64)
        y = np.sin(X[:, 0]) + 0.1 * rng.randn(3000)
        batch_size, epochs, label = 32, 5, "teaching-scale 3000x64, 128/64/1"

    model = build_model(mnist=mnist)
    model.fit(X[:32], y[:32], batch_size=32, epochs=1)   # warmup
    times = []
    for _ in range(reps):
        np.random.seed(0)
        model = build_model(mnist=mnist)
        t0 = time.perf_counter()
        model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True)
        times.append(time.perf_counter() - t0)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if reps > 1 else 0.0
    print(f"fit({label}, {epochs} epochs): {mean:.3f}s +/- {stdev:.3f}s "
          f"(mean of {reps} runs)")
    return mean, stdev


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist", action="store_true",
                        help="benchmark the MNIST-like configuration")
    parser.add_argument("--reps", type=int, default=5,
                        help="number of fit repetitions (default: 5)")
    args = parser.parse_args()

    mode = "compiled kernels ACTIVE" if FAST else "pure NumPy fallback (NUMPY_KERAS_DISABLE_CYTHON set or not built)"
    print(f"Mode: {mode}")
    if FAST:
        microbench()
    else:
        print("\nMicro-benchmarks skipped (compiled kernels not available).")
    model_bench(mnist=args.mnist, reps=args.reps)
