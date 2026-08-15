"""Benchmark for the optional Cython acceleration layer.

Run twice and compare the model-level times:

    python benchmarks/bench_cython.py
    NUMPY_KERAS_DISABLE_CYTHON=1 python benchmarks/bench_cython.py

The first run exercises the compiled kernels, the second the pure NumPy
fallback.  Model-level timings include the pure-Python hot-path fixes
(metrics-skip, cached activation lookups), which apply to both modes --
the gap between the two runs is the Cython-only speedup.

--cnn benchmarks a LeNet-style CNN (6@5x5 / 16@5x5, im2col) on the first
2000 samples of data/mnist_train_small.csv.
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


def _bench_conv_kernels():
    """col2im / maxpool against their pure NumPy equivalents.

    Sizes mirror one batch of the --cnn model: a 32x26x26x6 feature map for
    the 3x3 convolution, a 32x24x24x6 map for the 2x2 pooling.  (im2col is
    not compiled -- NumPy's strided copy is already optimal.)"""
    rng = np.random.RandomState(3)

    x = rng.randn(32, 26, 26, 6)
    kh = kw = 3
    cols = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))
    cols = cols.transpose(0, 1, 2, 4, 5, 3)
    cols = cols.reshape(-1, kh * kw * x.shape[3])
    grad_cols = rng.randn(*cols.shape)
    N, Hp, Wp, C = x.shape
    OH = (Hp - kh) // 1 + 1
    OW = (Wp - kw) // 1 + 1
    M = OH * OW
    n_idx = np.repeat(np.arange(N), M)
    i_idx = np.tile(np.repeat(np.arange(OH), OW), N)
    j_idx = np.tile(np.tile(np.arange(OW), OH), N)
    flat = grad_cols.reshape(N * M, kh * kw, C)

    def pure_col2im():
        out = np.zeros((N, Hp, Wp, C))
        for i in range(kh):
            for j in range(kw):
                np.add.at(out, (n_idx, i_idx + i, j_idx + j, slice(None)),
                          flat[:, i * kw + j, :])
        return out

    def fast_col2im():
        out = np.zeros((N, Hp, Wp, C))
        _kernels.col2im(grad_cols, out, kh, kw, 1, 1)
        return out

    f, p = _timeit_pair(fast_col2im, pure_col2im)
    print(f"{'col2im':<14} {p * 1e6:>9.2f} us -> {f * 1e6:>9.2f} us   {p / f:>5.2f}x")

    x = rng.randn(32, 24, 24, 6)

    def pure_maxpool():
        win = np.lib.stride_tricks.sliding_window_view(x, (2, 2), axis=(1, 2))[:, ::2, ::2]
        win = win.transpose(0, 1, 2, 4, 5, 3)
        N, OH, OW, _, _, C = win.shape
        out = np.max(win, axis=(3, 4))
        amax = np.argmax(win.reshape(N, OH, OW, 4, C), axis=3)
        return out, amax

    f, p = _timeit_pair(lambda: _kernels.maxpool_forward(x, 2, 2, 2, 2), pure_maxpool)
    print(f"{'maxpool':<14} {p * 1e6:>9.2f} us -> {f * 1e6:>9.2f} us   {p / f:>5.2f}x")


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
    _bench_conv_kernels()


# ---------------------------------------------------------------------------
# Model-level benchmarks
# ---------------------------------------------------------------------------

def build_model(mnist=False, cnn=False):
    from numpy_keras import Sequential
    from numpy_keras import layers

    if cnn:
        # LeNet-style: 6@5x5 -> pool -> 16@5x5 -> pool -> 120 -> 10
        model = Sequential()
        model.add(layers.Input((28, 28, 1)))
        model.add(layers.Conv2D(6, kernel_size=5, activation="relu"))
        model.add(layers.MaxPool2D(pool_size=2))
        model.add(layers.Conv2D(16, kernel_size=5, activation="relu"))
        model.add(layers.MaxPool2D(pool_size=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation="tanh"))
        model.add(layers.Dense(10, activation="softmax"))
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
                      metrics=["accuracy"])
        return model
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


def _load_mnist_subset(n):
    """Load the first n samples of data/mnist_train_small.csv (label, 784
    pixels), reshaped to (n, 28, 28, 1) and normalized to [0, 1]."""
    import csv
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "mnist_train_small.csv")
    with open(path) as f:
        rows = list(csv.reader(f))
    X = np.array([[float(v) for v in r[1:]] for r in rows]) / 255.0
    y = np.array([int(r[0]) for r in rows])
    return X.reshape(-1, 28, 28, 1)[:n], y[:n]


def model_bench(mnist=False, cnn=False, reps=5):
    np.random.seed(0)
    rng = np.random.RandomState(0)
    if cnn:
        n_samples = 2000
        X, y = _load_mnist_subset(n_samples)
        batch_size, epochs = 32, 2
        label = f"CNN 6@5x5/16@5x5 on MNIST {n_samples}x28x28"
    elif mnist:
        X = rng.randn(10000, 784)
        y = np.eye(10)[rng.randint(0, 10, 10000)]
        batch_size, epochs, label = 64, 3, "MNIST-like 10000x784, 256/256/10"
    else:
        X = rng.randn(3000, 64)
        y = np.sin(X[:, 0]) + 0.1 * rng.randn(3000)
        batch_size, epochs, label = 32, 5, "teaching-scale 3000x64, 128/64/1"

    model = build_model(mnist=mnist, cnn=cnn)
    model.fit(X[:32], y[:32], batch_size=32, epochs=1)   # warmup
    times = []
    for _ in range(reps):
        np.random.seed(0)
        model = build_model(mnist=mnist, cnn=cnn)
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
    parser.add_argument("--cnn", action="store_true",
                        help="benchmark the LeNet-style CNN on data/mnist_train_small.csv")
    parser.add_argument("--reps", type=int, default=5,
                        help="number of fit repetitions (default: 5)")
    args = parser.parse_args()

    mode = "compiled kernels ACTIVE" if FAST else "pure NumPy fallback (NUMPY_KERAS_DISABLE_CYTHON set or not built)"
    print(f"Mode: {mode}")
    if FAST:
        microbench()
    else:
        print("\nMicro-benchmarks skipped (compiled kernels not available).")
    model_bench(mnist=args.mnist, cnn=args.cnn, reps=args.reps)
