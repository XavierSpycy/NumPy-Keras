"""Benchmark the optional CuPy backend against the pure NumPy path.

Run twice and compare the printed numbers:

    NUMPY_KERAS_BACKEND=numpy python benchmarks/bench_cupy.py --mnist --cnn --rnn
    NUMPY_KERAS_BACKEND=cupy  python benchmarks/bench_cupy.py --mnist --cnn --rnn

The same script runs on whichever backend the environment variable
selects (or the default numpy backend when it is unset); the printed
header says which one is active.  GPU timings synchronize the CUDA
stream inside the timed region, so they measure the real device work.
The optional Cython kernels are disabled for the comparison so both
sides exercise the pure-Python compute path of the library.

Micro-benchmarks use realistic sizes: tiny fp64 ops are launch-overhead
bound on the GPU and do not represent real training workloads.  The
model-level fits include host-side data preparation and per-batch
host/device transfers -- exactly what a real fit() call pays.

--rnn benchmarks an LSTM reading sequences of 10 timesteps.  The GPU
path batches the input projection into one 3D matmul, but the
per-timestep recurrence still runs in a Python loop with small kernels;
at this teaching scale the launch overhead usually makes RNNs *slower*
on the GPU -- that is expected and reported honestly.  RNNs only pay off
on the device at much larger N/T/U.
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np

# select the backend BEFORE importing the library
os.environ.setdefault("NUMPY_KERAS_DISABLE_CYTHON", "1")
os.environ.setdefault("NUMPY_KERAS_BACKEND", "numpy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy_keras import backend as B


def _sync():
    if B.is_cupy_mode():
        import cupy
        cupy.cuda.get_current_stream().synchronize()


def _timeit(fn, steps=200):
    """Median per-call time of fn; device work is stream-synchronized."""
    for _ in range(10):            # warmup
        fn()
    _sync()
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _bench(name, fn, steps=200):
    print(f"{name:<16} {_timeit(fn, steps) * 1e6:>10.2f} us")


# ---------------------------------------------------------------------------
# Micro-benchmarks
# ---------------------------------------------------------------------------

def _bench_activations():
    from numpy_keras.activations import functional as F

    x = B.asarray(np.random.RandomState(0).randn(4096, 784))
    _bench("relu", lambda: F.relu(x))
    _bench("sigmoid", lambda: F.sigmoid(x))
    _bench("tanh", lambda: F.tanh(x))
    _bench("softmax", lambda: F.softmax(x))


def _bench_dense():
    from numpy_keras import layers

    rng = np.random.RandomState(1)
    layer = layers.Dense(512, activation="tanh")
    layer.set_input_shape((784,))
    layer.init_params(784)
    X = B.asarray(rng.randn(4096, 784))
    grad = B.asarray(rng.randn(4096, 512))
    layer.forward(X, is_training=True)   # warm the caches for backward
    _bench("dense_forward", lambda: layer.forward(X, is_training=True))
    _bench("dense_backward", lambda: layer.backward(grad))


def _bench_conv():
    from numpy_keras import layers

    rng = np.random.RandomState(2)
    layer = layers.Conv2D(8, kernel_size=3, activation="relu")
    layer.set_input_shape((28, 28, 1))
    X = B.asarray(rng.randn(64, 28, 28, 1))
    grad = B.asarray(rng.randn(64, 26, 26, 8))
    layer.forward(X, is_training=True)
    _bench("conv_forward", lambda: layer.forward(X, is_training=True))
    _bench("conv_backward", lambda: layer.backward(grad))

    pool = layers.MaxPool2D(pool_size=2)
    pool.set_input_shape((26, 26, 8))
    Xp = layer.forward(X, is_training=True)   # (64, 26, 26, 8)
    gradp = B.asarray(rng.randn(64, 13, 13, 8))
    pool.forward(Xp, is_training=True)
    _bench("maxpool_forward", lambda: pool.forward(Xp, is_training=True))
    _bench("maxpool_backward", lambda: pool.backward(gradp))


def _bench_rnn():
    from numpy_keras import layers

    rng = np.random.RandomState(3)
    X = B.asarray(rng.randn(64, 20, 32))
    for cls in (layers.SimpleRNN, layers.LSTM, layers.GRU):
        layer = cls(units=64)
        layer.set_input_shape((20, 32))
        grad = B.asarray(rng.randn(64, 64))
        layer.forward(X, is_training=True)
        _bench(f"{cls.__name__}_forward", lambda: layer.forward(X, is_training=True))
        _bench(f"{cls.__name__}_backward", lambda: layer.backward(grad))


def _bench_adam_update():
    from numpy_keras import optimizers

    rng = np.random.RandomState(4)
    layer = type("_L", (), {})()
    layer.params = {"W": B.asarray(rng.randn(784, 512))}
    layer.grads = {"W": B.asarray(rng.randn(784, 512) * 0.1)}
    opt = optimizers.Adam(learning_rate=0.01, weight_decay=0.05)
    opt.init_moment([layer])
    _bench("adam_update", lambda: opt.update([layer]))


def microbench():
    print(f"\nMicro-benchmarks (backend={B.get_backend()}, median per call):")
    _bench_activations()
    _bench_dense()
    _bench_conv()
    _bench_rnn()
    _bench_adam_update()


# ---------------------------------------------------------------------------
# Model-level benchmarks
# ---------------------------------------------------------------------------

def build_model(mnist=False, cnn=False, rnn=False):
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
    if rnn:
        model = Sequential()
        model.add(layers.Input((10, 64)))
        model.add(layers.LSTM(64, return_sequences=False))
        model.add(layers.Dense(10, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam")
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

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "mnist_train_small.csv")
    with open(path) as f:
        rows = list(csv.reader(f))
    X = np.array([[float(v) for v in r[1:]] for r in rows]) / 255.0
    y = np.array([int(r[0]) for r in rows])
    return X.reshape(-1, 28, 28, 1)[:n], y[:n]


def model_bench(mnist=False, cnn=False, rnn=False, reps=5):
    rng = np.random.RandomState(0)
    if cnn:
        n_samples = 2000
        X, y = _load_mnist_subset(n_samples)
        batch_size, epochs = 32, 2
        label = f"CNN 6@5x5/16@5x5 on MNIST {n_samples}x28x28"
    elif rnn:
        X = rng.randn(500, 10, 64)
        y = np.eye(10)[rng.randint(0, 10, 500)]
        batch_size, epochs, label = 32, 3, "LSTM 10x64 -> 64 -> 10, 500 samples"
    elif mnist:
        X = rng.randn(10000, 784)
        y = np.eye(10)[rng.randint(0, 10, 10000)]
        batch_size, epochs, label = 64, 3, "MNIST-like 10000x784, 256/256/10"
    else:
        X = rng.randn(3000, 64)
        y = np.sin(X[:, 0]) + 0.1 * rng.randn(3000)
        batch_size, epochs, label = 32, 5, "teaching-scale 3000x64, 128/64/1"

    model = build_model(mnist=mnist, cnn=cnn, rnn=rnn)
    model.fit(X[:32], y[:32], batch_size=32, epochs=1)   # warmup
    times = []
    for _ in range(reps):
        np.random.seed(0)
        model = build_model(mnist=mnist, cnn=cnn, rnn=rnn)
        t0 = time.perf_counter()
        model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True)
        _sync()
        times.append(time.perf_counter() - t0)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if reps > 1 else 0.0
    print(f"fit({label}, {epochs} epochs): {mean:.3f}s +/- {stdev:.3f}s "
          f"(mean of {reps} runs)")
    return mean, stdev


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist", action="store_true",
                        help="benchmark the MNIST-like MLP configuration")
    parser.add_argument("--cnn", action="store_true",
                        help="benchmark the LeNet-style CNN on data/mnist_train_small.csv")
    parser.add_argument("--rnn", action="store_true",
                        help="benchmark the LSTM configuration")
    parser.add_argument("--micro", action="store_true",
                        help="run the micro-benchmarks")
    parser.add_argument("--reps", type=int, default=5,
                        help="number of fit repetitions (default: 5)")
    args = parser.parse_args()

    print(f"backend: {B.get_backend()}")
    if args.micro:
        microbench()
    if not (args.mnist or args.cnn or args.rnn or args.micro):
        parser.error("pick at least one of --micro/--mnist/--cnn/--rnn")
    if args.mnist:
        model_bench(mnist=True, reps=args.reps)
    if args.cnn:
        model_bench(cnn=True, reps=args.reps)
    if args.rnn:
        model_bench(rnn=True, reps=args.reps)
