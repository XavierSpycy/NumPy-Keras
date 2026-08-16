"""12_cython.py — Cython 加速：从 NumPy 到编译内核

运行方式（在任意目录均可）：
    pip install -e .        # 仓库根目录执行一次
    python build_cython.py build_ext --inplace   # 编译内核（可选）
    python tutorials/code/12_cython.py           # 默认（编译内核）模式
    NUMPY_KERAS_DISABLE_CYTHON=1 python tutorials/code/12_cython.py  # 纯 NumPy 模式

说明：
- 打印当前内核状态（已编译 / 纯 NumPy），并计时同一个 MLP
  （10,000×784，[256,256,10]，3 轮）的训练耗时——两种模式下
  分别跑一遍即可得到速度对比
- 固定种子 np.random.seed(0)，数字可复现
- 环境：Apple M2 Pro / macOS / Python 3.12 / NumPy 1.26.4
"""

import csv
import itertools
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

import numpy_keras as keras
from numpy_keras.cython import _kernels as _ck

np.random.seed(0)

print(f"内核状态: {'已编译（默认模式）' if _ck is not None else '纯 NumPy（NUMPY_KERAS_DISABLE_CYTHON=1）'}")

def load_mnist(path, n_rows=None):
    with open(path) as f:
        rows = list(itertools.islice(csv.reader(f), n_rows))
    y = np.array([int(r[0]) for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows]) / 255.0
    return X, y


X, y = load_mnist(ROOT / "data" / "mnist_train_small.csv", n_rows=10000)
print(f"数据: {X.shape}")

model = keras.Sequential()
model.add(keras.layers.Input(784))
model.add(keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])

t0 = time.time()
history = model.fit(X, y, batch_size=64, epochs=3, verbose=0)
elapsed = time.time() - t0
print(f"3 轮训练耗时: {elapsed:.2f} s（平均每轮 {elapsed / 3:.2f} s）")
print(f"训练准确率: {history['metrics']['train_accuracy'][-1]:.4f}")
