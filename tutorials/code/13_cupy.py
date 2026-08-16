"""13_cupy.py — CuPy GPU 加速：可选的后端开关

运行方式（在仓库根目录，先按 CUDA 版本安装 CuPy）：
    pip install "cupy-cuda12x>=13.6.0"      # 驱动 >= 12.x 的机器通用
    python tutorials/code/13_cupy.py                     # NumPy 后端（若编译了 Cython 内核则自动叠加）
    NUMPY_KERAS_BACKEND=cupy python tutorials/code/13_cupy.py  # GPU 后端

说明：
- 打印当前后端，并演示三种用法：同种子下两个后端的初始权重逐位
  相同（奇偶校验）、模型自动上卡训练、set_backend 随时切换后端
- 训练计时跟随启动时的后端——两种模式下分别跑一遍即可得到对比；
  计时前先跑一次小模型预热，把 CUDA context 初始化与内核即时编译
  排除在计时之外；配置带 accuracy 指标，是真实使用场景
- 固定种子 np.random.seed(0)，数字可复现
- 实测环境：2× NVIDIA A800 80GB / Python 3.12 / NumPy 1.26.4
"""

import csv
import itertools
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

import numpy_keras as keras
from numpy_keras import backend as B

np.random.seed(0)

ACTIVE = B.get_backend()
print(f"当前后端: {ACTIVE}")

# 演示 1：随机数始终在主机生成——同种子下，两个后端的初始权重逐位相同
B.set_backend("numpy")
np.random.seed(7)
model_cpu = keras.Sequential([
    keras.layers.Input(784),
    keras.layers.Dense(256, activation="relu"),
])
W_cpu = model_cpu.parameters["dense_1"]["W"]

B.set_backend("cupy")
np.random.seed(7)  # 同样的种子，同样的权重，只是出生在 GPU 上
model_gpu = keras.Sequential([
    keras.layers.Input(784),
    keras.layers.Dense(256, activation="relu"),
])
W_gpu = model_gpu.parameters["dense_1"]["W"]
print(f"同种子初始权重逐位相同: {np.array_equal(W_cpu, B.asnumpy(W_gpu))}")

# 演示 2：回到启动时的后端训练——数据留在主机，模型自动上卡，
# fit / predict 的接口完全不变
B.set_backend(ACTIVE)

def load_mnist(path, n_rows=None):
    with open(path) as f:
        rows = list(itertools.islice(csv.reader(f), n_rows))
    y = np.array([int(r[0]) for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows]) / 255.0
    return X, y


X, y = load_mnist(ROOT / "data" / "mnist_train_small.csv", n_rows=10000)
print(f"数据: {X.shape}")

# 预热：首次上卡要初始化 CUDA context 并即时编译内核，不计入计时——
# 否则这部分一次性开销会不公平地记在 GPU 头上
warmup = keras.Sequential([
    keras.layers.Input(784),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
warmup.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
warmup.fit(X[:64], y[:64], batch_size=64, epochs=1, verbose=0)

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

# 演示 3：训练后随时切到对方后端跑一次推理，再切回——fit/predict
# 入口会把参数、梯度、BatchNorm 统计量与优化器状态自动双向同步
other = "cupy" if ACTIVE == "numpy" else "numpy"
B.set_backend(other)
_ = model.predict(X[:8])
B.set_backend(ACTIVE)
_ = model.predict(X[:8])   # 再次进入 predict，状态同步回启动后端
print(f"切到 {other} 跑过 predict 并切回 {ACTIVE}，"
      f"参数在 GPU: {B.on_gpu(model.parameters['dense_1']['W'])}")
