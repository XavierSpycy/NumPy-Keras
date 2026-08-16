"""07_dropout.py — Dropout：最简单的正则化

运行方式（在任意目录均可）：
    pip install -e .   # 仓库根目录执行一次
    python tutorials/code/07_dropout.py

说明：
- 第一部分验证"倒置 Dropout"的核心性质：mask 的期望值为 1，
  所以训练时随机置零并按 1/(1-rate) 放大，推理时什么都不用做
- 第二部分在 MNIST 子集上训练同一个大 MLP（两个隐层 512/256），
  一个不加 Dropout、一个加 0.5，对比训练/验证准确率的差距
- 固定种子 np.random.seed(0)，数字可复现
- 环境：Apple M2 Pro / macOS / Python 3.12 / NumPy 1.26.4
"""

import csv
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体回退链：macOS 用苹方、Windows 用雅黑，其他系统退化为英文渲染
plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS",
    "Microsoft YaHei", "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]

import numpy_keras as keras

ASSETS = ROOT / "tutorials" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

np.random.seed(0)

# 1. 倒置 Dropout 的核心性质：mask 期望值为 1
rate = 0.5
mask = (np.random.rand(100000) > rate) / (1.0 - rate)
print(f"rate={rate} 时 mask 的期望值: {mask.mean():.4f}")
print("  （期望不变：训练时随机置零并按 1/(1-rate) 放大，推理时无需任何缩放）\n")

# 2. 过拟合演示：同一个大 MLP，with/without Dropout
def load_mnist(path, n_rows=None):
    with open(path) as f:
        rows = list(itertools.islice(csv.reader(f), n_rows))
    y = np.array([int(r[0]) for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows]) / 255.0
    return X, y


X_train, y_train = load_mnist(ROOT / "data" / "mnist_train_small.csv", n_rows=1500)
X_test, y_test = load_mnist(ROOT / "data" / "mnist_test.csv", n_rows=1000)
print(f"数据: 训练 {X_train.shape}, 测试 {X_test.shape}")


def build(dropout_rate=None):
    m = keras.Sequential()
    m.add(keras.layers.Input(784))
    m.add(keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal"))
    if dropout_rate:
        m.add(keras.layers.Dropout(dropout_rate))
    m.add(keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal"))
    if dropout_rate:
        m.add(keras.layers.Dropout(dropout_rate))
    m.add(keras.layers.Dense(10, activation="softmax"))
    m.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
    return m


histories = {}
for name, rate in [("no dropout", None), ("dropout 0.5", 0.5)]:
    np.random.seed(0)                    # 两个模型同种子、同初始点
    m = build(rate)
    h = m.fit(X_train, y_train, batch_size=64, epochs=50, verbose=0,
              validation_data=(X_test, y_test))
    histories[name] = h
    ta = h["metrics"]["train_accuracy"][-1]
    va = h["metrics"]["val_accuracy"][-1]
    print(f"{name:>12}: train_acc={ta:.4f}, val_acc={va:.4f}, "
          f"gap={ta - va:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
epochs = range(1, 51)
for name in histories:
    axes[0].plot(epochs, histories[name]["metrics"]["train_accuracy"], label=name)
axes[0].set_title("Train accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)
for name in histories:
    axes[1].plot(epochs, histories[name]["metrics"]["val_accuracy"], label=name)
axes[1].set_title("Validation accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / "07_overfit_compare.png", dpi=150)
plt.close(fig)

print("图片已保存: tutorials/assets/07_overfit_compare.png")
