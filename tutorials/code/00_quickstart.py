"""00_quickstart.py — 五分钟上手 numpy_keras：训练你的第一个 MNIST 模型

运行方式（在任意目录均可）：
    pip install -e .   # 仓库根目录执行一次
    python tutorials/code/00_quickstart.py

说明：
- 固定全局随机种子 np.random.seed(0)：网络初始化、数据洗牌全部可复现
- 训练集取 data/mnist_train_small.csv 前 5000 行，
  测试集取 data/mnist_test.csv 前 1000 行（两个不同的文件，真实泛化测试）
- 本文所有数字在纯 NumPy 模式下测量（NUMPY_KERAS_DISABLE_CYTHON=1），
  启用 Cython 加速时训练轨迹一致、仅耗时不同
- 环境：Apple M2 Pro / macOS / Python 3.12 / NumPy 1.26.4
"""

import csv
import itertools
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # 无图形界面（如服务器）也能生成图片
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # tutorials/code/ 的上级的上级 = 仓库根

import numpy_keras as keras    # pip install -e . 之后任意目录都可导入

ASSETS = ROOT / "tutorials" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

np.random.seed(0)              # 全局种子：参数初始化、shuffle 全部可复现


def load_mnist(path, n_rows=None):
    """读取 label-first 的 MNIST CSV：第一列是标签，其余 784 列是像素（0-255）。"""
    with open(path) as f:
        rows = list(itertools.islice(csv.reader(f), n_rows))
    y = np.array([int(r[0]) for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows]) / 255.0
    return X, y


# 1. 数据：训练与测试来自两个不同的文件 —— 这是"测试集"二字的底线
X_train, y_train = load_mnist(ROOT / "data" / "mnist_train_small.csv", n_rows=5000)
X_test, y_test = load_mnist(ROOT / "data" / "mnist_test.csv", n_rows=1000)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 2. 模型：两层 MLP。relu 配合 he_normal 初始化（原因见 01 激活函数一文）
model = keras.Sequential()
model.add(keras.layers.Input(784))
model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)
model.summary()

# 3. 训练：5 个 epoch，验证集独立于训练集
t0 = time.time()
history = model.fit(
    X_train, y_train,
    batch_size=64, epochs=5, verbose=0,
    validation_data=(X_test, y_test),
)
print(f"\n训练耗时: {time.time() - t0:.1f} s\n")

print("每个 epoch 的指标:")
for e in range(len(history["loss"])):
    print(
        f"epoch {e + 1}: loss={history['loss'][e]:.4f}, "
        f"train_acc={history['metrics']['train_accuracy'][e]:.4f}, "
        f"val_acc={history['metrics']['val_accuracy'][e]:.4f}"
    )

# 4. 测试：compile 时传了 metrics，evaluate 返回的是准确率而不是 loss
test_acc = model.evaluate(X_test, y_test, batch_size=64)
print(f"\n测试集准确率: {test_acc:.4f}")

# 5. 配图
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
epochs = range(1, len(history["loss"]) + 1)
axes[0].plot(epochs, history["loss"], "o-", label="train")
axes[0].plot(epochs, history["metrics"]["val_loss"], "o-", label="val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].plot(epochs, history["metrics"]["train_accuracy"], "o-", label="train")
axes[1].plot(epochs, history["metrics"]["val_accuracy"], "o-", label="val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / "00_history.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(5, 5, figsize=(6, 6))
for ax, img, label in zip(axes.ravel(), X_test[:25], y_test[:25]):
    ax.imshow(img.reshape(28, 28), cmap="gray")
    ax.set_title(str(label), fontsize=10)
    ax.axis("off")
fig.suptitle("First 25 test samples")
fig.tight_layout()
fig.savefig(ASSETS / "00_mnist_samples.png", dpi=150)
plt.close(fig)

print("图片已保存: tutorials/assets/00_history.png, tutorials/assets/00_mnist_samples.png")
