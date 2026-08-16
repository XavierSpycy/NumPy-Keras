"""09_cnn.py — CNN 解剖：im2col、卷积与 LeNet 实战

运行方式（在任意目录均可）：
    pip install -e .   # 仓库根目录执行一次
    python tutorials/code/09_cnn.py

说明：
- 第一部分：4×4 输入、2×2 卷积核的 im2col 展开（卷积 = 矩阵乘法）
- 第二部分：LeNet 风格 CNN 在 MNIST 2,000 行子集上训练 2 轮，
  打印训练/测试准确率
- 第三部分：把第一个卷积层的 6 个输出通道画成特征图
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

# 1. im2col：卷积 = 把每个滑窗拉成一列，再做矩阵乘法
x = np.arange(16).reshape(4, 4).astype(float)
cols = np.lib.stride_tricks.sliding_window_view(x, (2, 2)).reshape(-1, 4)
print("4×4 输入:")
print(x)
print(f"\nim2col 后的列矩阵 ({cols.shape[0]} 个滑窗位置 × {cols.shape[1]} 个像素):")
print(cols)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(x, cmap="gray_r")
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, str(int(x[i, j])), ha="center", va="center", fontsize=9)
axes[0].set_title("4x4 input")
axes[1].imshow(cols, cmap="gray_r", aspect="auto")
axes[1].set_title("im2col: 9 windows x 4 pixels")
fig.tight_layout()
fig.savefig(ASSETS / "09_im2col.png", dpi=150)
plt.close(fig)

# 2. LeNet 实战：MNIST 2,000 行子集，2 轮
def load_mnist(path, n_rows=None):
    with open(path) as f:
        rows = list(itertools.islice(csv.reader(f), n_rows))
    y = np.array([int(r[0]) for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows]) / 255.0
    return X, y


X_train, y_train = load_mnist(ROOT / "data" / "mnist_train_small.csv", n_rows=2000)
X_test, y_test = load_mnist(ROOT / "data" / "mnist_test.csv", n_rows=1000)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
print(f"\n数据: 训练 {X_train.shape}, 测试 {X_test.shape}")

model = keras.Sequential()
model.add(keras.layers.Input((28, 28, 1)))
model.add(keras.layers.Conv2D(6, kernel_size=5, activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(16, kernel_size=5, activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(120, activation="tanh"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
model.summary()

history = model.fit(X_train, y_train, batch_size=32, epochs=2, verbose=0)
print(f"训练集准确率: {model.evaluate(X_train, y_train, batch_size=64):.4f}")
print(f"测试集准确率: {model.evaluate(X_test, y_test, batch_size=64):.4f}")

# 3. 第一个卷积层的 6 个特征图（取测试集第一张图）
# 注意：层的字典键来自 camel_to_snake("Conv2D") = "conv2_d"，
# 用类型查找更稳（见《引擎室》一篇对层命名的说明）
conv1 = next(l for l in model.layers.values() if isinstance(l, keras.layers.Conv2D))
fm = conv1.forward(X_test[0:1], is_training=False)   # (1, 24, 24, 6)
fig, axes = plt.subplots(1, 6, figsize=(15, 2.8))
for i in range(6):
    axes[i].imshow(fm[0, :, :, i], cmap="viridis")
    axes[i].set_title(f"filter {i}")
    axes[i].axis("off")
fig.suptitle("Feature maps of conv2d_1 on one test digit")
fig.tight_layout()
fig.savefig(ASSETS / "09_feature_maps.png", dpi=150)
plt.close(fig)

print("图片已保存: tutorials/assets/09_im2col.png, tutorials/assets/09_feature_maps.png")
