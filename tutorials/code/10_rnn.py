"""10_rnn.py — RNN 三部曲：SimpleRNN / LSTM / GRU 与 BPTT

运行方式（在任意目录均可）：
    pip install -e .   # 仓库根目录执行一次
    python tutorials/code/10_rnn.py

说明：
- 第一部分：把 MNIST 每张图按行扫描成 28 个时间步的序列，
  同一配置下对比 SimpleRNN 与 LSTM（800 训练 / 200 测试样本）
- 第二部分：可视化 LSTM 扫描一个数字时隐状态随时间的演化
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

# 1. 行扫描 MNIST：SimpleRNN vs LSTM
def load_mnist(path, n_rows=None):
    with open(path) as f:
        rows = list(itertools.islice(csv.reader(f), n_rows))
    y = np.array([int(r[0]) for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows]) / 255.0
    return X, y


X_train, y_train = load_mnist(ROOT / "data" / "mnist_train_small.csv", n_rows=800)
X_test, y_test = load_mnist(ROOT / "data" / "mnist_test.csv", n_rows=200)
# (N, T, F)：每行 28 个像素是一个时间步，一张图是 28 个时间步的序列
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)
print(f"数据: 训练 {X_train.shape}, 测试 {X_test.shape}（T=28 时间步, F=28 特征）")


def build(rnn_layer):
    m = keras.Sequential()
    m.add(keras.layers.Input((28, 28)))
    m.add(rnn_layer)
    m.add(keras.layers.Dense(10, activation="softmax"))
    m.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
    m.optimizer.learning_rate = 0.01
    return m


histories = {}
for name, layer in [("SimpleRNN", keras.layers.SimpleRNN(32)),
                    ("LSTM", keras.layers.LSTM(32))]:
    np.random.seed(0)                    # 同种子、同初始点
    m = build(layer)
    h = m.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0,
              validation_data=(X_test, y_test))
    histories[name] = h
    print(f"{name:>11}: train_acc={h['metrics']['train_accuracy'][-1]:.4f}, "
          f"val_acc={h['metrics']['val_accuracy'][-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name in histories:
    axes[0].plot(histories[name]["loss"], label=name)
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)
for name in histories:
    axes[1].plot(histories[name]["metrics"]["val_accuracy"], label=name)
axes[1].set_title("Validation accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / "10_rnn_compare.png", dpi=150)
plt.close(fig)

# 2. LSTM 扫描一个数字时，隐状态随时间的演化
np.random.seed(0)
m = build(keras.layers.LSTM(32))
m.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
lstm = next(l for l in m.layers.values() if isinstance(l, keras.layers.LSTM))
lstm.forward(X_test[0:1], is_training=False)
h_seq = lstm._LSTM__h_seq          # 层内缓存的隐状态序列（名字改写后可见）
fig, ax = plt.subplots(figsize=(10, 3))
ax.imshow(h_seq[0].T, aspect="auto", cmap="viridis")
ax.set_xlabel("timestep (pixel row)")
ax.set_ylabel("hidden unit")
ax.set_title(f"LSTM hidden state while scanning digit {y_test[0]}")
fig.tight_layout()
fig.savefig(ASSETS / "10_hidden_state.png", dpi=150)
plt.close(fig)

print("图片已保存: tutorials/assets/10_rnn_compare.png, tutorials/assets/10_hidden_state.png")
