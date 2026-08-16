"""05_learning_rate.py — 学习率与九大调度器

运行方式（在任意目录均可）：
    pip install -e .   # 仓库根目录执行一次
    python tutorials/code/05_learning_rate.py

说明：
- 第一部分：同一个模型在 4 档学习率下的训练曲线（lr 扫描）
- 第二部分：9 个调度器的 lr-epoch 曲线（程序直接驱动 scheduler，
  不训练网络；ReduceLROnPlateau 喂的是合成的 val_loss 序列）
- 第三部分：ReduceLROnPlateau + EarlyStopping 在 MNIST 子集上的实战
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

# 1. lr 扫描：同一个模型，4 档学习率
def make_blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[1.0, 1.0], [-1.0, -1.0]])
    X = np.vstack([rng.normal(c, 0.9, (n, 2)) for c in centers])
    y = np.array([0] * n + [1] * n)
    idx = rng.permutation(2 * n)
    return X[idx], y[idx]


def build_model(lr):
    m = keras.Sequential()
    m.add(keras.layers.Input(2))
    m.add(keras.layers.Dense(8, activation="relu", kernel_initializer="he_normal"))
    m.add(keras.layers.Dense(2, activation="softmax"))
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=lr))
    return m


X, y = make_blobs()
print("lr 扫描（SGD, 150 epochs, 同一个初始点）:")
sweep = {}
for lr in (0.001, 0.01, 0.1, 5.0):
    np.random.seed(0)
    m = build_model(lr)
    h = m.fit(X, y, batch_size=32, epochs=150, verbose=0)
    sweep[lr] = h["loss"]
    print(f"  lr={lr:<5}: 最终 loss = {h['loss'][-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
for lr, loss in sweep.items():
    ax.semilogy(loss, label=f"lr={lr}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (log scale)")
ax.set_title("Same model, four learning rates (SGD)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / "05_lr_sweep.png", dpi=150)
plt.close(fig)

# 2. 九个调度器的 lr-epoch 曲线（直接驱动调度器，不训练）
def tiny_model(lr0=0.1):
    m = keras.Sequential()
    m.add(keras.layers.Input(2))
    m.add(keras.layers.Dense(1))
    m.compile(loss="mse", optimizer="sgd")
    m.optimizer.learning_rate = lr0
    return m


def lr_curve(scheduler, epochs=30, lr0=0.1):
    m = tiny_model(lr0)
    if hasattr(scheduler, "on_train_begin"):
        scheduler.on_train_begin(model=m)
    curve = []
    for _ in range(epochs):
        scheduler.on_epoch_end(model=m)
        curve.append(m.optimizer.learning_rate)
    return curve


def plateau_curve(scheduler, epochs=30, lr0=0.1):
    """ReduceLROnPlateau 需要 history 里的监控值：喂一条合成的曲线，
    前 10 轮停在 0.5（平台），第 11 轮起每轮降 0.02（稳步改善）。"""
    m = tiny_model(lr0)
    vals = [0.5] * 10 + [0.5 - 0.02 * i for i in range(1, epochs - 10 + 1)]
    curve = []
    for v in vals:
        if "val_loss" not in m.history.metrics:
            m.history.metrics["val_loss"] = []
        m.history.metrics["val_loss"].append(v)   # 与 fit 内的顺序一致
        scheduler.on_epoch_end(model=m)
        curve.append(m.optimizer.learning_rate)
    return curve


SCHEDULERS = [
    ("MultiplicativeLR 0.95^e", keras.callbacks.MultiplicativeLR(lambda e: 0.95)),
    ("StepLR(5, 0.5)", keras.callbacks.StepLR(step_size=5, gamma=0.5)),
    ("MultiStepLR([8,16], 0.3)", keras.callbacks.MultiStepLR([8, 16], gamma=0.3)),
    ("ConstantLR(0.3, 5)", keras.callbacks.ConstantLR(factor=0.3, total_iters=5)),
    ("LinearLR(0.3→1.0, 8)", keras.callbacks.LinearLR(start_factor=0.3, end_factor=1.0, total_iters=8)),
    ("ExponentialLR(0.9)", keras.callbacks.ExponentialLR(gamma=0.9)),
    ("PolynomialLR(20, 2)", keras.callbacks.PolynomialLR(total_iters=20, power=2)),
    ("CosineAnnealingLR(20, 0.01)", keras.callbacks.CosineAnnealingLR(T_max=20, eta_min=0.01)),
    ("ReduceLROnPlateau(0.5, p=3)", keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)),
]

fig, axes = plt.subplots(3, 3, figsize=(15, 9))
for ax, (name, sched) in zip(axes.ravel(), SCHEDULERS):
    curve = plateau_curve(sched) if name.startswith("Reduce") else lr_curve(sched)
    ax.plot(range(1, len(curve) + 1), curve, "o-", markersize=3)
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("lr")
    ax.grid(alpha=0.3)
fig.suptitle("Nine LR schedulers driven epoch by epoch (initial lr = 0.1)")
fig.tight_layout()
fig.savefig(ASSETS / "05_schedulers.png", dpi=150)
plt.close(fig)

# 3. ReduceLROnPlateau + EarlyStopping 实战（MNIST 子集）
def load_mnist(path, n_rows=None):
    with open(path) as f:
        rows = list(itertools.islice(csv.reader(f), n_rows))
    y = np.array([int(r[0]) for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows]) / 255.0
    return X, y


X_train, y_train = load_mnist(ROOT / "data" / "mnist_train_small.csv", n_rows=3000)
X_test, y_test = load_mnist(ROOT / "data" / "mnist_test.csv", n_rows=1000)
print(f"\nMNIST 子集: 训练 {X_train.shape}, 测试 {X_test.shape}")


class LRTracker:
    """每个 epoch 记录当前学习率（调度器是原地改属性的）。"""

    def __init__(self):
        self.lrs = []

    def on_epoch_end(self, model=None):
        self.lrs.append(model.optimizer.learning_rate)


np.random.seed(0)
m = keras.Sequential()
m.add(keras.layers.Input(784))
m.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
m.add(keras.layers.Dense(10, activation="softmax"))
m.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
          metrics=["accuracy"])
m.optimizer.learning_rate = 0.01

lr_tracker = LRTracker()
h = m.fit(
    X_train, y_train,
    batch_size=64, epochs=30, verbose=0,
    validation_data=(X_test, y_test),
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        # 监控准确率时务必 mode="max"：默认 mode="min" 会把"准确率下降"
        # 当成改善，恢复最佳权重时会恢复到最差的一轮
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                      patience=8, restore_best_weights=True),
        lr_tracker,
    ],
)
print(f"实际训练了 {len(h['loss'])} 个 epoch（EarlyStopping 提前终止）")
print(f"最佳 val_accuracy: {max(h['metrics']['val_accuracy']):.4f}")
print(f"测试集准确率（恢复最佳权重后）: {m.evaluate(X_test, y_test, batch_size=64):.4f}")
print(f"学习率轨迹: {' '.join(f'{lr:.4f}' for lr in lr_tracker.lrs)}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
epochs = range(1, len(h["loss"]) + 1)
axes[0].plot(epochs, h["loss"], label="train")
axes[0].plot(epochs, h["metrics"]["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].plot(epochs, h["metrics"]["train_accuracy"], label="train")
axes[1].plot(epochs, h["metrics"]["val_accuracy"], label="val")
axes[1].set_title("Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[2].plot(epochs, lr_tracker.lrs, "o-", markersize=3)
axes[2].set_title("Learning rate")
axes[2].set_xlabel("Epoch")
axes[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / "05_plateau.png", dpi=150)
plt.close(fig)

print("图片已保存: tutorials/assets/05_lr_sweep.png, "
      "tutorials/assets/05_schedulers.png, tutorials/assets/05_plateau.png")
