"""06_mlp.py — MLP 深入：初始化器与深层网络

运行方式（在任意目录均可）：
    pip install -e .   # 仓库根目录执行一次
    python tutorials/code/06_mlp.py

说明：
- 第一部分：各初始化器在同一形状下的实际尺度（std 与最大绝对值）
- 第二部分：10 层 ReLU 网络在三种初始化下的逐层激活标准差
- 第三部分：README §3.2 的 12 层深网在 data/train_data.npy 的前 10,000 行
  上训练（数据集本为 50000×128 十分类；10k 子集让实验在普通笔记本上
  几分钟内可复现，用全部 50k 样本只需去掉切片）。
  EarlyStopping + ReduceLROnPlateau 监控 val_loss；不传 metrics 以免
  每轮全量 predict，最终准确率训练后一次测出。
- 固定种子 np.random.seed(42)，数字可复现
- 环境：Apple M2 Pro / macOS / Python 3.12 / NumPy 1.26.4
"""

import time
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
from numpy_keras.initializers import functional as F

ASSETS = ROOT / "tutorials" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

np.random.seed(0)

# 1. 各初始化器的实际尺度（同一个形状）
print("初始化器尺度 (shape=(64, 128), fan_in=64, fan_out=128):")
INITIALIZERS = [
    ("glorot_uniform", F.xavier_uniform),
    ("glorot_normal", F.xavier_normal),
    ("he_uniform", F.kaiming_uniform),
    ("he_normal", F.kaiming_normal),
    ("random_uniform", lambda s: F.uniform(s, -0.05, 0.05)),
    ("random_normal", lambda s: F.normal(s, 0.0, 0.05)),
]
for name, init_fn in INITIALIZERS:
    w = init_fn((64, 128))
    print(f"  {name:>16}: std={w.std():.4f}, |max|={np.abs(w).max():.4f}")

# 2. 10 层 ReLU 网络：三种初始化，逐层激活标准差
def build_relu_chain(initializer):
    m = keras.Sequential()
    m.add(keras.layers.Input(64))
    for _ in range(10):
        m.add(keras.layers.Dense(64, activation="relu", kernel_initializer=initializer))
    return m


def layer_stds(model, x):
    stds = [float(x.std())]
    a = x
    for layer in model.layers.values():
        if not hasattr(layer, "forward"):
            continue
        a = layer.forward(a, is_training=False)
        stds.append(float(a.std()))
    return stds


x = np.random.randn(1000, 64)
chains = [
    ("glorot_uniform", build_relu_chain("glorot_uniform")),
    ("he_normal", build_relu_chain("he_normal")),
    ("random_normal(std=0.05)", build_relu_chain("random_normal")),
]
print("\n10 层 ReLU 网络逐层激活标准差:")
for name, m in chains:
    stds = layer_stds(m, x)
    print(f"  {name:>22}: " + " ".join(f"{s:.2f}" for s in stds))

fig, ax = plt.subplots(figsize=(8, 4.5))
for name, m in chains:
    ax.plot(range(11), layer_stds(m, x), "o-", label=name)
ax.axhline(0.1, color="gray", ls=":", lw=1)
ax.text(0, 0.1, "  std 0.1（信号几乎消失）", va="bottom", fontsize=9, color="gray")
ax.set_xlabel("Layer index (0 = input)")
ax.set_ylabel("Std of activations")
ax.set_title("10-layer ReLU net: signal scale vs initializer")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / "06_initializer_compare.png", dpi=150)
plt.close(fig)

# 3. 12 层深网实战（README §3.2 的架构）
np.random.seed(42)                     # README 同款种子
X_train = np.load(ROOT / "data" / "train_data.npy")[:10000]   # 10k 子集，几分钟内可复现
y_train = np.load(ROOT / "data" / "train_label.npy").squeeze()[:10000]
X_test = np.load(ROOT / "data" / "test_data.npy")
y_test = np.load(ROOT / "data" / "test_label.npy").squeeze()
print(f"\n数据集: train {X_train.shape} {y_train.shape}, test {X_test.shape} {y_test.shape}")

model = keras.Sequential()
model.add(keras.layers.Input(shape=X_train.shape[1]))
model.add(keras.layers.Dense(120, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(112, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dropout(0.20))
model.add(keras.layers.Dense(96, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.Dense(64, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dropout(0.10))
model.add(keras.layers.Dense(32, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(24, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(16, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

# 刻意不传 metrics：metrics 会在每个 epoch 结束后对全量数据做 predict，
# 对 50k 样本的 12 层网络几乎等于把训练再做一遍。监控改用 val_loss
# （只要有验证数据就存在），最终准确率训练结束后一次测出。
early_stop = keras.callbacks.EarlyStopping('val_loss', mode='min', patience=5, restore_best_weights=True)
lr_scheduler = keras.callbacks.ReduceLROnPlateau('val_loss', mode='min', factor=0.5, patience=3, min_lr=1e-6)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

class EpochLogger:
    """每轮打印验证损失与学习率（调度器原地改属性，能看到降 lr 的时刻）。"""

    def on_epoch_end(self, model=None):
        n = len(model.history.loss)
        print(f"  epoch {n:>2}: val_loss={model.history.metrics['val_loss'][-1]:.4f}, "
              f"lr={model.optimizer.learning_rate:.6f}")


t0 = time.time()
history = model.fit(X_train, y_train, epochs=60, batch_size=128, verbose=0,
                    callbacks=[early_stop, lr_scheduler, EpochLogger()],
                    validation_split=0.1)
print(f"\n训练耗时: {time.time() - t0:.1f} s")
print(f"实际 epoch: {len(history['loss'])}（EarlyStopping 提前终止）")
print(f"最终学习率: {model.optimizer.learning_rate:.6f}")
# predict 会把 softmax 输出解码回整数标签（fit 时已记录类别映射）
train_acc = float(np.mean(model.predict(X_train, batch_size=128) == y_train))
test_acc = float(np.mean(model.predict(X_test, batch_size=128) == y_test))
print(f"训练集准确率（恢复最佳权重后，一次全量 predict）: {train_acc:.4f}")
print(f"测试集准确率（恢复最佳权重后，一次全量 predict）: {test_acc:.4f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
epochs = range(1, len(history["loss"]) + 1)
ax.plot(epochs, history["loss"], label="train")
ax.plot(epochs, history["metrics"]["val_loss"], label="val")
ax.axvline(len(history["loss"]), color="gray", ls=":", lw=1)
ax.text(len(history["loss"]), ax.get_ylim()[1], " early stop", va="top", fontsize=9, color="gray")
ax.set_title("Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / "06_deep_mlp_history.png", dpi=150)
plt.close(fig)

print("图片已保存: tutorials/assets/06_initializer_compare.png, "
      "tutorials/assets/06_deep_mlp_history.png")
