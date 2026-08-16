"""08_batch_norm.py — BatchNormalization：训练/推理双模式与滑动统计量

运行方式（在任意目录均可）：
    pip install -e .   # 仓库根目录执行一次
    python tutorials/code/08_batch_norm.py

说明：
- 同一个 8 层 MLP 用同一个高学习率（SGD lr=0.7）训练两遍：
  一层不加 BN、一层在每层激活后加 BN，对比训练曲线
- 训练结束后打印 BN 层的滑动均值/方差：训练模式用 batch 统计量，
  推理模式（evaluate/predict）用滑动统计量
- 固定种子 np.random.seed(0)，数字可复现
- 环境：Apple M2 Pro / macOS / Python 3.12 / NumPy 1.26.4
"""

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

def make_blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[1.0, 1.0], [-1.0, -1.0]])
    X = np.vstack([rng.normal(c, 0.9, (n, 2)) for c in centers])
    y = np.array([0] * n + [1] * n)
    idx = rng.permutation(2 * n)
    return X[idx], y[idx]


X, y = make_blobs()
print(f"玩具数据: {X.shape}, 标签 {np.unique(y)}")


def build(with_bn):
    m = keras.Sequential()
    m.add(keras.layers.Input(2))
    for _ in range(8):
        m.add(keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"))
        if with_bn:
            m.add(keras.layers.BatchNormalization())
    m.add(keras.layers.Dense(2, activation="softmax"))
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=0.7),
              metrics=["accuracy"])
    return m


histories = {}
for name, with_bn in [("without BN", False), ("with BN", True)]:
    np.random.seed(0)                    # 同种子、同初始点
    m = build(with_bn)
    # 无 BN 时高学习率让梯度爆炸、预测崩塌（损失停在高位、准确率钉死
    # 随机水平）——压掉 relu 的 invalid 警告噪音
    with np.errstate(invalid="ignore"):
        h = m.fit(X, y, batch_size=64, epochs=200, verbose=0)
    histories[name] = h
    print(f"{name:>11}: 最终 loss={h['loss'][-1]:.4f}, "
          f"train_acc={h['metrics']['train_accuracy'][-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name in histories:
    axes[0].plot(histories[name]["loss"], label=name)
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)
for name in histories:
    axes[1].plot(histories[name]["metrics"]["train_accuracy"], label=name)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / "08_bn_compare.png", dpi=150)
plt.close(fig)

# 滑动统计量：训练模式用 batch 统计量，推理模式用它们
np.random.seed(0)
m = build(with_bn=True)
m.fit(X, y, batch_size=64, epochs=50, verbose=0)
bn_layer = next(l for l in m.layers.values() if hasattr(l, "moving_mean"))
print(f"\n训练 50 轮后，第一个 BN 层的滑动均值（前 5 个通道）: {np.round(bn_layer.moving_mean[:5], 4)}")
print(f"滑动方差（前 5 个通道）: {np.round(bn_layer.moving_variance[:5], 4)}")
print("（evaluate/predict 走推理模式，用的就是这两个统计量而非 batch 统计量）")

print("图片已保存: tutorials/assets/08_bn_compare.png")
