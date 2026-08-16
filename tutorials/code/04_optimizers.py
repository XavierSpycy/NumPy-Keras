"""04_optimizers.py — 优化器进化史：SGD → Momentum → NAG → Adagrad → Adadelta → Adam

运行方式（在任意目录均可）：
    pip install -e .   # 仓库根目录执行一次
    python tutorials/code/04_optimizers.py

说明：
- 第一部分在同一个二分类玩具数据、同一个模型上对比 6 种优化器
  （每个都用独立种子重新初始化，保证公平）
- 第二部分用一维二次函数 f(x)=0.5x^2 验证 NAG 的更新公式
  （闭式解对照：x1=0.81, x2=0.5751）
- 固定种子，数字可复现；纯 NumPy 模式与 Cython 模式轨迹一致
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

# 1. 同一模型、同一数据，6 种优化器对比
def make_blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[1.0, 1.0], [-1.0, -1.0]])
    X = np.vstack([rng.normal(c, 0.9, (n, 2)) for c in centers])
    y = np.array([0] * n + [1] * n)
    idx = rng.permutation(2 * n)
    return X[idx], y[idx]


def build_model(optimizer):
    m = keras.Sequential()
    m.add(keras.layers.Input(2))
    m.add(keras.layers.Dense(8, activation="relu", kernel_initializer="he_normal"))
    m.add(keras.layers.Dense(2, activation="softmax"))
    m.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)
    return m


X, y = make_blobs()
print(f"玩具数据: {X.shape}, 标签 {np.unique(y)}")

OPTIMIZERS = [
    ("sgd lr=0.1", keras.optimizers.SGD(learning_rate=0.1)),
    ("momentum 0.9", keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)),
    ("nesterov 0.9", keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)),
    ("adagrad lr=0.5", keras.optimizers.Adagrad(learning_rate=0.5)),
    ("adadelta", keras.optimizers.Adadelta(learning_rate=1.0)),
    ("adam", keras.optimizers.Adam(learning_rate=1e-3)),
]

histories = {}
for name, opt in OPTIMIZERS:
    np.random.seed(0)                    # 每个优化器从同一个初始点出发
    m = build_model(opt)
    h = m.fit(X, y, batch_size=32, epochs=150, verbose=0)
    histories[name] = h["loss"]
    print(f"{name:>16}: ep1={h['loss'][0]:.3f} ep5={h['loss'][4]:.3f} "
          f"ep10={h['loss'][9]:.3f} ep150={h['loss'][-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
for name, loss in histories.items():
    ax.plot(loss, label=name)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Same model, same init: six optimizers on the same toy problem")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / "04_optimizers_compare.png", dpi=150)
plt.close(fig)

# 2. NAG 公式验证：f(x) = 0.5 x^2 上应有界收敛到 0
class FakeLayer:
    def __init__(self, x0):
        self.params = {"w": np.array([x0])}
        self.grads = {"w": np.zeros(1)}


def nag_trace(mu=0.9, lr=0.1, steps=10, x0=1.0):
    layer = FakeLayer(x0)
    opt = keras.optimizers.SGD(learning_rate=lr, momentum=mu, nesterov=True)
    trace = []
    for _ in range(steps):
        layer.grads["w"] = layer.params["w"].copy()   # f=0.5x^2 的梯度就是 x
        opt.update([layer])
        trace.append(float(layer.params["w"][0]))
    return trace


print("\nNAG 在 f(x)=0.5x^2 上的轨迹（lr=0.1, momentum=0.9, x0=1.0）:")
print("  " + " ".join(f"{x:+.4f}" for x in nag_trace()[:6]) + " ...")
print(f"  10 步后: {nag_trace()[-1]:+.4f}（应保持在 |x| < 1 内收敛）")

print("\n图片已保存: tutorials/assets/04_optimizers_compare.png")
