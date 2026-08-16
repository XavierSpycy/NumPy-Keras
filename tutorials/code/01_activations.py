"""01_activations.py — 激活函数全解：17 个函数、导数与梯度消失

运行方式（在任意目录均可）：
    pip install -e .   # 仓库根目录执行一次
    python tutorials/code/01_activations.py

说明：
- 画出库里全部 17 个激活函数及其导数；导数按库的约定定义在
  **后激活值** a = f(x) 上（f'(a)），图中虚线即 f'(f(x))
- 用一个 10 层 MLP 的纯前向实验展示梯度消失：sigmoid+glorot 与
  relu+he_normal 两条链，逐层激活标准差对比
- 固定种子 np.random.seed(0)，数字可复现；纯 NumPy 模式与
  Cython 模式结果一致（本文只有前向，不涉及内核）
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
from numpy_keras.activations import functional as F

ASSETS = ROOT / "tutorials" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

np.random.seed(0)

# 1. 17 个激活函数及其导数（softmax 无导数，见 02 损失函数）
X = np.linspace(-4, 4, 401)
ACTIVATIONS = [
    "linear", "sigmoid", "tanh", "relu", "relu6", "leaky_relu",
    "elu", "celu", "selu", "softplus", "softsign", "log_sigmoid",
    "hardsigmoid", "hardtanh", "hardshrink", "softshrink", "softmax",
]

fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for ax, name in zip(axes.ravel(), ACTIVATIONS):
    f = getattr(F, name)
    y = f(X)
    ax.plot(X, y, label="f(x)")
    if name != "softmax":           # softmax 没有导数（设计如此，见正文）
        deriv = getattr(F, name + "_deriv")
        d = deriv(y)
        # linear_deriv 返回标量 1（导数恒为 1，写成标量也成立）
        d = np.full_like(y, d) if np.isscalar(d) else d
        ax.plot(X, d, "--", label="f'(a)")
    ax.set_title(name)
    ax.grid(alpha=0.3)
    ax.set_ylim(-4, 4)
axes.ravel()[-1].axis("off")
axes.ravel()[-1].text(0.1, 0.5,
    "实线：f(x)\n虚线：f'(a)，定义在\n后激活值 a=f(x) 上\n\nsoftmax 在 1D 上退化为\n均匀分布（~1/401），\n它只有逐样本归一化\n才有意义",
    fontsize=11, va="center")
fig.tight_layout()
fig.savefig(ASSETS / "01_activations.png", dpi=150)
plt.close(fig)

# 2. 梯度消失的数学根源：导数的上界
grid = np.linspace(0, 1, 10001)
sigmoid_max_deriv = float(np.max(F.sigmoid_deriv(grid)))
print(f"sigmoid 导数的最大值: {sigmoid_max_deriv:.4f}")
print(f"tanh 导数在 a=0 处: {F.tanh_deriv(0.0):.4f}")
print(f"relu 导数在 a>0 处: {F.relu_deriv(1.0):.4f}")
print(f"10 层 sigmoid 链的导数乘积上界: {sigmoid_max_deriv ** 10:.2e}")

# 3. 前向实验：10 层 MLP，逐层激活标准差
def build_chain(activation, initializer):
    model = keras.Sequential()
    model.add(keras.layers.Input(64))
    for _ in range(10):
        model.add(keras.layers.Dense(
            64, activation=activation, kernel_initializer=initializer))
    return model


def layer_stds(model, x):
    """逐层前向，记录每层输出的标准差（Input 层无 forward，直接跳过）。"""
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
    ("sigmoid + glorot_uniform", build_chain("sigmoid", "glorot_uniform")),
    ("relu + he_normal", build_chain("relu", "he_normal")),
]
for name, model in chains:
    stds = layer_stds(model, x)
    print(f"{name:>22} 各层激活标准差: " + " ".join(f"{s:.2f}" for s in stds))

fig, ax = plt.subplots(figsize=(8, 4.5))
for name, model in chains:
    ax.plot(range(11), layer_stds(model, x), "o-", label=name)
ax.axhline(0.1, color="gray", ls=":", lw=1)
ax.text(0, 0.1, " 标准差 0.1（信息几乎消失）", va="bottom", fontsize=9, color="gray")
ax.set_xlabel("Layer index (0 = input)")
ax.set_ylabel("Std of activations")
ax.set_title("10-layer MLP: how activations scale through the network")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / "01_vanishing.png", dpi=150)
plt.close(fig)

print("图片已保存: tutorials/assets/01_activations.png, tutorials/assets/01_vanishing.png")
