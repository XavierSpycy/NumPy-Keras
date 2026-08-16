"""03_backprop.py — 反向传播逐行拆解：链式法则、数值梯度校验与 autograd 对照

运行方式（在任意目录均可）：
    pip install -e .   # 仓库根目录执行一次
    python tutorials/code/03_backprop.py

说明：
- 第一部分追踪一次完整反向传播：criterion 给出的种子梯度如何在
  各层之间逐层变形（每层返回的 dX 与累积的参数梯度形状）
- 第二部分是有限差分梯度校验：逐一扰动 17 个参数，用中央差分
  (loss(p+eps) - loss(p-eps)) / 2eps 逼近真梯度，与解析梯度对比
- 第三部分把同一个模型用 numpy_keras.autograd（基于 autograd 库
  的自动微分实现）训练一遍，两条 loss 曲线应重合
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

# 1. 追踪一次反向传播：梯度在各层之间如何变形
model = keras.Sequential()
model.add(keras.layers.Input(2))
model.add(keras.layers.Dense(3, activation="tanh"))
model.add(keras.layers.Dense(2, activation="linear"))
model.compile(loss="mse", optimizer="sgd")

X = np.array([[0.5, -0.8], [1.2, 0.3], [-0.4, 0.9], [0.7, -0.2]])
y = np.array([[0.3, 0.8], [0.1, -0.5], [-0.6, 0.4], [0.9, 0.2]])

# 私有方法（名字改写为 _Sequential__forward 等）：教学与测试用，
# 等价于 fit 内部的 前向 -> criterion -> backward 三步
y_hat = model._Sequential__forward(X, is_training=True)
loss, grad = model._Sequential__criterion(y, y_hat)
print(f"loss = {loss:.6f}")
print(f"criterion 给出的种子梯度 grad 形状: {grad.shape}  (= y_hat 形状)\n")

for name in reversed(list(model.layers.keys())):
    layer = model.layers[name]
    if not hasattr(layer, "backward"):
        continue
    grad = layer.backward(grad)
    grads = {k: v.shape for k, v in layer.grads.items()} if hasattr(layer, "grads") else {}
    print(f"{name:<9} 返回 dX 形状 {grad.shape}, 参数梯度 {grads}")
print()

# 2. 有限差分梯度校验：逐一扰动每个参数
def gradcheck(model, X, y, eps=1e-6):
    """中央差分 (loss(p+eps)-loss(p-eps))/2eps 逼近解析梯度。"""

    def loss_of():
        y_hat = model._Sequential__forward(X, is_training=True)
        loss, _ = model._Sequential__criterion(y, y_hat)
        return loss

    y_hat = model._Sequential__forward(X, is_training=True)
    _, seed_grad = model._Sequential__criterion(y, y_hat)
    model._Sequential__backward(seed_grad)

    max_rel = 0.0
    n_checked = 0
    for name, params in model.parameters.items():
        for key, p in params.items():
            g = model.layers[name].grads[key]
            flat_p, flat_g = p.ravel(), g.ravel()
            for i in range(flat_p.size):
                old = flat_p[i]
                flat_p[i] = old + eps
                loss_plus = loss_of()
                flat_p[i] = old - eps
                loss_minus = loss_of()
                flat_p[i] = old
                num = (loss_plus - loss_minus) / (2 * eps)
                rel = abs(num - flat_g[i]) / max(1e-12, abs(num) + abs(flat_g[i]))
                max_rel = max(max_rel, rel)
                n_checked += 1
    return max_rel, n_checked


max_rel, n_checked = gradcheck(model, X, y)
print(f"梯度校验: 共检查 {n_checked} 个参数, 解析梯度与数值梯度的最大相对误差 = {max_rel:.2e}")
print("(误差在 1e-8 以下 => 手写的反向传播是正确的)\n")

# 3. autograd 对照：同一模型、同一种子，手写反向 vs 自动微分
def make_blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[1.0, 1.0], [-1.0, -1.0]])
    X = np.vstack([rng.normal(c, 0.9, (n, 2)) for c in centers])
    y = np.array([0] * n + [1] * n)
    idx = rng.permutation(2 * n)
    return X[idx], y[idx]


X, y = make_blobs()

np.random.seed(0)
hand = keras.Sequential()
hand.add(keras.layers.Input(2))
hand.add(keras.layers.Dense(8, activation="relu", kernel_initializer="he_normal"))
hand.add(keras.layers.Dense(2, activation="softmax"))
hand.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
h_hand = hand.fit(X, y, batch_size=32, epochs=200, verbose=0)

np.random.seed(0)
auto = keras.autograd.Sequential()
auto.add(keras.autograd.layers.Input(2))
auto.add(keras.autograd.layers.Dense(8, activation="relu", kernel_initializer="he_normal"))
auto.add(keras.autograd.layers.Dense(2, activation="softmax"))
auto.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
h_auto = auto.fit(X, y, batch_size=32, epochs=200, verbose=0)

print(f"手写反向传播 200 轮后 loss: {h_hand['loss'][-1]:.6f}")
print(f"autograd 自动微分 200 轮后 loss: {h_auto['loss'][-1]:.6f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(h_hand["loss"], label="hand-written backprop")
ax.plot(h_auto["loss"], "--", label="autograd (auto-diff)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Same model, same seed: hand-written vs automatic differentiation")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / "03_autograd_compare.png", dpi=150)
plt.close(fig)

print("图片已保存: tutorials/assets/03_autograd_compare.png")
