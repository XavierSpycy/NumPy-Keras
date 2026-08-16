"""02_losses.py — 损失函数：MSE 与交叉熵，softmax+CE 的隐藏合体

运行方式（在任意目录均可）：
    pip install -e .   # 仓库根目录执行一次
    python tutorials/code/02_losses.py

说明：
- 第一部分验证 softmax+CE 的"合体梯度"：CE 对 softmax 输出的原始梯度
  乘上 softmax 的雅可比矩阵，化简后就是 (y_hat - y)/N —— 库里由
  softmax 层在自己的 backward 里做这个雅可比乘积（CE 只返回对 ŷ 的
  原始梯度），所以 softmax 不需要独立的逐元素导数
- 第二部分在同一个二分类玩具数据上训练两个同样的单神经元模型，
  一个用 MSE、一个用稀疏交叉熵，对比收敛速度与最终准确率
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
from numpy_keras.activations import functional as F
from numpy_keras.activations._mapper import _ActivationMapper

ASSETS = ROOT / "tutorials" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

np.random.seed(0)

# 1. softmax 的数值稳定：先减最大值，exp 永不溢出
print("softmax 数值稳定:")
with np.errstate(over="ignore"):
    print(f"  np.exp(1000) = {np.exp(1000)}")
print(f"  softmax([1000, 1000, 1000]) = {F.softmax(np.array([[1000., 1000., 1000.]]))[0]}")

# 2. softmax + CE 的合体梯度：雅可比矩阵 × CE 梯度 = (y_hat - y)/N
print("\nsoftmax+CE 合体梯度验证:")
z = np.array([[-1.0, 2.0, 0.5]])            # 预激活 logits（单个样本）
y = np.array([[0.0, 1.0, 0.0]])             # one-hot 标签
y_hat = F.softmax(z)
print(f"  y_hat = softmax(z) = {y_hat[0]}")

# CE 对 softmax 输出的梯度: ∂L/∂ŷ = -y / ŷ / N
grad_wrt_yhat = -y / np.clip(y_hat, 1e-10, 1 - 1e-10) / y.shape[0]
# softmax 的雅可比: J[i, j] = ŷ_i * (δ_ij - ŷ_j)
J = y_hat[:, :, None] * (np.eye(3)[None, :, :] - y_hat[:, None, :])
# 链式法则: ∂L/∂z = (∂L/∂ŷ) @ J
grad_by_chain = np.einsum("bi,bij->bj", grad_wrt_yhat, J)
# 库的两步走：CE 返回对 ŷ 的原始梯度，softmax 层在自己的 backward 里
# 做雅可比乘积（每层自持激活导数的约定）
ce = keras.losses.CategoricalCrossEntropy()
grad_by_lib = _ActivationMapper().backward("softmax", y_hat, ce.grad(y, y_hat), {})
print(f"  链式法则手算 ∂L/∂z = {grad_by_chain[0]}")
print(f"  库（softmax 层 backward）∂L/∂z = {grad_by_lib[0]}")
print(f"  两者一致: {np.allclose(grad_by_chain, grad_by_lib)}")

# 3. 学习减速实验：MSE vs 交叉熵
#    经典设计（Nielsen《Neural Networks and Deep Learning》第 3 章同款）：
#    单神经元、全批量 SGD、并且刻意"自信地错"——W=0、b 拉大，
#    初始预测 ≈ 0.993 却错一半样本，让 sigmoid 导数接近 0。
#    MSE 的梯度带着 sigmoid'(ŷ) 因子，会被压得几乎不动；
#    交叉熵的梯度没有这个因子，起步就快。
def make_blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[1.0, 1.0], [-1.0, -1.0]])
    X = np.vstack([rng.normal(c, 0.9, (n, 2)) for c in centers])
    y = np.array([0] * n + [1] * n)
    idx = rng.permutation(2 * n)
    return X[idx], y[idx]


X, y = make_blobs()
print(f"\n玩具数据: {X.shape}, 标签 {np.unique(y)}")


class AccTracker:
    """每个 epoch 结束时评估准确率（callback 的 on_epoch_end 钩子）。"""

    def __init__(self, X, y, mode):
        self.X, self.y, self.mode = X, y, mode
        self.accs = []

    def on_epoch_end(self, model=None):
        pred = model.predict(self.X)
        if self.mode == "mse":
            pred = (pred > 0.5).astype(int)              # sigmoid 输出按 0.5 定类
        self.accs.append(float(np.mean(pred == self.y)))


def build_single_neuron(mode):
    m = keras.Sequential()
    m.add(keras.layers.Input(2))
    if mode == "mse":
        m.add(keras.layers.Dense(1, activation="sigmoid"))
        m.compile(loss="mse", optimizer="sgd")
    else:
        m.add(keras.layers.Dense(2, activation="softmax"))
        m.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")
    m.optimizer.learning_rate = 0.6
    # 刻意"自信地错"：W=0、b 拉大，初始预测 ≈ 0.993 却错一半样本
    last = list(m.layers.values())[-1]
    last.params["W"][:] = 0.0
    last.params["b"][:] = np.array([5.0] if mode == "mse" else [5.0, -5.0])
    return m


histories, acc_curves = {}, {}
for name, mode in [
        ("mse + sigmoid", "mse"),
        ("crossentropy + softmax", "ce"),
    ]:
    np.random.seed(0)                                    # 两个模型同种子、同初始化条件
    m = build_single_neuron(mode)
    tracker = AccTracker(X, y, mode)
    y_fit = y.astype(float).reshape(-1, 1) if mode == "mse" else y
    h = m.fit(X, y_fit, batch_size=400, epochs=300, verbose=0, callbacks=[tracker])
    histories[name] = h
    acc_curves[name] = tracker.accs
    for ep in (4, 19, 99, 299):
        print(f"{name:>24}: 第 {ep + 1:>3} 轮准确率 {tracker.accs[ep]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name in histories:
    axes[0].plot(histories[name]["loss"], label=name)
axes[0].set_title("Loss（两条曲线尺度不同，只看趋势）")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)
for name in acc_curves:
    axes[1].plot(range(1, len(acc_curves[name]) + 1), acc_curves[name], label=name)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / "02_mse_vs_ce.png", dpi=150)
plt.close(fig)

print("\n图片已保存: tutorials/assets/02_mse_vs_ce.png")
