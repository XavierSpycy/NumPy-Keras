"""11_engine_room.py — 引擎室：如何新增一层

运行方式（在任意目录均可）：
    pip install -e .   # 仓库根目录执行一次
    python tutorials/code/11_engine_room.py

说明：
- 第一部分：从零实现一个库中没有的层（Scale：逐特征可学习缩放），
  只实现 4 个方法 + 2 个属性就接入 Sequential
- 第二部分：整模型有限差分梯度校验，证明自定义层的 backward 正确
- 第三部分：把自定义层放进真实训练跑通
- 固定种子 np.random.seed(0)，数字可复现
- 环境：Apple M2 Pro / macOS / Python 3.12 / NumPy 1.26.4
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

import numpy_keras as keras

np.random.seed(0)


class Scale:
    """自定义层：y = x * s，s 是逐特征的可学习缩放参数。

    接入 Sequential 不需要继承任何基类——鸭子类型契约：
    实现 set_input_shape / init_params / forward / backward 四个方法，
    提供 output_dim / output_shape 两个属性，并把可训练参数放进
    params / grads 字典即可（见正文的契约清单）。

    每层只对自己的变换负责：backward 只算 dL/ds 与 dL/dx，
    上一层的激活由上一层自己处理，无需任何额外代码。"""

    def __init__(self, initial_scale=1.0):
        self.__initial_scale = initial_scale
        self.__input_shape = None
        self.__output_dim = None

    def set_input_shape(self, shape):
        self.__input_shape = tuple(shape)

    def init_params(self, input_dim):
        self.__output_dim = input_dim
        self.params = {"s": np.full((input_dim,), self.__initial_scale)}
        self.grads = {"s": np.zeros_like(self.params["s"])}

    def forward(self, inputs, is_training):
        self.inputs = inputs
        return inputs * self.params["s"]

    def backward(self, grad):
        # dL/ds = sum(grad ⊙ x)，dL/dx = grad ⊙ s —— 本层的变换链到此为止
        self.grads["s"] = np.sum(grad * self.inputs, axis=0)
        return grad * self.params["s"]

    @property
    def output_dim(self):
        return self.__output_dim

    @property
    def output_shape(self):
        return self.__input_shape

    def __str__(self):
        return "Scale()"


# 1. 接入 Sequential：和内置层完全一样
model = keras.Sequential()
model.add(keras.layers.Input(2))
model.add(keras.layers.Dense(3, activation="tanh"))
model.add(Scale(initial_scale=1.0))
model.add(keras.layers.Dense(2, activation="linear"))
model.compile(loss="mse", optimizer="sgd")
model.summary()

# 2. 有限差分梯度校验（与《反向传播逐行拆解》同款方法）
X = np.array([[0.5, -0.8], [1.2, 0.3], [-0.4, 0.9], [0.7, -0.2]])
y = np.array([[0.3, 0.8], [0.1, -0.5], [-0.6, 0.4], [0.9, 0.2]])


def gradcheck(model, X, y, eps=1e-6):
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
print(f"\n梯度校验: 共检查 {n_checked} 个参数（含自定义层的 s），"
      f"最大相对误差 = {max_rel:.2e}")
print(f"自定义层 Scale 的参数: {list(model.layers.values())[2].params['s']}")

# 3. 放进真实训练：跑通即证明接入成功
def make_blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[1.0, 1.0], [-1.0, -1.0]])
    X = np.vstack([rng.normal(c, 0.9, (n, 2)) for c in centers])
    y = np.array([0] * n + [1] * n)
    idx = rng.permutation(2 * n)
    return X[idx], y[idx]


Xb, yb = make_blobs()
np.random.seed(0)
m = keras.Sequential()
m.add(keras.layers.Input(2))
m.add(keras.layers.Dense(8, activation="relu", kernel_initializer="he_normal"))
m.add(Scale())
m.add(keras.layers.Dense(2, activation="softmax"))
m.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
h = m.fit(Xb, yb, batch_size=32, epochs=100, verbose=0)
print(f"\n含自定义层的模型训练 100 轮后 loss = {h['loss'][-1]:.4f}")
