"""11_engine_room.py — 引擎室：如何新增一层

运行方式（在任意目录均可）：
    pip install -e .   # 仓库根目录执行一次
    python tutorials/code/11_engine_room.py

说明：
- 第一部分：从零实现一个库中没有的层（Scale：逐特征可学习缩放），
  只实现 6 个钩子就接入 Sequential
- 第二部分：整模型有限差分梯度校验，证明自定义层的 backward 正确
- 第三部分：把自定义层放进真实训练跑通
- 固定种子 np.random.seed(0)，数字可复现
- 环境：Apple M2 Pro / macOS / Python 3.12 / NumPy 1.26.4
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

import numpy_keras as keras
from numpy_keras.activations._mapper import _ActivationMapper

np.random.seed(0)


class Scale:
    """自定义层：y = x * s，s 是逐特征的可学习缩放参数。

    接入 Sequential 不需要继承任何基类——鸭子类型契约：
    实现 set_input_shape / init_params / forward / backward 四个方法，
    提供 output_dim / output_shape 两个属性，并把可训练参数放进
    params / grads 字典即可（见正文的契约清单）。

    关键：本层插在带激活的 Dense 之后，必须自己接管"上一层激活的
    导数"——实现 set_activation_deriv 接收上一层的导数函数、在
    backward 里于自身输入处应用，并让 activation 属性返回 None
    使通用链在层边界归零（与 RNN 层同款约定，见正文）。"""

    def __init__(self, initial_scale=1.0):
        self.__initial_scale = initial_scale
        self.__input_shape = None
        self.__output_dim = None
        self.__prev_deriv = None
        self.__prev_config = {}

    def set_input_shape(self, shape):
        self.__input_shape = tuple(shape)

    def set_activation_deriv(self, prev_activation, prev_config):
        if prev_activation:
            self.__prev_deriv = _ActivationMapper()[prev_activation + "_deriv"]
            self.__prev_config = prev_config
        else:
            self.__prev_deriv = None

    def init_params(self, input_dim):
        self.__output_dim = input_dim
        self.params = {"s": np.full((input_dim,), self.__initial_scale)}
        self.grads = {"s": np.zeros_like(self.params["s"])}

    def forward(self, inputs, is_training):
        self.inputs = inputs
        return inputs * self.params["s"]

    def backward(self, grad):
        # 先走完本层的变换链：dL/ds = sum(grad ⊙ x)，dL/dx = grad ⊙ s
        self.grads["s"] = np.sum(grad * self.inputs, axis=0)
        dx = grad * self.params["s"]
        # 再应用上一层激活的导数（在自身输入处取值）
        if self.__prev_deriv:
            dx *= self.__prev_deriv(self.inputs, **self.__prev_config)
        return dx

    @property
    def activation(self):
        # 自持输出链：返回 None，让下一层跳过通用 deriv 链
        return None

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
