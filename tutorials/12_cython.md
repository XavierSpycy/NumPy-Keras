# 12 Cython 加速：从 NumPy 到编译内核（可选）

> **前置知识**：本系列《反向传播逐行拆解》与《CNN 解剖》
> **运行环境**：numpy_keras v2.1.0 / Python 3.12 / NumPy 1.26.4（Apple M2 Pro 实测）
> **运行时间**：约 10 秒 × 两种模式（纯 NumPy 模式）/
> **随机种子**：`np.random.seed(0)`

## 纯 NumPy 慢在哪

NumPy 的矩阵乘法本身已是 BLAS，慢的是**矩阵乘之外**的部分：激活函数、bias、优化器更新这些逐元素操作，每一次都是一趟独立的内存遍历。一趟 784×256 的矩阵乘之后，relu、bias 加法、Adam 更新各扫一遍内存——瓶颈在内存带宽，不在计算。Cython 内核的思路：**把相邻的几趟融合成一趟**。

## 自动探测与降级

内核是**可选**的：编译出的 `.so` 存在就用，不存在就退回纯 NumPy，行为完全一致。探测逻辑（`numpy_keras/cython/__init__.py`）：

```python
# excerpt: numpy_keras/cython/__init__.py
if os.environ.get("NUMPY_KERAS_DISABLE_CYTHON"):
    _kernels = None
else:
    try:
        from . import _kernels
    except ImportError:
        _kernels = None
```

两个开关：环境变量 `NUMPY_KERAS_DISABLE_CYTHON=1` 强制纯模式（读者机器上没有 `.so` 时的默认状态）；`python build_cython.py build_ext --inplace` 编译并安装内核：

```python
# excerpt: build_cython.py
"""Build the optional Cython acceleration kernels in place.

Usage (from the repository root):

    python build_cython.py build_ext --inplace

This drops ``_kernels.cpython-<version>-<platform>.so`` next to the ``.pyx``
file.  The library auto-detects the compiled module at import time and falls
back to the pure NumPy implementations when it is absent (or when
``NUMPY_KERAS_DISABLE_CYTHON`` is set).
"""
```

调用点都是同款模式（以 Dense 为例）：满足 dtype/连续性/激活白名单才走内核，否则走原样保留的纯 NumPy 代码——**一条路径验证过，另一条永远可退**。

## 哪些操作值得编译——以及哪些不值得

| 有内核 | 为什么 |
|---|---|
| Dense 前向/反向 | 矩阵乘 + 激活 + bias 融合成一趟 |
| 四个优化器的 update | 逐元素更新融合成单循环 |
| col2im 散射 | 纯 NumPy 的 scatter-add 很慢，编译版约 16× |
| maxpool 反向 | 窗口扫描 + 散射融合 |

| 刻意没有内核 | 为什么 |
|---|---|
| im2col | 已是 `sliding_window_view` 的零拷贝 memcpy，无利可图 |
| RNN 各层 | 每时间步已是 BLAS 矩阵乘，Python 时间步循环的收益未经实测，不预设 |

这个"哪些不编译"的清单和"哪些编译"同样重要——加速的前提是**先测出瓶颈在哪**。README §2.1 的完整基准表（同 session 双模式、多次取均值、硬件记录）给出了历史数字；本文用一个最小实验复现其方法学：

```text
=== 已编译（默认模式）===
3 轮训练耗时: 2.81 s（平均每轮 0.94 s）
训练准确率: 0.9782

=== 纯 NumPy（NUMPY_KERAS_DISABLE_CYTHON=1）===
3 轮训练耗时: 3.84 s（平均每轮 1.28 s）
训练准确率: 0.9782
```

约 **1.37×** 的加速、准确率逐位一致（0.9782）。这是 MLP 的典型收益区间（README 表的 1.3× 左右）；卷积路径的收益更大（col2im 是纯 NumPy 的真正瓶颈，~1.7× 整体）。两点方法学提醒：绝对耗时受机器负载影响（同一会话内对比才可信），以及**两模式的数字一致性由奇偶测试钉死**（`tests/test_cython_kernels.py` 对每条内核路径断言与纯 NumPy 等价）。

## 完整代码

```python
"""12_cython.py — Cython 加速：从 NumPy 到编译内核

运行方式（在任意目录均可）：
    pip install -e .        # 仓库根目录执行一次
    python build_cython.py build_ext --inplace   # 编译内核（可选）
    python tutorials/code/12_cython.py           # 默认（编译内核）模式
    NUMPY_KERAS_DISABLE_CYTHON=1 python tutorials/code/12_cython.py  # 纯 NumPy 模式

说明：
- 打印当前内核状态（已编译 / 纯 NumPy），并计时同一个 MLP
  （10,000×784，[256,256,10]，3 轮）的训练耗时——两种模式下
  分别跑一遍即可得到速度对比
- 固定种子 np.random.seed(0)，数字可复现
- 环境：Apple M2 Pro / macOS / Python 3.12 / NumPy 1.26.4
"""

import csv
import itertools
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

import numpy_keras as keras
from numpy_keras.cython import _kernels as _ck

np.random.seed(0)

print(f"内核状态: {'已编译（默认模式）' if _ck is not None else '纯 NumPy（NUMPY_KERAS_DISABLE_CYTHON=1）'}")

def load_mnist(path, n_rows=None):
    with open(path) as f:
        rows = list(itertools.islice(csv.reader(f), n_rows))
    y = np.array([int(r[0]) for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows]) / 255.0
    return X, y


X, y = load_mnist(ROOT / "data" / "mnist_train_small.csv", n_rows=10000)
print(f"数据: {X.shape}")

model = keras.Sequential()
model.add(keras.layers.Input(784))
model.add(keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])

t0 = time.time()
history = model.fit(X, y, batch_size=64, epochs=3, verbose=0)
elapsed = time.time() - t0
print(f"3 轮训练耗时: {elapsed:.2f} s（平均每轮 {elapsed / 3:.2f} s）")
print(f"训练准确率: {history['metrics']['train_accuracy'][-1]:.4f}")
```

完整运行输出（两种模式各跑一遍）：

```text
内核状态: 已编译（默认模式）
数据: (10000, 784)
3 轮训练耗时: 2.81 s（平均每轮 0.94 s）
训练准确率: 0.9782
```

```text
内核状态: 纯 NumPy（NUMPY_KERAS_DISABLE_CYTHON=1）
数据: (10000, 784)
3 轮训练耗时: 3.84 s（平均每轮 1.28 s）
训练准确率: 0.9782
```

## 小结

- 纯 NumPy 的瓶颈是逐元素操作的内存遍历；Cython 内核把它们融合成单趟
- 架构三件套：`.so` 自动探测 + 环境变量降级 + 调用点白名单守卫，纯 NumPy 路径永远可退
- 哪些不编译和哪些编译同样重要：im2col 是零拷贝 memcpy、RNN 每步已是 BLAS
- 基准方法学：同会话双模式对比、多次取均值、硬件记录；奇偶测试保证两模式数字一致
- 本机复现：MLP 3 轮 2.81s vs 3.84s（~1.37×），准确率逐位一致

**练习**：在带卷积的模型上跑同样对比，速度比和 MLP 差多少？把 batch_size 改成 256 再测——两种模式的差距变大还是变小，为什么？

至此，从五分钟上手到编译内核，三部曲的每一块拼图都有了逐行可读的实现与本系列的拆解。祝你在白盒里玩得开心。
