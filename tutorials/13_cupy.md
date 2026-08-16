# 13 CuPy GPU 加速：从 NumPy 到 GPU 后端（可选）

> **前置知识**：本系列《Cython 加速》与《引擎室》
> **运行环境**：numpy_keras v2.2.0 / Python 3.12 / NumPy 1.26.4 / cupy-cuda12x 13.6.0（2× NVIDIA A800 80GB 实测）
> **运行时间**：约 5 秒 × 两种模式
> **随机种子**：`np.random.seed(0)`

## 从 Cython 到 CuPy：瓶颈不同

第 12 章的 Cython 加速回答的问题是：**矩阵乘之外**的逐元素操作每一趟都是一次内存遍历，瓶颈在内存带宽，解法是把相邻几趟**融合成一趟**。它不换硬件，只是减少遍历次数。

本章的 CuPy 后端换了一个思路：**把整个计算搬到 GPU 上**。矩阵乘交给 cuBLAS，逐元素操作天然大规模并行——同样的代码，只是因为数据住在 GPU 显存里，算子就换成了 GPU 内核。两种加速的对比：

| | Cython（第 12 章） | CuPy（本章） |
|---|---|---|
| 瓶颈 | CPU 内存带宽 | CPU 算力/带宽（换设备解决） |
| 手段 | 融合内存遍历 | 换设备 + 大规模并行 |
| 代价 | 编译一个 `.so` | 装 CuPy、每步多一次内核启动开销 |
| 新瓶颈 | 无（收益稳定） | 内核启动开销、主机/设备往返（见下文） |

关键差别在最后一行：GPU 不是免费的。每个算子都是一次内核启动（微秒级），主机与显存之间搬数据要过 PCIe——**当算子够大时 GPU 碾压 CPU，算子太小时反而更慢**。所以本章和上一章一样，重点不是"怎么加速"，而是"哪些地方值得加速、哪些地方不值得"。

## 一个全局开关

和 Cython 内核一样，CuPy 后端是**可选**的：不装 CuPy、不设开关，行为与之前完全一致。开启方式两种：环境变量（import 时读取），或 `set_backend`（随时可调，notebook 友好）。探测逻辑（`numpy_keras/backend.py`）：

```python
# excerpt: numpy_keras/backend.py
# Module-level selection: read once at import time.
_requested = os.environ.get("NUMPY_KERAS_BACKEND", "numpy")
if _requested.lower() not in ("numpy", "cupy"):
    raise ValueError(
        f"Unknown NUMPY_KERAS_BACKEND={_requested!r}; expected 'numpy' or 'cupy'.")
if _requested.lower() == "cupy":
    set_backend("cupy")
```

```bash
export NUMPY_KERAS_BACKEND=cupy      # 方式一：环境变量
```

方式二是一行 `keras.set_backend("cupy")`（`keras.get_backend()` 随时查询当前后端），完整脚本的演示 1 里就有真实的切换调用：

```python
# excerpt: 本脚本·演示 1
B.set_backend("cupy")
np.random.seed(7)  # 同样的种子，同样的权重，只是出生在 GPU 上
```

`set_backend` 的实现是一个全局别名：所有计算模块（层、激活、损失、优化器、初始化器）统一写 `from ..backend import xp as np`，`set_backend` 把 `xp` 重绑到 numpy 或 cupy，并同步改写各消费模块的 `np` 别名——模块全局变量在**调用时**解析，所以已导入的模块也会立即跟着切换。未安装 CuPy 时请求 `"cupy"` 会发出警告并留在 NumPy 后端，与 Cython 的降级哲学一致。

## 什么上 GPU、什么留在主机

设计上有一条明确的边界：

| 上 GPU | 留在主机 |
|---|---|
| 参数、梯度、层间激活、损失 | 随机数生成（初始化器、Dropout 掩码） |
| BatchNorm 滑动统计量、优化器状态 | 数据预处理（洗牌、one-hot 编码） |
| （`fit`/`predict`/`evaluate` 入口自动同步） | 标签解码与指标运算 |

**随机数留在主机**是这条设计里最值得琢磨的一笔：`numpy.random` 的序列与设备无关，所以**相同种子下，CPU 与 GPU 的初始权重和 Dropout 掩码逐位相同**。这让两条路径可以直接对拍校验（见下节），也让调试体验接近"同一条代码"。搬运成本可以忽略——一个 batch 的掩码只有几十 KB，而初始化只发生一次。

模型状态的同步是**双向**且自动的：`fit`/`predict`/`evaluate` 的入口会调用 `__sync_backend`，把参数、梯度、BatchNorm 统计量与优化器状态搬到当前设备（或搬回主机）——先建模型再切后端、训练中途来回切换，都不需要手动搬任何东西：

```python
# excerpt: numpy_keras/models/sequential.py
        def _sync(a):
            if on_gpu():
                return a if is_cupy_array(a) else asarray(a)
            return asnumpy(a) if is_cupy_array(a) else a
```

## 一致性怎么保证

有了"同种子逐位相同"的前提，奇偶校验测试就非常强：同一段代码在两个后端各跑一遍，直接比较数值。`tests/test_cupy.py` 共 39 个测试，模块头先探测 CuPy 是否可用，不可用则整体跳过：

```python
# excerpt: tests/test_cupy.py
B.set_backend("cupy")
CUPY = B.is_cupy_mode()
B.set_backend("numpy")

pytestmark = pytest.mark.skipif(not CUPY, reason="cupy not available")
```

覆盖范围：全部激活函数、softmax 反传、Dense/Conv2D/MaxPool2D/Dropout/BatchNormalization 的前向反向（含参数梯度）、三种 RNN × 两种输出模式、四个优化器各 5 步（含 SGD Nesterov 分支）、同种子端到端训练（loss 轨迹 rtol=1e-5）、GPU 上的有限差分梯度检查、以及"建好模型再切后端"。容差设计成三档：前向 `1e-10`（CUDA libm 与 NumPy 的 exp/tanh 差 ~1 ulp）、梯度 `1e-9`（GPU 归约顺序与 CPU 逐步累加不同）、轨迹 `1e-5`。钉死 CPU 逐时间步语义的 `rtol=1e-12` 参考测试全部只走未改动的 NumPy 路径——GPU 分支是**新增**的，不是替换。

## RNN 的 GPU 分支：批量输入投影

三种 RNN 的前向都是时间步循环，其中输入投影项 `x_t @ W_xh` 与递推无关，可以把 T 步合成**一次 3D 矩阵乘**，循环里只剩递推项（SimpleRNN 的 GPU 分支）：

```python
# excerpt: numpy_keras/layers/simple_rnn.py
        if is_cupy_array(inputs):
            # GPU path: batch the input projection over all timesteps into a
            # single 3D matmul, so only the recurrence stays in the loop.
            pre_x = inputs @ self.params["W_xh"]           # (N, T, U)
```

反向同理：BPTT 循环里把每步的 `d_pre` 存进序列张量，循环结束后用一次 `tensordot` 累加 `W_xh` 的梯度、一次 3D 矩阵乘算完整个 `dX`——每时间步省掉多次内核启动。

但**诚实地说：教学规模下 RNN 在 GPU 上反而更慢**（本章实测 2-4 倍）。T=20、U=64 时每步的矩阵乘只有 64×64，内核启动开销和逐时间步的 Python 循环完全主导，批量投影省下的启动次数补不回来。第 12 章"哪些不编译"的清单精神同样适用于 GPU：**RNN 只有 N/T/U 都很大时才值得上卡**。

## 实测数字

微基准（`benchmarks/bench_cupy.py`，A800，中位数；Cython 已禁用，对比的是纯 NumPy 与纯 CuPy 路径）：

| 算子（规模） | 纯 NumPy | CuPy | 加速比 |
|---|---|---|---|
| relu（4096×784） | 5.26 ms | 86 µs | ~61× |
| tanh（4096×784） | 5.32 ms | 48 µs | ~112× |
| sigmoid（4096×784） | 117.8 ms | 810 µs | ~145× |
| softmax（4096×784） | 35.7 ms | 175 µs | ~203× |
| Dense 前向（4096×784→512） | 11.5 ms | 258 µs | ~45× |
| Dense 反向 | 38.4 ms | 653 µs | ~59× |
| Conv2D 前向（64×28×28, 8@3×3） | 2.41 ms | 428 µs | ~5.6× |
| Conv2D 反向 | 10.8 ms | 2.18 ms | ~4.9× |
| MaxPool 前向（64×28×28） | 5.05 ms | 398 µs | ~13× |
| MaxPool 反向 | 2.82 ms | 470 µs | ~6.0× |
| Adam 更新（784×512，融合内核） | 3.49 ms | 42 µs | ~83× |
| SimpleRNN 前向（64×20×32→64） | 775 µs | 1.92 ms | ~0.4×（更慢） |

模型级（`fit` 全程计时，含主机侧数据准备；MLP 10000×784 [256,256,10]，3 轮，batch 64，预热后 3 次取中位数）：

| 配置 | 纯 NumPy | CPU + Cython | CuPy | GPU/纯 NumPy |
|---|---|---|---|---|
| 无 metrics | 2.47 s | 1.67 s | 0.78 s | ~3.2× |
| + accuracy 指标 | 2.20 s | 2.13 s | 1.05 s | ~2.1× |
| 教程脚本（带指标，见下） | — | 2.20 s | 1.20 s | ~1.8× |

（Cython 与 CuPy 是库的两个可选加速层，机器上编译了 Cython 内核时 CPU 侧自动受益，所以"教程脚本"一行列出的是同一脚本在本机的两种模式下实际打印的数字。）

模型级数字不是一步到位的，**瓶颈转移**的过程本身值得展开——和第 12 章一样，先测出瓶颈在哪：

1. **初版 GPU 端到端只有 ~1.1×**。cProfile 显示：Adam 更新占 40%（纯路径每个参数数组 ~8 个微小内核 × 471 次更新，启动开销主导）；`predict` 每批把全量输出矩阵 `asnumpy` 回主机逐行 argmax，每批强制一次同步；softmax 反传的 `cupy.einsum` 每批 ~0.36 ms；此外首次上卡的 context 初始化与内核 JIT 也被计入了时间。
2. **逐个消灭**：给 GPU 写融合优化器内核（与 Cython 内核同构，逐语句镜像纯路径，一个参数数组一次内核，`_gpu_kernels.py`）；`predict` 改为设备端 argmax + 单次大前传（设备上小批量循环只会徒增启动开销）；softmax 反传换成收缩形式 `y ⊙ (g − g·y)`（与 einsum 数学等价，但不再物化 (n, C, C) 雅可比）；教程脚本加预热把一次性开销排除出计时。
3. **剩余的时间去哪了**：GPU 每轮 ~0.26s（无指标）里，前向/反向的逐层小内核与主机侧数据准备各占一部分——批量只有 64 时，这一步的收益边界就到这里。批量更大、层更宽，GPU 的优势会继续扩大；教学规模下，这些数字就是诚实答案。

## 完整代码

```python
"""13_cupy.py — CuPy GPU 加速：可选的后端开关

运行方式（在仓库根目录，先按 CUDA 版本安装 CuPy）：
    pip install "cupy-cuda12x>=13.6.0"      # 驱动 >= 12.x 的机器通用
    python tutorials/code/13_cupy.py                     # NumPy 后端（若编译了 Cython 内核则自动叠加）
    NUMPY_KERAS_BACKEND=cupy python tutorials/code/13_cupy.py  # GPU 后端

说明：
- 打印当前后端，并演示三种用法：同种子下两个后端的初始权重逐位
  相同（奇偶校验）、模型自动上卡训练、set_backend 随时切换后端
- 训练计时跟随启动时的后端——两种模式下分别跑一遍即可得到对比；
  计时前先跑一次小模型预热，把 CUDA context 初始化与内核即时编译
  排除在计时之外；配置带 accuracy 指标，是真实使用场景
- 固定种子 np.random.seed(0)，数字可复现
- 实测环境：2× NVIDIA A800 80GB / Python 3.12 / NumPy 1.26.4
"""

import csv
import itertools
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

import numpy_keras as keras
from numpy_keras import backend as B

np.random.seed(0)

ACTIVE = B.get_backend()
print(f"当前后端: {ACTIVE}")

# 演示 1：随机数始终在主机生成——同种子下，两个后端的初始权重逐位相同
B.set_backend("numpy")
np.random.seed(7)
model_cpu = keras.Sequential([
    keras.layers.Input(784),
    keras.layers.Dense(256, activation="relu"),
])
W_cpu = model_cpu.parameters["dense_1"]["W"]

B.set_backend("cupy")
np.random.seed(7)  # 同样的种子，同样的权重，只是出生在 GPU 上
model_gpu = keras.Sequential([
    keras.layers.Input(784),
    keras.layers.Dense(256, activation="relu"),
])
W_gpu = model_gpu.parameters["dense_1"]["W"]
print(f"同种子初始权重逐位相同: {np.array_equal(W_cpu, B.asnumpy(W_gpu))}")

# 演示 2：回到启动时的后端训练——数据留在主机，模型自动上卡，
# fit / predict 的接口完全不变
B.set_backend(ACTIVE)

def load_mnist(path, n_rows=None):
    with open(path) as f:
        rows = list(itertools.islice(csv.reader(f), n_rows))
    y = np.array([int(r[0]) for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows]) / 255.0
    return X, y


X, y = load_mnist(ROOT / "data" / "mnist_train_small.csv", n_rows=10000)
print(f"数据: {X.shape}")

# 预热：首次上卡要初始化 CUDA context 并即时编译内核，不计入计时——
# 否则这部分一次性开销会不公平地记在 GPU 头上
warmup = keras.Sequential([
    keras.layers.Input(784),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
warmup.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
warmup.fit(X[:64], y[:64], batch_size=64, epochs=1, verbose=0)

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

# 演示 3：训练后随时切到对方后端跑一次推理，再切回——fit/predict
# 入口会把参数、梯度、BatchNorm 统计量与优化器状态自动双向同步
other = "cupy" if ACTIVE == "numpy" else "numpy"
B.set_backend(other)
_ = model.predict(X[:8])
B.set_backend(ACTIVE)
_ = model.predict(X[:8])   # 再次进入 predict，状态同步回启动后端
print(f"切到 {other} 跑过 predict 并切回 {ACTIVE}，"
      f"参数在 GPU: {B.on_gpu(model.parameters['dense_1']['W'])}")```
