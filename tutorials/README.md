# numpy_keras 系列教程

从零到三部曲（MLP / CNN / RNN）的中文教程系列。每篇独立成文：理论（公式）→ 逐行读源码 → 完整可运行代码 → 实测数字。所有数字都是实测（种子、硬件、模式全部注明），不虚报。

## 阅读顺序

| # | 文章 | 主题 | 代码 | 预计运行时间 |
|---|---|---|---|---|
| 00 | [五分钟上手](00_quickstart.md) | 诚实划分、第一个 MNIST 模型 | [code/00_quickstart.py](code/00_quickstart.py) | ~1 分钟 |
| 01 | [激活函数全解](01_activations.md) | 17 函数、后激活值导数、梯度消失 | [code/01_activations.py](code/01_activations.py) | 数秒 |
| 02 | [损失函数](02_losses.md) | MSE vs CE、softmax+CE 合体梯度 | [code/02_losses.py](code/02_losses.py) | ~1 分钟 |
| 03 | [反向传播逐行拆解](03_backprop.md) | 链式法则、有限差分校验、autograd 对照 | [code/03_backprop.py](code/03_backprop.py) | ~30 秒 |
| 04 | [优化器进化史](04_optimizers.md) | SGD → Momentum → NAG → Adagrad → Adadelta → Adam | [code/04_optimizers.py](code/04_optimizers.py) | ~1 分钟 |
| 05 | [学习率与九大调度器](05_learning_rate.md) | lr 扫描、调度器、EarlyStopping | [code/05_learning_rate.py](code/05_learning_rate.py) | 1-2 分钟 |
| 06 | [MLP 深入](06_mlp.md) | 初始化器尺度、12 层深网 | [code/06_mlp.py](code/06_mlp.py) | ~8 分钟 |
| 07 | [Dropout](07_dropout.md) | 倒置 Dropout、过拟合对比 | [code/07_dropout.py](code/07_dropout.py) | 1-2 分钟 |
| 08 | [BatchNormalization](08_batch_norm.md) | 训练/推理双模式、滑动统计量 | [code/08_batch_norm.py](code/08_batch_norm.py) | ~1 分钟 |
| 09 | [CNN 解剖](09_cnn.md) | im2col、LeNet、特征图 | [code/09_cnn.py](code/09_cnn.py) | ~1 分钟 |
| 10 | [RNN 三部曲](10_rnn.md) | SimpleRNN/LSTM/GRU、BPTT | [code/10_rnn.py](code/10_rnn.py) | ~1 分钟 |
| 11 | [引擎室](11_engine_room.md)（可选） | 鸭子类型契约、新增一层 | [code/11_engine_room.py](code/11_engine_room.py) | 数秒 |
| 12 | [Cython 加速](12_cython.md)（可选） | 编译内核、基准方法学 | [code/12_cython.py](code/12_cython.py) | ~10 秒 × 2 模式 |
| 13 | [CuPy GPU 加速](13_cupy.md)（可选） | 后端开关、主机/设备边界、瓶颈转移 | [code/13_cupy.py](code/13_cupy.py) | ~5 秒 × 2 模式 |

前置环境（每篇 docstring 也有）：仓库根目录 `pip install -e .`，数据文件随仓库提供。除 06（编译内核模式测量）外，所有数字以纯 NumPy 模式为准（`NUMPY_KERAS_DISABLE_CYTHON=1`），两模式轨迹由奇偶测试锁定一致。

## 写作约定（维护本系列时遵守）

1. **代码单一来源**：`code/XX.py` 是权威；md 中的完整代码块与它逐字节一致，节选块以 `# excerpt:` 首行标记（指向仓库文件）。`python check_snippets.py` 校验全部 13 篇。
2. **数字纪律**：所有数字来自脚本实测输出（原样粘贴）；种子、模式（纯/编译）、硬件写进每篇元信息；旧配方或旧版本的数字不得作为比较基准。
3. **无死链**：文章间只用《书名号》软提及，不挂章节链接（知乎/CSDN 上会失效）；GitHub 导航靠本索引。
4. **不讲 bug 史**：实验只展示正确行为；库的修复过程不进文章（版本日志按 changelog 惯例记录）。
5. **运行预算**：每篇脚本设计目标 ≤ 3-5 分钟（读者可等待的心理预算）。

## 发布到知乎 / CSDN 清单

- **图片**：`assets/` 下的 PNG 需手动上传到平台图床，md 里的相对路径 `assets/xx.png` 替换为上传后的 URL
- **LaTeX 公式**：知乎支持 `$$...$$`；CSDN 支持差——建议把公式截图转图片，或改写为行内文本
- **外链**：GitHub 项目链接保留；`tutorials/code/xx.py` 等仓库路径改成 GitHub 直达链接（`https://github.com/XavierSpycy/NumPy-Keras/blob/main/tutorials/code/xx.py`）
- **元信息块**：发布时把"前置知识"改成软提及版本（当前写法已兼容）
- **顺序建议**：按 00 → 13 顺序发布，每篇之间留出传播窗口；01（激活函数）和 06（MLP 深入）自带强实验钩子，适合作为首发引流篇
