# PhysicsNeMo PINNs 训练机制总结

## 一、PINNs 的本质

PINNs（Physics-Informed Neural Networks）用神经网络来求解偏微分方程（PDE）。它不依赖传统意义上的"数据集"，而是通过让神经网络满足物理方程（如 Navier-Stokes）来学习流场。

---

## 二、训练数据：实时采样的配点

### 没有数据集，只有几何体

训练数据不是从文件读取的，而是每次迭代时从**计算域的几何体上随机采样的坐标点**（配点，collocation points）。

### 约束类型与采样方式

| 约束类型 | 作用 | 采样区域 |
|---------|------|---------|
| `PointwiseBoundaryConstraint` | 强制边界条件（入口、出口、壁面） | 几何体边界 |
| `PointwiseInteriorConstraint` | 强制 PDE 残差（NS 方程、对流扩散） | 计算域内部 |
| `IntegralBoundaryConstraint` | 强制积分守恒（质量流量） | 多条积分线 |

### heat_sink 案例每步采样量

```
inlet（入口边界）：          64 点
outlet（出口边界）：         64 点
hs_wall（散热器壁面）：     500 点
channel_wall（通道壁面）： 2500 点
interior_flow（流场内部）： 4800 点  ← NS 方程残差
interior_heat（热场内部）： 4800 点  ← 对流扩散方程残差
integral lines：           4×128 点  ← 质量守恒积分
─────────────────────────────────
合计：约 12,928 点 / 每步
```

---

## 三、训练流程

```
定义计算域（10cm × 5cm 散热器通道）
  ↓
for step in range(500,000):

    1. 从各约束对应区域随机采样（共 ~12,928 点，每步都是全新的点）

    2. 将坐标 (x, y) 输入神经网络 → 得到预测值 (u, v, p, T)

    3. 对预测值自动微分，代入 PDE 计算残差：
         入口：预测速度是否等于给定入口速度？
         壁面：预测速度是否为 0（无滑移条件）？
         内部：∂u/∂x + ∂v/∂y = 0？（连续性方程）
               u·∂u/∂x + ... = 0？（动量方程）

    4. 所有残差加权求和 → 总 Loss

    5. 反向传播，更新神经网络权重

训练完成
```

### 与传统神经网络训练的对比

| | 传统 NN | PINNs |
|---|---|---|
| 数据来源 | 磁盘文件 | 每步实时采样 |
| Epoch 概念 | 有 | **没有**，只有 Step |
| Loss 含义 | 预测与标签的差距 | PDE 残差 + 边界条件违反量 |
| 数据重复 | 每 Epoch 重复使用 | **每步都是新的随机点** |
| 训练目标 | 拟合已有数据 | 让网络满足物理方程 |

---

## 四、采样机制详解

### 默认采样：`np.random.uniform()`

```python
# parameterization.py 核心代码
rand_param = np.random.uniform(value[0], value[1], size=(batch_size, 1))
```

纯均匀随机采样，无去重逻辑。

### 精确重复的概率

float32 在连续域内可表示约 **800 万**个不同值，二维坐标组合达 **64 万亿**种。从 64 万亿个位置中随机选 4800 个点，精确重复的概率可忽略不计。

### Clustering（点扎堆）

精确重复虽然不会发生，但**局部聚集（clustering）**是随机采样的固有现象：

- 某些区域点密集（被过度优化）
- 某些区域点稀疏（欠优化）
- 边界层等梯度剧烈区域尤其容易被"漏掉"

**框架提供的缓解方案：** 设置 `quasirandom=True` 启用 Halton 低差异序列，使点分布更均匀。默认不开启，因为每步重新采样时 Halton 的优势不明显。

### 关于 batch size 的理解

- 提高 batch size 的真正价值：**提高每步对关键区域（如边界层）的覆盖概率**
- 当 batch size 已能稳定覆盖所有关键区域后，继续增大的边际效益趋近于零
- 真正有效的方案是**自适应采样**：把更多点动态分配到 PDE 残差高的区域

---

## 五、边界层与采样质量

**边界层**：紧贴壁面、速度从 0 快速增大到主流值的薄层区域，NS 方程在此梯度极大。

```
管壁 |→→→→→→→→→→→  主流（速度最大）
     |→→→→→→→→→
     |→→→→→→→     ← 边界层（速度梯度剧烈）
     |→→→
     |→
     |            （速度 = 0）
```

随机采样容易在均匀主流区浪费大量点，而边界层采样不足，导致壁面附近预测精度下降。这是 PINNs 随机采样的主要精度隐患。

---

## 六、分布式训练（DDP）

### 支持情况

PhysicsNeMo Sym 的 `Solver` 已**内置完整的 DDP 支持**，无需修改任何用户代码。

核心机制（`constraint.py`）：

```python
if self.manager.distributed:
    self.model = DistributedDataParallel(
        self.model,
        device_ids=[self.manager.local_rank],
        ...
    )
```

`DistributedManager` 在框架初始化时自动读取环境变量，判断是否处于分布式环境，并自动完成 DDP 包装和 All-Reduce 同步。

### DDP 工作原理

```
每个 Step：
  GPU 0: 独立采样 12,928 点 → 计算梯度
  GPU 1: 独立采样 12,928 点 → 计算梯度
  GPU 2: 独立采样 12,928 点 → 计算梯度
  GPU 3: 独立采样 12,928 点 → 计算梯度
                                  ↓ All-Reduce（梯度取平均）
                         4块 GPU 同步更新权重（保持一致）
```

等效于每步用 **4 × 12,928 = 51,712 个点**训练，覆盖计算域更全面。

### 启用方式

**代码不做任何修改**，只改启动命令：

```bash
# 单 GPU（现状）
python heat_sink.py

# 单机 4 卡 DDP
torchrun --nproc_per_node=4 heat_sink.py

# 多机多卡（2台机器，每台4卡）
# 机器 A（主节点）：
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=192.168.1.1 --master_port=29500 heat_sink.py

# 机器 B（从节点）：
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr=192.168.1.1 --master_port=29500 heat_sink.py
```

### 多进程发现机制

`torchrun` 启动时自动为每个进程注入环境变量：

```
MASTER_ADDR   主节点 IP（所有进程向它报到）
MASTER_PORT   主节点监听端口
WORLD_SIZE    全局总进程数
RANK          当前进程的全局编号
LOCAL_RANK    当前进程在本机的编号
```

所有进程启动后向主节点"握手"，确认全员到齐后开始同步训练。
