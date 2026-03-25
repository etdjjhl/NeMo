# `compare_openfoam_csv.py` 使用说明

本文档说明如何使用 [`compare_openfoam_csv.py`](../compare_openfoam_csv.py) 对训练后的 PINN 模型和官方 OpenFOAM 参考结果做数值对比。

适用对象：

- 基线模型：[`cases/three_fin_2d/heat_sink.py`](../cases/three_fin_2d/heat_sink.py)
- 参数化模型：[`cases/three_fin_2d/heat_sink_param.py`](../cases/three_fin_2d/heat_sink_param.py)

官方参考 CSV 默认路径：

- [`cases/three_fin_2d/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv`](../cases/three_fin_2d/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv)

## 1. 这个脚本是做什么的

`compare_openfoam_csv.py` 的目标是回答一个直接问题：

> 当前训练出来的 PINN，在 OpenFOAM 参考点上，和官方 CFD 结果差多少？

它不会重新训练模型，也不会做新的 CFD 仿真。它做的是：

1. 读取你已经训练好的 PINN checkpoint。
2. 读取官方 OpenFOAM CSV 中的参考点。
3. 在这些同样的 `(x, y)` 点上运行 PINN 推理。
4. 将 PINN 输出和 OpenFOAM 真值逐点比较。
5. 输出误差指标和可视化图。

## 2. 测试方法说明

### 2.1 输入数据来源

脚本使用官方 OpenFOAM CSV 作为参考真值。对于 `three_fin_2d`，官方 CSV 对应的是基线工况，也就是：

- 入口风速 `inlet_vel = 1.5`
- 与 `heat_sink.py` 中 validator 对齐的参考数据

这也是为什么参数化模型在和该 CSV 对比时，应当传入：

```bash
--inlet-vel 1.5
```

### 2.2 比较流程

脚本内部的比较流程如下：

1. 从 `--run-dir` 递归查找模型文件：
   - `flow_network*.pth`
   - `heat_network*.pth`
2. 尝试从 `.hydra/config.yaml` 判断模型类型：
   - `baseline`
   - `param`
3. 读取 OpenFOAM CSV 中的坐标和真值字段：
   - `x`
   - `y`
   - `u`
   - `v`
   - `p`
   - `T`
4. 将 OpenFOAM 的温度 `T` 转换为训练/验证时一致的归一化温度：

```python
c = (T - 293.498) / 273.15
```

5. 在全部 CSV 点上运行 PINN 推理：
   - baseline 模型输入：`(x, y)`
   - param 模型输入：`(x, y, inlet_vel)`
6. 对四个场逐点比较：
   - `u`
   - `v`
   - `p`
   - `c`
7. 输出误差表与图像。

### 2.3 这个测试的意义

这个测试是“点级别真值对比”，比只看训练 monitor 更直接。

它主要适合回答：

- 模型有没有学到合理的速度场、压力场、温度场。
- 参数化模型在 `inlet_vel=1.5` 这一官方工况下，是否接近基线/官方参考。
- 不同 checkpoint 之间，哪个更接近 OpenFOAM。

它不直接回答：

- 模型在整个参数区间 `1.0 ~ 2.5` 上的泛化能力如何。
- 其它未提供 CFD 参考数据的工况是否也正确。

## 3. 脚本使用说明

### 3.1 运行环境

建议使用项目已经准备好的 PhysicsNeMo 环境：

```bash
/home/featurize/work/env_conda/nemo/bin/python
```

### 3.2 最常用命令

#### 基线模型对比

```bash
/home/featurize/work/env_conda/nemo/bin/python compare_openfoam_csv.py \
  --run-dir outputs/20260226_094257 \
  --model-type baseline \
  --csv-path cases/three_fin_2d/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv \
  --out-dir outputs/openfoam_compare_base
```

#### 参数化模型对比

```bash
/home/featurize/work/env_conda/nemo/bin/python compare_openfoam_csv.py \
  --run-dir outputs/20260317_033129 \
  --csv-path cases/three_fin_2d/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv \
  --inlet-vel 1.5 \
  --out-dir outputs/openfoam_compare_param
```

### 3.3 参数说明

#### `--run-dir`

训练输出目录。可以传：

- `outputs/20260317_033129`
- `outputs/latest`
- `cases/three_fin_2d/outputs/run`
- `cases/three_fin_2d/outputs/run_param`

脚本会递归查找包含 checkpoint 的目录。

#### `--csv-path`

OpenFOAM 参考 CSV 路径。默认值是：

```text
cases/three_fin_2d/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv
```

如果你使用默认官方 CSV，这个参数可以省略。

#### `--model-type`

可选值：

- `auto`
- `baseline`
- `param`

默认是 `auto`。

行为说明：

- 如果 `.hydra/config.yaml` 里有 `custom.parameterized: true`，脚本会判为 `param`
- 否则判为 `baseline`

如果自动判断失败，建议手动指定。

#### `--inlet-vel`

只对参数化模型有意义。表示推理时输入给网络的 `inlet_vel`。

对当前官方 CSV，应使用：

```bash
--inlet-vel 1.5
```

对于 baseline 模型，这个参数会被忽略。

#### `--batch-size`

控制在多少个 CSV 点上分批做一次推理。默认是：

```text
8192
```

如果显存不够，可以适当调小。

#### `--plot-max-points`

控制画图时最多采样多少个点。默认是：

```text
20000
```

注意：

- 指标计算永远使用全部 CSV 点
- 只有图像可能做下采样

#### `--out-dir`

输出目录。默认是：

```text
openfoam_compare
```

建议显式指定到仓库内，例如：

```bash
--out-dir outputs/openfoam_compare_param
```

## 4. 输出结果说明

脚本会在 `--out-dir` 下生成：

- `comparison_report.md`
- `u_spatial.png`
- `u_parity.png`
- `v_spatial.png`
- `v_parity.png`
- `p_spatial.png`
- `p_parity.png`
- `c_spatial.png`
- `c_parity.png`

## 5. 报告里每个字段的含义

### 5.1 报告头部字段

#### `Generated`

报告生成时间，UTC。

#### `Model type`

模型类型：

- `baseline`：来自 `heat_sink.py`
- `param`：来自 `heat_sink_param.py`

#### `Artifact dir`

实际加载 checkpoint 的目录。  
注意它不一定等于你传入的 `--run-dir`，因为脚本会在其下递归查找真正含 `.pth` 文件的目录。

#### `CSV path`

本次对比使用的 OpenFOAM CSV 路径。

#### `Reference points`

CSV 中用于对比的总点数。  
当前官方 `heat_sink_zeroEq_Pr5_mesh20.csv` 是 `29152` 个点。

#### `inlet_vel`

只有在 `param` 模型时出现。表示参数化网络在推理时使用的入口风速。

### 5.2 指标表字段

表格对每个场 `u / v / p / c` 都会给出以下指标。

#### `Field`

物理量名称：

- `u`：x 方向速度
- `v`：y 方向速度
- `p`：压力
- `c`：归一化温度

#### `MAE`

平均绝对误差：

```text
mean(abs(pred - ref))
```

越小越好。

优点是直观，单位和原变量一致。  
通常这是最容易直接解释的误差。

#### `RMSE`

均方根误差：

```text
sqrt(mean((pred - ref)^2))
```

越小越好。

相比 MAE，RMSE 对局部大误差更敏感。

#### `Max Abs`

最大绝对误差：

```text
max(abs(pred - ref))
```

越小越好。

这个指标适合发现“局部坏点”或“某个区域误差特别大”的情况。

#### `Rel L2`

相对 L2 误差：

```text
||pred - ref||_2 / ||ref||_2
```

越小越好。

它是整体场误差的无量纲比例，适合比较不同 checkpoint。  
但要注意：

- 如果参考场本身整体数值很小，这个指标会显得偏大。
- `v` 场经常比 `u` 更容易出现这种情况，因为 `v` 的整体量级通常更小。

#### `PINN mean`

PINN 预测值在所有参考点上的平均值。

它主要用于粗略 sanity check，例如：

- 预测整体偏大还是偏小
- 和 OpenFOAM 平均值是否有系统偏差

它不能单独代表模型好坏。

#### `OpenFOAM mean`

OpenFOAM 真值在所有参考点上的平均值。

通常用于和 `PINN mean` 对照看整体偏差。

## 6. 图像怎么读

### 6.1 `*_spatial.png`

每个 `*_spatial.png` 都有三张并排图：

1. `PINN prediction`
2. `OpenFOAM reference`
3. `Difference (PINN - OpenFOAM)`

第三张图最关键，表示空间误差分布。

读法：

- 如果误差主要集中在入口、散热片边缘、尾流区，通常说明这些区域更难学。
- 如果整个通道都偏红或偏蓝，通常说明有系统性偏差。
- 如果误差图整体颜色很浅，说明场整体拟合较好。

### 6.2 `*_parity.png`

横轴是 OpenFOAM，纵轴是 PINN。

图中的虚线表示理想情况：

```text
y = x
```

读法：

- 点越贴近虚线，说明逐点预测越准确。
- 如果点云整体在虚线上方，表示 PINN 整体偏大。
- 如果点云整体在虚线下方，表示 PINN 整体偏小。
- 如果点云散得很开，说明局部误差较大。

## 7. 如何解读测试结果

### 7.1 基线模型

如果你比较的是 `heat_sink.py` 的 baseline 模型，这份报告主要回答：

- 当前这个 checkpoint 是否已经接近官方 OpenFOAM 解
- 当前训练步数是否明显不够

例如你之前的 10k 步 baseline 结果里：

- `u MAE = 2.2671e-01`
- `p MAE = 1.6103e+00`
- `c MAE = 5.1515e-02`

这说明它距离 OpenFOAM 还有明显差距，属于“能跑通但还没充分收敛”的状态。

### 7.2 参数化模型

如果比较的是参数化模型，并且使用：

```bash
--inlet-vel 1.5
```

那么这份报告回答的是：

- 参数化模型在官方基线工况下，是否学到了正确解
- 它和 baseline 模型相比是更好还是更差

例如你当前一份参数化模型结果里：

- `u MAE = 1.0361e-02`
- `p MAE = 1.2400e-01`
- `c MAE = 3.2770e-03`

这说明在 `1.5` 这个工况点上，它已经比 10k 步 baseline 更接近 OpenFOAM。

### 7.3 不要只看单个数字

推荐的读报告顺序是：

1. 先看 `u / p / c` 的 `MAE`
2. 再看 `Rel L2`
3. 再看 `*_spatial.png` 的误差分布
4. 最后看 `*_parity.png` 判断系统偏差

原因是：

- `MAE` 最直观
- `Rel L2` 适合做整体比较
- 空间图能告诉你“误差出在哪”
- parity 图能告诉你“误差是随机的还是系统偏差”

## 8. 常见用法

### 8.1 比较两个不同 checkpoint

你可以对两个输出目录分别跑脚本，再比较两份 `comparison_report.md`。

例如：

```bash
/home/featurize/work/env_conda/nemo/bin/python compare_openfoam_csv.py \
  --run-dir outputs/ckpt_a \
  --out-dir outputs/openfoam_compare_a

/home/featurize/work/env_conda/nemo/bin/python compare_openfoam_csv.py \
  --run-dir outputs/ckpt_b \
  --out-dir outputs/openfoam_compare_b
```

### 8.2 比较参数化模型在某个固定工况的表现

```bash
/home/featurize/work/env_conda/nemo/bin/python compare_openfoam_csv.py \
  --run-dir outputs/param_run \
  --inlet-vel 1.5 \
  --out-dir outputs/openfoam_compare_param_15
```

注意：当前官方 CSV 只对应 `1.5`，所以如果你传别的 `inlet_vel`，从物理上就不是和这份 CSV 对应的工况，结果没有比较意义。

## 9. 常见问题

### 9.1 为什么 baseline 不需要 `--inlet-vel`

因为 baseline 网络输入只有 `(x, y)`，没有 `inlet_vel` 这一维。

### 9.2 为什么 `c` 不是直接用 OpenFOAM 的温度 `T`

因为训练脚本里的 validator 比较的是归一化温度 `c`，不是绝对温度 `T`。  
为了和训练过程保持一致，测试脚本也用了同样的转换：

```python
c = (T - 293.498) / 273.15
```

### 9.3 为什么图像和指标点数不一致

不是指标少算了，而是图像可能做了下采样。  
指标总是基于全部 CSV 点计算。

### 9.4 如果脚本提示找不到 CSV

确认官方 CSV 在：

- [`cases/three_fin_2d/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv`](../cases/three_fin_2d/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv)

或者手动传：

```bash
--csv-path /your/path/to/heat_sink_zeroEq_Pr5_mesh20.csv
```

### 9.5 如果脚本提示找不到 checkpoint

确认 `--run-dir` 下最终能找到这些文件之一：

- `flow_network.0.pth`
- `heat_network.0.pth`

通常它们位于：

- `outputs/<timestamp>/hydra_outputs/`
- `cases/three_fin_2d/outputs/run/`
- `cases/three_fin_2d/outputs/run_param/`

## 10. 推荐工作流

建议后续固定按下面顺序使用这个脚本：

1. 训练结束后，先跑一次 OpenFOAM 对比。
2. 先看 `u / p / c` 的 MAE 是否明显下降。
3. 再看误差空间分布是否仍集中在散热片附近和尾流区。
4. 对参数化模型，固定先评估 `inlet_vel = 1.5`，确保和官方基准工况对齐。
5. 只有在 `1.5` 工况已经表现合理后，再考虑扩展到其它参数点做额外评估。

一句话总结：

`compare_openfoam_csv.py` 是当前仓库里最直接、最可信的“模型对官方参考真值”的定量评估工具，优先级高于只看训练 monitor。
