# `heat_sink_param.py` 参数化实现审查

## 1. 目标与结论

本文档审查 [`cases/three_fin_2d/heat_sink_param.py`](../cases/three_fin_2d/heat_sink_param.py) 相对 [`cases/three_fin_2d/heat_sink.py`](../cases/three_fin_2d/heat_sink.py) 的参数化改造是否完整、是否正确，并结合 NVIDIA 官方文档与上游示例源码，总结 PhysicsNeMo 中“把非参数化 case 改成参数化 case”通常需要做哪些事情。

先给结论：

- 你的 `heat_sink_param.py` 在方法论上是正确的。
- 对于“入口风速参数化”这个问题，核心必做步骤基本都做了，没有看到明显的硬错误。
- 你当前实现没有照搬 3D 示例里的“几何参数化”步骤，这不是遗漏，而是因为这里参数化的是入口工况，不是散热片几何。
- 如果训练效果不理想，当前更可疑的是训练预算、监控覆盖面、参数空间带来的优化难度，而不是参数化思路本身错误。

## 2. 审查依据

本次审查综合了以下材料，审查日期为 2026-03-17：

- 2D 基线教程：<https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/foundational/scalar_transport.html>
- 3D 非参数化教程（Conjugate Heat Transfer）：<https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/advanced/conjugate_heat_transfer.html>
- 3D 参数化教程（Parameterized 3D Heat Sink）：<https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/advanced/parametrized_simulations.html>
- 官方 GitHub 示例目录：<https://github.com/NVIDIA/physicsnemo-sym/tree/main/examples/three_fin_3d>
- 重点源码路径：
  - `examples/three_fin_3d/three_fin_geometry.py`
  - `examples/three_fin_3d/three_fin_flow.py`
  - `examples/three_fin_3d/three_fin_thermal.py`
- 本地基线与参数化实现：
  - [`cases/three_fin_2d/heat_sink.py`](../cases/three_fin_2d/heat_sink.py)
  - [`cases/three_fin_2d/heat_sink_param.py`](../cases/three_fin_2d/heat_sink_param.py)

## 3. PhysicsNeMo 中参数化 case 的通用改法

### 3.1 先区分“参数化的是什么”

官方 3D 参数化示例参数化的是散热片几何尺寸，例如长度、高度、厚度。这种情况下，几何对象本身会随参数变化，因此几何定义、采样、约束、验证和推理都要围绕参数化几何展开。

而你的 2D case 参数化的是入口风速 `inlet_vel`。这属于“工况参数”或“边界条件参数”，不是“几何参数”。因此：

- 几何形状本身不需要改。
- PDE 形式本身也不需要因为这个参数而改写。
- 需要改的是网络输入、采样点携带的参数、边界条件表达式，以及任何依赖入口流量的目标量。

这是本次审查里最重要的一点。`Parameterized 3D Heat Sink` 和你的方法是同一类思想，但不是逐行一一对应。

### 3.2 参数化 case 的最小闭环

结合官方教程和源码，一个非参数化 case 改成参数化 case，至少要检查下面这些点：

1. 定义参数符号与参数范围  
   使用 `Parameter(...)` 和 `Parameterization(...)` 描述参数空间。

2. 把参数加入网络输入  
   如果解 `u, v, p, c` 依赖该参数，网络输入就必须从空间坐标扩展为“空间坐标 + 参数”。

3. 让训练采样显式覆盖参数空间  
   相关的 `PointwiseBoundaryConstraint`、`PointwiseInteriorConstraint`、`IntegralBoundaryConstraint` 等需要带上 `parameterization=...`，否则训练时不会真正看到该参数维度。

4. 把边界条件里的常量改成参数表达式  
   如果入口剖面原来写死为某个常数，参数化后要改成由符号参数驱动的表达式。

5. 检查积分约束、派生目标量是否也要改  
   如果某个目标值本质上来自入口条件积分，参数化后不能继续写死为常数，必须同步改成参数函数。

6. 检查 validator、inferencer、monitor 是否还能工作  
   参数化网络要求输入包含参数值。  
   如果参考数据只对应某一个工况，则 validator/monitor 固定在一个代表性参数点是合理做法。  
   如果以后增加 inferencer，同样必须显式给出参数值。

7. 重新评估训练预算  
   参数空间增加后，模型学习的是一个解族，而不是单工况解。步数、batch size、监控策略通常都要提高。

### 3.3 几何参数化与工况参数化的区别

官方 3D 参数化示例比你的 2D 实现多出的内容，主要来自“几何本身会变”：

- 几何对象带参数化定义。
- 不同参数值下，壁面、流道、散热片位置都在变化。
- 推理和验证时，需要固定若干参数组合去采样和可视化。

这些步骤在你的 2D 入口风速参数化里不是必须项。因为你的通道和散热片几何完全没变，变化的只是入口边界条件。

## 4. `heat_sink.py -> heat_sink_param.py` 逐项对照

### 4.1 参数定义

你做了：

```python
inlet_vel_sym = Parameter("inlet_vel")
inlet_vel_range = (1.0, 2.5)
param_ranges = Parameterization({inlet_vel_sym: inlet_vel_range})
```

结论：正确，且是必做步骤。

### 4.2 网络输入维度

基线版本：

```python
input_keys=[Key("x"), Key("y")]
```

参数化版本：

```python
input_keys=[Key("x"), Key("y"), Key("inlet_vel")]
```

结论：正确，且是必做步骤。  
如果这一步漏掉，网络就无法学习“不同风速下的不同解”。

### 4.3 几何定义

`channel`、`heat_sink`、`geo` 都没有改成参数化几何。

结论：正确，不是遗漏。  
原因是 `inlet_vel` 不改变几何，只改变边界条件与对应解。

### 4.4 入口边界条件

基线版本入口抛物线高度是固定常量 `1.5`，参数化版本改成：

```python
inlet_parabola = parabola(
    y, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel_sym
)
```

并且入口约束增加了：

```python
parameterization=param_ranges
```

结论：正确，且是必做步骤。

### 4.5 其他边界条件与 interior 约束

你给下面这些约束都加了 `parameterization=param_ranges`：

- `outlet`
- `hs_wall`
- `channel_wall`
- `interior_flow`
- `interior_heat`

结论：这是正确的，而且很关键。  
虽然这些条件本身的目标值未必直接依赖 `inlet_vel`，但网络现在接收的是 `(x, y, inlet_vel)`，训练采样必须覆盖整个参数空间，否则参数维度只在入口出现，网络会学得不完整。

### 4.6 积分连续性约束

基线版本：

```python
outvar={"normal_dot_vel": 1}
```

参数化版本：

```python
outvar={"normal_dot_vel": (2 / 3) * inlet_vel_sym}
```

并把积分约束的参数空间改成：

```python
Parameterization({
    x_pos: channel_length,
    inlet_vel_sym: inlet_vel_range,
})
```

结论：这是正确的，而且是最容易漏掉但你没有漏掉的一步。

原因如下：

- 入口抛物线速度剖面高度被参数化后，截面总流量不再是常数。
- 原基线 case 中 `inlet_vel = 1.5`，该抛物线在 `[-0.5, 0.5]` 上的积分正好是 `1.0`，所以原脚本把目标写成常量 `1` 是成立的。
- 参数化后，目标流量必须跟着 `inlet_vel` 一起变，否则会把入口边界和积分连续性约束互相打架。

这一步是你当前实现最关键、也最说明思路正确的地方。

### 4.7 Validator

你做了两件事：

1. 保留原 OpenFOAM 参考数据作为真值来源。
2. 给 `openfoam_invar_numpy` 增加固定列：

```python
openfoam_var["inlet_vel"] = np.full((n_pts, 1), 1.5)
```

结论：正确。

原因是现有参考数据只对应单一工况，参数化网络做 validator 时，必须显式告诉网络“这批数据属于哪个参数点”。固定到 `1.5` 与基线工况一致，是合理做法。

### 4.8 Monitor

你将 monitor 采样固定到：

```python
fixed_params = Parameterization({inlet_vel_sym: 1.5})
```

并用在：

- `geo.sample_interior(...)`
- `heat_sink.sample_boundary(...)`

结论：正确，也符合官方参数化示例常见做法。

原因是参数化模型训练时，如果 monitor 不固定到某个代表工况，那么曲线的物理意义会变得不稳定，不利于纵向比较。

### 4.9 Config 调整

`conf_param/config.yaml` 相比基线版本：

- `max_steps` 从 `500000` 提高到 `600000`
- 多个 batch size 翻倍

结论：方向正确，但是否足够需要实验验证。  
参数维度增加后，训练难度上升是正常现象。单纯“逻辑正确”不等于“预算已经足够”。

### 4.10 当前没有做的事，哪些是遗漏，哪些不是

#### 不是遗漏

- 没有把几何对象改成参数化几何  
  不是遗漏，因为你参数化的不是几何。

- 没有改 PDE 本体  
  不是遗漏，因为 `inlet_vel` 是边界条件参数，不是方程系数变化。

- 没有增加 inferencer  
  不是遗漏，因为原始 2D case 也没有 inferencer。只是如果将来补推理导出，就必须给 inferencer 加上固定 `inlet_vel`。

#### 可优化但不是硬错误

- 目前 monitor 只固定在 `1.5` 一个参数点  
  这有利于和基线对比，但不利于看模型在参数区间端点的表现。

- 当前 validator 也只有 `1.5` 一个工况  
  如果没有额外 CFD 参考数据，这是现实限制，不是代码逻辑问题。

## 5. 最终审查结论

### 5.1 是否“完整且正确”

结论：基本完整，且方法正确。

对“入口风速参数化”这个问题，以下核心闭环都已经存在：

- 参数已定义
- 网络已接收参数输入
- 入口边界条件已参数化
- 相关约束采样已覆盖参数空间
- 与入口流量相关的积分约束已同步参数化
- validator 已固定到参考工况
- monitor 已固定到代表工况
- 训练预算已做上调

因此，从实现逻辑上看，`heat_sink_param.py` 并没有缺失明显的关键步骤。

### 5.2 最值得强调的判断

你最初的理解是“虽然 3D 官方示例参数化的是散热片尺寸，而我这里参数化的是入口风速，但方法应该是相同的”。

这个判断只能说“框架层面相同，具体改动项不完全相同”：

- 相同的是：都要把参数纳入网络输入与采样流程，让模型学习参数到解的映射。
- 不同的是：几何参数化要改几何与几何采样；入口风速参数化不需要。

所以，不能因为 2D 代码里没有出现 3D 示例中的参数化几何逻辑，就把它判成遗漏。

### 5.3 为什么训练效果仍可能不理想

如果 `heat_sink_param.py` 的效果不如预期，更可能的原因包括：

- 参数空间引入后，训练目标从单解变成解族，优化难度明显上升。
- 单一 monitor 固定在 `1.5`，可能掩盖了端点参数 `1.0`、`2.5` 的误差。
- 现有 validator 只有一个工况，无法直接判断模型在整个参数空间上的泛化质量。
- 步数和 batch 虽然上调了，但仍可能不足以支撑同等精度。

这些属于“训练与评估设计问题”，不是“参数化步骤写错了”。

## 6. 建议的后续检查顺序

以下建议按优先级排序，目的是先区分“实现错误”和“训练不足”。

### 6.1 做一个最小一致性实验

把参数化模型的训练工况临时收缩到单点 `inlet_vel = 1.5`，或者很窄的区间，例如 `[1.45, 1.55]`，再与基线 `heat_sink.py` 做对比。

预期：

- 如果此时结果接近基线，说明参数化实现本身没问题，主要矛盾是参数空间扩大后的训练难度。
- 如果此时仍显著劣化，再回头查训练配置或实现细节。

### 6.2 补多参数点监控

建议在不改变现有 `1.5` monitor 的前提下，额外增加几个固定参数点的 monitor，例如：

- `1.0`
- `1.5`
- `2.5`

这样可以快速判断模型是“整体偏弱”，还是“只在区间端点退化”。

### 6.3 补多参数点推理或导出

如果后续要看流场/温度场，建议增加若干固定参数值的 inferencer 或导出脚本。  
参数化网络在推理阶段同样必须显式输入 `inlet_vel`，否则无法得到可解释的结果。

### 6.4 继续提高训练预算

如果前面几步确认实现正确，下一步再考虑：

- 增加训练步数
- 继续提高 interior 与 inlet 相关 batch
- 必要时增加网络容量

这类动作应放在“逻辑正确已经确认”之后做，否则会把实现问题和算力问题混在一起。

### 6.5 如果仍不稳定，再考虑课程式训练

例如先在较窄速度区间训练，再扩展到完整区间 `1.0 ~ 2.5`。  
这不是官方示例的硬要求，但对工况参数化问题常常有效。

## 7. 一句话总结

`heat_sink_param.py` 相对 `heat_sink.py` 的参数化改造，核心步骤是完整且正确的；当前最该怀疑的不是“少改了什么必做项”，而是“参数空间扩大后，训练和评估是否还足够”。
