# 参数化 PINN 审查报告

> **审查对象**: `heat_sink_param.py`（参数化入口风速）基于 `heat_sink.py`（固定风速）的改动
> **审查结论**: **实现正确且完整**，9 项检查全部通过 ✅
> **日期**: 2026-03-17

---

## Part 1: NVIDIA 官方参数化方法论总结

### 官方案例对比

NVIDIA PhysicsNeMo Sym 提供了两个典型案例的对比路径：

| 维度 | 非参数化案例 (Conjugate Heat Transfer) | 参数化案例 (Parameterized 3D Heat Sink) |
|------|---------------------------------------|----------------------------------------|
| 参数类型 | 固定几何 + 固定边界条件 | 散热片高度/间距等几何参数可变 |
| 网络输入 | `(x, y)` 或 `(x, y, z)` | `(x, y, z, fin_height, fin_length, ...)` |
| 几何构造 | 固定尺寸 | 需传入 `parameterization=` 使几何随参数变化 |
| 约束采样 | 每次采样固定域 | 每次采样时从参数范围中随机抽取一组参数值 |
| 模型能力 | 仅能推断单一工况 | 一个模型覆盖整个参数空间 |

### 两种参数化类型的关键区别

| | 几何参数化 | 边界条件参数化（本案例） |
|---|-----------|----------------------|
| 参数示例 | 散热片高度、间距 | 入口风速 `inlet_vel` |
| 几何是否变化 | 是 — 参数改变形状 | 否 — 几何固定不变 |
| 几何构造函数是否传 `parameterization=` | **必须** — 否则采样的几何形状不会随参数变化 | **不需要** — 几何形状不依赖参数 |
| 约束是否传 `parameterization=` | 必须 | 必须 |

### 9 步参数化改动方法论

从 NVIDIA 官方非参数化→参数化的改动中，提炼出以下 9 步通用方法：

1. **定义参数** — 使用 `Parameter()` 创建 sympy 符号，用 `Parameterization()` 定义取值范围
2. **网络 `input_keys` 加入参数** — 两个网络（flow_net / heat_net）都需添加 `Key("param_name")`
3. **几何构造传入 `parameterization=`** — 仅几何参数化时需要；边界条件参数化不需要
4. **所有约束加入 `parameterization=`** — 每个 `PointwiseBoundaryConstraint` / `PointwiseInteriorConstraint` / `IntegralBoundaryConstraint`
5. **边界条件表达式符号化** — 原来的硬编码数值改为含参数符号的表达式（如 `parabola(..., height=inlet_vel_sym)`）
6. **积分约束目标值符号化** — 目标值从常数改为参数表达式，并合并多组参数字典
7. **验证器 invar 加入固定参数值** — 验证数据对应特定参数值，需在 invar 中注入该值
8. **监控器采样传入固定参数** — `sample_interior()` / `sample_boundary()` 传入 `parameterization=fixed_params`
9. **配置增大 batch_size / max_steps** — 参数空间更大，需要更多采样点和训练步数

---

## Part 2: 参数化改动检查清单

| # | 改动类别 | 适用范围 | 说明 |
|---|---------|---------|------|
| 1 | 参数定义 (`Parameter` + `Parameterization`) | 通用 | 定义参数符号及其范围 |
| 2 | 网络 `input_keys` 加入参数 Key | 通用 | 让神经网络接收参数作为额外输入 |
| 3 | 几何构造函数传入 `parameterization=` | **仅几何参数化** | 使几何形状随参数变化 |
| 4 | 所有约束加入 `parameterization=` | 通用 | 使约束采样覆盖参数空间 |
| 5 | 边界条件表达式符号化 | 通用 | 边界值从常量变为参数的函数 |
| 6 | 积分约束目标值符号化 + 合并参数字典 | 通用 | 积分目标随参数变化 |
| 7 | 验证器 invar 加入固定参数值 | 通用 | 验证数据对应特定参数工况 |
| 8 | 监控器采样传入固定参数 | 通用 | 监控指标在固定工况下可比 |
| 9 | 配置增大 batch_size / max_steps | 通用 | 覆盖更大的参数空间 |

---

## Part 3: `heat_sink_param.py` 逐项对比

### 检查项 1: 参数定义 ✅

**原版** (`heat_sink.py:56`):
```python
inlet_vel = 1.5
```

**新版** (`heat_sink_param.py:65-67`):
```python
inlet_vel_sym = Parameter("inlet_vel")
inlet_vel_range = (1.0, 2.5)
param_ranges = Parameterization({inlet_vel_sym: inlet_vel_range})
```

**判定**: ✅ 正确
**备注**: 使用 `Parameter("inlet_vel")` 创建 sympy 符号，范围 `(1.0, 2.5)` m/s，并创建了 `Parameterization` 对象 `param_ranges` 供后续约束复用。原版的 `inlet_vel = 1.5` 被完全替换。

---

### 检查项 2: 网络 `input_keys` 加入参数 Key ✅

**原版** (`heat_sink.py:111-119`):
```python
flow_net = instantiate_arch(
    input_keys=[Key("x"), Key("y")],
    ...
)
heat_net = instantiate_arch(
    input_keys=[Key("x"), Key("y")],
    ...
)
```

**新版** (`heat_sink_param.py:120-129`):
```python
flow_net = instantiate_arch(
    input_keys=[Key("x"), Key("y"), Key("inlet_vel")],
    ...
)
heat_net = instantiate_arch(
    input_keys=[Key("x"), Key("y"), Key("inlet_vel")],
    ...
)
```

**判定**: ✅ 正确
**备注**: 两个网络（flow_net 和 heat_net）都添加了 `Key("inlet_vel")`，使网络能够根据不同的入口风速产生不同的预测结果。

---

### 检查项 3: 几何构造函数传入 `parameterization=` — 不适用 ✅

**原版** (`heat_sink.py:66-101`): 几何构造无 `parameterization` 参数
**新版** (`heat_sink_param.py:73-108`): 几何构造同样无 `parameterization` 参数

**判定**: ✅ 不适用（正确地未传入）
**备注**: 本案例是**边界条件参数化**（入口风速），不是几何参数化。`inlet_vel` 不影响 channel、heat_sink 等几何形状，因此几何构造函数**不应该**传入 `parameterization`。如果错误地传入，会导致几何采样时多出一个无用的维度。`integral_line` 的 `parameterization` 仅包含 `x_pos`（原版就有），与入口风速参数化无关。

---

### 检查项 4: 所有约束加入 `parameterization=` ✅

逐个约束检查：

| 约束名称 | 原版 | 新版 | 状态 |
|---------|------|------|------|
| inlet (`heat_sink_param.py:153`) | 无 | `parameterization=param_ranges` | ✅ |
| outlet (`heat_sink_param.py:163`) | 无 | `parameterization=param_ranges` | ✅ |
| heat_sink_wall (`heat_sink_param.py:173`) | 无 | `parameterization=param_ranges` | ✅ |
| channel_wall (`heat_sink_param.py:183`) | 无 | `parameterization=param_ranges` | ✅ |
| interior_flow (`heat_sink_param.py:199`) | 无 | `parameterization=param_ranges` | ✅ |
| interior_heat (`heat_sink_param.py:212`) | 无 | `parameterization=param_ranges` | ✅ |
| integral_continuity (`heat_sink_param.py:233-236`) | 无 | 合并字典（见检查项6） | ✅ |

**判定**: ✅ 全部 7 个约束都已正确加入参数化
**备注**: 6 个常规约束直接使用 `param_ranges`；积分约束因需合并 `x_pos` 参数，使用了独立的 `Parameterization` 字典。

---

### 检查项 5: 边界条件表达式符号化 ✅

**原版** (`heat_sink.py:136-138`):
```python
inlet_parabola = parabola(
    y, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel
)
```
其中 `inlet_vel = 1.5`（Python float 常量）。

**新版** (`heat_sink_param.py:145-147`):
```python
inlet_parabola = parabola(
    y, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel_sym
)
```
其中 `inlet_vel_sym = Parameter("inlet_vel")`（sympy 符号）。

**判定**: ✅ 正确
**备注**: `parabola(y, a, b, height=H)` 生成表达式 `H * (y-a)(y-b) / ((a+b)/2 - a)((a+b)/2 - b)`。当 `H` 从 float 变为 sympy 符号后，整个表达式自动成为符号表达式，PhysicsNeMo Sym 在采样时会用参数的具体值替换。

---

### 检查项 6: 积分约束目标值符号化 + 合并参数字典 ✅

**原版** (`heat_sink.py:206-214`):
```python
integral_continuity = IntegralBoundaryConstraint(
    ...
    outvar={"normal_dot_vel": 1},
    ...
)
```

**新版** (`heat_sink_param.py:225-237`):
```python
integral_continuity = IntegralBoundaryConstraint(
    ...
    outvar={"normal_dot_vel": (2 / 3) * inlet_vel_sym},
    ...
    parameterization=Parameterization({
        x_pos: channel_length,
        inlet_vel_sym: inlet_vel_range,
    }),
)
```

**判定**: ✅ 正确

**数学推导**（关键验证点）:

入口抛物线速度分布为：
```
u(y) = inlet_vel * parabola(y, -0.5, 0.5)
```

`parabola(y, -0.5, 0.5, height=H)` 的归一化形式为：

```
u(y) = H * 4 * (y + 0.5)(0.5 - y)   [在 y ∈ [-0.5, 0.5] 上]
     = H * 4 * (0.25 - y²)
```

对其在通道宽度上积分：
```
∫_{-0.5}^{0.5} u(y) dy = H * 4 * ∫_{-0.5}^{0.5} (0.25 - y²) dy
                        = H * 4 * [0.25y - y³/3]_{-0.5}^{0.5}
                        = H * 4 * [(0.125 - 0.04167) - (-0.125 + 0.04167)]
                        = H * 4 * [0.08333 - (-0.08333)]
                        = H * 4 * 0.16667
                        = H * 2/3
```

因此 `normal_dot_vel` 目标值 = `(2/3) * inlet_vel_sym` ✅

**验证**: 当 `inlet_vel = 1.5` 时，`(2/3) * 1.5 = 1.0`，与原版的固定值 `1` 完全一致 ✅

**参数字典合并**: `IntegralBoundaryConstraint` 需要同时包含 `x_pos`（积分线位置）和 `inlet_vel_sym`（入口风速）两个参数，因此使用独立的 `Parameterization` 字典合并两者，而非直接复用 `param_ranges`。这是正确的做法。

---

### 检查项 7: 验证器 invar 加入固定参数值 ✅

**原版** (`heat_sink.py:234-238`):
```python
openfoam_invar_numpy = {
    key: value
    for key, value in openfoam_var.items()
    if key in ["x", "y", "sdf"]
}
```

**新版** (`heat_sink_param.py:258-263`):
```python
n_pts = openfoam_var["x"].shape[0]
openfoam_var["inlet_vel"] = np.full((n_pts, 1), 1.5)
openfoam_invar_numpy = {
    key: value
    for key, value in openfoam_var.items()
    if key in ["x", "y", "sdf", "inlet_vel"]
}
```

**判定**: ✅ 正确
**备注**: OpenFOAM 参考数据对应 `inlet_vel = 1.5` 的工况。新版在 invar 中注入了 `inlet_vel = 1.5` 的 numpy 数组（形状 `(n_pts, 1)`），并将 `"inlet_vel"` 加入 invar 的过滤键中。这样参数化网络在验证时能得到正确的参数输入。

---

### 检查项 8: 监控器采样传入固定参数 ✅

**原版** (`heat_sink.py:257-290`):
```python
global_monitor = PointwiseMonitor(
    geo.sample_interior(100),
    ...
)
force = PointwiseMonitor(
    heat_sink.sample_boundary(100),
    ...
)
peakT = PointwiseMonitor(
    heat_sink.sample_boundary(100),
    ...
)
```

**新版** (`heat_sink_param.py:285-321`):
```python
fixed_params = Parameterization({inlet_vel_sym: 1.5})

global_monitor = PointwiseMonitor(
    geo.sample_interior(100, parameterization=fixed_params),
    ...
)
force = PointwiseMonitor(
    heat_sink.sample_boundary(100, parameterization=fixed_params),
    ...
)
peakT = PointwiseMonitor(
    heat_sink.sample_boundary(100, parameterization=fixed_params),
    ...
)
```

**判定**: ✅ 正确
**备注**:
- 创建了 `fixed_params = Parameterization({inlet_vel_sym: 1.5})` 用于固定监控工况
- 所有 3 个监控器的 `sample_interior()` / `sample_boundary()` 都传入了 `parameterization=fixed_params`
- 固定在 `inlet_vel = 1.5` 使监控指标在训练过程中可比，且与原版对应同一工况

---

### 检查项 9: 配置增大 batch_size / max_steps ✅

**原版** (`conf/config.yaml`):
```yaml
training:
  max_steps: 500000

batch_size:
  inlet: 64
  outlet: 64
  hs_wall: 500
  channel_wall: 2500
  interior_flow: 4800
  interior_heat: 4800
  integral_continuity: 128
  num_integral_continuity: 4
```

**新版** (`conf_param/config.yaml`):
```yaml
training:
  max_steps: 600000

batch_size:
  inlet: 128
  outlet: 128
  hs_wall: 1000
  channel_wall: 5000
  interior_flow: 9600
  interior_heat: 9600
  integral_continuity: 256
  num_integral_continuity: 4
```

**判定**: ✅ 正确

**逐项对比**:

| 参数 | 原版 | 新版 | 倍率 |
|------|------|------|------|
| max_steps | 500,000 | 600,000 | 1.2× |
| inlet | 64 | 128 | 2× |
| outlet | 64 | 128 | 2× |
| hs_wall | 500 | 1,000 | 2× |
| channel_wall | 2,500 | 5,000 | 2× |
| interior_flow | 4,800 | 9,600 | 2× |
| interior_heat | 4,800 | 9,600 | 2× |
| integral_continuity | 128 | 256 | 2× |
| num_integral_continuity | 4 | 4 | 1× |

**备注**: 所有 batch_size 全面翻倍（2×），`max_steps` 增加 20%（500K → 600K）。`num_integral_continuity` 保持不变。新版还添加了 `custom.parameterized: true` 标记。

---

## Part 4: 结论与训练调优建议

### 结论

`heat_sink_param.py` 相对于 `heat_sink.py` 的参数化改动 **正确且完整**。

| 检查项 | 状态 |
|--------|------|
| 1. 参数定义 | ✅ |
| 2. 网络 input_keys | ✅ |
| 3. 几何 parameterization | ✅ 不适用（正确） |
| 4. 所有约束 parameterization | ✅ (7/7) |
| 5. 边界条件符号化 | ✅ |
| 6. 积分目标符号化 + 参数合并 | ✅ |
| 7. 验证器固定参数 | ✅ |
| 8. 监控器固定参数 | ✅ (3/3) |
| 9. 配置 batch_size/max_steps | ✅ |

无遗漏步骤。可以放心开始长时间训练。

### 训练调优建议（若效果不理想时参考）

1. **训练步数不足**
   `max_steps=600,000` 仅比原版多 20%，但参数空间从 1 个点扩展到一个连续区间。若收敛不充分，可考虑增至 `1,000,000` 或更多。

2. **网络容量未增加**
   两个网络仍使用默认的 `fully_connected` 配置。参数空间维度增加后，可考虑：
   - 增大层宽（如 512 → 768）
   - 增加层数（如 6 层 → 8 层）

3. **学习率衰减策略未调整**
   `decay_rate=0.95`、`decay_steps=5000` 与原版相同。更长的训练可能需要：
   - 减缓衰减（如 `decay_rate=0.97`）
   - 增大 `decay_steps`（如 `8000`）

   以避免学习率过早衰减到极小值。

4. **积分连续性采样数未增加**
   `num_integral_continuity=4` 保持不变。在参数化场景下，每批次只采样 4 条积分线，可能不足以覆盖不同 `inlet_vel` 下的流量守恒。可考虑增至 `8` 或 `12`。

5. **参数范围设计**
   当前范围 `(1.0, 2.5)` 跨度为原值 `1.5` 的 ±67%，是合理的。若需更大范围，建议先在子区间上验证收敛性。
