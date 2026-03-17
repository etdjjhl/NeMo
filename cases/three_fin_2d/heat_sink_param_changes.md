## heat_sink_param.py 改动说明
基于 `heat_sink.py`，将入口风速 `inlet_vel` 从固定标量改为参数化变量，使单个训练模型覆盖 **1.0 ~ 2.5 m/s** 的完整风速空间。
### 1. 配置目录
| | heat_sink.py | heat_sink_param.py |
|---|---|---|
| `config_path` | `"conf"` | `"conf_param"` |
### 2. inlet_vel 参数化
**原版**：`inlet_vel = 1.5`（硬编码标量）
**新版**：
```python
inlet_vel_sym = Parameter("inlet_vel")
inlet_vel_range = (1.0, 2.5)
param_ranges = Parameterization({inlet_vel_sym: inlet_vel_range})
```
### 3. 网络输入维度
两个网络均增加 `inlet_vel` 作为输入坐标：
```python
# 原版
input_keys=[Key("x"), Key("y")]
# 新版
input_keys=[Key("x"), Key("y"), Key("inlet_vel")]
```
### 4. 所有约束加入 parameterization
每个 `PointwiseBoundaryConstraint` / `PointwiseInteriorConstraint` 均加入 `parameterization=param_ranges`，涵盖：inlet、outlet、heat_sink_wall、channel_wall、interior_flow、interior_heat。
入口抛物面剖面的 `height` 也从 `1.5` 改为 `inlet_vel_sym`：
```python
inlet_parabola = parabola(y, ..., height=inlet_vel_sym)
```
### 5. 积分连续性约束目标值
| | 原版 | 新版 |
|---|---|---|
| `normal_dot_vel` 目标 | `1`（固定） | `(2/3) * inlet_vel_sym` |
| `parameterization` | `{x_pos: channel_length}` | `{x_pos: channel_length, inlet_vel_sym: inlet_vel_range}` |
推导：抛物面在 [-0.5, 0.5] 上积分 = (2/3) × inlet_vel，inlet_vel=1.5 时恰好等于原来的 1.0。
### 6. 验证器（Validator）
加入 `inlet_vel=1.5` 的固定列，与 OpenFOAM 参考工况对齐：
```python
openfoam_var["inlet_vel"] = np.full((n_pts, 1), 1.5)
# invar 中增加 "inlet_vel"
```
### 7. 监控器（Monitor）采样固定在 inlet_vel=1.5
为使训练曲线可比，三个 Monitor 的采样点锁定到参考风速：
```python
fixed_params = Parameterization({inlet_vel_sym: 1.5})
geo.sample_interior(100, parameterization=fixed_params)
heat_sink.sample_boundary(100, parameterization=fixed_params)
```
### 总结
| 改动点 | 原版 | 新版 |
|---|---|---|
| inlet_vel | 固定 1.5 | 参数 1.0~2.5 |
| 网络输入维度 | (x, y) | (x, y, inlet_vel) |
| 约束 parameterization | 无 | 全部加入 param_ranges |
| 积分约束目标 | 1 | (2/3)×inlet_vel_sym |
| 验证器 invar | (x, y, sdf) | 额外加 inlet_vel=1.5 |
| Monitor 采样 | 直接采样 | 固定 inlet_vel=1.5 |
