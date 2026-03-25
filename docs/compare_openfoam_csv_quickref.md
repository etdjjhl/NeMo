# `compare_openfoam_csv.py` 速查

完整说明见：

- [`docs/compare_openfoam_csv_usage.md`](./compare_openfoam_csv_usage.md)

这份文档只保留日常最常用的信息。

## 1. 这个脚本干什么

把训练好的 PINN checkpoint 和官方 OpenFOAM CSV 做逐点对比，输出：

- `comparison_report.md`
- `u/v/p/c` 的空间误差图
- `u/v/p/c` 的 parity plot

默认官方 CSV：

- [`cases/three_fin_2d/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv`](../cases/three_fin_2d/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv)

## 2. 最常用命令

### 基线模型

```bash
/home/featurize/work/env_conda/nemo/bin/python compare_openfoam_csv.py \
  --run-dir outputs/20260226_094257 \
  --model-type baseline \
  --out-dir outputs/openfoam_compare_base
```

### 参数化模型

```bash
/home/featurize/work/env_conda/nemo/bin/python compare_openfoam_csv.py \
  --run-dir outputs/20260317_033129 \
  --inlet-vel 1.5 \
  --out-dir outputs/openfoam_compare_param
```

### 用最新一次输出

```bash
/home/featurize/work/env_conda/nemo/bin/python compare_openfoam_csv.py \
  --run-dir outputs/latest \
  --inlet-vel 1.5 \
  --out-dir outputs/openfoam_compare_latest
```

## 3. 必记规则

- baseline 模型不用关心 `--inlet-vel`
- param 模型和官方 CSV 对比时，必须用 `--inlet-vel 1.5`
- `--out-dir` 不写时，结果默认输出到当前目录的 `openfoam_compare/`
- 如果自动识别模型类型失败，就手动加：
  - `--model-type baseline`
  - `--model-type param`

## 4. 输出目录里有什么

- `comparison_report.md`
- `u_spatial.png`
- `u_parity.png`
- `v_spatial.png`
- `v_parity.png`
- `p_spatial.png`
- `p_parity.png`
- `c_spatial.png`
- `c_parity.png`

## 5. 报告里最该看什么

优先看这 5 个东西：

1. `u` 的 `MAE`
2. `p` 的 `MAE`
3. `c` 的 `MAE`
4. `Rel L2`
5. `*_spatial.png` 里的误差分布

## 6. 每个指标一句话解释

- `MAE`
  平均绝对误差，越小越好，最直观

- `RMSE`
  对局部大误差更敏感，越小越好

- `Max Abs`
  最大绝对误差，适合看有没有局部坏点

- `Rel L2`
  整体相对误差比例，适合比较不同 checkpoint

- `PINN mean`
  PINN 预测整体均值，只能做粗略 sanity check

- `OpenFOAM mean`
  OpenFOAM 真值整体均值，用来和 `PINN mean` 对照

## 7. 图怎么读

### `*_spatial.png`

三张图：

1. PINN
2. OpenFOAM
3. Difference

重点看第 3 张。  
误差越浅越好；如果某些区域明显偏红/偏蓝，说明这些区域误差集中。

### `*_parity.png`

横轴 OpenFOAM，纵轴 PINN。  
点越贴近虚线 `y = x` 越好。

## 8. 快速判断结果好坏

经验上按这个顺序判断：

1. `u / p / c` 的 `MAE` 是否比旧 checkpoint 更小
2. `Rel L2` 是否下降
3. 空间误差是否只剩局部区域
4. parity 图是否更贴近对角线

## 9. 常见问题

### 报错找不到 CSV

确认文件存在：

- [`cases/three_fin_2d/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv`](../cases/three_fin_2d/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv)

### 报错找不到 checkpoint

确认 `--run-dir` 下最终能找到：

- `flow_network.0.pth`
- `heat_network.0.pth`

### 参数化模型能不能用别的 `inlet_vel`

能推理，但不能和这份官方 CSV 做有效对比。  
因为这份 CSV 对应的就是 `1.5` 工况。

## 10. 一句话记忆

如果你只记一件事：

> 对参数化模型，用官方 CSV 做对比时，固定传 `--inlet-vel 1.5`，优先看 `u/p/c` 的 `MAE` 和误差图。
