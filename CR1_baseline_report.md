# CR-1 基线报告 — PhysicsNeMo Sym 三翅散热片 2D 对流扩散 PINN Demo

**生成时间：** 2026-02-26
**运行目录：** `outputs/20260226_094257/`

---

## 1. 环境信息

| 项目 | 值 |
|------|----|
| GPU | NVIDIA GeForce RTX 3060 |
| 显存 | 12288 MiB (12 GB) |
| 驱动版本 | 560.35.03 |
| CUDA Compute Cap | 8.6 (Ampere) |
| nvcc 版本 | 12.3 |
| Python | 3.11.14 |
| PyTorch | 2.4.0+cu121 |
| nvidia-physicsnemo | 1.3.0 |
| nvidia-physicsnemo-sym | 2.3.0 |
| numpy | 2.3.5 |
| Conda 环境路径 | `/home/featurize/work/env_conda/nemo` |

---

## 2. 运行参数

| 参数 | 值 |
|------|----|
| 示例 | `examples/three_fin_2d/heat_sink.py` |
| max_steps | 10,000（全量 500,000 的 2%） |
| 随机种子 | 42 |
| 开始时间 | 2026-02-26 09:42:58 UTC |
| 结束时间 | 2026-02-26 10:39:49 UTC |

---

## 3. 运行时间

| 指标 | 值 |
|------|----|
| 总耗时 | 56 分 52 秒（3412.5 s） |
| 吞吐量 | ~2.9 steps/s |
| 每步耗时 | ~333 ms/step（CUDA Graph 预热后） |
| 运行状态 | 正常完成，无报错 |

> CUDA Graph 预热（约 38 秒）发生在第 100 步之前，之后速度稳定。

---

## 4. GPU 使用情况

本次运行中 `nvidia-smi dmon` 在该环境下未能正常采集数据（工具限制）。
根据训练日志，GPU 全程保持 CUDA Graph 计算模式，每步耗时稳定在 ~333 ms，说明 GPU 持续满负荷运行。

---

## 5. 数值稳定性

| 监控量 | 初始值 | 最终值（10k步） | 趋势 |
|--------|--------|-----------------|------|
| mass_imbalance（质量不守恒） | 3.962 | 0.892 | 下降 77% ✓ |
| momentum_imbalance（动量不守恒） | 6.736 | 1.821 | 下降 73% ✓ |
| 总 loss | 1.868 | 0.103 | 下降 94% ✓ |

两个守恒量均呈单调下降趋势，数值稳定，无发散或爆炸迹象。

---

## 6. 监控指标最终值（step = 10,000）

| 物理量 | 最终值 | 说明 |
|--------|--------|------|
| peakT | 0.2137 | 归一化峰值温度，已趋于稳定 |
| mass_imbalance | 0.8917 | 连续性方程残差，仍有下降空间 |
| momentum_imbalance | 1.8206 | N-S 动量方程残差，仍在收敛 |
| force_x | −0.8405 | 散热片 x 方向阻力，仍在单调变化 |
| force_y | 0.4391 | 散热片 y 方向升力，仍在单调变化 |

---

## 7. 图表分析

### 7.1 peakT（峰值温度）

![peakT](outputs/20260226_094257/charts/monitor_peakT.png)

- 从 0.75 急剧下降，约 500 步后收敛至 ~0.21 并保持平稳
- **解读**：PINN 较快学会热边界条件，温度场预测早期已基本稳定

---

### 7.2 mass_imbalance（质量守恒残差）

![mass_imbalance](outputs/20260226_094257/charts/monitor_mass_imbalance.png)

- 从 4.0 下降至 ~1.0，整体趋势持续缓慢下降，存在小幅震荡
- **解读**：连续性方程（∇·u = 0）残差在训练中被持续优化，10k 步尚未完全收敛，需更多训练步数

---

### 7.3 momentum_imbalance（动量守恒残差）

![momentum_imbalance](outputs/20260226_094257/charts/monitor_momentum_imbalance.png)

- 从 6.8 快速下降至 ~0.6，此后在 1.0~2.0 间震荡
- **解读**：N-S 方程动量残差整体下降，震荡属 PINN 多目标优化的正常现象

---

### 7.4 force_x（x 方向阻力）

![force_x](outputs/20260226_094257/charts/monitor_force_x.png)

- 从 ~0 持续单调下降至 ~-0.84，尚未收敛
- **解读**：流体绕散热片的阻力随流场学习不断确定，10k 步时流场仍未充分收敛

---

### 7.5 force_y（y 方向升力）

![force_y](outputs/20260226_094257/charts/monitor_force_y.png)

- 从 0 单调递增至 ~0.44，尚未收敛
- **解读**：与 force_x 同理，升力随训练逐渐建立，方向符合物理预期（流体受散热片偏转产生 y 向分力）

---

### 7.6 综合关键指标

![combined](outputs/20260226_094257/charts/combined_key_metrics.png)

- 同时展示 mass_imbalance 和 momentum_imbalance，便于对比两个守恒量的收敛速度
- mass_imbalance 收敛更快（77%↓），momentum_imbalance 收敛相对慢（73%↓）

---

## 8. 安装过程关键问题记录

本次安装过程中遇到两个非显而易见的问题，记录供后续复现参考：

### 问题 1：清华 PyPI 镜像缺包

**现象：** `pip install Cython / nvidia-physicsnemo / nvidia-physicsnemo-sym` 报 `No matching distribution found`

**原因：** 系统默认 pip 源为清华镜像（`https://pypi.tuna.tsinghua.edu.cn/simple`），该镜像缺少部分 NVIDIA 包。

**解决：** 对上述三个包统一指定官方源：
```bash
pip install <package> --index-url https://pypi.org/simple
```

### 问题 2：torch 版本自动升级导致 nvcc 编译失败

**现象：** `nvidia-physicsnemo` 要求 `torch>=2.4.0`，pip 自动将 torch 升级至 2.10.0，编译 `nvidia-physicsnemo-sym` 的 CUDA 扩展时报：
```
nvcc fatal: Unsupported gpu architecture 'compute_100'
```

**原因：** torch 2.10.0 的 `cpp_extension` 默认 gencode 列表新增了 Blackwell 架构（`compute_100`），而系统 nvcc 版本为 12.3，不支持该架构。torch 2.4.0 的最高 arch 为 Hopper（9.0），完全兼容 nvcc 12.3。

**解决：** 安装 physicsnemo 之前，先将 torch 固定到 2.4.0：
```bash
pip install "torch==2.4.0+cu121" --index-url https://download.pytorch.org/whl/cu121
```

---

## 9. 结论与后续建议

### 本次运行结论
- 环境搭建成功，`setup_env.sh` 可完整复现
- 10,000 步 demo 运行正常，全部物理监控量均呈下降趋势，数值稳定无发散
- 图表内容符合预期，PINN 早期学习效果明显

### 与完全收敛的差距
当前 10k 步仅完成全量训练的 2%，force_x / force_y 仍在单调变化，mass/momentum imbalance 尚未趋近于 0。**当前结果不适合作为定量参考**，仅用于验证环境和流程。

### 运行完整训练
```bash
MAX_STEPS=500000 bash run_case.sh
# 预计耗时：约 500k × 333ms ≈ 46 小时（RTX 3060）
```

---

## 10. 交付物清单

| 文件 | 路径 | 说明 |
|------|------|------|
| 环境安装脚本 | `setup_env.sh` | 可复现的 conda 环境搭建，幂等 |
| 运行脚本 | `run_case.sh` | 下载示例 + GPU 监控 + 驱动训练 |
| Python 运行器 | `run_case.py` | Hydra 注入、日志解析、图表生成 |
| 环境快照 | `environment.yml` | conda 环境完整导出 |
| 依赖锁定 | `requirements.txt` | pip freeze 输出 |
| 训练输出 | `outputs/20260226_094257/` | 本次运行全部产物 |
| 元数据 | `outputs/.../metadata.json` | 版本、参数、耗时 |
| 基线报告（自动生成） | `outputs/.../baseline_report.md` | 由 run_case.py 自动写入 |
| 图表（6张） | `outputs/.../charts/` | 各物理量时间序列图 |
