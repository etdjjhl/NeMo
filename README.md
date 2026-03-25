# PhysicsNeMo Sym — 三翅散热片 2D PINN Demo

基于 [NVIDIA PhysicsNeMo Sym](https://github.com/NVIDIA/modulus-sym) 的物理信息神经网络（PINN）示例，求解二维三翅散热片的耦合流-热问题。**无需仿真数据**，仅通过 PDE 残差约束训练神经网络。

支持两种模式：
- **baseline** — 固定入口流速的标准 PINN
- **param** — 参数化 PINN：将入口流速 `inlet_vel`（1.0~2.5 m/s）作为网络额外输入，一次训练覆盖整个速度区间

---

## 物理问题

```
入口 (抛物线流速) ──► [散热片 ×3] ──► 出口 (p=0)
                        ↑ 高温壁面 (350 K)
```

求解区域：5×1 矩形通道内嵌三个矩形翅片（热源）。两个神经网络同时训练：

| 网络 | 输入 (baseline) | 输入 (param) | 输出 | 控制方程 |
|------|----------------|-------------|------|---------|
| `flow_network` | x, y | x, y, **inlet_vel** | u, v, p | Navier-Stokes + 零方程湍流模型 |
| `heat_network` | x, y | x, y, **inlet_vel** | c (温度) | 对流扩散方程 |

---

## 快速开始

### 1. 环境安装（首次）

```bash
bash setup_env.sh           # 创建 conda 环境，安装依赖，验证导入
bash setup_env.sh --force   # 强制重建环境
```

> **注意：** `nvidia-physicsnemo` 系列包仅在 PyPI 官方源（`https://pypi.org/simple`）可用，清华镜像缺包。

#### 自定义 conda 环境路径

`run_case.sh` 开头的 `ENV_DIR` 变量决定使用哪个 Python 环境：

```bash
ENV_DIR="/opt/conda/envs/nemo" bash run_case.sh
```

### 2. 运行训练

```bash
# baseline 快速验证（10k 步，~57 分钟，RTX 3060）
bash run_case.sh

# baseline 完整训练（500k 步）
MAX_STEPS=500000 bash run_case.sh

# 参数化 PINN 快速验证
MODE=param bash run_case.sh

# 参数化 PINN 完整训练（推荐 300k 步）
MODE=param MAX_STEPS=300000 bash run_case.sh

# 使用本地快速磁盘加速 I/O
LOCAL_DISK=/home/featurize/data MAX_STEPS=50000 bash run_case.sh
```

### 3. 直接调用 Python（灵活参数）

```bash
/home/featurize/work/env_conda/nemo/bin/python run_case.py \
    --case-dir cases/three_fin_2d \
    --out-dir  outputs/my_run \
    --max-steps 10000 \
    --seed 42 \
    --mode baseline          # 或 param

# 本地磁盘加速
python run_case.py \
    --case-dir cases/three_fin_2d \
    --out-dir  outputs/my_run \
    --max-steps 50000 \
    --mode param \
    --local-disk /home/featurize/data
```

### 4. 可视化推理结果（参数化模式）

训练完成后，可以用 `visualize.py` 对不同入口流速生成场图：

```bash
cd cases/three_fin_2d
python visualize.py
```

### 5. 从检查点恢复训练

```bash
bash restore_checkpoint.sh   # 从归档 outputs 恢复检查点后继续训练
```

---

## 输出结构

每次运行生成一个带时间戳的目录，按模式分隔：

```
outputs/
├── baseline/
│   ├── YYYYMMDD_HHMMSS/
│   │   ├── metadata.json          # 版本、参数、GPU 信息、耗时
│   │   ├── gpu_stats.log          # nvidia-smi 采样（每 5s）
│   │   ├── baseline_report.md     # 自动生成的训练报告
│   │   ├── charts/                # PNG 图表（各监控量 + 综合 + GPU）
│   │   └── hydra_outputs/
│   │       └── monitors/*.csv     # PhysicsNeMo 训练监控时间序列
│   └── latest -> YYYYMMDD_HHMMSS  # 符号链接指向最近一次运行
└── param/
    ├── YYYYMMDD_HHMMSS/
    │   └── ...                    # 同上结构
    └── latest -> YYYYMMDD_HHMMSS
```

---

## 项目结构

```
.
├── run_case.sh                        # 主入口：环境检查 → 下载 → GPU 监控 → 训练
├── run_case.py                        # Python 运行器（参数注入、日志解析、图表生成）
├── setup_env.sh                       # conda 环境一键搭建
├── restore_checkpoint.sh              # 从归档 outputs 恢复检查点继续训练
├── compare_openfoam_csv.py            # OpenFOAM 对比验证工具
├── requirements.txt                   # pip freeze 依赖锁定
├── environment.yml                    # conda 环境导出
├── cases/three_fin_2d/
│   ├── heat_sink.py                   # baseline PINN 脚本（NVIDIA 官方）
│   ├── heat_sink_param.py             # 参数化 PINN 脚本（inlet_vel 作为输入）
│   ├── heat_sink_inverse.py           # 反问题 PINN 脚本
│   ├── visualize.py                   # 推理可视化（多流速场图对比）
│   ├── conf/config.yaml               # baseline Hydra 配置
│   ├── conf_param/config.yaml         # 参数化 Hydra 配置
│   └── conf_inverse/                  # 反问题配置
├── reqs/                              # 需求文档（req0~req3）
├── docs/
│   ├── study.md                       # PINN 学习笔记
│   ├── PINN_Study_Notes.pdf           # 英文版 PINN 学习笔记
│   └── PINN学习笔记.pdf               # 中文版 PINN 学习笔记
├── outputs/                           # 所有运行产物（.gitignore 排除大文件）
└── CR1_baseline_report.md             # 首次基线运行报告（10k 步）
```
