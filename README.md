# PhysicsNeMo Sym — 三翅散热片 2D PINN Demo

基于 [NVIDIA PhysicsNeMo Sym](https://github.com/NVIDIA/modulus-sym) 的物理信息神经网络（PINN）示例，求解二维三翅散热片的耦合流-热问题。**无需仿真数据**，仅通过 PDE 残差约束训练神经网络。

---

## 物理问题

```
入口 (抛物线流速) ──► [散热片 ×3] ──► 出口 (p=0)
                        ↑ 高温壁面 (350 K)
```

求解区域：5×1 矩形通道内嵌三个矩形翅片（热源）。两个神经网络同时训练：

| 网络 | 输入 | 输出 | 控制方程 |
|------|------|------|---------|
| `flow_network` | x, y | u, v, p | Navier-Stokes + 零方程湍流模型 |
| `heat_network` | x, y | c (温度) | 对流扩散方程 |

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
# run_case.sh 第 10 行
ENV_DIR="/home/featurize/work/env_conda/nemo"
PYTHON="${ENV_DIR}/bin/python"   # 自动由 ENV_DIR 推导，无需单独修改
```

如果你的环境安装在其他路径（例如换了机器或用户目录不同），通过环境变量传入即可：

```bash
ENV_DIR="/opt/conda/envs/nemo" bash run_case.sh
```

脚本使用 `${ENV_DIR:-默认路径}` 语法，未设置时才使用默认值，与 `MAX_STEPS` 的行为一致。

### 2. 运行训练

```bash
# 快速验证（10k 步，~57 分钟，RTX 3060）
bash run_case.sh

# 完整训练（500k 步，~46 小时，RTX 3060）
MAX_STEPS=500000 bash run_case.sh
```

### 3. 直接调用 Python（灵活参数）

```bash
/home/featurize/work/env_conda/nemo/bin/python run_case.py \
    --case-dir cases/three_fin_2d \
    --out-dir  outputs/my_run \
    --max-steps 10000 \
    --seed 42
```

---

## 输出结构

每次运行生成一个带时间戳的目录：

```
outputs/YYYYMMDD_HHMMSS/
├── metadata.json          # 版本、参数、GPU 信息、耗时
├── gpu_stats.log          # nvidia-smi 采样（每 5s）
├── baseline_report.md     # 自动生成的训练报告
├── charts/                # PNG 图表（各监控量 + 综合 + GPU）
└── hydra_outputs/
    └── monitors/*.csv     # PhysicsNeMo 训练监控时间序列
```

`outputs/latest` 符号链接指向最近一次运行。

---

## 项目结构

```
.
├── run_case.sh                        # 主入口：环境检查 → 下载 → GPU 监控 → 训练
├── run_case.py                        # Python 运行器（参数注入、日志解析、图表生成）
├── setup_env.sh                       # conda 环境一键搭建
├── requirements.txt                   # pip freeze 依赖锁定
├── environment.yml                    # conda 环境导出
├── cases/three_fin_2d/
│   ├── heat_sink.py                   # NVIDIA 官方 PINN 脚本（不修改）
│   └── conf/config.yaml               # Hydra 配置（batch size、LR 调度）
├── outputs/                           # 所有运行产物（.gitignore 排除大文件）
├── docs/
│   ├── study.md                       # PINN 学习笔记
│   ├── PINN_Study_Notes.pdf           # 英文版 PINN 学习笔记
│   └── PINN学习笔记.pdf               # 中文版 PINN 学习笔记
└── CR1_baseline_report.md             # 首次基线运行报告（10k 步）
```
