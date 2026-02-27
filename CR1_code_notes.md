# CR-1 代码说明文档

本文档说明 CR-1 中各脚本的来源、职责与关键实现细节。

---

## 1. heat_sink.py — 未修改的官方示例

### 来源

从 NVIDIA 官方仓库 `NVIDIA/modulus-sym` 通过 git sparse-checkout 下载，**未做任何修改**。

```
仓库：https://github.com/NVIDIA/modulus-sym
路径：examples/three_fin_2d/heat_sink.py
下载方式：git clone --filter=blob:none --depth=1 --sparse（避免克隆 2GB+ 完整仓库）
```

MD5 校验（本地 = 上游）：`a53ac63b361f027401fb1ef35886f1fd`

### 文件做了什么

`heat_sink.py` 是一个使用 PhysicsNeMo Sym 框架编写的 PINN（物理信息神经网络）训练脚本，求解二维三翅散热片的**对流扩散**问题，物理方程包括：

- **Navier-Stokes 方程**（湍流零方程模型）：求解速度场 (u, v) 和压力场 p
- **Advection-Diffusion 方程**：以速度场为输入求解温度场 T

脚本核心流程：

1. 定义几何域（矩形通道 + 三个矩形散热片）
2. 设置边界条件（入口抛物线速度、出口压力、壁面无滑移、散热片温度）
3. 构建两个全连接神经网络（`flow_network`、`heat_network`）
4. 定义物理约束（PDE 残差、边界条件、积分约束）
5. 定义监控器（peakT、mass_imbalance、momentum_imbalance、force_x、force_y）
6. 由 `Solver` 驱动训练循环，使用 Adam + 指数衰减学习率调度

入口为 `@physicsnemo.sym.main(config_path="conf", config_name="config")` 装饰的 `run()` 函数，Hydra 负责加载 `conf/config.yaml` 并注入超参数。

---

## 2. run_case.sh — Shell 编排脚本

### 来源

本项目自行编写，位于 `run_case.sh`。

### 做了什么（5个步骤）

```
[1/5] 预检  →  [2/5] 下载示例  →  [3/5] 准备输出目录  →  [4/5] 启动GPU监控  →  [5/5] 运行训练
```

#### [1/5] 预检

检查 `${ENV_DIR}/bin/python` 是否存在，不存在则直接退出并提示先运行 `setup_env.sh`。

#### [2/5] 下载示例（幂等）

若 `cases/three_fin_2d/heat_sink.py` 已存在则跳过；否则执行 git sparse-checkout：

```bash
git clone --filter=blob:none --no-checkout --depth=1 --sparse <repo> <tmp>
git sparse-checkout set examples/three_fin_2d
git checkout
```

`--filter=blob:none` 只下载目录树，不下载文件内容；`--depth=1` 只取最新一次提交。两者合用可将下载量从 2GB+ 压缩到几 MB。

#### [3/5] 准备输出目录

创建 `outputs/YYYYMMDD_HHMMSS/`，同时更新 `outputs/latest` 软链接指向本次输出。每次运行产物独立存放，不互相覆盖。

#### [4/5] 启动 GPU 监控

```bash
nvidia-smi dmon -s ugmt -d 5 > gpu_stats.log &
GPU_MON_PID=$!
trap cleanup_gpu_mon EXIT
```

- `-s ugmt`：采集 SM 利用率、显存、温度、功耗
- `-d 5`：每 5 秒采样一次
- `trap EXIT`：无论训练是否正常结束，退出时都会 kill 监控进程，避免后台进程残留

#### [5/5] 调用 run_case.py

```bash
"$PYTHON" run_case.py \
    --case-dir  "$CASE_DIR"   \
    --out-dir   "$OUT_DIR"    \
    --max-steps "$MAX_STEPS"  \
    --seed      42
```

`MAX_STEPS` 可通过环境变量覆盖：
```bash
MAX_STEPS=500000 bash run_case.sh   # 完整 500k 步训练
```

---

## 3. run_case.py — Python 运行器

### 来源

本项目自行编写，位于 `run_case.py`（约 650 行）。

### 做了什么（9个函数）

#### `parse_args()`

解析命令行参数：`--case-dir`、`--out-dir`、`--max-steps`（默认 10000）、`--seed`（默认 42）。

#### `set_seeds(seed)`

在训练开始前统一设置所有随机源，确保可复现性：

```python
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

#### `collect_metadata(args, start_time)`

收集运行元数据并写入 `metadata.json`，包括：
- GPU 信息（通过 `nvidia-smi --query-gpu` 查询）
- 所有关键包的版本（torch、physicsnemo、physicsnemo.sym 等）
- 运行参数（max_steps、seed、路径）
- git hash（best-effort，失败时记为 "unavailable"）

#### `run_heat_sink(case_dir, max_steps, out_dir)`

**最关键的函数**，负责以正确方式调用 `heat_sink.py`。

**为什么用 subprocess 而不是 importlib？**

`heat_sink.py` 的入口是 `@physicsnemo.sym.main(config_path="conf", config_name="config")`，这个装饰器在底层调用 Hydra 的 `_run_hydra()`。Hydra 解析 `config_path="conf"` 时，是相对于**脚本文件所在目录**来查找配置文件夹的，而这一路径解析依赖 Python 的 `__file__` 属性和调用栈。

用 `importlib.util.exec_module()` 动态加载时，Hydra 无法正确推断脚本路径，导致报错：
```
Primary config module 'conf' not found
```

改用 `subprocess.run(cmd, cwd=case_dir)` 后，`heat_sink.py` 作为独立进程运行，Hydra 的路径解析行为与用户直接运行完全一致，问题消失。

**为什么 hydra.run.dir 用相对路径？**

PhysicsNeMo Sym 的 `add_hydra_run_path()` 内部调用了：
```python
hydra_dir.relative_to(org_dir)   # org_dir = case_dir（进程 cwd）
```
如果 `hydra.run.dir` 是绝对路径（如 `/home/.../outputs/xxx`），而 `org_dir` 是 `case_dir`，两者没有父子关系，会抛出：
```
ValueError: '/home/.../outputs/xxx' is not in the subpath of '.../three_fin_2d'
```
因此必须使用相对路径（`"outputs/run"`），让 Hydra 输出落在 `case_dir/outputs/run/` 下，训练完成后再 `shutil.copytree` 到 `out_dir/hydra_outputs/`。

**注入的 Hydra overrides：**

```python
overrides = [
    f"training.max_steps={max_steps}",          # 训练步数
    f"training.rec_monitor_freq={freq}",         # monitor 记录频率（步数/20）
    f"training.rec_validation_freq={freq}",      # validation 频率
    f"training.rec_inference_freq={freq}",       # inference 频率
    f"hydra.run.dir=outputs/run",                # 输出目录（相对路径）
]
```

#### `parse_monitor_logs(out_dir)`

在多个可能路径下用 glob 搜索 CSV 文件（兼容不同版本的 PhysicsNeMo 输出目录结构），读取后将所有数值字符串转换为 float，返回 `{monitor_name: [行字典, ...]}` 结构。

#### `parse_gpu_stats(out_dir)`

解析 `nvidia-smi dmon` 输出的 `gpu_stats.log`。dmon 的输出格式是两行 `#` 注释作为表头，后跟数据行，需要手动解析列名并对齐数值。对每个数值列计算 mean/max/min 汇总。

#### `generate_charts(monitors, gpu_stats, out_dir)`

使用 `matplotlib`（Agg 后端，无需显示器）生成三类图表：

1. **per-monitor 图**：每个 CSV 文件一张，自动检测 step 列，其余数值列各一个子图
2. **combined_key_metrics 图**：自动识别列名中含 `loss/residual/imbalance/error/temperature/velocity` 的列，汇总到一张图
3. **GPU 图**：若 gpu_stats 有数据，绘制 SM 利用率、显存、温度、功耗随时间变化

#### `generate_baseline_report(...)`

将所有信息写入 `baseline_report.md`，章节包括：环境表、运行参数、运行时间、GPU 用量、数值稳定性（自动判断 imbalance 趋势）、各监控器最终值表格、图表引用、注意事项。

#### `main()`

串联所有步骤：set_seeds → collect_metadata → run_heat_sink → parse_monitor_logs → parse_gpu_stats → generate_charts → generate_baseline_report。训练异常时捕获错误写入报告但不立即崩溃，保证报告和图表始终能生成。

---

## 4. 三个脚本的协作关系

```
bash run_case.sh
  │
  ├─ [检查环境]  →  ENV_DIR/bin/python
  ├─ [下载示例]  →  cases/three_fin_2d/heat_sink.py   (NVIDIA 官方，未修改)
  ├─ [准备输出]  →  outputs/TIMESTAMP/
  ├─ [GPU 监控]  →  nvidia-smi dmon → gpu_stats.log
  │
  └─ python run_case.py --case-dir ... --out-dir ... --max-steps ...
       │
       ├─ set_seeds()
       ├─ collect_metadata()           →  metadata.json
       ├─ run_heat_sink()              →  subprocess: python heat_sink.py [overrides]
       │    └─ heat_sink.py (NVIDIA)       cwd=case_dir, hydra.run.dir=outputs/run
       │         └─ 训练完成后 shutil.copytree → out_dir/hydra_outputs/
       ├─ parse_monitor_logs()         →  读取 hydra_outputs/monitors/*.csv
       ├─ parse_gpu_stats()            →  读取 gpu_stats.log
       ├─ generate_charts()            →  charts/*.png
       └─ generate_baseline_report()   →  baseline_report.md
```

---

## 5. 关键设计决策汇总

| 决策点 | 选择 | 原因 |
|--------|------|------|
| 调用 heat_sink.py 的方式 | `subprocess.run()` | Hydra `config_path` 解析依赖脚本的真实文件路径，importlib 动态加载会破坏这一机制 |
| `hydra.run.dir` | 相对路径 `"outputs/run"` | PhysicsNeMo 内部 `Path.relative_to(cwd)` 要求该路径是 cwd 的子路径 |
| 示例下载方式 | git sparse-checkout + `--filter=blob:none` | 完整仓库超过 2GB，sparse 方式只下载目标子目录，仅几 MB |
| matplotlib 后端 | `Agg` | 服务器无显示器，Agg 是无头渲染后端，不依赖 X11/GUI |
| 随机种子设置时机 | `main()` 第一行 | 在任何随机操作（包括 numpy/torch 初始化）之前设置，确保完整可复现 |
| GPU 监控清理 | `trap cleanup_gpu_mon EXIT` | 无论训练是否报错，EXIT trap 都会执行，避免 dmon 进程泄漏 |
| 训练异常处理 | `try/except` 捕获但不立即 reraise | 即使训练失败，仍能生成部分报告和已有的图表，方便调试 |
