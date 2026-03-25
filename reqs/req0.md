## CR-1
本需求的主要目标是搭建nvidia NeMo环境，并运行一个简单的demo: 2D Convection Diffusion — Scalar Transport: 2D Advection Diffusion，具体要求如下：

### 交付物
- 可运行的环境配置（conda + 版本锁定文件）。
- 可运行的示例：执行 run_case.sh / run_case.py 后能产生输出结果。
- 一份简短的基线报告，包含：运行时间、GPU 使用情况、数值稳定性、关键图表（温度/流场指标）。

### 完成标准
- 在全新机器上，单条命令即可复现整个运行流程。（包含环境配置和运行demo, 分成两个脚本）
- 结果与日志已附带清晰的元数据（随机种子、提交哈希、参数）进行存储。

### 其他要求
- 环境安装在/home/featurize/work/env_conda的某个子目录下（比如/home/featurize/work/env_conda/nemo），因为只有这个目录的数据不会被删除，是安全的。
- 在安装过程中请检查GPU相关的环境要求：GPU型号、驱动等，如果有问题则直接退出。
- 默认不启用桌面，输出尽量存成文件形式。

### 资料
- 文档入口页面：https://docs.nvidia.com/physicsnemo/latest/index.html
- 安装页面：https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html
- 2D Convection Diffusion介绍页面：https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/foundational/scalar_transport.html
