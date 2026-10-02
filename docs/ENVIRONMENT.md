# NightUAV Environment

记录日期：2026-10-02。硬件与系统信息来自此前本机体检，软件版本来自已完成的 nightuav 安装与 GPU 实测。

## 硬件与系统

| 项目 | 已确认配置 |
| --- | --- |
| 系统 | Windows 11 家庭中文版 25H2，x64，Build 26200.9457 |
| CPU | Intel Core i7-14650HX，16 核 / 24 线程 |
| 内存 | 约 16 GB（系统报告约 15.64 GiB） |
| NVIDIA GPU | 1 块 NVIDIA GeForce RTX 5060 Laptop GPU |
| 显存 | 8151 MiB，约 8 GB |
| NVIDIA Driver | 592.27 |
| GPU Compute Capability | 12.0 |

## 专用 Python 环境

| 项目 | 已验证版本或路径 |
| --- | --- |
| Conda 环境 | nightuav |
| 环境目录 | `E:\anaconda3\envs\nightuav` |
| Python | 3.10.22 |
| 解释器 | `E:\anaconda3\envs\nightuav\python.exe` |
| pip | 26.2.1 |
| pip 包位置 | `E:\anaconda3\envs\nightuav\Lib\site-packages\pip` |
| torch | 2.13.0+cu130 |
| torchvision | 0.28.0+cu130 |
| torch.version.cuda | 13.0 |
| 官方安装源 | `https://download.pytorch.org/whl/cu130`（稳定版） |

这些包已安装，不需要重复安装。此前 `python -m pip check` 已通过。

## CUDA 版本含义

- `torch.version.cuda = 13.0`：当前 PyTorch 预编译包所使用的 CUDA 构建版本。
- `nvidia-smi` 显示 CUDA 13.1：驱动报告的 CUDA 支持版本，不代表安装了 Toolkit 13.1。
- 本项目未安装完整 CUDA Toolkit；此前检查未发现系统 `nvcc`。
- 另一个名为 cuda 的 micromamba 环境不用于 NightUAV。

## 重复验证

在 PowerShell 中直接使用专用解释器，绕过环境激活对脚本执行策略的依赖：

```powershell
& 'E:\anaconda3\envs\nightuav\python.exe' -B 'E:\NightUAV\scripts\verify_gpu.py'
```

脚本输出实际 Python / PyTorch / torchvision 版本、CUDA 状态、GPU 名称和计算能力，并执行一次 1000×1000 GPU 矩阵乘法及同步。输出 `GPU_MATRIX_TEST: PASS` 且退出码为 0 才表示验证通过；失败退出码为 1，不回退到 CPU。

这是基础 GPU 运算检查，不是训练、性能基准或完整模型兼容性验证。

## 使用约定

- VS Code 选择解释器 `E:\anaconda3\envs\nightuav\python.exe`。
- 当前普通 PowerShell 的默认 Python 可能仍是 `D:\python0314\python.exe`。
- 后续获准安装依赖时，统一使用目标解释器的 `python -m pip`，不要裸用 `pip`。
- 不需要为运行验证脚本修改系统 PATH、执行策略或 base 环境。

## U3D 首次推理输入

2026-10-02 只读检查 `E:\NightUAV\datasets\U3D\lowlight\test`：

- 100 个文件，100 张可识别的 JPEG 图片，扩展名均为 `.jpg`。
- 分辨率均为 3840×2160，颜色模式均为 RGB。
- Pillow 图片打开和 `verify()` 检查未报告错误；这不等同于逐像素质量评估。
- 按路径排序选取 `E:\NightUAV\datasets\U3D\lowlight\test\0901.jpg` 作为首次模型推理输入。
- 只记录原始路径，不复制、移动或修改数据；本次未执行模型推理。
