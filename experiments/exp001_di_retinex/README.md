# EXP001：DI-Retinex 首次单图 GPU 推理

## 目的与结论

使用官方 LOL-v1 normal 权重，对 U3D 0901.jpg 进行 1280×720 CUDA 单图推理。
**工程运行成功，但存在明显局部过曝和噪声放大。** 本次不是完整论文复现，不是原生 4K 推理，不包含训练或方法比较。没有使用 GT，不计算 PSNR/SSIM。
运行时间（UTC）：2026-10-02T17:09:30.419755+00:00；本地日期为 2026-10-03。

## 模型与权重

- 官方仓库：https://github.com/sunshangquan/DI-retinex-low-light-enhancement
- 使用已有克隆，commit：`3f94f95ffb181904a837532a70adc064bee793a2`。
- 只导入未修改的 mymodel.py，构造 `enhance_net_nopool(ch=6, number_f=64, linear=False)`。
- 参数数量：42182。
- 权重：`E:\NightUAV\checkpoints\di_retinex\latest (21.54 lolv1).pth`。
- 来源：[固定 commit 的官方端点](https://raw.githubusercontent.com/sunshangquan/DI-retinex-low-light-enhancement/3f94f95ffb181904a837532a70adc064bee793a2/weights/latest%20%2821.54%20lolv1%29.pth)。
- 文件大小：170775 字节；SHA256：`f7c7dce7c92dd1652947ae02c12047535d52908ee64554a94d175d2a6788e108`。
- 内容与上游 Git blob `1c7f845e46b59815430905bc4a4051ac4a27a029` 一致。
- 实际通过 `torch.load(map_location='cpu', weights_only=True)` 和 `load_state_dict(strict=True)`，没有使用不安全反序列化或忽略不匹配键。
- 三层卷积权重形状：[64,3,3,3]、[64,64,3,3]、[6,64,3,3]，与 normal 结构匹配；完整形状见 metrics。

## 环境与输入

- Windows 11 x64；Driver 592.27（此前体检记录）。
- Python 3.10.22；解释器：`E:\anaconda3\envs\nightuav\python.exe`。
- torch 2.13.0+cu130；torchvision 0.28.0+cu130；CUDA 13.0。
- numpy 2.2.6；Pillow 12.3.0。
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，计算能力 [12, 0]，显存 8150.56 MiB。
- 原图：`E:\NightUAV\datasets\U3D\lowlight\test\0901.jpg`，3840×2160。
- 只读打开、转 RGB，在内存中用 Pillow LANCZOS 缩放至 1280×720；[0,1]、NCHW、batch=1、FP32。
- 使用 eval()、inference_mode()，CUDA:0；无 CPU 回退；TF32 与 cuDNN benchmark 关闭。
- 原图操作前后 SHA256 一致：`c5a052b4be3a03896f431c59f3d54d91c0d28ba3a3b112cbcd0bdfd5fdcd9244`。

## 实际执行命令

```powershell
& 'E:\anaconda3\envs\nightuav\python.exe' '-B' 'E:\NightUAV\scripts\infer_di_retinex.py' '--input' 'E:\NightUAV\datasets\U3D\lowlight\test\0901.jpg' '--output-dir' 'E:\NightUAV\results\demos\di_retinex_exp001' '--metrics' 'E:\NightUAV\experiments\exp001_di_retinex\metrics\inference.json' '--width' '1280' '--height' '720' '--warmup' '3' '--runs' '5'
```

日志：`logs/run.log`。再次执行时必须选择新输出目录和新 metrics 路径；脚本拒绝覆盖已有产物。

## 实测结果

| 项目 | 实测 |
| --- | ---: |
| 权重读取、模型构造/加载并传 GPU | 321.027 ms |
| 首次前向 | 1647.165 ms |
| 首次后额外预热次数 | 3 |
| 预热后测量次数 | 5 |
| 预热后平均前向 | 25.803 ms |
| 预热后中位前向 | 25.789 ms |
| 首次 peak allocated | 688.709 MiB |
| 首次 peak reserved | 692.000 MiB |
| 预热后 peak allocated | 688.709 MiB |
| 预热后 peak reserved | 692.000 MiB |

预热后逐次耗时（ms）：25.308, 26.155, 25.671, 25.789, 26.093。

每次模型前向前后均 CUDA 同步，用 perf_counter 计时。前向计时不包含加载、图像预处理、数据传输、结果校验或保存；首次前向包含首次调用开销，不等于进程启动时间。短测量不作为正式吞吐 Benchmark。
显存是 PyTorch allocator 峰值，包含常驻模型与输入，不是系统总显存；reserved 包括缓存，可能保留前几次运行的缓存。

## 输出与检查

目录：`E:\NightUAV\results\demos\di_retinex_exp001`。

- [输入副本](../../results/demos/di_retinex_exp001/input_1280x720.png)
- [增强图](../../results/demos/di_retinex_exp001/enhanced_1280x720.png)
- [完整指标](metrics/inference.json)
- [运行配置](config/run.json)

保存实际首次前向结果，浮点范围 [0,1]，按 round(x*255) 转 uint8 PNG。四个官方输出张量均无 NaN/Inf。保存后的两张图已重新打开并完整解码：1280×720、RGB。增强图不是全黑/全白/常量。
增强图通道均值 165.504，标准差 65.874；达到 255 的通道元素比例 24.072%（不是全白像素比例，也不是质量评价指标）。
视觉检查：暗部大幅提亮，商场外墙和道路灯光附近明显过曝，暗部噪声放大。保留原始模型输出，未额外后处理、未针对测试图调参。不能凭单图宣称方法优越或概括整个 U3D。

## 兼容性、问题与保护

- 模型在现有环境直接运行成功，无需安装包或降级 torch/torchvision/CUDA。
- 独立入口避免原测试脚本的 Windows split('/') 输出路径问题；不导入 dataloader，因此不需要 cv2/natsort。
- 校验源码 commit 和 blob、权重哈希。解析绝对路径后，图片只能写 results，指标只能写 experiments；拒绝 datasets、code、_model_audit、checkpoints 下的输出及已存在目标，使用排他文件创建。
- 使用 -B，避免在第三方源码目录产生 pycache。原始数据与第三方源码未修改。
- 上游许可证尚不明确；本次只执行用户授权的本机研究验证，没有再分发源码/权重，后续发布需核实许可。
- 实验文档生成的第一次尝试因 PowerShell 传递 Python -c 引号导致 SyntaxError；没有写出文档，也未影响成功的推理结果，改为临时 Python 文件生成记录。

## Git 管理与下一步

.gitignore 新增 `/_model_audit/` 和 `/results/demos/di_retinex_exp001/`。数据集、权重和日志已有忽略规则；结果图保存在本机，不进入建议提交范围。
建议仅定向提交 .gitignore、scripts/infer_di_retinex.py、本 README、config/run.json、metrics/inference.json。此前的 docs/BASELINE_SELECTION.md 未改写，是否提交另行决定。本次未 git add/commit/push。
下一步先人工验收，再用固定方案检查更多样本，关注高光饱和、噪声和跨场景泛化；之后另行授权原生 4K、分块或 small 模型实验。重新训练应使用 U3D 训练划分，不对 test/0901.jpg 调参。本次成功不保证 4K 显存可行或训练已复现。
