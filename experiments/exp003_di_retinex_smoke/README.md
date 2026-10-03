# EXP003：数据划分审计与DI-Retinex训练 smoke test

## 结果与边界

**实际完成20步GPU优化，前向/两项损失/反向/参数更新全部通过。** 当前未解决train/val独立性问题，本实验没有运行验证或评价增强效果，更不证明收敛/泛化/论文结果复现。

[划分审计报告](../../docs/U3D_SPLIT_AUDIT.md)：跨划分存在1对字节级重复train0209/val0880，另有多组相同场景目视初判。原始目录、图片和官方权重未改动。EXP002文档及未提交文件均保留。

## 实际代码及调用链

官方commit `3f94f95ffb181904a837532a70adc064bee793a2`。项目入口 [train_di_retinex_smoke.py](../../scripts/train_di_retinex_smoke.py)只读导入官方lowlight_train模块，使用其mymodel、weights_init、set_seed、dataloader；不调用会写tmp.png和默认snapshots的官方train/CLI。官方模型、数据读取和初始化没有重写。

两项loss及曝光采样/ratio/mask直接从该commit的lowlight_train.py提取对应赋值AST，按原表达执行；抽取表达式和源码SHA256见 [config.json](training/config.json)。wrapper SHA256 `df7e0555bb655969e875782469a460edbfe937c072fdc7443bae0830a0f8838f`，所有第三方源码hash与commit匹配。

实际命令：

```powershell
& 'E:\anaconda3\envs\nightuav\python.exe' -B 'E:\NightUAV\scripts\audit_u3d_split.py'
& 'E:\anaconda3\envs\nightuav\python.exe' -B 'E:\NightUAV\scripts\train_di_retinex_smoke.py'
```

两脚本拒绝覆盖本次产物；不能直接重跑同名实验。若以后运行新测试，必须显式改为新的实验及checkpoint目录并记录脚本版本，不能删除已有结果后盲目重复。

## 数据、裁剪与初始化

只使用官方train/0102.jpg（3840×2160，已审核train来源、可读取、哈希无重复、未命中自动跨划分候选），不是20张不同图片。其与NCC最近val图目视无明显对应，但不保证所有场景独立，所以仅做技术测试。

官方加载器先在原始uint8 RGB整图做OpenCV GaussianBlur(9×9,sigmaX/Y=1.7)，再转float32/255。入口对X与G做相同坐标的256×256均匀随机裁剪，不先缩放、不使用GT、不做颜色增强。20个xywh已逐步记录。种子1149，独立Python RNG固定裁剪序列。CUDA、FP32、batch1，workers0；随机初始化，未载入LOL预训练权重。

## 原配置与本次差异

| 配置 | 官方当前代码 | EXP003 |
|---|---|---|
| 模型 | normal 3→64→64→6，三层卷积 | 完全相同，官方类 |
| 初始化 | Conv weight N(0,0.02)，bias默认 | 官方函数；seed1149 |
| 数据路径 | Linux LOL-v2 Test/Low默认路径 | 仅U3D/train/0102；禁用val/test |
| 加载/尺寸 | 原图，无随机裁剪；size256未使用 | 官方全图滤波后成对256随机裁剪 |
| 数据范围 | RGB，[0,1] | 同；非有限/全黑输入报错，不静默跳过 |
| batch/workers | 1/4 | 1/0，无多进程 |
| 优化 | Adam lr1e-3，wd1e-4，clip0.1 | 相同；betas(0.9,0.999)，eps1e-8；clip_grad_norm_等价接口 |
| 损失 | RD+VS，系数1:1 | 同，直接执行官方AST |
| 曝光 | Normal(0.5,scale0.001) | 同，CPU采样后CUDA；Gamma2.2 |
| 时长 | 200 epochs默认 | 20次optimizer.step，仅一次运行 |
| 保存 | tmp.png；snapshots/Epoch/latest | 无图像保存；独立exp003_smoke/step020.pth |
| checkpoint | 纯model state_dict | model+optimizer+步数/种子/RNG；安全加载核对 |
| 验证/选模 | 无内置验证 | 禁止验证，不选best |

论文v1§5.1写1000 epochs、曝光分布方差0.001；代码默认200，Normal第二参数为标准差0.001（方差1e-6）。论文RD公式写原低光，代码用平滑低光。详见 [既有方法笔记](../../docs/DI_RETINEX_METHOD_STUDY.md)。本轮不擅自修正论文/代码差异。

与此前EXP002训练方案建议的全视野720p不同，本次按用户明确授权改用小尺寸随机裁剪。裁剪改变图均值及ratio，不能称严格复现训练协议。

## 损失、截断与路径审查

mask=(Y<1)，ratio=E/mean(X.detach())；RD=mean(abs(mask*(G**2.2-(1/ratio)**2.2*Y**2.2)))；VS=mean(var(k*b-k+b+1,spatial,correction=1))。无新损失，Adam weight_decay不作为图像第三损失。

官方模型本就将z裁剪为[0,1]，饱和通道不参与RD；这里没有额外非预期截断，保留原行为并记录每步饱和比例。模型参数、四个输出、两项损失和全部参数梯度都检查有限。全黑输入mean=0会导致ratio问题，入口在计算前停止而不改原loss加epsilon。

Windows风险：官方split('/')仅用于日志名，在本入口用Path；默认Linux路径不可照用。官方tmp.png和snapshots/latest可能覆盖旧结果，因此不调用这些写入路径。入口拒绝已有run/checkpoint目录；不修改第三方源码，-B禁止生成pycache。

## 依赖与环境

Python 3.10.22，torch 2.13.0+cu130，torchvision 0.28.0+cu130，CUDA运行时 13.0，GPU NVIDIA GeForce RTX 5060 Laptop GPU。

本次只安装 opencv-python-headless 4.12.0.88、natsort8.4.0；先pip dry-run确认仅新增两包，再--no-deps安装；numpy2.2.6满足OpenCV要求。torch/torchvision/CUDA包未更改。pip check通过。OpenCV无GUI版本足够GaussianBlur，未安装CUDA Toolkit。详见 [dependencies.json](dependencies.json) 与依赖预检计划。

## 实测耗时与显存

| 项目 | 实测 |
|---|---:|
| 首步同步墙钟 | 1770.691 ms |
| 第2～20步平均 | 31.282 ms |
| 第2～20步中位 | 29.662 ms |
| 最大已分配显存 | 1205.728 MiB |
| 最大保留显存 | 1864.000 MiB |
| 平均整图读取/平滑/裁剪/传输 | 517.588 ms |

前后CUDA同步，并另记CUDA events。step包含前向、loss、backward、梯度检查、裁剪及Adam，还有诊断同步开销；不含加载/传输/checkpoint/参数差检查。不是纯内核训练吞吐基准。第2～20步没有额外无记录预热，首步单列。显存为PyTorch allocator峰值，包含驻留量/工作区；不代表单模型参数大小，也不能推导4K训练可行。

## 每步记录

[原始JSONL](training/steps.jsonl)、[CSV](training/steps.csv)、[汇总](training/summary.json)。每步都确认参数差L1>0，不只检查loss可导。

| step | RD | VS | total | 同步ms | 参数更新 |
|---|---:|---:|---:|---:|---|
| 1 | 0.01118845 | 0.00000234 | 0.01119079 | 1770.691 | 成功 |
| 2 | 0.02210632 | 0.00000704 | 0.02211336 | 60.678 | 成功 |
| 3 | 0.00011013 | 0.00001770 | 0.00012783 | 28.906 | 成功 |
| 4 | 0.00833351 | 0.00006599 | 0.00839951 | 29.666 | 成功 |
| 5 | 0.00080480 | 0.00007980 | 0.00088460 | 31.342 | 成功 |
| 6 | 0.01303515 | 0.00018889 | 0.01322404 | 27.808 | 成功 |
| 7 | 0.07807317 | 0.00067584 | 0.07874902 | 26.577 | 成功 |
| 8 | 0.00319436 | 0.00048356 | 0.00367791 | 29.662 | 成功 |
| 9 | 0.00900776 | 0.00073939 | 0.00974716 | 31.847 | 成功 |
| 10 | 0.00475705 | 0.00134630 | 0.00610335 | 29.562 | 成功 |
| 11 | 0.00187941 | 0.00144742 | 0.00332683 | 29.947 | 成功 |
| 12 | 0.00594439 | 0.00160329 | 0.00754767 | 30.767 | 成功 |
| 13 | 0.00654862 | 0.00149040 | 0.00803902 | 30.544 | 成功 |
| 14 | 0.00013953 | 0.00120540 | 0.00134493 | 30.730 | 成功 |
| 15 | 0.00828663 | 0.00098128 | 0.00926790 | 29.436 | 成功 |
| 16 | 0.00152983 | 0.00077048 | 0.00230031 | 29.072 | 成功 |
| 17 | 0.00020018 | 0.00061836 | 0.00081854 | 29.100 | 成功 |
| 18 | 0.00161660 | 0.00046859 | 0.00208519 | 31.561 | 成功 |
| 19 | 0.00113216 | 0.00034374 | 0.00147590 | 29.469 | 成功 |
| 20 | 0.00170785 | 0.00024415 | 0.00195200 | 27.684 | 成功 |

## Checkpoint

`E:\NightUAV\checkpoints\di_retinex\exp003_smoke\step020.pth`；SHA256 `bc40f1a75b72242321ae2b39bf25a565dea898a824bc3c0a55474e581b25f6b5`。

实际用torch.load(weights_only=True,map_location='cpu')重载，逐张量与末步模型完全相同，且与初始化不同。没有运行额外恢复训练步骤，所以安全可读取不等于已验证完整断点续训重现。

## 警告、失败与未解决事项

没有OOM、NaN或依赖冲突，没有失败后重复训练。日志float(loss_tensor)触发requires_grad张量转标量警告；发生在backward/step之后，仅日志数值读取，未改变计算图用于训练的路径。保留本次实际脚本与hash，未来可用detach().item()清理日志警告。natsort.exe不在PATH是安装提示，模块import已成功，不修改PATH。

严重未解决项为场景独立性；需人工审核同场景分组、确定新清单。没有证明原生4K或长时间训练稳定，没有验证指标、质量结论、收敛结论。下一步可准备完整训练入口与分组协议；在独立验证方案确定前，不应开始正式选模/泛化实验。

## Git

未add/commit/push。建议定向纳入两份新脚本、docs/U3D_SPLIT_AUDIT.md、本README、audit清单/目视记录、training配置/JSONL/CSV/汇总、dependencies.json。outputs被现有experiments/**/outputs规则忽略；checkpoint在checkpoints下忽略；第三方仓库和数据仍忽略。EXP001/EXP002未提交内容保持原状，本次不代为提交。
