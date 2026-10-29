# EXP006B 论文与代码证据索引

审计日期：2026-10-11。已读指全文方法及相关实验部分，不代表复现验证。文件条目/下载链接公开不代表二进制成功下载或安全加载。本轮未获取新权重、未运行第三方代码。机制解释见 [主报告](../../docs/RESEARCH_GAP_LOCAL_DYNAMIC_RANGE.md)。

|ID / 论文与来源|原文及阅读位置|官方仓库|代码与权重状态|paired GT|
|---|---|---|---|---|
|P01 DI-Retinex: Digital-Imaging Retinex Theory for Low-Light Image Enhancement（2024 preprint; IJCV 2025 per official repository）|[原文](https://arxiv.org/html/2404.03327v1)；3.4; 4.1–4.3 equations17,20–23;5.1|[官方仓库](https://github.com/sunshangquan/DI-retinex-low-light-enhancement)|官方模型/训练/推理与权重目录公开；本项目历史推理完成；本次未加载|不需要；低光-only|
|P02 Unsupervised Ultra-High-Resolution UAV Low-Light Image Enhancement: A Benchmark, Metric and Framework（2025 arXiv v1）|[原文](https://arxiv.org/html/2509.01373v1)；Proposed U3LIE Framework; equations8–12; Training Configuration; Appendix C|[官方仓库](https://github.com/AeroVILab-AHU/U3D_Toolkit)|部分代码；模型链接404；配置链接configs/u3lie.py也404；不可完整复现|不需要；APA派生图不是正常光GT|
|P03 Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement（2020 CVPR）|[原文](https://arxiv.org/pdf/2001.06826)；3.1–3.3;4.1 training-data and loss ablations|[官方仓库](https://github.com/Li-Chongyi/Zero-DCE)|模型/训练/推理和Epoch99.pth条目公开；未下载/加载；论文与代码曝光loss不同|不需要；原训练包含不同曝光图像|
|P04 Unsupervised Retinex Exposure Control: A Novel Approach to Image Enhancement（2025 IET Image Processing 19(1), e70077）|[原文](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70077)；3.1–3.4 equations7–12;4.2.1;Appendix;Data Availability|未确认公开GitHub；按作者要求申请|作者说明代码/预训练模型按合理请求提供；未确认公开官方仓库，未获取实现|不需要|
|P05 Toward Fast, Flexible, and Robust Low-Light Image Enhancement（2022 CVPR）|[原文](https://arxiv.org/pdf/2204.10137)；2.1–2.3 equations4–5;3|[官方仓库](https://github.com/vis-opt-group/SCI/tree/main/CVPR)|train/test/model/loss及easy/medium/difficult.pt公开；README称不是论文报告结果所用模型；未加载|不需要；不要混入TPAMI扩展|
|P06 Deep Retinex Decomposition for Low-Light Enhancement（2018 BMVC）|[原文](https://arxiv.org/pdf/1808.04560)；2.1–2.3|[官方仓库](https://github.com/weichen582/RetinexNet)|TensorFlow模型/入口及Decom/Relight模型目录公开；未下载/加载|需要成对低/正常光；无需分解分量GT不等于无需paired data|
|P07 Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement（2023 ICCV）|[原文](https://arxiv.org/html/2303.06705v2)；3.1–3.2;4.1; released LOL-v1 configuration|[官方仓库](https://github.com/caiyuanhao1998/Retinexformer)|模型/训练/推理公开；README提供云盘权重入口；本轮未下载/加载；禁用GT_mean作无参考比较|原训练需要paired GT；预训练单图推理不需要|
|P08 EnlightenGAN: Deep Light Enhancement without Paired Supervision（2021 IEEE TIP; 2019 preprint）|[原文](https://arxiv.org/html/1906.06972v2)；III-A–C equations7–8;IV-A|[官方仓库](https://github.com/VITA-Group/EnlightenGAN)|仓库和模型下载入口公开；本次只核查README/目录和论文，未逐行审查训练实现；未下载/加载|不需要paired GT，但需要非配对正常光域及预训练VGG|

## 具体源码与权重入口

### P01

- [源码 1](https://github.com/sunshangquan/DI-retinex-low-light-enhancement/blob/3f94f95ffb181904a837532a70adc064bee793a2/mymodel.py)
- [源码 2](https://github.com/sunshangquan/DI-retinex-low-light-enhancement/blob/3f94f95ffb181904a837532a70adc064bee793a2/lowlight_train.py)
- [权重目录/入口（状态见上表）](https://github.com/sunshangquan/DI-retinex-low-light-enhancement/tree/master/weights)
- 本地核实commit：`3f94f95ffb181904a837532a70adc064bee793a2`。

### P02

- [源码 1](https://github.com/AeroVILab-AHU/U3D_Toolkit/blob/main/u3lie_loss.py)
- [权重目录/入口（状态见上表）](https://github.com/AeroVILab-AHU/U3D_Toolkit/blob/main/weights/u3lie)
- 未固定远端commit；链接为审计日分支快照，后续执行前须固定版本。

### P03

- [源码 1](https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py)
- [源码 2](https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/lowlight_train.py)
- [权重目录/入口（状态见上表）](https://github.com/Li-Chongyi/Zero-DCE/tree/master/Zero-DCE_code/snapshots)
- 未固定远端commit；链接为审计日分支快照，后续执行前须固定版本。

### P04

- 无已核实的公开权重端点。
- 未固定远端commit；链接为审计日分支快照，后续执行前须固定版本。

### P05

- [源码 1](https://github.com/vis-opt-group/SCI/blob/main/CVPR/model.py)
- [源码 2](https://github.com/vis-opt-group/SCI/blob/main/CVPR/loss.py)
- [权重目录/入口（状态见上表）](https://github.com/vis-opt-group/SCI/tree/main/CVPR/weights)
- 未固定远端commit；链接为审计日分支快照，后续执行前须固定版本。

### P06

- [源码 1](https://github.com/weichen582/RetinexNet/blob/master/model.py)
- [权重目录/入口（状态见上表）](https://github.com/weichen582/RetinexNet/tree/master/model)
- 未固定远端commit；链接为审计日分支快照，后续执行前须固定版本。

### P07

- [源码 1](https://github.com/caiyuanhao1998/Retinexformer/blob/master/Options/RetinexFormer_LOL_v1.yml)
- [权重目录/入口（状态见上表）](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV)
- 未固定远端commit；链接为审计日分支快照，后续执行前须固定版本。

### P08

- [源码 1](https://github.com/VITA-Group/EnlightenGAN/blob/master/train.py)
- [权重目录/入口（状态见上表）](https://drive.google.com/file/d/1AkV-n2MdyfuZTFvcon8Z4leyVb0i7x63/view)
- 未固定远端commit；链接为审计日分支快照，后续执行前须固定版本。

## 检索线索与未完成事项

检索词族：DI-Retinex、U3LIE、Retinex exposure control、highlight protection、local exposure loss、structure preserving low-light、unsupervised illumination，以及论文标题+official code。采用作者arXiv全文、出版机构全文、作者链接的GitHub；不以综述/摘要替代已审方法。CVF若干PDF读取失败后以作者arXiv全文交叉读取；EnlightenGAN PDF因大小限制改读v2 HTML。

- [Region-Adaptive Low-Light Image Enhancement with Light Effect Suppression and Detail Preservation](https://diglib.eg.org/server/api/core/bitstreams/2ad23d33-0750-4283-9d4f-a182ae64ccea/content)：检索显示2025；全文端点读取失败。未据标题推断技术机制、监督类型或开源状态，不计入8篇已读。
- UREC引用的CuDI/LCD、DI引用的NightEnhance、SCI的TPAMI2025扩展尚未完成同等深度审查；进一步原创性判断必须补齐。
- U3LIE README的[配置入口](https://github.com/AeroVILab-AHU/U3D_Toolkit/blob/main/configs/u3lie.py)与上表模型入口本轮404。完整工具链缺失不会削弱其作为先例的地位。

这是有边界的机制审计，不是全面的新颖性检索证明。研究报告中的潜在不足与三个方向均是本项目推断/待测假设。
