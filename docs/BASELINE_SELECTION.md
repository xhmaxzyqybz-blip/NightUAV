# NightUAV Baseline Selection

审查日期：2026-10-03。结论基于源码静态审查及小字节范围的权重访问验证；未安装候选依赖、未加载模型权重、未执行推理或训练。GPU 环境此前通过矩阵运算检查，不代表两个模型已运行成功。

## 结论

优先选择 **DI-Retinex 的 LOL-v1 normal 模型**完成第一张图片的工程验证，再考虑其 small 版本和 U3D 无监督训练；Retinexformer 作为后续有监督预训练跨数据集推理对照。

理由：DI-Retinex 模型只依赖 torch 和 numpy，单图读写可使用现有 Pillow；官方权重可直接获取，模型与调用链很短，训练损失不需要配对 GT。可以新增独立单图封装，避免原脚本的 Windows 输出路径问题及无关导入，不必立即安装额外依赖。限制：仓库没有发现明确许可证，使用、修改和再分发范围需向作者确认；技术推荐不代表已确认授权条件。

| 目标 | DI-Retinex | Retinexformer |
| --- | --- | --- |
| 官方权重单图推理 | 技术上具备条件，需小型封装与加载验证；尚未执行 | 技术上具备条件，需至少补 einops 并编写无 GT 单图入口；尚未执行 |
| U3D 无配对 GT 重新训练 | 现有损失支持仅低光图训练；需修改数据路径、裁剪与训练配置，属于新数据集实验 | 现有训练需要配对 GT；不能直接用不配对的日间图替代 GT，需新设计训练目标 |
| 完整复现论文结果 | 不能承诺；缺少完整评估工具与逐实验协议，许可证不明，现代环境未验证 | 代码更齐全，但仍需原基准配对数据、对应权重/配置、协议与环境验证；不等于 U3D 推理 |

以本机“首次自定义图片推理”为范围，两者均属 **B：少量修改后具备尝试条件**。这不是已经完成模型复现的结论。完整论文数值复现仍有待核实条件，不给出 A。

## 1. 基础设施 Git 提交

- 已审阅 `docs/ENVIRONMENT.md` 与 `scripts/verify_gpu.py`：内容与此前实测一致；未发现密码、访问令牌或私钥。环境绝对路径是任务要求的本机记录。
- 提交前只有上述两个未跟踪文件，无其他意外修改。
- 已执行定向 git add、git diff --cached 与差异检查。
- Commit：`7fd4ace45bd4ed664d6683562698a33744d0d804`
- Message：`Document GPU environment and add verification script`
- `git push origin main` 成功，远程从 `4686ca9` 更新到 `7fd4ace`。
- 本报告及 `_model_audit` 审查副本不属于该提交，不自动暂存、提交或推送。

## 2. 审查范围与可追溯版本

| 候选 | 本地目录 | 审查 commit |
| --- | --- | --- |
| A | `E:\NightUAV\_model_audit\DI-retinex-low-light-enhancement` | `3f94f95ffb181904a837532a70adc064bee793a2` |
| B | `E:\NightUAV\_model_audit\Retinexformer` | `1e9a0efce4b306b6701b824768370ff26066c32a` |

两者均从用户给定官方 GitHub 地址浅克隆，采用 blob 延迟下载和源码/配置/文档稀疏检出，未检出数据、图片、完整权重。已查看完整 Git 文件树：稀疏副本没有检出的文件不等于上游缺失。Retinexformer 的 VERSION 在上游存在，内容为 1.2.0。

补检出 `.gitignore` 等少量元数据时遇到 GitHub 443 超时，未修改任何模型源码。GitHub API 曾返回 403 限流，文件存在性以已克隆的 Git 树和源码核实；权重端点用独立 HTTP 范围请求验证。未全量下载权重或验证其内部参数形状。

## 3. DI-Retinex：实际代码结构

| 文件 | 作用及依据 |
| --- | --- |
| `mymodel.py` | `enhance_net_nopool`，三层 3×3 卷积及亮度/对比度映射；第 8–64 行包含完整构造与 forward |
| `lowlight_test.py` | 第 59–86 行为目录推理、权重加载和命令行入口；第 34–57 行图片读写 |
| `lowlight_train.py` | Adam、反向传播、梯度裁剪、checkpoint 保存及 main 入口 |
| `dataloader.py` | 读取低光 RGB，生成同图 GaussianBlur 版本，返回输入/模糊图/路径 |
| `README.md` | Python 3.8、PyTorch 1.9 的作者测试环境；测试和训练示例 |
| `weights/*.pth` | 上游 Git 树有 LOL-v1/v2 normal 与 small 四个权重，未在本次完整下载 |
| `data/lolv1/`、`result/lolv1/`、`assets/` | 上游示例图、结果图、展示图；不属于本次源码检出范围 |

没有单独 config 文件：参数在 argparse 中；loss 直接写在训练脚本中，不是缺少 loss。没有完整的 PSNR/SSIM 数据集评估脚本；推理中粗略计时没有 GPU 同步，不能当作可靠 FPS 基准。

### 无监督训练证据

`dataloader.py:41–54` 只返回低光图、其高斯模糊版本和路径，没有读取正常光 GT。`lowlight_train.py:51–61` 使用 masked reverse degradation 项和方差项；曝光目标来自设定分布。可用 U3D lowlight/train 作为训练输入，不需要 U3D 的非配对日间图。

但 `self.size=256` 并未实际生效：`dataloader.py:49` 的 resize 被注释，当前加载整图。4K 训练必须另行设计裁剪和批量大小。默认训练路径还指向作者本机 LOL-v2 Test/Low，不能照搬；不得用 NightUAV test/0901.jpg 参与训练或调参。

### 推理与 Windows 问题

- 原脚本接收目录，不是单文件；列出目录全部内容且没有扩展名过滤。
- `lowlight_test.py:53` 使用 `image_path.split('/')[-1]`。若输入 Windows 反斜杠绝对路径，提取结果可能仍为绝对路径，随后 `os.path.join(save_path, image_name)` 会丢弃输出目录，存在写回原图的风险。**不要直接对 U3D 原始目录运行此脚本。**
- 原入口导入 dataloader，间接要求 cv2，同时直接要求 natsort；当前环境均未安装。
- 未调用 eval()，当前网络没有 BatchNorm/Dropout，预计影响有限，但独立封装仍应使用 eval() 与 inference_mode()。
- 测试支持 normal (`ch=6, number_f=64`) 和 small (`ch=2, number_f=4`)；训练脚本默认固定普通结构，small 重新训练需明确调整构造参数。

## 4. Retinexformer：实际代码结构

| 文件/目录 | 作用 |
| --- | --- |
| `basicsr/models/archs/RetinexFormer_arch.py` | 完整光照估计、IG_MSA、Denoiser、RetinexFormer 网络 |
| `Enhancement/test_from_dataset.py` | 权重加载、数据集推理、PSNR/SSIM、可选 self-ensemble 和 GT_mean |
| `basicsr/train.py` | 训练入口；有 main guard |
| `basicsr/models/image_restoration_model.py` | 模型封装、优化、验证；第 185 行对输出与 self.gt 计算像素损失 |
| `basicsr/data/paired_image_dataset.py` | 配对输入与 GT 数据加载、裁剪和增广 |
| `basicsr/data/single_image_dataset.py` | 单图数据类，但不代表现有 test_from_dataset 入口已经无 GT |
| `basicsr/models/losses/` | L1 等 loss |
| `basicsr/metrics/`、`Enhancement/utils.py` | PSNR、SSIM、NIQE 等评价工具；部分路径依赖额外包/资源 |
| `Options/*.yml` | LOL、SID、SMID、SDSD、FiveK、NTIRE 等模型/训练配置 |
| `setup.py`、`VERSION`、`train_multigpu.sh` | BasicSR 打包和分布式训练入口 |
| `LICENSE.txt` | 根目录 MIT 许可证 |

### 无 GT 推理与训练必须分开

`Options/RetinexFormer_LOL_v1.yml:12–14` 指定 Dataset_PairedImage、GT 和低光目录，94–97 行指定 L1Loss。原训练需要配对 GT，不能直接用 U3D 的不对应日间图片作为监督目标。

`Enhancement/test_from_dataset.py:207–225` 从 YAML 读取低光/GT 目录并 zip 配对，GT 为空时不会产生正常的单图推理结果。虽然定义了 --input_dir，但该分支实际使用 YAML 路径。--GT_mean 用 GT 均值校正输出，U3D 无配对 GT 情况必须禁用。独立无 GT 单图入口可以只调用网络，不必创建假 GT 或伪造 PSNR/SSIM。

LOL-v1 网络应按配置构造 `RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2])`；不要使用模型类默认的 n_feat=31、stage=3。权重加载入口预期 checkpoint['params']。

### CUDA 扩展与 Windows

核心架构依赖 torch 和 einops，没有自定义 CUDA 算子要求。README 明确采用 `setup.py develop --no_cuda_ext`；setup.py 默认分支却声明 DCN 等扩展，完整 Git 树没有对应 ops 源文件，不能盲目执行默认安装。

首个独立推理封装可按文件路径加载原始架构模块，避免 BasicSR 自动导入额外组件。若使用完整工具包，须补足其依赖并处理旧式 setup.py develop 与当前 setuptools 的兼容性，不能假定安装 BasicSR 就会自动补依赖：get_requirements() 直接返回空列表。

多 GPU shell 脚本使用 Bash，配置使用 NCCL；这些不是原生 Windows 单 GPU 推理的必需项，不能直接照搬。部分测试分支调用 CUDA IPC、使用 '/' 拆路径，需在 Windows 上进一步验证或从单图封装中移除不必要逻辑。

## 5. 权重获取与本次核验

### DI-Retinex

官方固定版本目录：[weights](https://github.com/sunshangquan/DI-retinex-low-light-enhancement/tree/3f94f95ffb181904a837532a70adc064bee793a2/weights)

| 文件 | HTTP Content-Range 报告总字节数 | 本次结果 |
| --- | ---: | --- |
| `latest (21.54 lolv1).pth` | 170775 | 206，ZIP 文件头，非 LFS 指针 |
| `latest (lolv1 small).pth` | 3479 | 同上 |
| `latest (lolv2 21.35).pth` | 170775 | 同上 |
| `latest (lolv2 small).pth` | 3479 | 同上 |

每个权重仅读取 128 字节；没有完整下载、反序列化或验证参数形状。文件体积小且端点可访问，权重可获取的证据强于 README 占位链接。

### Retinexformer

- [官方 Google Drive 权重目录](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV)
- [官方百度网盘](https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2)，提取码 cyh2；本次没有确认百度网盘下载可用性。
- Google Drive 目录实际返回 200，列出 LOL_v1、LOL_v2_real、LOL_v2_synthetic、SID、SMID、SDSD_indoor、SDSD_outdoor、FiveK、NTIRE 权重。
- `LOL_v1.pth` 文件 ID：`1xDwQtTCj3tlAVCTJgYrzonBGVwqeOhKu`。
- 其下载端点范围请求返回 206，Content-Range 为 `bytes 0-127/6475021`，ZIP 文件头；即约 6.48 MB，可访问。只读取 128 字节，未完整下载或加载。

两者均未验证 checkpoint 内部键名和模型 strict load；这些属于确认后执行的下一步。

## 6. 环境兼容性

当前环境：Python 3.10.22 / torch 2.13.0+cu130 / torchvision 0.28.0+cu130 / numpy 2.2.6 / Pillow 12.3.0。此次只查包元数据，没有导入或运行候选模型。

| 项目 | DI-Retinex | Retinexformer |
| --- | --- | --- |
| 作者环境 | Python 3.8 / PyTorch 1.9 | ICCV 推荐 Python 3.7 / torch 1.11；NTIRE 给出 Python 3.9 / torch 2 路线 |
| 本机 3.10 / 2.13 | 基础 Conv2d、激活、逐点运算预计容易适配，未实测 | 核心网络预计可适配，BasicSR 工具链更复杂，未实测 |
| 最小单图额外依赖 | 独立封装预计不需要新增包 | 核心网络至少需要 einops（当前缺失） |
| 原官方入口依赖 | cv2 / natsort 当前缺失 | cv2 / einops / yaml / tqdm / lmdb / scipy / skimage 等当前缺失，完整集合按调用链进一步锁定 |
| 自定义 CUDA 编译 | 未发现需求 | 核心网络不需要；完整 setup 默认扩展分支不要运行 |

两者的原始 torch.load 调用未明确 weights_only。本机 PyTorch 源码说明 2.6 起默认值变化；后续先用 `torch.load(..., map_location='cpu', weights_only=True)`，校验结构并严格加载。不能为了消除错误就自动改成不受限 pickle 加载。

不降级现有 PyTorch，不安装作者旧 CUDA 环境。现有 GPU 已通过基础测试，但不能据此保证任何历史模型/扩展都兼容。当前 README 中的旧训练环境也不保证能支持本机新 GPU。

## 7. 许可证和文档不一致

### DI-Retinex

完整 Git 树没有 LICENSE/COPYING 文件，README 也未提供明确授权条款；不能标为 MIT 或“无限制开源”。后续再分发、修改后发布或商业使用前，应联系作者确认适用授权；引用论文不能替代授权确认。

README 的引用提示写成 Histoformer、arXiv 链接格式嵌套、示例使用 pythonC，含空格的权重路径未加引号。这些不影响网络代码存在，但说明示例不能不经审阅直接复制执行。

### Retinexformer

根 LICENSE.txt 为 MIT，要求保留版权和许可文本；仓库内 BasicSR 的 setup.py 却声明 Apache-2.0。报告保留此差异，不将根 MIT 自动扩展为所有第三方代码、数据集和网盘权重的统一授权。权重文件是否有单独条款仍待下载时核查。

README 的超高分辨率支持是作者实现与实验说明，不能当作 8 GB 显卡实测承诺。LOL-v1 YAML 的注释仍写 300k 等旧计划，而实际 total_iter 和 scheduler 合计为 150k；复现实验要以明确的最终配置和作者协议为准。

## 8. 3840×2160 与 8 GB 显存风险

以下为静态内存估计，不是实测峰值。

- DI normal：一个 64 通道 4K FP32 特征图约 1.98 GiB，forward 还保留多个中间张量并使用卷积工作区。权重小不代表 4K 整图一定能装进 8 GB。
- DI small：4 通道中间特征显著减小，适合后续效率对照；不同结构必须配套 small 权重。
- Retinexformer：一个 40 通道 4K FP32 特征约 1.24 GiB；光照分支、Q/K/V、FFN 与多尺度缓存叠加，8 GB 下整图风险高。
- Retinexformer 使用通道注意力，不能错误声称它构造完整 HW×HW 注意力矩阵；主要风险仍是多份高分辨率激活。
- 官方高分辨率分支 (`test_from_dataset.py:238–255`) 采用奇偶列拆分，不是带重叠的通用滑窗拼接，也不保证与整图输出等价；不要把它描述成无损分块。

首测使用原图的内存中等比缩小版本（例如 1280×720），batch=1、FP32、eval/inference_mode，记录它不是原生 4K Benchmark。通过后再另行验证原分辨率和分块策略，不开启 self-ensemble。

DI 默认 linear=False、三层 3×3 局部卷积，后续可以设计带至少 3 像素邻域的裁块推理、裁去边缘后拼接；需保持原始全图边界填充方式并与小图整图结果验证一致性。Retinexformer 注意力含跨空间聚合，裁块不能默认与整图等价。

## 9. 首次推理具体方案（等待确认，不执行）

1. 保留并固定 DI 审查 commit。确认许可证/作者允许的研究使用范围；若需要明确宽松许可优先的路线，可改选 Retinexformer，但需核对其权重条款。
2. 下载官方 LOL-v1 normal 权重 `latest (21.54 lolv1).pth` 至 `checkpoints/di_retinex/`，记录来源、字节数、SHA256；该目录已被主仓库忽略。
3. 新增 NightUAV 独立单图封装，按路径只加载 mymodel.py；构造 `enhance_net_nopool(ch=6, number_f=64, linear=False)`。不导入 lowlight_test.py/dataloader.py，不执行原脚本的路径拼接。
4. 严格验证权重是匹配的 state_dict；使用 weights_only=True、CPU 加载后移动模型到 cuda:0，调用 eval()。若不匹配则停止，不静默 strict=False。
5. 只读打开 `E:\NightUAV\datasets\U3D\lowlight\test\0901.jpg`，转 RGB，内存中缩小至 1280×720，归一化到 [0,1]、NCHW、FP32。使用 inference_mode()，取模型四元组中的 enhanced_image。
6. 显式指定独立 PNG 输出路径，例如 `results/demos/di_retinex_0901_1280x720.png`。在写出前验证解析后的输出路径与输入不同、且不位于 datasets 下。
7. GPU 同步后记录耗时、峰值 allocated/reserved 显存、输入输出尺寸、数值范围、checkpoint hash、源码 commit；保持输出分辨率描述真实，不把缩小结果称作 4K 推理。
8. 用户验收后再决定原生 4K、小模型、U3D 批量评价及 EXP001 的配置。U3D 没有配对 GT，不计算伪造的 PSNR/SSIM；不把不同日间图用于参考质量指标。
9. U3D 重新训练作为独立后续阶段：只用训练划分，明确裁剪、loss、种子、曝光目标和验证协议。单图推理成功不代表训练或论文复现完成。

备选 Retinexformer 路线：确认后仅补核心所需 einops，按文件路径加载 RetinexFormer_arch.py，按 LOL-v1 YAML 构造网络，加载 LOL_v1.pth 的 params；用无 GT、无 self-ensemble、无 GT_mean 的安全单图封装，输入尺寸补齐为 4 的倍数并在输出裁回。不要为了首次推理直接安装整个旧 BasicSR 环境。

## 10. 完成边界与 Git 后续注意

已完成：两个环境文件提交/推送、候选源码审查、权重端点小范围验证、本选型报告。

未执行：候选依赖安装、PyTorch/CUDA 重配、完整权重下载、候选模型导入/推理/训练、正式实验创建、本报告提交。

`_model_audit/` 是本次明确要求创建的审查目录，内部有两个独立 Git 克隆，当前主项目 .gitignore 未排除它。后续不要直接 git add . 以免纳入嵌套仓库；应在用户确认后决定忽略、移到项目外或规范为其他管理方式。本次不擅自修改忽略规则。

## 依据链接

- [DI-Retinex 固定源码](https://github.com/sunshangquan/DI-retinex-low-light-enhancement/tree/3f94f95ffb181904a837532a70adc064bee793a2)
- [DI 模型](https://github.com/sunshangquan/DI-retinex-low-light-enhancement/blob/3f94f95ffb181904a837532a70adc064bee793a2/mymodel.py)
- [DI 训练](https://github.com/sunshangquan/DI-retinex-low-light-enhancement/blob/3f94f95ffb181904a837532a70adc064bee793a2/lowlight_train.py)
- [DI 推理路径处理](https://github.com/sunshangquan/DI-retinex-low-light-enhancement/blob/3f94f95ffb181904a837532a70adc064bee793a2/lowlight_test.py#L53)
- [Retinexformer 固定源码](https://github.com/caiyuanhao1998/Retinexformer/tree/1e9a0efce4b306b6701b824768370ff26066c32a)
- [Retinexformer 测试入口](https://github.com/caiyuanhao1998/Retinexformer/blob/1e9a0efce4b306b6701b824768370ff26066c32a/Enhancement/test_from_dataset.py)
- [Retinexformer LOL-v1 配置](https://github.com/caiyuanhao1998/Retinexformer/blob/1e9a0efce4b306b6701b824768370ff26066c32a/Options/RetinexFormer_LOL_v1.yml)
- [Retinexformer 许可证](https://github.com/caiyuanhao1998/Retinexformer/blob/1e9a0efce4b306b6701b824768370ff26066c32a/LICENSE.txt)
