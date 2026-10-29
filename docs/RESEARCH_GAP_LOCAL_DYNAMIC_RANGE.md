# EXP006B：局部动态范围控制相关工作审计

审计日期：2026-10-11。状态：文献与静态代码审查完成，等待人工研究决策；没有实现候选方法或开展新训练。本报告不是原创性证明或穷尽式系统综述。

## 1. 判断与研究边界

H1-v3 可以作为待检验问题，但“局部曝光约束”“亮区抑制”“输出范围惩罚”“结构感知平滑”均已有明确先例。最直接的重合来自 U3LIE 的局部亮度区间和 UREC 的曝光引导、边界损失。不能把这些机制改名、组合或移植到 UAV 后直接称为原创。可研究的是：在冻结的 DI-Retinex adapted baseline 上，哪种机制与新增饱和存在可检验的联系，以及简单已有约束是否已足以改善暗区可见性与高亮层次之间的权衡。[U3LIE §方法](https://arxiv.org/html/2509.01373v1)，[UREC §3.3](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70077)。

已读取 8 篇论文的全文方法部分及相关实验/消融部分；其中 7 项有官方仓库，UREC 仅确认作者按请求提供代码/模型。全文不可访问的新线索单独列为未完成核查，不据其标题判断机制。逐项入口、定位及证据等级见 [SOURCES.md](../experiments/exp006b_prior_art_review/SOURCES.md) 和 [sources.json](../experiments/exp006b_prior_art_review/sources.json)。分支链接可能变动，审计日期不是代码永久版本保证。

## 2. 本地证据与不能推出的结论

冻结权重：`checkpoints/di_retinex/exp005b/epoch015.pth`。本轮重新计算 SHA256：
`73bd2f1a50acc17aebb8c2fbc6e7b585cf8b38cc6b62573e2a2de0058a3f39d0`。

依据 [EXP005B 配置](../experiments/exp005b_di_retinex_u3d_baseline/config/run.json) 和 [EXP006A 报告](../experiments/exp006a_failure_analysis/README.md)：

- 训练使用 v2.1 train 770；dev 8；quarantine 22；原始网络、RD:VS=1:1，15 epochs，随机初始化。256×256 patch 与官方完整图像训练不同，属于 adapted baseline。
- EXP006A 使用固定 1280×720 输入及保存的 E15 PNG；Y′=0.2126R+0.7152G+0.0722B，属于编码 RGB 亮度统计，不是线性物理亮度。
- 像素加权暗区平均增益约 0.17670；亮区约 0.11203。亮区输入已有任一通道255的比例约4.3164%，输出约97.5716%；新增饱和占整个亮区约93.2552%。
- 上述是 8-bit 任一通道端点统计。PNG 舍入到255不等于精确的浮点越界；单通道饱和也不等于全白。颜色差异不等于质量下降。
- 用户已报告人工观察到亮暗交界可能层次压缩；本次没有新增人工标签，也没有证明真实结构细节损失。仍不能据此宣称 H1 已成立或 DI-Retinex 理论错误。

## 3. 逐篇技术审查

### P01 DI-Retinex

**DI-Retinex: Digital-Imaging Retinex Theory for Low-Light Image Enhancement**：2024 arXiv v1；官方仓库标注 IJCV 2025（本轮技术公式依据公开 v1 和冻结源码，不冒充已比对最终期刊全文）。[论文 §3.4、§4.1–4.3](https://arxiv.org/html/2404.03327v1)，[官方仓库](https://github.com/sunshangquan/DI-retinex-low-light-enhancement)。

无需正常光 GT 或非配对正常光集合。normal 模型预测逐像素对比度/亮度系数，并非全图固定增益；理论本身已讨论动态范围溢出。核心为掩码 Gamma 空间反向退化和偏置方差抑制。

冻结 commit `3f94f95ffb181904a837532a70adc064bee793a2` 的 [模型](https://github.com/sunshangquan/DI-retinex-low-light-enhancement/blob/3f94f95ffb181904a837532a70adc064bee793a2/mymodel.py) 与 [训练入口](https://github.com/sunshangquan/DI-retinex-low-light-enhancement/blob/3f94f95ffb181904a837532a70adc064bee793a2/lowlight_train.py) 显示：
`z=kX+0.5(kb-k+b+1); Y=clamp(z,0,1)`；
`M=(Y<1); r=E/mean(X.detach())`；
`RD=mean(abs(M*(G**2.2-r**(-2.2)*Y**2.2)))`；
`VS=mean(var_spatial(kb-k+b+1))`。G 是输入的 Gaussian 平滑图。VS 不是局部 TV。

**与 H1-v3：** RD 饱和掩码不是主动防饱和惩罚；clamp 外侧 RD 路径无梯度是代码事实，但 VS 和共享参数仍可影响对应位置。由此推测训练约束可能不足，尚非因果证明。不得声称首次提出逐像素增益、动态范围问题或 Retinex 偏置。完整 train/test/model 已公开；官方权重目录存在，本项目曾完成预训练推理，本轮没有重下载或加载权重。

### P02 U3LIE

**Unsupervised Ultra-High-Resolution UAV Low-Light Image Enhancement: A Benchmark, Metric and Framework**：2025 arXiv v1；本轮未确认正式录用来源。[论文方法、式8–12](https://arxiv.org/html/2509.01373v1)，[官方仓库](https://github.com/AeroVILab-AHU/U3D_Toolkit)。

DCE++ 派生的轻量曲线网络；训练用 APA、区间损失、空间一致性、颜色和 TV。无需成对正常光 GT；原图与 APA 派生图不是低/正常光实拍 GT 对。区间损失对局部均值低于下界、高于上界分别平方惩罚，并约束全局均值。APA 包括自适应 gamma、CLAHE、颜色校正与 V 缩放。

[公开 loss](https://github.com/AeroVILab-AHU/U3D_Toolkit/blob/main/u3lie_loss.py) 中实际组合使用 patch16、暗目标0.5、亮目标0.6、全局0.6、全局权重0.4；不是独立 L_INT 类的默认0.3/0.7。代码“亮度”为 RGB 通道均值，与本项目 Y′定义不同。仅公开 APA.py、EEI.py、u3lie_loss.py 等；缺模型和完整入口，loss 还引用缺失的工厂/基类。README 模型与配置链接本轮均返回404。

**与 H1-v3：** 最直接先例。局部双侧曝光约束和高亮抑制不能作为原创。潜在待测不足：区域平均值不直接约束每个 RGB 通道的尾部；该数学区别不等于已证实它在 U3D 上失败。代码不完整不消除论文先例。

### P03 Zero-DCE

**Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement**：CVPR 2020。[全文 §3、§4.1](https://arxiv.org/pdf/2001.06826)，[官方仓库](https://github.com/Li-Chongyi/Zero-DCE)。

无需 GT；逐像素迭代曲线 `I+aI(1-I)`，a 在[-1,1]，对合法输入保持范围。损失包含局部曝光、空间一致性、颜色恒常及系数平滑；曝光目标0.6、16×16区域。论文训练还包含过曝图，不能把低光-only重训当原协议。空间一致性已经涉及相邻区域对比关系。

[代码 Myloss.py](https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py) 的曝光项是平方差，论文式5为绝对差；[训练代码](https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/lowlight_train.py) 的 TV/spatial/color/exposure 权重为200/1/5/10，也不能直接套用论文数字。train/test/model及 snapshots/Epoch99.pth 公开，本轮未下载验证。

**与 H1-v3：** 有界映射、局部曝光、空间一致性均为先例。保持数值范围不保证不会接近1并量化为255；固定区域均值也不等于逐像素高亮保护。后者是需要实验核实的局限推测。

### P04 UREC

**Unsupervised Retinex Exposure Control: A Novel Approach to Image Enhancement**：IET Image Processing 19(1), e70077，2025。[全文 §3及式7–12、附录](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70077)。

无需 paired GT。原图及反相图分解得到反射率，用 `I_en=I+K_I R_I-K_inv R_inv` 同时调亮与压亮。分解使用保真和结构加权平滑；曝光模块使用 K 平滑、空间 guidance map及其全局均值约束、上下界 hinge 惩罚。还有 guided filter 修正边缘。附录讨论自动生成 guidance，不能说它只能人工指定。

**与 H1-v3：** 局部曝光目标、平滑增益、越界惩罚同时存在，是很高的重合风险。简单 `relu(z-1)` 加入 DI 不构成新机制。尚需检验 guidance、平滑与小尺度 UAV 灯光边缘的适配；本文并不能保证恢复输入已截断的信息。

论文 Data Availability 写明代码及预训练模型按合理请求提供；本轮未找到可核实的官方公开 GitHub，未获取或运行实现。此项是论文机制核查，不能称代码复现完成。

### P05 SCI（仅 CVPR 2022 版本）

**Toward Fast, Flexible, and Robust Low-Light Image Enhancement**：CVPR 2022。[全文 §2.1–2.3、式4–5](https://arxiv.org/pdf/2204.10137)，[官方 CVPR 子目录](https://github.com/vis-opt-group/SCI/tree/main/CVPR)。

无需正常光 GT；共享照明估计与训练期自校准，推理用单阶段；损失是校准输入保真和空间变化加权平滑。不是显式逐通道高亮新增饱和损失。结构权重与照明自适应已是先例；对小光源的适配优劣没有在本项目验证。

[loss.py](https://github.com/vis-opt-group/SCI/blob/main/CVPR/loss.py) 返回1.5*MSE+SmoothLoss；论文平滑参数σ=0.1，公开代码σ=10，颜色变换与邻域实现也须单独核对，不能直接宣称等价。train/test/model/loss和easy/medium/difficult.pt存在；README明确模型不是论文报告结果所用模型。仓库还包含 TPAMI 2025 扩展，本报告不混用其结论。

### P06 Retinex-Net

**Deep Retinex Decomposition for Low-Light Enhancement**：BMVC 2018。[全文 §2.1–2.3](https://arxiv.org/pdf/1808.04560)，[作者项目页](https://daooshee.github.io/BMVC2018website/)，[官方仓库](https://github.com/weichen582/RetinexNet)。

需要成对低/正常光训练图，虽不需反射率/照明真值，不能称低光-only无监督。分解损失有重建、跨曝光反射率一致性、结构感知照明 TV；增强阶段有正常光重建与平滑，多尺度融合调节照明。

[model.py](https://github.com/weichen582/RetinexNet/blob/master/model.py) 用反射率梯度引导照明平滑（指数衰减），main.py及 Decom/Relight checkpoint目录公开；本轮未加载。TensorFlow 实现不能直接当现有 PyTorch 训练插件。

**与 H1-v3：** 边缘感知照明平滑、多尺度局部照明调整不可宣称原创；不是专门的“输入未饱和→输出新饱和”约束。直接拿其监督重训结果与低光-only方案比较有监督信息不对等问题。

### P07 Retinexformer

**Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement**：ICCV 2023。[全文 §3](https://arxiv.org/html/2303.06705v2)，[官方仓库](https://github.com/caiyuanhao1998/Retinexformer)。

照明估计先提亮，照明引导注意力及残差恢复处理退化；考虑欠/过曝等扰动，不能说现有 Retinex 网络从不关注曝光。其 [LOL-v1配置](https://github.com/caiyuanhao1998/Retinexformer/blob/master/Options/RetinexFormer_LOL_v1.yml) 使用 Dataset_PairedImage 和 L1Loss，需要配对正常光 GT。不是可直接对 U3D lowlight-only 按原目标重训的方案。

模型/训练/测试公开；README有 Google Drive/Baidu 权重入口，本轮只核实链接，未下载/重载。部分测试选项 GT_mean 利用 GT 均值，本项目无 GT 不可采用。

**与 H1-v3：** 照明引导的区域交互已有先例；没有在所审查配置看到新增饱和专用损失，不等于它必然出现该问题。不能把引入照明 attention 直接视为原创，且大网络替换会混入容量因素。

### P08 EnlightenGAN

**EnlightenGAN: Deep Light Enhancement without Paired Supervision**：IEEE TIP 2021（arXiv初版2019）。[全文 §III](https://arxiv.org/html/1906.06972v2)，[官方仓库](https://github.com/VITA-Group/EnlightenGAN)。

不需要一一配对 GT，但需非配对正常光域，不能和仅用低光输入等同。U-Net 的1-I注意力偏向暗处；全局/局部对抗损失配合全局/局部 VGG 自特征保持。不是单独的逐通道输出上界损失。

仓库 train.py、模型及数据代码公开；README有增强模型和VGG权重下载链接，增强模型 Drive 页面本轮可解析到文件名，未下载、未验证加载。

**与 H1-v3：** 按输入亮度区分增强强度、局部约束、输入特征保持已有先例。额外正常光数据与VGG先验改变信息条件；对抗局部真实性不直接给出物理细节恢复证明。不得把暗区 attention 或输入感知一致性当新贡献。

## 4. 机制对照与不可作为原创的内容

下表的机制依据 P01–P08 的原文/代码；“潜在不足”是本项目待测推断，不是已测排名。

|已有方法|解决的问题 / 具体机制|适用条件|潜在不足或比较边界|不可作为我们的原创|
|---|---|---|---|---|
|DI-Retinex|数字成像偏置；像素仿射映射、masked RD、VS|低光-only|饱和掩码与局部高亮控制的关系待测|像素增益/偏置、动态范围问题|
|U3LIE|UHR非均匀曝光；APA、局部双侧区间+全局均值|低光及其派生APA输入|patch均值不能直接约束通道尾部；完整实现缺失|亮度区间、局部亮暗双约束|
|Zero-DCE|无GT曝光调整；有界曲线、曝光/空间/颜色/TV|无参考；原训练含多曝光|范围有效≠8-bit不饱和；单目标可能不适合所有区域|有界曲线、局部曝光/对比保持|
|UREC|欠曝与过曝共存；反相Retinex、引导图、边界惩罚|无paired GT；代码按请求|引导目标/空间尺度在UAV待测|上界惩罚、空间曝光引导、平滑K|
|SCI|低成本照明增强；自校准、保真+加权平滑|低光-only|公开模型/参数与论文有差异|结构加权平滑、自适应照明|
|Retinex-Net|分解及照明调节；反射率一致性、结构TV|paired训练|不能直接按原目标无监督重训|边缘感知照明正则、多尺度调整|
|Retinexformer|提亮伴随退化；照明引导注意力+恢复|paired训练|容量、数据、预算与DI不同|照明引导attention|
|EnlightenGAN|非配对域增强；亮度attention、局部GAN/SFP|需正常光域和VGG先验|先验和训练稳定性不同|暗区加权、局部感知一致性|

公开代码或权重不代表已验证兼容本机，也不代表可完整复现论文。未公开实现不代表对应思想可重新作为原创。

## 5. 代码与论文差异的处理原则

1. DI：v1正文写1000 epochs、不裁patch；代码默认200 epochs，Normal第二参数是std=0.001，RD使用Gaussian图。沿用冻结源码描述现有baseline，不把本项目15-epoch patch训练冒充原论文配置。
2. Zero-DCE：曝光绝对差/平方差以及loss权重存在差异。今后若移植先例对照，必须注明 paper-form 或 released-code-form；不得挑选有利版本却不报告。
3. SCI：只审CVPR目录；版本扩展、公开权重免责声明、平滑参数差异均需保留。
4. U3LIE：README结构不代表实际文件，Model/config链接404；L_INT类默认参数与组合实际传参不同。EEI是质量与效率组合评价，不是纯图像质量或本次候选loss。
5. Retinexformer：禁用GT_mean作无参考比较；预训练跨数据集推理、U3D重训、完整论文复现是三种不同实验。
6. 所有方法的“亮度”定义不一定相同。RGB均值、Y′、HSV V、线性亮度不可直接混用。

## 6. 最多三个候选方向（仅假设，未授权实施）

### H-A：在相近暗区增益下，针对截断前越界的约束是否有独立收益？

- **具体问题/证据：** E15同时存在暗区增益与大量亮区新增8-bit端点；须先区分float越界、输入已有饱和、量化舍入。现有记录没有证明哪类主导局部层次。
- **现有机制可能不足：** DI的masked RD并非防越界项；但UREC式12已经有明确boundary loss。因此这是机制验证，重合风险很高。
- **最小技术部分（待批准）：** 保持网络，单独研究训练期对可重构的pre-clamp输出的约束；RD/VS仍保留。先做普通已有边界项对照，再考虑按输入可用余量区别处理，不能一开始包装为新loss。
- **公平实验：** 原DI vs 普通边界先例 vs 余量条件版本；另加仅整体压暗的固定后处理对照。必须同时报告新增饱和和暗区增益/人工可见性，防止靠全图变暗“获胜”。
- **消融：** 无附加项 / 无条件边界 / 条件边界；RGB通道与亮度版本分开；输入已有饱和与未饱和分开。权重搜索预算相同。
- **否定与风险：** 若收益不优于普通边界或整体曝光降低，或暗区明显回退，应否定条件化收益；可能因clamp后计算而无有效梯度，也可能引入色偏。仅减少>1比例不能证明结构恢复或原创。

### H-B：局部目标的空间尺度是否决定小光源与暗背景的权衡？

- **具体问题/证据：** 相同图像包含暗背景与亮灯；现有全图/大区域均值可能掩盖稀疏亮通道尾部，这是待测数学解释。
- **现有机制可能不足：** U3LIE已有16×16区间；Zero-DCE已有局部曝光；UREC已有空间guidance。不能说“此前只有全局曝光”。
- **最小技术部分（待批准）：** 网络不变，仅比较已有局部区间约束的尺度与输入条件权重；先检验单尺度先例是否足够。任何自适应版本都需说明与这些方法的具体差异。
- **公平实验：** 原DI、固定区间先例、条件化版本，保持训练数据及预算。固定720p dev，不能用不同分辨率制造优势。
- **消融：** 固定单尺度/固定多尺度/条件化；有无全局项；相同额外约束强度与参数搜索预算；区域按输入冻结而非按输出重新选。
- **否定与风险：** 若简单单尺度已达相同权衡，复杂条件化无理由；均值约束可能继续漏掉小灯、压制真实亮物体或造成光晕。多尺度/分区本身高度重合，尚无原创性依据。

### H-C：在真实结构压缩确认后，结构条件增益平滑能否避免交界处层次退化？

- **前置门槛：** 先用固定输入选取、完整公开的边界剖面与人工标注，区分曝光变化、噪声和真实边缘/细节。若不能确认压缩，不开展“恢复结构损失”的方法实验。
- **现有机制可能不足：** Retinex-Net/SCI已有结构加权平滑，Zero-DCE已有空间一致性，UREC已有K平滑及guided filter；这些未必适配当前DI系数和UAV小目标，但尚未实测。
- **最小技术部分（待批准）：** 不加大型模块，仅比较已有无条件平滑与输入结构条件的系数/局部对比约束。结构定义先固定，不把伪影高梯度当细节。
- **公平实验：** 原DI、普通空间一致性先例、结构条件版本；曝光与暗区增益共同报告。禁止只截取改善的边界。
- **消融：** 亮度条件与结构条件拆开；无条件/结构条件平滑；去掉边界权重。噪声区域与人工确认边缘分别统计。
- **否定与风险：** 若效果仅是锐化噪声、出现光晕，或与普通TV/空间一致性无差别，否定该方向。结构保持、边缘感知或梯度loss不具自动原创性。

## 7. 后续公平实验的共同前提（本轮未执行）

- 冻结E15保留为历史基准；任何proposed从相同随机初始化重新训练，不从E15微调后与随机训练baseline比较。seed1149配对；若初步结果支持，再预注册额外种子验证波动。
- 同一v2.1、770/8/22、全黑crop同图重采样协议、Gaussian/crop顺序、Adam/lr/clip、15 epochs/11,550 updates。除明确研究变量外保持一致；禁止训练官方val/test或quarantine。
- 固定E15比较，不自动选best；所有新增超参数在训练前登记，给对照相同搜索预算，不能反复看8张dev后宣称泛化。
- 联合报告：每图与区域的暗区增益、输入已有/输出新增饱和、float越界和8-bit端点、颜色变化、人工结构观察、耗时/显存。空区域记null，不能当0；同时给逐图统计，避免像素加权掩盖个例。
- 先登记“可接受暗区可见性变化”及人工审核标准，再研究饱和权衡；当前不虚构阈值或把任一无参考统计称质量分数。无paired GT不能算与输入对比的PSNR/SSIM当增强质量。
- dev只有8张且已用于假设形成，缺完整拍摄序列元数据。后续独立泛化验证需要另行确认场景独立的评估设计。720p结果不外推原生4K细节。
- UREC不可用代码时，自己实现boundary项只能称机制对照，不能称复现完整UREC；同理仅实现L_INT不能称完整U3LIE。
- 训练新方法、修改loss或网络均等待另一次明确授权，本报告没有执行这些动作。

## 8. 未闭合的文献风险与人工决策点

高相关线索 [Region-Adaptive Low-Light Image Enhancement with Light Effect Suppression and Detail Preservation（2025，Eurographics站点）](https://diglib.eg.org/server/api/core/bitstreams/2ad23d33-0750-4283-9d4f-a182ae64ccea/content) 的全文端点本轮重复访问失败；不推断其loss、GT需求、代码或权重状态，也不将其计入8篇已读。NightEnhance、CuDI、LCD及SCI期刊扩展等引文链尚未逐篇完成同深度审查。后续宣称新颖性前必须补齐这些直接相关工作及更新检索。

建议人工先决定是否开展“浮点截断—量化饱和—边界层次”的补充诊断，以及是否以普通boundary/区间/空间一致性作为必要先例对照。当前可以确定存在研究问题与较高先例重合；不能确定任何候选方向具有原创性。
