# DI-Retinex 方法学习笔记

状态：源码审查 + 固定预训练权重诊断；没有训练。依据固定 commit `3f94f95ffb181904a837532a70adc064bee793a2`，不将 README 宣传或论文结果当成本项目测量。

## 依据与范围

- [模型源码](../_model_audit/DI-retinex-low-light-enhancement/mymodel.py)：第 18–20、26–39、41–67 行。
- [训练源码](../_model_audit/DI-retinex-low-light-enhancement/lowlight_train.py)：第 27–76、92–105 行。
- [加载器](../_model_audit/DI-retinex-low-light-enhancement/dataloader.py)：第 17–24、39–57 行。
- [论文 arXiv v1](https://arxiv.org/html/2404.03327v1)：§4.1–4.3 式17、20–23，§5.1。这里只核实此公开版本，未核实 IJCV 最终排版版本。论文支持“亮度/对比度映射、掩码重退化、偏置方差约束”这一方法设计；以下精确实现以本地代码为准。

## 网络输入、输出和变换

输入 X 是 RGB、float32、[0,1]、N×3×H×W。normal 网络由三层 3×3、stride=1、padding=1 卷积组成：3→64→64→6，前两层 ReLU。没有显式反射率/照明图分解、BatchNorm 或 Dropout。第三层分为三个通道的 c 与 b，分别使用 tanh 与 sigmoid；因此 c∈(-1,1)，b∈(0,1)。linear=False，无额外线性分支。

与 mymodel.py 一致的逐元素表达：

$$k=\tan((45-44.8c)\pi/180),\quad d=\tfrac12(kb-k+b+1),\quad z=kX+d,\quad Y=\mathrm{clip}(z,0,1).$$

forward 返回 `(Y,c,b,k)`，不能误把参数图保存为增强图。这里 k 是代码的增益；论文有不同符号约定，不能将论文像素上界 k 与代码增益混淆。三层卷积预测系数的理论感受野为 7×7（边缘受 padding 影响）。k 的理论范围约 0.00349 到 286.48，不代表实测始终达到该范围。

## 训练中实际使用的两项损失

定义 N 为 batch，P=HW，C=3；所有数值在 [0,1] RGB 编码域。加载器先对原始 uint8 RGB 做 OpenCV `GaussianBlur((9,9), sigmaX=1.7, sigmaY=1.7)`，再除以255，得 G。它不是正常光参考，也不是随机生成的高斯噪声。

每个 batch 采样一个标量 E，训练默认曝光均值0.5：

$$E\sim\mathcal N(0.5,0.001^2),\quad \mu=\mathrm{mean}_{N,C,H,W}(X),\quad r=E/\mathrm{stopgrad}(\mu),\quad M=\mathbf1[Y<1].$$

`Normal` 的第二参数是标准差，故代码方差为 10^-6。ratio 使用整个 batch 的均值，不是逐图均值；batch=1 时才与逐图一致。

### 1. 掩码重退化（lowlight_train.py:59）

$$L_{RD}=\frac1{NCP}\sum_{n,c,p}\left|M_{ncp}\left(G_{ncp}^{2.2}-r^{-2.2}Y_{ncp}^{2.2}\right)\right|.$$

实现 `(mask * (img_gauss**2.2 - (1/ratio)**2.2 * enhanced_image**2.2)).abs().mean()`。

分母是全部通道元素，不是有效 mask 数。mask 是基于模型输出的硬比较，比较本身不可导。Y≥1 的位置不贡献这一项；clamp 外侧对 z 的梯度为零。不要笼统说整个网络在饱和像素完全没有梯度：方差损失仍直接作用于 b、k，共享卷积也影响其他像素。

### 2. 偏置方差抑制（lowlight_train.py:60）

令 q=kb-k+b+1=2d：

$$\bar q_{nc}=P^{-1}\sum_pq_{ncp},\quad L_{VS}=\frac1{NC}\sum_{n,c}\frac1{P-1}\sum_p(q_{ncp}-\bar q_{nc})^2.$$

实现 `q.var((2,3)).mean()`；当前 PyTorch 默认 correction=1，因此是样本方差。没有约束均值为零；它是每图每通道的全空间方差，不是 TV 或局部梯度平滑。数学上等于4倍 d 的相同定义方差，不能直接以 Var(d) 替换后还声称权重不变。

总损失 `L=L_RD+L_VS`，两项系数均1。Adam 的 weight_decay=1e-4 另作用于优化过程，不是代码中显式第三项图像损失。没有显式颜色一致性、曝光均值误差、感知、GT重建或噪声监督项。

## 为何无需正常光参考

两项仅依赖低光 X、其平滑图 G、预测 b/k/Y 和随机曝光先验 E。训练不需要 paired GT，也不需要非配对正常光图集；推理同样不需要 GT。这里“无监督/零参考”不意味着“免训练”：公开脚本有共享网络训练循环。EXP001/EXP002 使用 LOL-v1 已训练权重，尚未在 U3D 更新任何参数。

## 高亮、噪声和细节：代码推论，尚非因果结论

- 增益和偏置可以令 z>1；clamp 丢弃超范围细节。mask 排除这些通道，使 RD 本身不惩罚它们继续变亮。VS 可能间接约束，故不能断言饱和无代价或 mask 必然造成实际全部过曝。
- E/mean(X) 是全局标量：明亮灯牌与暗背景共存时可能存在局部曝光需求冲突。这只是 U3D 域上的待检验解释。
- 条件固定系数时，输入噪声被 k 放大；真实网络的 c/b 也随输入变化，因此局部雅可比不是简单 k。需扰动实验而不能凭纹理高频就称噪声。
- 平滑的 G 提供一定抑噪倾向，也可能弱化细节。VS 抑制偏置空间变化但不等于直接抑制增益或估计传感器噪声。
- RGB 已经过 ISP/JPEG，4K→720p LANCZOS 本身会改变噪声与细节；当前诊断不能推断 RAW 性质，也不能保证原生4K效果。

## 论文、代码和默认参数差异

| 项目 | 公开论文 v1 | 当前代码 | 处理 |
|---|---|---|---|
| 训练轮数 | §5.1写1000 | 默认200 | 两者分别标注，不声称默认脚本复现论文 |
| 曝光随机量 | 文本将0.001称为方差 | Normal scale=0.001，方差1e-6 | 先保持代码语义；论文一致性留待核实 |
| RD目标 | 式20写低光 I_l | 实际用高斯平滑 G | 不擅自删除平滑 |
| 数据路径 | 训练集训练 | 默认指向 LOL-v2 Test/Low | 必须显式指定 U3D/train；禁止照用默认 |
| 尺寸 | 不裁剪、batch1 | size=256未使用，resize被注释 | 不能误以为默认256输入 |
| 断点/验证 | 本地未有完整协议证据 | 只存 state_dict，每100 epoch，无验证早停 | U3D平台单独补充，明确工程扩展 |

额外风险：ratio 没有 epsilon、全黑输入可能非有限；bool类型CLI易误解析；torch.load旧写法不应照搬；当前环境缺 cv2/natsort，原训练入口暂不可直接运行。以上是静态发现，并未执行原训练脚本。
