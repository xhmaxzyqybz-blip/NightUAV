# EXP006B：局部动态范围控制相关工作审计

状态：完成本轮文献与静态代码审查，等待人工研究决策。没有修改网络、loss、权重、划分，没有新训练或候选模块。

## 目的与结果

针对H1-v3，核查局部曝光、高亮保护和结构保持先例。完成8篇全文方法部分及相关实验/消融阅读；代码核查深度逐项列明。U3LIE、UREC、Zero-DCE等已覆盖大量通用机制，不能将添加局部区间、越界惩罚或结构平滑直接宣称原创。报告提出最多3个待验证方向及公平对照、消融和否定条件，没有选定创新方案。

## 文件

- [研究报告](../../docs/RESEARCH_GAP_LOCAL_DYNAMIC_RANGE.md)：逐篇机制、对照表、重合风险及3个条件性研究方向。
- [SOURCES.md](SOURCES.md)：可点击论文、官方代码与权重入口、阅读定位及访问局限。
- [sources.json](sources.json)：机器可读证据索引；未确认官方仓库用null，避免伪造链接。
- [verification.json](verification.json)：本轮只读核查的E15、源码与split哈希及Git状态。

## 来源边界

EXP005B E15 SHA256：
`73bd2f1a50acc17aebb8c2fbc6e7b585cf8b38cc6b62573e2a2de0058a3f39d0`。

EXP006A统计属于固定8张dev和8-bit PNG诊断，不证明结构细节损失；用户报告的人工视觉结论作为研究背景，不虚构额外审核记录。未运行模型或重新生成输出。没有下载新模型权重，也没有验证其他方法在本机能否运行。

8篇：DI-Retinex、U3LIE、Zero-DCE、UREC、SCI（CVPR2022）、Retinex-Net、Retinexformer、EnlightenGAN。另有高相关2025论文全文端点不可读及引文链待补，已明确列出，不以摘要代替核实。

## 人工决策

先决定是否补充浮点越界/量化端点/边界结构诊断，以及是否批准普通boundary、局部区间、空间一致性等先例对照。任何新loss或训练需另行授权。未commit/push；原有未提交成果保持原状。
