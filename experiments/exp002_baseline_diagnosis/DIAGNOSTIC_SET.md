# U3D 小规模开发诊断集

固定官方权重推理已完成；没有训练、调参或使用test选样。清单保留原始train/val划分。

## 选样规则

只扫描 lowlight/train 与 lowlight/val 的900张图。用Pillow JPEG draft后缩至320×180的缩略图计算编码域亮度均值、标准差、暗区比例(Y′<0.1)、亮区比例(Y′>0.9)、相邻梯度绝对值。每个划分先取亮度最小/最大、亮区比例最大、标准差最大（去重），再对五维min-max归一化特征做确定性的最远点补齐至10张，路径字典序决定平局。所选20张SHA256均不同。

这些是照明覆盖样本，不是随机代表性抽样。缩略指标会受降采样影响；类别阈值仅作描述，不用来选择训练超参数。亮度均值<0.12记very_dark，<0.3记mixed_dark，否则relatively_bright，标准差>0.18加high_contrast。语义标签是本次查看联系表后的粗粒度人工标注，不是数据集官方标签。

## 固定清单

| 原划分 | 文件来源（项目相对路径） | 目视场景 | 统计照明标签 |
|---|---|---|---|
| train | `datasets/U3D/lowlight/train/0187.jpg` | 极暗建筑屋顶/少量窗灯 | very_dark |
| train | `datasets/U3D/lowlight/train/0472.jpg` | 相对明亮城市工业建筑/天空 | relatively_bright_high_contrast |
| train | `datasets/U3D/lowlight/train/0008.jpg` | 亮球场与暗植被 | mixed_dark_high_contrast |
| train | `datasets/U3D/lowlight/train/0740.jpg` | 亮球场/橙色路灯/暗背景 | mixed_dark_high_contrast |
| train | `datasets/U3D/lowlight/train/0234.jpg` | 商业街/店铺灯光/树木 | mixed_dark_high_contrast |
| train | `datasets/U3D/lowlight/train/0595.jpg` | 园区道路/植被/水面或空地 | mixed_dark |
| train | `datasets/U3D/lowlight/train/0148.jpg` | 暗住宅区/远处灯带 | very_dark |
| train | `datasets/U3D/lowlight/train/0115.jpg` | 道路车流/暖色灯光 | mixed_dark_high_contrast |
| train | `datasets/U3D/lowlight/train/0349.jpg` | 商铺沿街道路/紫色混合光 | mixed_dark |
| train | `datasets/U3D/lowlight/train/0775.jpg` | 住宅高楼/亮球场 | mixed_dark_high_contrast |
| val | `datasets/U3D/lowlight/val/0813.jpg` | 极暗屋顶/少量路灯 | very_dark |
| val | `datasets/U3D/lowlight/val/0852.jpg` | 商业街/道路/暖白混合灯光 | relatively_bright_high_contrast |
| val | `datasets/U3D/lowlight/val/0815.jpg` | 亮球场与暗植被 | mixed_dark_high_contrast |
| val | `datasets/U3D/lowlight/val/0850.jpg` | 暖色路灯/暗道路边缘 | mixed_dark_high_contrast |
| val | `datasets/U3D/lowlight/val/0894.jpg` | 园区道路/植被/水面 | mixed_dark |
| val | `datasets/U3D/lowlight/val/0825.jpg` | 商场/高楼/道路灯光 | mixed_dark_high_contrast |
| val | `datasets/U3D/lowlight/val/0856.jpg` | 城市高楼/冷暖混合光 | mixed_dark |
| val | `datasets/U3D/lowlight/val/0834.jpg` | 暗建筑/局部灯光 | very_dark |
| val | `datasets/U3D/lowlight/val/0801.jpg` | 高楼夹道/暖色道路灯 | mixed_dark_high_contrast |
| val | `datasets/U3D/lowlight/val/0876.jpg` | 商业街/密集招牌与人流 | mixed_dark_high_contrast |

完整尺寸、哈希、选样特征见 [JSON清单](diagnostic_manifest.json)。900张缩略统计见 [inventory](metrics/train_val_inventory.json)。[联系表](outputs/diagnostic_contact.png)；[成对图目录](outputs/diagnostic)。

## 划分风险与用途

人工目视发现 train/0008 与 val/0815 都是近似球场构图；train/0595 与 val/0894 是相似园区道路和水面；多个商业街样本也可能同场景。仅凭图像不能确定是否同一视频相邻帧，故标记“疑似场景重叠”，不声称已经证实泄漏或独立。未移动、删除或改写原始数据。

训练只使用原train；原val用于开发诊断与预先规定的早停，不参与梯度。当前val不能未经场景审计就宣称跨场景/跨数据集泛化。正式训练前需元数据、近重复检索和人工确认；若需分组清单，应另行记录排除/分组，保持原始文件不动。诊断集不能代替全部val100，更不是独立最终测试集。
