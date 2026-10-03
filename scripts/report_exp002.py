import json,csv
from pathlib import Path
import numpy as np
import argparse
ap=argparse.ArgumentParser(description='Build EXP002 reports from existing measured JSON; refuses overwrite')
ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
ap.add_argument('--experiment',default='exp002_baseline_diagnosis')
args=ap.parse_args();r=args.root.resolve();e=(r/'experiments'/args.experiment).resolve();m=e/'metrics'
if not e.is_relative_to((r/'experiments').resolve()):raise ValueError('Invalid experiment folder')
d=json.loads((m/'diagnostic_metrics.json').read_text(encoding='utf-8-sig'));s=json.loads((m/'exp001_statistics.json').read_text(encoding='utf-8-sig'));a=json.loads((m/'exp001_audit.json').read_text(encoding='utf-8-sig'));manifest=json.loads((e/'diagnostic_manifest.json').read_text(encoding='utf-8-sig'))
def put(p,text):
    with p.open('x',encoding='utf-8') as f:f.write(text)
scenes=['极暗建筑屋顶/少量窗灯','相对明亮城市工业建筑/天空','亮球场与暗植被','亮球场/橙色路灯/暗背景','商业街/店铺灯光/树木','园区道路/植被/水面或空地','暗住宅区/远处灯带','道路车流/暖色灯光','商铺沿街道路/紫色混合光','住宅高楼/亮球场','极暗屋顶/少量路灯','商业街/道路/暖白混合灯光','亮球场与暗植被','暖色路灯/暗道路边缘','园区道路/植被/水面','商场/高楼/道路灯光','城市高楼/冷暖混合光','暗建筑/局部灯光','高楼夹道/暖色道路灯','商业街/密集招牌与人流']
lines=['# U3D 小规模开发诊断集\n','固定官方权重推理已完成；没有训练、调参或使用test选样。清单保留原始train/val划分。\n','## 选样规则\n','只扫描 lowlight/train 与 lowlight/val 的900张图。用Pillow JPEG draft后缩至320×180的缩略图计算编码域亮度均值、标准差、暗区比例(Y′<0.1)、亮区比例(Y′>0.9)、相邻梯度绝对值。每个划分先取亮度最小/最大、亮区比例最大、标准差最大（去重），再对五维min-max归一化特征做确定性的最远点补齐至10张，路径字典序决定平局。所选20张SHA256均不同。\n','这些是照明覆盖样本，不是随机代表性抽样。缩略指标会受降采样影响；类别阈值仅作描述，不用来选择训练超参数。亮度均值<0.12记very_dark，<0.3记mixed_dark，否则relatively_bright，标准差>0.18加high_contrast。语义标签是本次查看联系表后的粗粒度人工标注，不是数据集官方标签。\n','## 固定清单\n','| 原划分 | 文件来源（项目相对路径） | 目视场景 | 统计照明标签 |','|---|---|---|---|']
for row,scene in zip(manifest['images'],scenes):lines.append(f"| {row['split']} | `{row['source']}` | {scene} | {row['illumination_type']} |")
lines.extend(['\n完整尺寸、哈希、选样特征见 [JSON清单](diagnostic_manifest.json)。900张缩略统计见 [inventory](metrics/train_val_inventory.json)。[联系表](outputs/diagnostic_contact.png)；[成对图目录](outputs/diagnostic)。\n','## 划分风险与用途\n','人工目视发现 train/0008 与 val/0815 都是近似球场构图；train/0595 与 val/0894 是相似园区道路和水面；多个商业街样本也可能同场景。仅凭图像不能确定是否同一视频相邻帧，故标记“疑似场景重叠”，不声称已经证实泄漏或独立。未移动、删除或改写原始数据。\n','训练只使用原train；原val用于开发诊断与预先规定的早停，不参与梯度。当前val不能未经场景审计就宣称跨场景/跨数据集泛化。正式训练前需元数据、近重复检索和人工确认；若需分组清单，应另行记录排除/分组，保持原始文件不动。诊断集不能代替全部val100，更不是独立最终测试集。\n'])
put(e/'DIAGNOSTIC_SET.md','\n'.join(lines))
rows=[]
for x in d:
    rows.append(dict(source=x['source'],split=x['split'],input_mean_y=x['input']['mean_y'],output_mean_y=x['enhanced']['mean_y'],input_saturated_channel_pct=x['input']['channel_255']*100,output_saturated_channel_pct=x['enhanced']['channel_255']*100,preclamp_gt1_channel_pct=x['preclamp_gt1_channels']*100,highpass_std_ratio=x['enhanced']['highpass_std']/x['input']['highpass_std']))
with (m/'diagnostic_summary.csv').open('x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
summary={split:{'n':len(g),'input_mean_y':float(np.mean([x['input_mean_y'] for x in g])),'output_mean_y':float(np.mean([x['output_mean_y'] for x in g])),'input_saturated_channel_pct_mean':float(np.mean([x['input_saturated_channel_pct'] for x in g])),'output_saturated_channel_pct_mean':float(np.mean([x['output_saturated_channel_pct'] for x in g])),'highpass_increased_images':sum(x['highpass_std_ratio']>1 for x in g)} for split in ('train','val') for g in [[x for x in rows if x['split']==split]]}
put(m/'diagnostic_aggregate.json',json.dumps(summary,indent=2))
report='''# EXP002：实现复核与基线诊断

状态：已实际运行；没有训练、优化或修改模型。EXP001/test 的分析只用于实现完整性检查，不进入创新假设证据集；开发诊断使用独立选择的 train10 + val10。

## 重复运行

```powershell
python -B scripts/analyze_exp002.py --experiment exp002_repeat_NEW
```

Portable template from the project root; replace the experiment name. Actual historical command: metrics/provenance.json. 脚本要求新实验目录，重复执行使用 `--experiment exp002_baseline_diagnosis_repeat01`，避免覆盖已有分析。需要现有 numpy/Pillow/torch，官方源码与EXP001权重/原始结果仍须在原路径；没有安装新依赖。来源commit、权重hash、脚本hash、解释器版本记录在 [provenance](metrics/provenance.json)。图像只写本实验outputs（Git已忽略）。

## EXP001 流程核查（测试样本，仅实现审计）

重新读取原图、RGB→LANCZOS→1280×720，与保存input逐像素一致；核对原图、两张PNG、官方源码和权重哈希。用同一官方模型重放一次，增强PNG与旧结果逐像素相同（最大差0）。重构未裁剪z=k(X+0.5(b−1))+0.5(b+1)，裁剪后与官方输出一致；全部张量有限。RGB/[0,1]/NCHW、eval/inference_mode/CUDA、取第一个返回值均正确。

numpy.rint 和官方 torchvision save_image 的 floor(x*255+0.5) 在半整数上最多差1灰度级，本次已实测，并非严重偏亮的来源。无需修改已完成的EXP001。缩放是明确实验选择：不能据720p结论推断4K；没有做额外gamma矫正或自动曝光。

裁剪前已有 **24.0191%通道>1**，故饱和不是保存PNG引入。详细检查见 [audit](metrics/exp001_audit.json)。没有发现足以解释明显过曝的RGB/BGR、归一化、输出索引或保存错误；这不保证模型训练与论文完全一致。

## 同坐标裁剪与统计

[裁剪对比](outputs/exp001_aligned_crops.png)，左输入右增强；坐标以1280×720左上角为原点，xyxy右/下边界不包含，保持1:1局部像素，不分别自动调亮。

| ROI | xyxy | 输入均值Y′ | 增强均值Y′ | 输入全白像素% | 增强全白像素% |
|---|---|---:|---:|---:|---:|
'''
for name,v in s.items():
    report+=f"| {name} | {v.get('box_xyxy_exclusive','全图')} | {v['input']['mean_y']:.4f} | {v['enhanced']['mean_y']:.4f} | {v['input']['all_channels_255']*100:.3f} | {v['enhanced']['all_channels_255']*100:.3f} |\n"
report+='''
亮度定义Y′=(0.2126R+0.7152G+0.0722B)/255，直接作用于编码RGB，不做sRGB线性化，不是物理亮度或曝光量。[直方图SVG](metrics/exp001_histogram.svg)、[64箱CSV](metrics/histogram.csv)、[完整统计](metrics/exp001_statistics.json)均已保存。箱范围[0,1]，最后一箱包含1。

统计分别区分“任意通道255的像素”“所有通道255的像素”“通道元素255”，不能互换。图中暗部提升但颗粒变明显，亮立面大片细节被白色截断，道路局部灯光溢出。梯度L1是水平/垂直相邻差绝对值均值的平均；高频代理是内部像素减四邻域均值的标准差，均以Y′计算。它们混合纹理、边缘、噪声，不是SNR，也不能直接判断细节恢复。暗屋顶ROI的高频代理甚至略降，不能用主观颗粒描述替代计量。

## train/val 开发诊断（用于假设准备）

[20张清单与场景标签](DIAGNOSTIC_SET.md)，[机器清单](diagnostic_manifest.json)，[逐图指标](metrics/diagnostic_summary.csv)。选样仅看输入的缩略图统计，之后固定官方LOL-v1权重与1280×720流程推理，无模型调参。900张缩略读取成功；20张均完成实际CUDA前向、有限值检查和输出保存，原图前后哈希一致。未把这些统计包装为图像质量评价分数。

| 子集 | 张数 | 输入Y′均值的平均 | 输出Y′均值的平均 | 输入饱和通道%平均 | 输出饱和通道%平均 | 高频代理增加张数 |
|---|---:|---:|---:|---:|---:|---:|
'''
for split,v in summary.items():report+=f"| {split} | {v['n']} | {v['input_mean_y']:.4f} | {v['output_mean_y']:.4f} | {v['input_saturated_channel_pct_mean']:.3f} | {v['output_saturated_channel_pct_mean']:.3f} | {v['highpass_increased_images']} |\n"
report+='''
这些平均数为逐图等权，不是随机样本总体估计。检查了原图联系表及 train0472、0595、0187、val0894 的成对图：明亮建筑/园区出现大面积发白，极暗屋顶出现提亮后的颗粒；没有GT或真实噪声标定，不能判断全部高频变化都是噪声。输出均保留，不挑选最好结果。

重要限制：发现球场、园区、商业街疑似跨train/val同场景。原始划分未修改；正式泛化结论前必须核查。此诊断集仅作开发集，绝不反复利用test/0901选损失或超参数。

## 后续与文件安全

参见 [方法学习](../../docs/DI_RETINEX_METHOD_STUDY.md)、[U3D训练方案](../../docs/U3D_DI_RETINEX_TRAINING_PLAN.md)、[假设草案](../../docs/DI_RETINEX_RESEARCH_HYPOTHESES.md)。优先原模型短训练可行性与场景分组核查，不直接设计大网络。

本实验没有修改数据/权重/第三方源码或PyTorch环境，没有commit/push。输出图保存在本实验outputs，已由现有Git规则忽略。建议提交分析脚本、文档、清单、metrics，保留之前未提交内容等待用户决定。
'''
put(e/'README.md',report)
print(json.dumps(summary,indent=2))
