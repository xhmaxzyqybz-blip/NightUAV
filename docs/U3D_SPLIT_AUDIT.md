# U3D train/val 划分审计（EXP003）

## 范围与结论

只读扫描本机 `datasets/U3D/lowlight/train` 800张和 `val`100张，对全部80,000个跨划分组合计算相似性。未将test用于训练、选参或相似检索；仅枚举目录文件数。目录实际为lowlight train800/val100/test100，light train800/test200、未见val目录。light仅做目录检查，未假设其与lowlight成对，也未用于训练。本机目录按已有官方数据下载布局记录，未独立验证远端归档校验和。

**确认跨划分字节重复1对：train/0209.jpg 与 val/0880.jpg。** SHA256相同，配对图也一致。train内部另有4组重复：0022/0694、0063/0491、0085/0217、0402/0443。共5组、每组2张，未移动/删除任何文件。

因此当前官方val不能被视作与train严格独立。精确重复是确定事实；同场景是另一个维度；是否视频连续帧尚无时间或序列元数据证明。不能把每个相似候选都叫作数据泄漏。

## 相似算法与可重复性

脚本 [audit_u3d_split.py](../scripts/audit_u3d_split.py)，指定nightuav Python执行。全文件SHA256识别字节重复。感知特征用Pillow JPEG draft缩略读取、LANCZOS到32×32灰度：正交DCT取左上8×8去DC的63系数，与中位数比较生成63位pHash；另将32×32灰度去均值、单位L2归一化，内积为NCC。

预设候选阈值：pHash Hamming≤10 **或** NCC≥0.97；没有用模型增强效果调阈值。得到25对候选，涉及21张train、17张val。加上EXP002预先指出而未命中阈值的2对，共27对待审查；另输出4对技术测试候选及其NCC最近val供目视检查。全部31对已生成单独配对图和6张联系表。

局限：平移/旋转/裁切/视点和曝光变化可能漏检；黑暗、重复建筑和灯光形状可能误报。算法不具备场景身份识别能力。文件哈希只能认出相同编码字节，重新编码的相同内容靠相似筛选也不保证全检出。阈值不是独立性证明。高相似关系也不保证传递后的整个连通分量都同场景。

## 目视检查与明确分级

已查看31对缩略配对图。27对关注样本中：1对字节相同；21对布局具有强同场景关联的目视初判；5对暂不确定（pair004、013、014、023、024），不能排除仅建筑/照明相似。所有非精确重复的时间关系均未确认，最终场景分组仍待人工审核，不将初判视作已审核标签。

先前两对：0008/0815能对应独特球场、道路与树木布局；0595/0894能对应园区水面、道路及田块布局，视角有变化。它们分别pHash距离16/24，NCC0.845/0.424，均未命中自动阈值，直接说明“未命中不等于独立”。

[全部候选CSV](../experiments/exp003_di_retinex_smoke/audit/candidates.csv)、[SHA256和pHash清单](../experiments/exp003_di_retinex_smoke/audit/inventory.json)、[逐对目视初判](../experiments/exp003_di_retinex_smoke/audit/visual_review.json)、[配对图](../experiments/exp003_di_retinex_smoke/outputs/split_audit/)。CSV的pending保留为最终人工确认队列，目视初判另存不覆盖原筛选结果。

| 配对ID | train / val | 初判 | 人工需核实 |
|---|---|---|---|
| 000 | 0209 / 0880 | exact_bytes_confirmed | 场景/拍摄序列；见配对图 |
| 001 | 0131 / 0895 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 002 | 0127 / 0899 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 003 | 0216 / 0895 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 004 | 0656 / 0813 | uncertain_scene | 场景/拍摄序列；见配对图 |
| 005 | 0114 / 0899 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 006 | 0216 / 0856 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 007 | 0131 / 0856 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 008 | 0009 / 0895 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 009 | 0792 / 0804 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 010 | 0551 / 0876 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 011 | 0758 / 0876 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 012 | 0009 / 0856 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 013 | 0042 / 0884 | uncertain_scene | 场景/拍摄序列；见配对图 |
| 014 | 0656 / 0884 | uncertain_scene | 场景/拍摄序列；见配对图 |
| 015 | 0646 / 0876 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 016 | 0343 / 0818 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 017 | 0742 / 0864 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 018 | 0179 / 0849 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 019 | 0091 / 0812 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 020 | 0616 / 0837 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 021 | 0147 / 0879 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 022 | 0049 / 0821 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 023 | 0342 / 0847 | uncertain_scene | 场景/拍摄序列；见配对图 |
| 024 | 0336 / 0869 | uncertain_scene | 场景/拍摄序列；见配对图 |
| 025 | 0008 / 0815 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |
| 026 | 0595 / 0894 | same_scene_visual_strong | 场景/拍摄序列；见配对图 |

## 可采用的验证方案（提案，未执行新划分）

- 当前保留官方原目录用于可比性记录，但不得宣称其val为场景独立泛化。EXP003不运行任何val，也不报告增强质量。
- 建立版本化的scene manifest（字段path、original_split、SHA256、scene_id、sequence_id、审核人/证据、用途），先人工核验同场景关系，再以场景/拍摄序列为组划分训练和验证；无需移动原图。
- 可选择保留val的完整场景组，从训练清单排除这些场景的全部图；或者在train+val开发池按场景重新分组。两种方案需先确定，不能只删除1对精确重复后就认为已解决。原官方划分实验与新场景划分实验分别报告，不混用指标。
- 统计审核后的独立场景数与照明覆盖，再确定组级比例或留一场景验证；当前没有可靠scene_id，不能承诺某个精确新划分。没有足够场景时，应补充独立数据；正式跨数据集结论仍需要外部数据。
- test保持封存，仅在协议、模型及选择准则冻结后用于最终评价；本轮不审查其内容。正式实验前由受控数据审核流程检查开发集与最终测试是否同源，避免研发者用测试图设计模型。

## 对技术训练的处理

选择官方train/0102.jpg：完整哈希无重复，pHash对val最小距离18、最高NCC0.1198；与最近val对比未见对应结构。确认其来源、可解码及无当前规则命中的重复/相似问题；**未证明其与所有val场景独立**。因此仅用于单样本20次随机裁剪的GPU smoke test，不进行验证、选模或收敛判断。完整训练应等待上述场景协议落实。
