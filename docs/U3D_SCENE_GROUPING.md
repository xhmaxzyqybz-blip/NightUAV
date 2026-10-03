# U3D 场景分组与划分 v1（EXP004）

**状态：未发布可训练划分。** 本轮发现自动检索组同时存在误合并和漏关联，按用户要求停在审计；未开始多样本训练。空train/dev是防误用的占位清单，不是已经建立可靠训练/验证集。

## 可追溯证据与覆盖

复用EXP003的900张文件SHA256与63位pHash清单，不重跑原80,000对train/val审计。复看原27对重点样本图：1对字节重复、21对强布局关联、5对不确定的分级仍保留；不将缺少元数据的关系称为连续帧。原始val的旧证据仅用于关联审计，不进入训练、开发验证或模型选择；test未读取。

补充范围仅原train800：全部319,600个无序对，以缓存pHash距离≤14或32×32灰度NCC≥0.90检索，得到253对候选。扩大阈值为提高召回，不是同场景判定。另为每张train保留NCC前2邻居，共1600条有向检索记录。800张图均有分组联系表；253对均有并排图。本轮详细查看前6个大组共58张，以及旧27对；未谎称已逐张完成全部800张场景语义审核。

补充脚本：[prepare_u3d_scene_groups.py](../scripts/prepare_u3d_scene_groups.py)。32×32图由Pillow JPEG draft+LANCZOS得到，NCC去均值后单位L2内积；pHash算法沿用EXP003。直方图/结构相似受低光、视角与重复建筑影响，不能代替拍摄序列元数据。初次运行因Windows默认GBK读取中文JSON失败，显式UTF-8修复后完成；旧成果未改动。

## 场景组与风险组不是一回事

机器清单 [u3d_scene_groups_v1.json](../configs/splits/u3d_scene_groups_v1.json)包含900条来源记录。每条有完整项目相对路径、original_split、SHA256、scene_group_id、conservative_group_id、grouping_basis、confirmation_status、scene_category、usage。

- scene_group_id只沿字节重复、既有强目视关系和本轮明确布局关系做并查集传递闭包；A-B/B-C时A/B/C同组。未关联的单张也有ID，但标为未审核，不能解释为已确认独立场景。
- conservative_group_id将新检索边与旧待定边纳入隔离块，防止已知风险边被分到两侧。此组可能混入不同场景，只能用于保守隔离，不可作为真实场景标签。
- 本轮增加了亮顶办公楼、体育场正面、体育场斜侧道路等目视子组。具体边在 [exp004_visual_associations.json](../experiments/exp004_di_retinex_baseline/audit/exp004_visual_associations.json)，生成过程 [freeze_u3d_scene_v1.py](../scripts/freeze_u3d_scene_v1.py)可追溯。

## 重大风险的具体证据

1. 自动RISK_000中亮顶办公楼（0032/0046等）、商铺（0078/0282）和暗道路（0153/0629）混杂，说明pHash误关联不能直接当同场景。
2. 同一亮顶办公楼出现在原RISK_000、004（0770/0796）、005（0199/0704/0716），说明“不同自动组”不能保证不同场景。本轮已为这些明确同楼图建立传递关系，并保守合并相关风险块。
3. 体育场斜侧道路与正面两类视角也保守纳入同一隔离块；没有据此宣称拍摄序列相同。
4. 原pair004/013/014的暗部结构与023/024的视角差异仍不确定，必须复核原图或元数据。新的253候选中，151对尚未落入已有目视/精确关联分量，列为待人工审核；其余只是已有关系闭包内，不代表这253对全部逐对人工确认。

[审核队列CSV](../experiments/exp004_di_retinex_baseline/audit/review_queue.csv)、[前2邻居](../experiments/exp004_di_retinex_baseline/audit/train_top2_neighbors.json)、[并排图和全部联系表](../experiments/exp004_di_retinex_baseline/outputs/)。旧不确定对沿用EXP003配对图，避免改写或不必要重复生成。

## 当前v1清单与自动检查

| 清单 | 图片数 | 场景关联ID数 | 保守风险组数 |
|---|---:|---:|---:|
| train | 0 | 0 | 0 |
| dev | 0 | 0 | 0 |
| quarantine | 800 | 748 | 620 |

748与620都是当前图结构的ID数量，**不是确认存在748/620个真实场景**。724张仍无确认场景关联，69张有目视关联但无序列元数据，7张仅有字节重复关联。语义类别仅为审过样本标注：52张有粗类别、748张未分类；不能伪造隔离集的完整场景分布。

本轮采用全量待放行隔离，是因为未建立足够可靠的准入子集；不表示800张图片全部质量有问题。原val100在机器清单中为original_val_excluded，不出现在任何三份train来源划分清单。test不在机器清单中。

自动检查 [check_u3d_scene_split.py](../scripts/check_u3d_scene_split.py)逐项检查清单内重复、实际路径、原train范围、用途匹配、清单SHA256、800张源文件SHA256、train/dev/quarantine文件/scene ID/风险ID/哈希交集、800张完整覆盖。通过完整路径匹配，不用basename join，避免同名错误。

实际 [split_check.json](../experiments/exp004_di_retinex_baseline/split_check.json)：结构检查通过，ready_for_training=false。空集下“没有交集”在数学上成立，但不能据此放行；检查器额外要求非空train/dev及approved审核元数据，`--require-ready`实际拒绝此版本。检查不能替代序列元数据，也不能自动证明未发现的场景不存在。

## 待完成的可靠划分流程

优先取得拍摄地点/飞行路线/视频序列ID；若无元数据，审核所有拟准入图片、宽松检索近邻和代表性不同视角。逐边记录same_scene / different_scene / uncertain及证据，把不确定关系整个传递风险分量隔离。仅在人工审核完成的场景组之间分配新train/dev，优先覆盖不同照明、建筑/道路/场地，不机械追求8:2。对于单张未审核图不能直接新建独立scene_id后入集。

批准后发布新版本清单与哈希，重新运行自动检查和已知跨组边检查；不直接覆盖本次blocked v1。原始数据和官方划分始终不改。当前还没有可用于选择模型的新dev，不能在泄漏风险验证结果上推进创新或选模。
