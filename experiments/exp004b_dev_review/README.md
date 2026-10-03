# EXP004B：定向开发候选人工审核

状态：40张候选，全部pending_user_review，**没有发布正式train/dev v2**。0/0/800 v1和其检查未修改。

## 候选与检索范围

候选全部来源官方lowlight/train；避开EXP002全部20张诊断样本，排除已知精确重复端点。按已有助手目视布局记录覆盖13种粗类别/视角（不是13个经确认独立地点），每类最多5张，轮询选到40张；没有根据增强效果挑选。亮度缩略均值范围0.0706～0.2874，不能代表全U3D照明分布。正式人工确认权保留给用户，既有记录不冒充用户确认。

每个候选与其余799张官方train比较，共31,960个有向组合。采用缓存63位pHash距离与新32×32灰度NCC；保存各自top6及阈值Hamming≤14或NCC≥0.90命中的结果，并加入旧精确/目视/待审核边和传递风险组。所有既有JSON审核表、253行审核CSV和1600条近邻记录都已读取；输入资料哈希见source_reviews.json。原val仅作为此前关联证据来源，不重新读取其图片，不进入训练或模型选择；test未使用。

局限：没有旋转/视点不变性，极暗图和建筑重复纹理会误匹配，top6会漏掉真实关联；加入旧风险块是保守排除机制，不是同场景断言。不同视角类别可能属于同一地点，需要人工合并。

## 材料和审核顺序

1. 打开 [本机联系表入口](outputs/index.html)。D01～D40各有多页，顶部始终是候选图，下方是关联图片，标注原文件编号、Hamming/NCC及PRIORITY。完整路径和关系依据以CSV为准；低NCC仍可能因旧关系被纳入。
2. 为减轻首轮工作，先看 **17个负担较低候选**：D05, D07, D08, D09, D10, D12, D18, D19, D20, D21, D23, D28, D29, D30, D31, D36, D40。它们的优先队列去重后 **60对关系，涉及42张不同图片**，见 [priority_review.csv](priority_review.csv)。这只是优先顺序，不能只审这些就宣布独立。
3. 其余23个候选有≥8条未确认的强/旧风险关联，建议先放弃或延期，不强行保留。这不是自动排除，具体建议见candidate_decisions.csv。
4. 在 [candidate_decisions.csv](candidate_decisions.csv)给每张候选填keep_for_further_review / drop / defer；在 [association_decisions.csv](association_decisions.csv)填same_scene / different_scene / uncertain，并写依据。默认全部pending_user。保留不确定关系意味着仍需隔离或放弃候选，不能忽略。
5. 初步保留后，再检查该候选的全部关联页/原图和邻居。全部815条有向关系、671对去重关系见 [candidate_associations.csv](candidate_associations.csv) 与association_decisions.csv；不能把“没有检索到”视为独立证明。

机器候选来源/哈希/类别/风险见 [candidate_dev_manifest.json](candidate_dev_manifest.json)。完整31,960分数CSV保存在outputs，避免作为大体积Git产物。自动检查已确认无候选来自val/test或EXP002诊断集。

## 拟定训练池安全检查（不是划分发布）

仅为审核估算，将40候选及所有检索关联按已知scene/risk组和SHA256做传递排除，得到218张暂排除、582张剩余审核池；数据详见review_only_pool.json，其training_allowed=false。已知组没有跨越该池与排除集合，但仍可能存在未知场景关系。**582不是获准训练集，40不是正式dev**。没有写任何v2清单。

用户确认后需重新构建场景闭包，保留不确定组隔离，检查重复/场景跨组与照明覆盖，再发布新版本；正式模型必须重新随机初始化，禁止复用EXP004A checkpoint。

生成入口：scripts/build_dev_candidate_review.py（指定nightuav Python，拒绝覆盖已有目录）。本轮只生成审核材料，未训练、未验证、未进行checkpoint选择。
