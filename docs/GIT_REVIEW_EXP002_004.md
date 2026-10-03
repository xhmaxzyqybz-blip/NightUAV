# NightUAV 待提交文件审查（EXP002～EXP004B）

状态：只审查，不暂存/commit/push。当前工作区含旧EXP001/选型、EXP002、EXP003、EXP004以及新EXP004A/B成果，不能合并成一项已完成基线复现。

## 内容与体积

扫描当前82个非忽略未跟踪/已修改文件，均可按UTF-8读取，单文件均不超过1MiB；未发现常见token、私钥、AWS访问键或带口令URL模式。这是规则扫描加已知命中复核，不是绝对无敏感信息证明。

初次扫描命中的 `dependency_plan.json` 用户路径是上游OpenCV说明中的 `C:\Users\username` 占位示例，邮箱是natsort公开作者元数据，不是本机用户私人目录或邮箱。无需改写旧文件；建议不提交该冗长pip计划，用已整理的dependencies.json记录版本即可。

20个文件含绝对路径，主要为E:/NightUAV项目路径、E:/anaconda3解释器/包路径或公开元数据。未发现真实用户主目录泄露。它们会影响跨机器可移植性；本次保留历史实验原样，不能为了清理路径破坏已记录脚本hash。后续可在单独提交中参数化运行入口，历史日志仍保留来源说明。

## Git安全

.gitignore已忽略datasets、checkpoints、_model_audit、code/U3D_Toolkit、Python缓存、*.log、experiments/**/outputs及experiments/**/checkpoints。真实嵌套仓库包括DI-Retinex、Retinexformer、U3D_Toolkit，均处于忽略目录；未发现已跟踪gitlink。

不要提交原图、输出联系表/HTML/完整检索大表、权重和checkpoint。人工审核图片保存在本机outputs；Markdown/CSV保留相对链接，克隆仓库后如需看图应按脚本生成或另行共享合规图像。自动忽略并不等于可以随意git add .；请继续定向添加。

## 建议提交顺序及边界

1. 公共.gitignore与早期选型/EXP001分别核对，不冒充本轮新增。
2. EXP002：诊断脚本、方法学习/训练计划/假设笔记、诊断集清单、实验README与实际统计。
3. EXP003：原划分审计脚本/报告、20步技术训练入口、审核清单、训练配置和实际指标；dependency_plan.json建议省略。
4. EXP004：分组/冻结/检查脚本、blocked v1清单、分组报告、审计记录与配置草案。提交描述必须说明“尚未发布可靠划分”。
5. EXP004A：独立smoke训练入口、16图manifest、README、config、150步CSV/JSONL/summary。checkpoint和日志文件忽略，完整数值日志JSONL可提交；不得用于独立验证初始化。
6. EXP004B：候选检索脚本、40候选清单、关联表、用户决策空模板、审核说明与统计。始终标为pending_user，不能将待审核文件发布为approved v2。

下表为逐文件建议（不执行git add）；机器版为 [建议清单CSV](../experiments/exp004a_multisample_smoke/git_submission_manifest.csv)。

| 归属 | 路径 | 大小bytes | 建议 |
|---|---|---:|---|
| EXP001 | `experiments/exp001_di_retinex/README.md` | 6443 | include_after_user_review |
| EXP001 | `experiments/exp001_di_retinex/config/run.json` | 1184 | include_after_user_review |
| EXP001 | `experiments/exp001_di_retinex/metrics/inference.json` | 4383 | include_after_user_review |
| EXP001 | `scripts/infer_di_retinex.py` | 10414 | include_after_user_review |
| EXP002 | `docs/DI_RETINEX_METHOD_STUDY.md` | 6123 | include_after_user_review |
| EXP002 | `docs/DI_RETINEX_RESEARCH_HYPOTHESES.md` | 6664 | include_after_user_review |
| EXP002 | `docs/U3D_DI_RETINEX_TRAINING_PLAN.md` | 6195 | include_after_user_review |
| EXP002 | `experiments/exp002_baseline_diagnosis/DIAGNOSTIC_SET.md` | 4173 | include_after_user_review |
| EXP002 | `experiments/exp002_baseline_diagnosis/README.md` | 5251 | include_after_user_review |
| EXP002 | `experiments/exp002_baseline_diagnosis/REPRODUCE.md` | 1104 | include_after_user_review |
| EXP002 | `experiments/exp002_baseline_diagnosis/diagnostic_manifest.json` | 17186 | include_after_user_review |
| EXP002 | `experiments/exp002_baseline_diagnosis/metrics/diagnostic_aggregate.json` | 558 | include_after_user_review |
| EXP002 | `experiments/exp002_baseline_diagnosis/metrics/diagnostic_metrics.json` | 25952 | include_after_user_review |
| EXP002 | `experiments/exp002_baseline_diagnosis/metrics/diagnostic_summary.csv` | 3274 | include_after_user_review |
| EXP002 | `experiments/exp002_baseline_diagnosis/metrics/exp001_audit.json` | 698 | include_after_user_review |
| EXP002 | `experiments/exp002_baseline_diagnosis/metrics/exp001_histogram.svg` | 2379 | include_after_user_review |
| EXP002 | `experiments/exp002_baseline_diagnosis/metrics/exp001_statistics.json` | 4236 | include_after_user_review |
| EXP002 | `experiments/exp002_baseline_diagnosis/metrics/histogram.csv` | 6369 | include_after_user_review |
| EXP002 | `experiments/exp002_baseline_diagnosis/metrics/provenance.json` | 659 | include_after_user_review |
| EXP002 | `experiments/exp002_baseline_diagnosis/metrics/train_val_inventory.json` | 535278 | include_after_user_review |
| EXP002 | `scripts/analyze_exp002.py` | 11429 | include_after_user_review |
| EXP002 | `scripts/report_exp002.py` | 10433 | include_after_user_review |
| EXP003 | `docs/U3D_SPLIT_AUDIT.md` | 7289 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/README.md` | 9205 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/audit/candidates.csv` | 2758 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/audit/candidates.json` | 5733 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/audit/exact_duplicates.json` | 501 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/audit/inventory.json` | 265103 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/audit/smoke_candidates.json` | 1734 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/audit/summary.json` | 629 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/audit/visual_review.json` | 12722 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/audit/visualized_pairs.json` | 7106 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/dependencies.json` | 571 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/dependency_plan.json` | 45376 | omit_verbose_public_package_metadata; keep dependencies.json |
| EXP003 | `experiments/exp003_di_retinex_smoke/training/config.json` | 2407 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/training/steps.csv` | 5922 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/training/steps.jsonl` | 11876 | include_after_user_review |
| EXP003 | `experiments/exp003_di_retinex_smoke/training/summary.json` | 718 | include_after_user_review |
| EXP003 | `scripts/audit_u3d_split.py` | 5299 | include_after_user_review |
| EXP003 | `scripts/train_di_retinex_smoke.py` | 8391 | include_after_user_review |
| EXP004 | `configs/exp004_di_retinex_baseline.json` | 1025 | include_after_user_review |
| EXP004 | `configs/splits/u3d_scene_dev_v1.txt` | 78 | include_after_user_review |
| EXP004 | `configs/splits/u3d_scene_groups_v1.json` | 768415 | include_after_user_review |
| EXP004 | `configs/splits/u3d_scene_quarantine_v1.txt` | 30478 | include_after_user_review |
| EXP004 | `configs/splits/u3d_scene_train_v1.txt` | 78 | include_after_user_review |
| EXP004 | `configs/splits/u3d_scene_v1_metadata.json` | 765 | include_after_user_review |
| EXP004 | `docs/U3D_SCENE_GROUPING.md` | 6110 | include_after_user_review |
| EXP004 | `experiments/exp004_di_retinex_baseline/README.md` | 4780 | include_after_user_review |
| EXP004 | `experiments/exp004_di_retinex_baseline/audit/confirmed_or_visual_relations.json` | 6220 | include_after_user_review |
| EXP004 | `experiments/exp004_di_retinex_baseline/audit/exp004_visual_associations.json` | 11850 | include_after_user_review |
| EXP004 | `experiments/exp004_di_retinex_baseline/audit/previous_artifacts_sha256.json` | 5563 | include_after_user_review |
| EXP004 | `experiments/exp004_di_retinex_baseline/audit/retrieval_summary.json` | 5323 | include_after_user_review |
| EXP004 | `experiments/exp004_di_retinex_baseline/audit/review_queue.csv` | 47957 | include_after_user_review |
| EXP004 | `experiments/exp004_di_retinex_baseline/audit/scene_records_draft.json` | 579003 | include_after_user_review |
| EXP004 | `experiments/exp004_di_retinex_baseline/audit/train_similarity_candidates.json` | 84638 | include_after_user_review |
| EXP004 | `experiments/exp004_di_retinex_baseline/audit/train_top2_neighbors.json` | 259196 | include_after_user_review |
| EXP004 | `experiments/exp004_di_retinex_baseline/preservation_check.json` | 124 | include_after_user_review |
| EXP004 | `experiments/exp004_di_retinex_baseline/split_check.json` | 1576 | include_after_user_review |
| EXP004 | `scripts/check_u3d_scene_split.py` | 4049 | include_after_user_review |
| EXP004 | `scripts/freeze_u3d_scene_v1.py` | 5210 | include_after_user_review |
| EXP004 | `scripts/prepare_u3d_scene_groups.py` | 6284 | include_after_user_review |
| EXP004A | `experiments/exp004a_multisample_smoke/README.md` | 3449 | include_after_user_review |
| EXP004A | `experiments/exp004a_multisample_smoke/git_review_scan.json` | 23141 | optional_audit_snapshot |
| EXP004A | `experiments/exp004a_multisample_smoke/smoke_train_manifest.json` | 5337 | include_after_user_review |
| EXP004A | `experiments/exp004a_multisample_smoke/training/config.json` | 7169 | include_after_user_review |
| EXP004A | `experiments/exp004a_multisample_smoke/training/steps.csv` | 42946 | include_after_user_review |
| EXP004A | `experiments/exp004a_multisample_smoke/training/steps.jsonl` | 89200 | include_after_user_review |
| EXP004A | `experiments/exp004a_multisample_smoke/training/summary.json` | 1641 | include_after_user_review |
| EXP004A | `scripts/train_di_retinex_multisample_smoke.py` | 9491 | include_after_user_review |
| EXP004B | `experiments/exp004b_dev_review/README.md` | 3887 | include_after_user_review |
| EXP004B | `experiments/exp004b_dev_review/association_decisions.csv` | 63034 | include_after_user_review |
| EXP004B | `experiments/exp004b_dev_review/candidate_associations.csv` | 149121 | include_after_user_review |
| EXP004B | `experiments/exp004b_dev_review/candidate_decisions.csv` | 3551 | include_after_user_review |
| EXP004B | `experiments/exp004b_dev_review/candidate_dev_manifest.json` | 38083 | include_after_user_review |
| EXP004B | `experiments/exp004b_dev_review/priority_review.csv` | 5679 | include_after_user_review |
| EXP004B | `experiments/exp004b_dev_review/review_only_pool.json` | 36303 | include_after_user_review |
| EXP004B | `experiments/exp004b_dev_review/review_workload.json` | 416 | include_after_user_review |
| EXP004B | `experiments/exp004b_dev_review/source_reviews.json` | 1388 | include_after_user_review |
| EXP004B | `experiments/exp004b_dev_review/summary.json` | 900 | include_after_user_review |
| EXP004B | `scripts/build_dev_candidate_review.py` | 10174 | include_after_user_review |
| 公共 | `.gitignore` | 641 | include_after_user_review |
| 早期选型 | `docs/BASELINE_SELECTION.md` | 18337 | include_after_user_review |

本Git审查文档与建议清单属于公共审查记录，可单独纳入；尚未执行任何提交。
