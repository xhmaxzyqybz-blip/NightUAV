# EXP004：场景审计与原模型训练准备

状态：**审计完成到可追溯待审核清单；可靠训练/开发划分尚未完成，训练按条件停止。** 没有将EXP003的20步冒充EXP004结果。

## 实际执行

读取指定前置报告、检查工作区；复查旧重复/候选图；复用既有哈希补充800张train内全对检索；构建目视/精确关系传递闭包及保守隔离组；保存场景记录、审核队列、v1清单与自动检查。详情见 [U3D_SCENE_GROUPING](../../docs/U3D_SCENE_GROUPING.md)。

命令：

```powershell
& 'E:\anaconda3\envs\nightuav\python.exe' -B 'E:\NightUAV\scripts\prepare_u3d_scene_groups.py'
& 'E:\anaconda3\envs\nightuav\python.exe' -B 'E:\NightUAV\scripts\freeze_u3d_scene_v1.py'
& 'E:\anaconda3\envs\nightuav\python.exe' -B 'E:\NightUAV\scripts\check_u3d_scene_split.py' --require-ready --report 'E:\NightUAV\experiments\exp004_di_retinex_baseline\split_check.json'
```

第三条实际拒绝训练就绪状态（预期阻断），结构检查无错误。脚本拒绝覆盖完成的分组/清单/报告；若仅复核清单，可不带--report执行check脚本，不重新检索。训练命令没有执行。

## 划分与哈希

v1为blocked_pending_scene_review，train0、dev0、quarantine800。原val100排除，test未使用。空清单是安全占位，不是可用训练划分。场景分布与确认状态见 [split_check.json](split_check.json)。

| 文件 | SHA256 |
|---|---|
| u3d_scene_dev_v1.txt | `19867f966595a1482da7daf0e2e75d175503d75da924ffbcfda747aa75f78625` |
| u3d_scene_groups_v1.json | `36e23991ca3c5b21ad6fe8da5a05d01b786f69cf1255bb74e5bf84d0c3d6c695` |
| u3d_scene_quarantine_v1.txt | `fa928bf4f12445a320c0ad50473e29f3b5a0193d7a174cd7d6f90423728a8a3b` |
| u3d_scene_train_v1.txt | `19867f966595a1482da7daf0e2e75d175503d75da924ffbcfda747aa75f78625` |

## 计划配置与实际状态

[配置草案](../../configs/exp004_di_retinex_baseline.json)明确execute_training=false、actual_steps=0，计划150步仅作后续准备。

沿用EXP003官方normal模型、随机初始化seed1149、Adam lr1e-3/wd1e-4、batch1、梯度裁剪0.1；RD:VS=1:1、曝光Normal(0.5,标准差0.001)、FP32。没有新增网络或损失，也没有修改原有训练入口。

与官方协议的预定差异：清单数据而非目录扫描，256×256随机裁剪而非全图、workers0而非4、150步而非默认200epochs；验证是项目附加协议（新dev完整视野等比缩至1280×720，对16:9图；固定E0.5、eval/inference_mode、逐图loss等权平均，无GT指标）。这些均未执行，仍需在正式配置中冻结。不再使用原val100选模。

由于用户要求重大划分问题时只完成审计并停止，本轮**未实现/运行新的清单训练加载适配、未开始150步训练、未运行dev、未生成EXP004 checkpoint**。现有EXP003代码保持原样；不能拿其硬编码0102样本直接冒充清单多样本训练。后续入口应在导入/加载数据前调用严格就绪检查，拒绝quarantine或原val/test进入优化，安全读取清单完整路径，并保持原loss表达不变。

## 实测运行数据

优化步骤0；loss/梯度/参数更新、GPU耗时、峰值显存、验证loss、checkpoint均N/A，而非0或成功。没有本轮训练性能数据。没有安装依赖、修改PyTorch环境或运行长训练。

EXP003参考值为256裁剪单图重复训练：后19步同步均值31.282ms，整图读取/滤波/传输均值517.588ms，已分配显存1205.728MiB。其总和约0.549秒/步只是历史参考，不是本轮多样本实测。正式训练时间需新清单批准后先测100～200步及一次dev，用 `epochs*(N_train/batch的上取整)*实测端到端步时 + 验证/保存开销`估算；当前train为空，不给出虚假的正式总时长。

## 阻碍及下一步

自动组存在跨场景误合并与同场景漏关联；大部分图尚无可靠scene_id，先审核并发布非空场景隔离版本。建议保留全部原数据，只更新版本化清单；不以调阈值替代场景审核。

准备脚本初次UTF-8读取失败已修复，成功生成检索结果；这不是训练失败，没有因此重跑训练。多样本训练与验证的条件尚不成立。

## Git和产物

新增分组文档、configs/splits的清单和元数据、配置草案、本实验审核记录。新增脚本为prepare_u3d_scene_groups.py、freeze_u3d_scene_v1.py、check_u3d_scene_split.py共3个；没有新增训练脚本。

图片在outputs下被Git忽略，建议仅定向提交上述3个脚本、文档、配置、清单和轻量审核JSON/CSV。未git add/commit/push。旧EXP001/002/003文档脚本哈希快照用于验收，不覆盖旧成果。
