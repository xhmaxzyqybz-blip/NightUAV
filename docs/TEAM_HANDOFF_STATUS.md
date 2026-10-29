# NightUAV 团队交接状态

本次同步依据本地Git状态与实验完成记录；不运行实验。仓库分支：main。同步前远端HEAD为b5fae29，已经包含EXP001–EXP004C、场景审计及冻结U3D v2.1清单。本次新增后续代码、轻量配置与研究记录。

## 实际研究进度

| 阶段 | 文件证据与状态 | 入口/记录 |
|---|---|---|
| EXP005B | E15 adapted baseline，15 epochs、11550有效更新；正式权重单独保管 | scripts/train_di_retinex_exp005b.py；experiments/exp005b_di_retinex_u3d_baseline/metrics/summary.json |
| EXP006A/C | 保存图像诊断与浮点越界归因验证PASS；不等于增强质量改善 | scripts/analyze_exp006a_failure_modes.py、analyze_exp006c_float_saturation.py；对应实验README与metrics |
| EXP007A-0 | 200次有效更新，机制smoke PASS；不是正式训练 | scripts/smoke_exp007a0_boundary.py；实验SMOKE_RESULTS.md、metrics/smoke_summary.json |
| B0 | 25参数条件分支数值检查PASS，无参数更新 | code/src/enhancement/nightuav_v1_conditioned_di.py；scripts/check_nightuav_v1_conditioning.py |
| B1 | 500次配对尝试，两组各500有效更新 | scripts/probe_nightuav_v1_b1.py；summary.json、evaluation_final.json、attempts.jsonl |
| B1完整图 | full_dev/verification.json为PASS；16张增强图及12组固定ROI一致性记录存在；图像不随Git上传 | experiments/nightuav_v1_b1_probe/full_dev/verification.json |
| A0 | 10276参数显式I/R分解；数值与梯度PASS，无训练 | code/src/enhancement/nightuav_a0_retinex.py；scripts/check_nightuav_a0_retinex.py |
| A1-0 | 重建损失，200有效更新 | scripts/train_nightuav_a1_0_reconstruction.py；对应summary.json |
| A1-1 | 重建+TV，200有效更新；TV系数0.8653310986696595 | scripts/train_nightuav_a1_1_tv.py；对应README、comparison.json |
| A2-0 | 非配对反射约束可行性研究完成 | experiments/nightuav_a2_0_reflectance_alignment/README.md |
| A2-1 | 固定日间弱结构先验，200有效更新；结构系数5.068692550668577 | scripts/train_nightuav_a2_1_structural_prior.py；对应README、comparison.json |
| A2-2 | **仅完成独立实现与train-only梯度标定，0次参数更新，未训练**；结构系数**4.8010370240182** | scripts/prepare_nightuav_a2_2_highlight_exclusion.py、calibrate_nightuav_a2_2_highlight_exclusion.py |

A2-2以config_locked.json、calibration_locked.json、day_reference_locked.json为当前协议。旧config.json与numeric_checks.json保留早期待确认状态，不能据此覆盖锁定协议。原有效条件完整保留，并叠加输入亮度≤0.7，日夜各自计算。尚无A2-2训练入口、训练日志或训练checkpoint；不能运行标定入口当作训练。

A1-1减少I纹理但增加重建代价。A2-1全图MAE改善、结构统计更近参考，同时有亮区重建恶化、R近上界及高频上升风险。统计接近不证明物理解耦或细节恢复；A3尚未开展。日夜场景交叉未知，当前dev结论不能宣称独立泛化。

## 获取与复现边界

已有仓库且工作区干净时：`git switch main`，然后`git pull --ff-only origin main`。新成员可克隆 https://github.com/xhmaxzyqybz-blip/NightUAV 。先阅读本文件和对应实验README，不自动运行训练入口；多数入口拒绝覆盖已存在实验目录。

本次保留历史文件原始字节，不统一行尾/编码。复现前应核对配置中的源码与输入SHA256；Git客户端行尾转换可能影响哈希，需采用不转换行尾的检出方式。部分历史JSON含UTF-8 BOM，读取时需识别；本次不改写历史记录。

以下资源需要从项目负责人取得，Git拉取并不包含完整运行环境：

- U3D原始数据及其原目录布局；冻结train/dev/quarantine清单已在Git，不能重新划分。日间参考400 crop清单与哈希随配置提供，图片本身不上传。
- 正式E15权重、B1完整checkpoint，以及需要恢复或核验的A1/A2 checkpoint。B1本地checkpoints/attempt100.pth至attempt500.pth存在；本次只确认文件与既有记录，不重新加载模型验证恢复。
- `_model_audit/DI-retinex-low-light-enhancement`上游源码，按实验配置的commit与哈希取得；该独立仓库未打包上传。参考docs/ENVIRONMENT.md准备兼容环境，现有入口可能严格检查本地环境版本。
- 固定720p dev输入、E15图像、A1/A2初始/最终raw.npz与montage、B1完整图与ROI。部分检查入口依赖历史浮点输出或checkpoint，缺失时不能只靠Git直接重跑验证。
- A1/A2使用seed1149随机初始化，已有参数哈希/初始输出一致性记录；不能用最终checkpoint替代初始状态。

## 本次有意未提交的内容

- datasets、checkpoints、*.pth等被既有.gitignore排除；实验outputs目录也被忽略。不使用git add -f，不改变忽略规则。
- 未跟踪的PNG/JPEG、NPZ及其他二进制产物保留本地，即使某些未被.gitignore覆盖也不纳入本次提交。
- EXP005A0、EXP005A、EXP005B的大型train_steps轨迹，以及包含大量重复逐步记录的metrics/train_epochNNN.json不上传；配置、train_history、dev评价、summary仍保留。若需逐步完整审计应单独索取。
- docs/NIGHTUAV_MODEL_V1_PROPOSAL.md含不必要的个人桌面路径，本次保留本地不提交，不改写原设计记录。方案脉络可先阅读已提交的A2-0及各阶段README。
- 工作区已有.gitignore修改只增加若干历史演示输出排除项；本次保留未提交，不改动它。

这是代码与轻量记录交接，不是可一键恢复所有实验的完整数据/权重包。旧EXP005A失败/停止记录作为EXP005B前置历史保留，不应与正式E15基线混淆。