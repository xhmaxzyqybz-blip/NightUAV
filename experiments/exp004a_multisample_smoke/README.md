# EXP004A：多样本训练技术验证

状态：实际完成150步，16个官方train文件全部参与，每张9或10次；无验证、无选模、无泛化/收敛结论。该实验与blocked v1场景清单相互独立，未取消或绕过其正式训练检查。

## 数据与配置

[专用manifest](smoke_train_manifest.json)只用于technical_smoke_only。排除EXP003确认的全部字节重复端点；来源、SHA256逐文件检查，输入前后哈希一致。不读取val/test。样本有不同照明和内容，不声称16个独立场景。

模型commit `3f94f95ffb181904a837532a70adc064bee793a2`，只读导入官方normal网络、初始化、加载器；两项loss使用原训练源码AST表达，RD:VS=1:1。随机初始化seed1149，样本每轮洗牌seed1150，不加载EXP003或官方预训练checkpoint。官方uint8 RGB整图GaussianBlur(9×9,sigma1.7)后，X/G同步256×256随机裁剪；batch1，workers0，FP32；Adam lr1e-3、wd1e-4、betas(0.9,0.999)、eps1e-8、clip0.1；曝光Normal(0.5,标准差0.001)。详见 [完整config](training/config.json)，含源码/脚本/manifest哈希。

实际命令（由nightuav解释器执行）：

```powershell
& 'E:\anaconda3\envs\nightuav\python.exe' -B 'E:\NightUAV\scripts\train_di_retinex_multisample_smoke.py'
```

执行时stdout/stderr保存至 [logs/run.log](logs/run.log)。完整逐步来源、裁剪坐标、两loss、总loss、曝光、梯度范数、参数差、耗时及显存：[JSONL](training/steps.jsonl)、[CSV](training/steps.csv)。检查150步全部有限且参数L1更新量>0。日志已使用detach避免EXP003的张量转标量警告。脚本拒绝覆盖已有run/checkpoint，不自动重试。

## 实测

| 项目 | 数值 |
|---|---:|
| 实际优化步骤 |150|
| 实际不同文件数 |16|
| 训练循环墙钟（含数据和日志） |90.981秒|
| 首步同步计算计时 |1741.938ms|
| 后149步平均/中位 |33.518/33.265ms|
| 读取、全图滤波、裁剪、传输平均 |555.038ms|
| 峰值已分配/保留显存 |1205.728/1864.000MiB|

GPU为NVIDIA GeForce RTX 5060 Laptop GPU，torch 2.13.0+cu130，torchvision 0.28.0+cu130，CUDA运行时13.0。无环境安装或更改。同步step包含forward/loss/backward/检查/clip/Adam，不含读取传输和checkpoint；包含诊断同步开销，不是纯GPU内核吞吐。90秒循环不包含模型初始化/启动和checkpoint保存，不能凭此推断完整训练时长。

## Checkpoint及禁止用途

[step150.pth](checkpoints/step150.pth)在本实验独立checkpoints目录。SHA256 `c2b8efe00aad75349717471921b6874b25b5adf3740a601bd44025801a22f695`。保存后weights_only=True加载，逐模型张量一致，且不同于初始化。未测试恢复后继续优化，不能说断点续训已验证。

**严禁将该checkpoint作为后续独立验证实验的初始化。** checkpoint、配置和manifest均记录technical_smoke_only用途。本次未做best选择；没有任何模型质量评价。任务B候选与smoke文件交叉不妨碍未来重新设计，但正式实验必须在批准的划分上重新随机初始化，不能沿用本模型或优化器状态。

与官方训练协议不同：16图清单/随机裁剪/150步/windows单进程/诊断检查/独立保存，而非官方整图、全训练集、多epoch。不得称完整基线或论文复现。所有输出局限于本实验；checkpoint和logs由现有Git规则忽略。
