# EXP007A-0：已有普通上界机制与梯度验证

状态：梯度检查PASS；lambda待人工确认；smoke test未运行。不是原创方法，不是完整baseline训练。

## 计算图与源码对应

第三方mymodel.py的brightness_contrast原式为z=k*(x+0.5*(b-1))+0.5*(b+1)，随即torch.clamp(z,0,1)得到Y。NightUAV适配器使用TorchFunctionMode观察原torch.clamp调用，保存其输入Tensor引用，不detach、不重新计算z、不替换层、不改第三方方法。捕获返回值与官方返回Y为同一对象；若clamp次数/签名/输出身份不匹配立即停止。

新增upper=mean(relu(z-1)^2)，total=RD+VS+lambda*upper。RD/VS继续调用EXP005B的make_loss：从冻结官方lowlight_train.py提取原赋值AST，没有重写其表达式、比例、mask或Gaussian。精确表达式和源码哈希见config/run.json。

相对z的新增梯度为2*relu(z-1)/N；z<=1为0。含越界真实crop及合成边界控制均验证通过。upper通过原计算图到达共享网络参数。比较的是gradient clipping之前的L2范数；原始训练clip=0.1下的实际更新影响不等同于这些比例。

## 样本与一致性

固定train v2.1中16张：排序后Random(1149)打乱取前16，坐标独立Random(1149)固定。只使用官方train来源且不在dev/quarantine清单，原Gaussian预处理和NightUAV全黑同图重采样不变。所有测量在同一随机初始化、不更新参数的状态进行；没有加载E15或其他checkpoint。

每图lambda=0与无适配器官方forward比较：Y/c/b/k、RD、VS、total及所有参数梯度torch.equal。该检查不包含optimizer更新；本次没有optimizer.step。原模型/旧脚本/权重/split结束哈希未变。

## 梯度量级

|项|跨16crop梯度L2的RMS|
|---|---:|
|RD|0.06479972099|
|VS|0.0001274923176|
|upper|0.06090725166|
|original|0.06479609967|

## 有限lambda规则

预先规定候选r={1%,5%,10%}；lambda=r*RMS(||grad(RD+VS)||)/RMS(||grad(upper)||)。使用全部16个固定crop，包括upper为零的样本，不按dev画质或越界严重程度筛选。

|目标RMS比例|候选lambda|
|---|---:|
|1%|0.01063848686|
|5%|0.05319243432|
|10%|0.1063848686|

建议先确认5%候选，仅作为小幅机制干预的smoke起点，不是最优lambda。该规则未考虑梯度方向、后续训练漂移、裁剪和Adam缩放，不能保证训练中每步维持该比例。lambda_candidates.json记录逐样本比例，便于查看异质性。若批准后固定该数值，不在线自适应，不依据dev反复调参。

## 等待确认与smoke协议

未运行200步或完整15轮；无新checkpoint。用户确认具体lambda后，另行重新随机初始化seed1149，最多200步，train770、batch1、256crop、Adam lr.001/wd.0001/clip.1、exposure.5、原RD:VS=1:1，继续原同图全黑重采样。不得继承本次或历史状态，也不得作为正式实验结果。

## 文件

- config/run.json：预先固定的样本/坐标、规则、环境、源码和split哈希。
- metrics/synthetic.json：阈值边界与零惩罚控制。
- metrics/gradients.json：每样本loss、每参数梯度范数及数值一致性。
- metrics/crop_records.json：原采样及重采样记录。
- metrics/lambda_candidates.json：有限候选及依据。
- metrics/verification.json：PASS范围及smoke未执行状态。

没有开发集画质选择、网络/loss原项修改、训练、commit或push。

## 本次实测补充

16个固定训练crop中9个激活上界项；未执行任何优化器更新。无NaN/Inf或OOM。

|项|16个crop平均loss|未裁剪参数梯度L2 RMS|
|---|---:|---:|
|RD|0.02551610574|0.06479972099|
|VS|0.000002717007|0.000127492318|
|upper|0.002807847749|0.06090725166|
|RD+VS|0.02551882185|0.06479609967|

5% RMS候选在单个crop上加权upper梯度/原梯度最大约12.1934%，说明RMS规则不等于逐样本5%上限。小样本、单一初始化上的标定不能外推整个训练过程。

合成零惩罚断言在把需要梯度的标量转换为Python float时产生一次PyTorch提示；这是断言读取数值，不会替换用于求导的Tensor，也没有触发错误。梯度检查均在原Tensor图上完成。没有改动环境来消除该提示。

## 后续人工批准与200步结果

此前待确认状态为历史记录。用户现已批准lambda=0.053192434320470425；重新随机初始化的200步smoke已PASS，详见[SMOKE_RESULTS.md](SMOKE_RESULTS.md)及metrics/smoke_summary.json。没有启动正式训练或commit/push。
