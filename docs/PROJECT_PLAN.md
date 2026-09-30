# NightUAV Project Plan

## 当前阶段

Phase 1：实验平台与 Baseline 建设

## 当前目标

建立：

数据集 → 低光增强 → 图像质量评价 → 目标检测 → 泛化评价

的完整实验流程。

## 已完成

- U3D 数据集下载
- U3D 数据结构检查
- U3LIE 官方仓库下载
- U3LIE 仓库完整性检查
- 确认当前 U3LIE 官方代码尚未完整公开
- 第一轮团队分工完成

## 当前团队任务

- B：传统 Retinex（SSR / MSR / MSRCR）
- C：深度 Retinex Baseline 调研
- D：夜间 UAV 检测与泛化数据集调研
- E：实验平台与评价指标体系

## 负责人任务

- 项目结构管理
- 实验平台集成
- Retinex 理论学习
- 深度学习运行环境准备
- 团队进度检查
- 后续 Baseline 选择

## 下一阶段

等待第一轮任务完成后：

1. 确定第一个深度 Retinex Baseline
2. 配置统一深度学习环境
3. 跑通单图推理
4. 在 U3D 上进行批量测试
5. 建立第一版 Benchmark
