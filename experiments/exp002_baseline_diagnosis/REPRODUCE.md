# 重复分析说明

实际分析代码为 scripts/analyze_exp002.py，报告生成代码为 scripts/report_exp002.py。使用同一数据、EXP001结果、源码commit与权重，执行：

```powershell
& 'E:\anaconda3\envs\nightuav\python.exe' -B 'E:\NightUAV\scripts\analyze_exp002.py' --experiment exp002_repeat01
& 'E:\anaconda3\envs\nightuav\python.exe' -B 'E:\NightUAV\scripts\report_exp002.py' --experiment exp002_repeat01
```

均拒绝覆盖现有产物。第一个命令实际完成数值计算和图片；第二个基于JSON生成表格与固定协议文字。报告中的语义场景标签、目视描述和固定EXP001说明为本次人工审查注释，必须与固定来源对应；数据或模型变化时不得直接套用。跨机器细微数值差异需要人工检查。没有创建重复实验，本次实际执行的是默认exp002_baseline_diagnosis。

最终验收：保存的PNG全部重新打开解码，20个诊断输出确认RGB/1280×720，20张原图SHA256再次匹配；JSON可解析，两个脚本AST语法检查通过。图片在outputs下被Git忽略。
