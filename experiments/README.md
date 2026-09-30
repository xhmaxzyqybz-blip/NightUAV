# 正式实验管理规范

正式实验统一采用三位编号加名称的方式命名：

- `EXP001_Name`
- `EXP002_Name`
- `EXP003_Name`

编号依次递增，`Name` 替换为简洁的实验名称。

每个正式实验至少保存以下内容：

```text
EXPxxx_Name/
├── README.md
├── config/
├── logs/
├── outputs/
├── metrics/
└── figures/
```

- `README.md`：实验目的、方法、数据集、运行方式和结论。
- `config/`：本次实验使用的配置。
- `logs/`：运行日志。
- `outputs/`：增强图片、检测结果等输出。
- `metrics/`：评价指标与统计结果。
- `figures/`：本次实验的可视化图片。

当前尚无正式实验，暂不创建 `EXP001` 或其他实验目录。
