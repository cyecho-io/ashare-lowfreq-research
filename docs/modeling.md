# 模型训练层设计

当前模型层的目标是把因子面板转换成“可用于回测排序”的股票分数。

## 当前范围

- 读取因子面板 Parquet
- 按时间切分训练集和测试集
- 用 LightGBM 回归未来收益标签
- 输出测试期股票分数

## 约束

- 当前只做单标签回归
- 当前只做最小离线训练，不做自动调参
- 当前不做滚动训练和多期集成

## 输出

```text
research/models/
├── walk_forward_scores.parquet
└── walk_forward_metrics.json
```

其中：

- `walk_forward_scores.parquet` 包含 `trade_date`、`symbol`、`prediction`
- `walk_forward_metrics.json` 包含训练参数和简单误差指标
