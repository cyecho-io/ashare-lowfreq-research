# 研究流水线

当前项目已经形成一条最小可复现的研究链路：

1. 从 Parquet 日线数据构建因子面板
2. 使用横截面超额收益标签训练 LightGBM
3. 用 walk-forward 方式逐月滚动训练与预测
4. 对预测分数做分层收益检验
5. 用预测分数驱动回测引擎构建组合并评估

## 推荐配置

当前验证过的有效方向是：

- 标签：`excess_fwd_return_5`
- 训练方式：`12` 个月训练窗口，按月 walk-forward
- 组合约束：
  - `top_k = 5`
  - `rebalance_every = 3`
  - `min_hold_bars = 5`
  - `keep_buffer = 2`
  - `min_turnover_names = 3`

## 标准输出

```text
research/factors/
research/models/
results/
```

其中关键文件包括：

- 因子面板 Parquet
- walk-forward 分数 Parquet
- walk-forward 指标 JSON
- 分层分析 JSON
- 模型组合回测结果目录
