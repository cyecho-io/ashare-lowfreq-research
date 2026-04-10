# 研究流水线

当前项目已经形成一条最小可复现的研究链路：

1. 从 Parquet 日线数据构建因子面板
2. 使用横截面超额收益标签训练 LightGBM
3. 用 walk-forward 方式逐月滚动训练与预测
4. 对预测分数做分层收益检验
5. 用预测分数驱动回测引擎构建组合并评估

当前研究层已经开始支持双路径：

- native path：使用项目自己的因子和 LightGBM 训练逻辑
- Qlib path：使用 Qlib 生成特征、标签、训练和导出分数

两条路径当前都以 `scores.parquet` 作为与下游执行层的稳定边界。

Qlib 环境准备和命令示例见：[docs/qlib-integration.md](/Users/yongqiuwu/works/github/Trade/docs/qlib-integration.md)

如果你通过 Web 控制台运行研究任务，`/research` 页面现在可以直接选择 `native` 或 `qlib` 后端。选择 `qlib` 时，配置文件里可选的 `[qlib]` 段会提供 Qlib 专属参数，例如 `provider_uri`、`region`、`market`、`model_name`、`config_id`。

当前 research、backtest、paper、simulation 产物都会写出统一 provenance 字段：

- `backend`
- `model`
- `config_id`

这使得 native / qlib 两条路径在结果链路里可以被稳定识别和筛选。

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
