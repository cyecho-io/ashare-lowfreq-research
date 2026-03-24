# 因子模块设计

当前 `factors/` 模块的目标不是做完整研究平台，而是为后续接入 LightGBM 提供最小可用输入。

## 当前输出

基于日线 Parquet 数据生成以下结果：

- 基础因子面板
- 未来收益标签

## 基础因子

当前先支持：

- `mom_5`: 5 日动量
- `mom_10`: 10 日动量
- `mom_20`: 20 日动量
- `mom_60`: 60 日动量
- `ma_gap_5`: 收盘价相对 5 日均线偏离
- `ma_gap_10`: 收盘价相对 10 日均线偏离
- `ma_gap_20`: 收盘价相对 20 日均线偏离
- `ma_gap_60`: 收盘价相对 60 日均线偏离
- `volatility_10`: 10 日收益波动率
- `volatility_20`: 20 日收益波动率
- `volatility_60`: 60 日收益波动率
- `range_ratio_5`: 5 日平均振幅
- `volume_ratio_5_20`: 5/20 日成交量比
- `amount_ratio_5_20`: 5/20 日成交额比
- `amount_mom_10`: 10 日成交额变化
- `price_pos_20`: 收盘价在 20 日区间内的位置
- `volatility_ratio_10_60`: 短期/长期波动率比
- `trend_strength_20`: 20 日动量相对波动率
- `cross_rank_mom_20`: 当日横截面 20 日动量分位
- `cross_rank_amount_ratio_5_20`: 当日横截面成交额活跃度分位
- `cross_rank_volatility_20`: 当日横截面低波分位

## 标签

当前先支持：

- `fwd_return_3`
- `fwd_return_5`
- `fwd_return_10`
- `excess_fwd_return_3`
- `excess_fwd_return_5`
- `excess_fwd_return_10`
- `industry_excess_fwd_return_3`
- `industry_excess_fwd_return_5`
- `industry_excess_fwd_return_10`

其中：

- `fwd_return_*` 是绝对未来收益
- `excess_fwd_return_*` 是相对当日横截面均值的超额未来收益
- `industry_excess_fwd_return_*` 是相对当日所属行业均值的超额未来收益

## 输出文件

```text
research/
└── factors/
    └── basic_factor_panel.parquet
```

该文件包含：

- `trade_date`
- `symbol`
- 因子列
- 标签列

后续接 LightGBM 时，可直接读取该面板并做时间切分。
