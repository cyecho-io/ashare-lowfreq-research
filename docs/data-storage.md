# 数据存储设计

## 目标

当前项目的数据层采用“先导入到本地标准格式，再由回测模块统一读取”的模式：

- 先复用已有 SQLite 数据
- 将其转换成项目内统一的 Parquet 存储
- 后续再通过 Tushare 做增量补数

## 目录规范

```text
storage/
├── catalog.json
└── parquet/
    ├── bars/
    │   └── daily.parquet
    ├── calendar/
    │   └── ashare_trading_calendar.parquet
    ├── instruments/
    │   └── ashare_instruments.parquet
    └── universe/
        └── memberships.parquet
```

## 表与字段映射

### 日线行情 `bars/daily.parquet`

来源表：`equity_daily_bars`

- `symbol`
- `trade_date`
- `open`
- `high`
- `low`
- `close`
- `prev_close`
- `adj_factor`
- `volume`
- `amount`
- `turnover_rate`
- `limit_up_price`
- `limit_down_price`
- `is_suspended`
- `is_limit_up`
- `is_limit_down`
- `close_adj`

说明：

- `close_adj = close * adj_factor`
- MVP 先保留日线层的核心字段，成交价和信号价口径后续由引擎配置决定

### 标的信息 `instruments/ashare_instruments.parquet`

来源表：`equity_instruments`

- `symbol`
- `exchange`
- `name`
- `listing_date`
- `delisting_date`
- `board`
- `industry_level_1`
- `industry_level_2`
- `is_st`
- `is_active`

### 交易日历 `calendar/ashare_trading_calendar.parquet`

来源表：`trading_calendar`

- `trade_date`
- `is_open`
- `has_night_session`
- `notes`

### 股票池成员 `universe/memberships.parquet`

来源表：`equity_universe_memberships`

- `universe_name`
- `symbol`
- `effective_date`
- `expiry_date`
- `source`

## catalog.json

`storage/catalog.json` 用来记录：

- 数据来源路径
- 导入时间
- 各表行数
- 各表时间范围
- 当前 schema 版本

这样后续再接入 Tushare 增量同步时，可以基于目录和 catalog 做补数判断。
