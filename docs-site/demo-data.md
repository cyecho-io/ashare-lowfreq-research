# Demo 数据

`storage/demo/` 是仓库内置的小型 demo 数据源，用于让用户在没有 Tushare token、没有私有行情库、没有外部 qlib 数据下载的情况下体验项目。

## 已包含文件

```text
storage/demo/catalog.json
storage/demo/parquet/bars/daily.parquet
storage/demo/parquet/benchmarks/000300.SH.parquet
storage/demo/parquet/calendar/ashare_trading_calendar.parquet
storage/demo/parquet/instruments/ashare_instruments.parquet
storage/demo/parquet/universe/memberships.parquet
storage/demo/qlib_data/cn_data/calendars/day.txt
storage/demo/qlib_data/cn_data/instruments/demo.txt
storage/demo/qlib_data/cn_data/features/
```

## 用途

demo 数据用于支持：

- Web 首页的数据状态展示
- native 模型研究
- 模型分数回测
- qlib 模型研究
- qlib 训练后的下游回测

## 设计约束

demo 数据应该保持轻量：

- 股票数量控制在几十只
- 日线区间控制在可快速训练的范围
- 不包含任何密钥或私有数据
- 文件结构和真实 `storage/` 保持一致

## qlib provider

项目标准 Parquet 数据不能直接替代 qlib provider。仓库内置了由 demo Parquet 数据生成的小型 provider：

```text
storage/demo/qlib_data/cn_data/
```

qlib demo 配置会读取：

```toml
[qlib]
provider_uri = "storage/demo/qlib_data/cn_data"
market = "demo"
```

如果你更新了 `storage/demo/parquet`，可以重新生成 demo qlib provider：

```bash
python scripts/build_demo_qlib_provider.py \
  --storage-root storage/demo \
  --provider-uri storage/demo/qlib_data/cn_data \
  --market demo
```

## 切换到真实数据

demo 数据只用于体验流程。要接入真实数据，请阅读 [接入真实数据](real-data.md)。
