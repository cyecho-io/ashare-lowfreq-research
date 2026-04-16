# 接入真实数据

快速体验默认使用 `storage/demo/`。这份数据很小，只用于验证 Web、native 研究、qlib 研究和分数回测流程。

如果你要做真实研究，建议配置自己的数据源，例如 Tushare 同步出的本地数据，以及你自己的 qlib provider。

## 1. 准备项目 Parquet 数据

如果使用 Tushare，先复制环境变量模板并填写 token：

```bash
cp .env.example .env
```

在 `.env` 中配置：

```text
TUSHARE_TOKEN=你的 token
```

同步 SQLite 行情库：

```bash
ashare-backtest sync-tushare-sqlite \
  --sqlite-path storage/source/ashare_arena_sync.db \
  --start 20240101 \
  --end 20260331
```

可选：同步基准指数：

```bash
ashare-backtest sync-tushare-benchmark \
  --symbol 000300.SH \
  --start 20240101 \
  --end 20260331
```

导入项目 Parquet 存储：

```bash
ashare-backtest import-sqlite storage/source/ashare_arena_sync.db --storage-root storage
```

## 2. 修改 native 配置

native 链路读取项目 Parquet 数据。你可以从 `configs/native/demo_strategy.toml` 复制一份自己的配置，然后修改这些字段：

```toml
[storage]
root = "storage"

[factor_spec]
id = "your_strategy_native"
universe_name = "tradable_core"
start_date = "2024-01-02"

[research_snapshot]
as_of_date = "2026-03-31"

[factors]
output_path = "research/native/factors/your_strategy.parquet"

[training]
score_output_path = "research/native/models/your_strategy_scores.parquet"
metric_output_path = "research/native/models/your_strategy_metrics.json"

[analysis]
layer_output_path = "research/native/models/your_strategy_layer.json"

[model_backtest]
output_dir = "results/native/your_strategy_backtest"
start_date = "2025-01-02"
end_date = "2026-03-31"
```

关键点：

- `[storage].root` 指向你的项目 Parquet 数据根目录
- `[factor_spec].universe_name` 要存在于 `storage/parquet/universe/memberships.parquet`
- 研究输出建议放到 `research/native/...`
- 回测输出建议放到 `results/native/...`

## 3. 修改 qlib 配置

qlib 链路需要 qlib provider。demo 配置读取的是：

```toml
[qlib]
provider_uri = "storage/demo/qlib_data/cn_data"
market = "demo"
```

接入真实 qlib 数据时，复制 `configs/qlib/demo_strategy.toml`，然后修改：

```toml
[storage]
root = "storage"

[training]
score_output_path = "research/qlib/models/your_strategy_scores.parquet"
metric_output_path = "research/qlib/models/your_strategy_metrics.json"

[analysis]
layer_output_path = "research/qlib/models/your_strategy_layer.json"

[qlib]
provider_uri = "~/.qlib/qlib_data/cn_data"
market = "csi300"
config_id = "your_strategy_qlib"

[model_backtest]
output_dir = "results/qlib/your_strategy_backtest"
```

关键点：

- `[storage].root` 仍然用于项目 universe 过滤、行业标签和下游回测
- `[qlib].provider_uri` 指向你的真实 qlib provider
- `[qlib].market` 必须是 provider 中存在的 instruments 文件，例如 `csi300` 或 `all`
- qlib 研究输出建议放到 `research/qlib/...`
- qlib 回测输出建议放到 `results/qlib/...`

## 4. 在 Web 里使用

启动 Web 控制台：

```bash
ashare-backtest-web
```

然后：

1. 进入 `/research`
2. 顶部 workspace 选择 `native` 或 `qlib`
3. 选择你的配置文件
4. 运行研究
5. 进入 `/backtest`
6. 选择生成的 score parquet 运行回测
