# CLI 对照

这个项目推荐优先使用 Web 页面。下面的命令用于给习惯 CLI 的用户提供同等入口。

## 启动 Web 控制台

对应 Web 入口：所有页面

```bash
ashare-backtest-web
```

## 运行 native 研究

对应 Web 页面：`/research`，workspace 选择 `native`

```bash
ashare-backtest run-research-config configs/native/demo_strategy.toml
```

## 运行 native 分数回测

对应 Web 页面：`/backtest`，workspace 选择 `native`

```bash
ashare-backtest run-model-backtest \
  --scores-path research/native/demo/models/demo_scores.parquet \
  --storage-root storage/demo \
  --start-date 2025-01-02 \
  --end-date 2026-02-27 \
  --top-k 6 \
  --rebalance-every 5 \
  --min-hold-bars 8 \
  --output-dir results/native/demo_backtest
```

## 运行 qlib 研究

对应 Web 页面：`/research`，workspace 选择 `qlib`

```bash
ashare-backtest run-research-config configs/qlib/demo_strategy.toml
```

如果未安装 qlib 可选依赖：

```bash
python -m pip install -e ".[qlib]"
```

## 运行 qlib 分数回测

对应 Web 页面：`/backtest`，workspace 选择 `qlib`

```bash
ashare-backtest run-model-backtest \
  --scores-path research/qlib/demo/models/demo_scores.parquet \
  --storage-root storage/demo \
  --start-date 2025-01-02 \
  --end-date 2026-02-27 \
  --top-k 6 \
  --rebalance-every 5 \
  --min-hold-bars 8 \
  --output-dir results/qlib/demo_backtest
```

## 真实数据同步

对应 Web 页面：数据准备通常先通过 CLI 完成，再在 Web 查看。

```bash
ashare-backtest sync-tushare-sqlite \
  --sqlite-path storage/source/ashare_arena_sync.db \
  --start 20240101 \
  --end 20260331

ashare-backtest import-sqlite storage/source/ashare_arena_sync.db --storage-root storage
```
