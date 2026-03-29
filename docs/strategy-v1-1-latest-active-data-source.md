# Strategy v1.1 Active Score Source 数据源生产说明

这份文档专门说明像

- [premarket_scores_industry_v4_v1_1_latest_active.parquet](/Users/yongqiuwu/works/github/Trade/research/models/premarket_scores_industry_v4_v1_1_latest_active.parquet)

这样的 active score source 数据源是如何生产出来的，以及后续如何按日期向后扩展。

本文重点讲两件事：

1. 从无到有，如何正式生产一个 active score source
2. 当数据源已经存在时，如何按后续交易日继续补齐

## 先讲清楚：这个数据源本质上是什么

`premarket_scores_industry_v4_v1_1_latest_active.parquet` 只是历史遗留文件名。按新的语义，它代表的是：

- 针对当前策略 `research_industry_v4_v1_1`
- 基于同一 `factor spec` 在某个 `as_of_date` 下生成的 factor snapshot
- 面向最新活跃日期持续更新的一条 walk-forward score source

它在仓库中的元数据登记在：

- [score_source_manifest.json](/Users/yongqiuwu/works/github/Trade/research/models/score_source_manifest.json)

对应条目包含：

- `factor_panel_path`
- `label_column`
- `train_window_months`
- `source_kind = premarket_latest_active`
- `supports_incremental_update = true`

这几个字段决定了这条数据源后续如何续更。这里的 `latest_active` 应理解为“活跃维护状态”，不是另一套训练范式。

## 一条重要原则

当前代码里虽然还保留单日打分 helper，但对外应统一理解为 walk_forward 语义下的按 `as_of_date` 打分，而不是另一套独立模式。

对 `v1.1` 来说，正式盘前生产链路本身就是：

1. 更新存储层数据
2. 重建信号日 factor snapshot
3. 对信号日运行 walk_forward as-of-date 打分
4. 产出按日期命名的 walk_forward scores
5. 再生成盘前参考、策略状态和 latest manifest

参考：

- [run_v1_1_premarket.sh](/Users/yongqiuwu/works/github/Trade/scripts/run_v1_1_premarket.sh)
- [strategy-v1-1-premarket-runbook.md](/Users/yongqiuwu/works/github/Trade/docs/strategy-v1-1-premarket-runbook.md)

所以“走正式链路”与“底层按单个 `as_of_date` 生成分数”并不冲突，但对外不再需要单独强调 `latest_inference`。

## 一、从无到有生产 active score source

下面假设目标是首次建立一条与

- [premarket_scores_industry_v4_v1_1_latest_active.parquet](/Users/yongqiuwu/works/github/Trade/research/models/premarket_scores_industry_v4_v1_1_latest_active.parquet)

同类型的数据源。

### 1. 明确日期定义

- `T`：信号日，也就是最新完整收盘日
- `T+1`：执行日，也就是下一交易日

例如：

- `T = 2026-03-25`
- `T+1 = 2026-03-26`

### 2. 更新 SQLite 源库

先把行情与日历同步到 `T`：

```bash
./.venv/bin/ashare-backtest sync-tushare-sqlite \
  --sqlite-path storage/source/ashare_arena_sync.db \
  --end 2026-03-25
```

### 3. 从 SQLite 刷新 Parquet

```bash
./.venv/bin/ashare-backtest import-sqlite storage/source/ashare_arena_sync.db --storage-root storage
```

这里的目标是保证：

- [daily.parquet](/Users/yongqiuwu/works/github/Trade/storage/parquet/bars/daily.parquet)
- [ashare_trading_calendar.parquet](/Users/yongqiuwu/works/github/Trade/storage/parquet/calendar/ashare_trading_calendar.parquet)

都已经覆盖到 `T`。

### 4. 构建信号日 factor snapshot

正式链路不会直接在旧 factor panel 上“猜”一天，而是先把信号日对应的 factor snapshot 构建到 `T`：

```bash
./.venv/bin/ashare-backtest build-factors \
  --storage-root storage \
  --factor-spec-id research_industry_v4_v1_1 \
  --universe-name tradable_core \
  --start-date 2024-01-02 \
  --as-of-date 2026-03-25
```

产物是：

- [asof_2026-03-25.parquet](/Users/yongqiuwu/works/github/Trade/research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-25.parquet)

### 5. 对信号日运行 walk_forward as-of-date 打分

这是正式链路里生成信号日分数的标准步骤：

```bash
./.venv/bin/ashare-backtest train-lgbm-walk-forward-as-of-date \
  --factor-panel-path research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-25.parquet \
  --label-column industry_excess_fwd_return_5 \
  --train-window-months 12 \
  --as-of-date 2026-03-25 \
  --output-scores-path research/models/walk_forward_scores_industry_v4_v1_1_2026-03-25.parquet \
  --output-metrics-path research/models/walk_forward_metrics_industry_v4_v1_1_2026-03-25.json
```

这一步的含义是：

- 用 `2026-03-25` 之前最近 12 个月的训练样本训练模型
- 只对 `2026-03-25` 这一天的股票横截面做打分
- 输出这一天的最新 score 文件

产物是：

- [walk_forward_scores_industry_v4_v1_1_2026-03-25.parquet](/Users/yongqiuwu/works/github/Trade/research/models/walk_forward_scores_industry_v4_v1_1_2026-03-25.parquet)
- [walk_forward_metrics_industry_v4_v1_1_2026-03-25.json](/Users/yongqiuwu/works/github/Trade/research/models/walk_forward_metrics_industry_v4_v1_1_2026-03-25.json)

### 6. 生成盘前参考与策略状态

正式盘前链路接着用这份信号日 walk_forward scores 生成 `T+1` 的调仓参考和策略状态。

参考命令：

```bash
./.venv/bin/ashare-backtest generate-premarket-reference \
  --scores-path research/models/walk_forward_scores_industry_v4_v1_1_2026-03-25.parquet \
  --storage-root storage \
  --trade-date 2026-03-26 \
  --top-k 6 \
  --rebalance-every 5 \
  --lookback-window 20 \
  --min-hold-bars 8 \
  --keep-buffer 2 \
  --min-turnover-names 3 \
  --max-names-per-industry 2 \
  --output-path research/models/premarket_reference_industry_v4_v1_1_2026-03-26.json
```

### 7. 刷新 latest 目录与 manifest

正式脚本最后会把最新产物刷新到：

- [research/models/latest/research_industry_v4_v1_1](/Users/yongqiuwu/works/github/Trade/research/models/latest/research_industry_v4_v1_1)

并更新：

- [manifest.json](/Users/yongqiuwu/works/github/Trade/research/models/latest/research_industry_v4_v1_1/manifest.json)

这个 manifest 会记录：

- latest 展示路径
- source score 路径
- source metrics 路径
- strategy state 路径
- premarket reference 路径

### 8. 如何得到 active score source

从工程角度，`latest_active` 这个遗留名不应被理解为“孤立的一份 parquet”或“另一套训练模式”，而应被理解为：

- factor snapshot 持续更新
- walk_forward as-of-date 打分每日对 `T` 生成分数
- 将这些按日期生成的 walk_forward 分数组织成一条“活跃 score 源”

也就是说，像

- [premarket_scores_industry_v4_v1_1_latest_active.parquet](/Users/yongqiuwu/works/github/Trade/research/models/premarket_scores_industry_v4_v1_1_latest_active.parquet)

这样的文件，其正式来源应该是上面这条日常正式链路，而不是手工散落拼接。

## 二、往数据源中继续扩展后续日期

这一部分讨论的是：数据源已经存在，例如它当前最后只到 `2026-03-24`，现在要把 `2026-03-25` 这一天补进去。

### 推荐原则

默认应走“原正式生成链路”，而不是绕过正式链路直接随意写 parquet。

也就是说，标准流程应是：

1. 先确保 `storage` 更新到新的 `T`
2. 再把 factor snapshot 更新到新的 `T`
3. 再正式生成新的 `walk_forward_scores_industry_v4_v1_1_T.parquet`
4. 最后把这一天的 score 纳入 active score source

### 最小增量场景

如果当前状态是：

- latest_active 只差 1 个交易日
- factor snapshot 已经覆盖到新信号日
- 不需要重跑整个历史窗口

那么合理的增量流程是：

#### 步骤 A：确认缺口

先确认：

- 目标执行日 `T+1`
- 所需信号日 `T`
- 当前数据源 `score_end_date`

如果：

- `score_end_date < T`

就说明要补齐 `score_end_date + 1 ... T` 之间的缺失交易日。

#### 步骤 B：重建或确认 factor snapshot 已覆盖到 T

```bash
./.venv/bin/ashare-backtest build-factors \
  --storage-root storage \
  --factor-spec-id research_industry_v4_v1_1 \
  --universe-name tradable_core \
  --start-date 2024-01-02 \
  --as-of-date T
```

#### 步骤 C：对每个缺失交易日逐日运行正式 walk_forward as-of-date 打分

例如只缺 `2026-03-25`：

```bash
./.venv/bin/ashare-backtest train-lgbm-walk-forward-as-of-date \
  --factor-panel-path research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_T.parquet \
  --label-column industry_excess_fwd_return_5 \
  --train-window-months 12 \
  --as-of-date 2026-03-25 \
  --output-scores-path research/models/walk_forward_scores_industry_v4_v1_1_2026-03-25.parquet \
  --output-metrics-path research/models/walk_forward_metrics_industry_v4_v1_1_2026-03-25.json
```

这一步仍然属于正式链路的一部分，不应被理解为“临时补丁”。

#### 步骤 D：把新增日期纳入 latest_active 数据源

纳入规则应保持稳定：

- 以 `trade_date + symbol` 为唯一键
- 同一天重复生成时，新结果覆盖旧结果
- 旧日期不应无故被重写

如果是将多份按日期命名的 walk_forward score 文件汇总成一条活跃源，则应：

1. 读现有 latest_active parquet
2. 读新增的 `walk_forward_scores_industry_v4_v1_1_T.parquet`
3. 追加后按 `trade_date, symbol` 去重
4. 重新写回 latest_active parquet

#### 步骤 E：刷新依赖 latest_active 的上层展示

补齐后应刷新：

- readiness 检查
- paper 页面中的数据源覆盖区间
- 如果需要，也刷新 latest manifest 中的 source_scores_path 语义

## 三、什么叫“严格走原正式链路”

在这个仓库里，“严格走原正式链路”并不意味着要引入另一套训练范式。

它真正意味着：

- 必须先确保 `storage` 和 factor snapshot 是最新的
- 只能使用正式定义的 `factor_panel_path / label_column / train_window_months`
- 只能按正式脚本的日期关系来生成 `T` 的分数和 `T+1` 的盘前状态
- 只能按稳定规则把新日期纳入 `latest_active` 数据源

换句话说：

- 按 `as_of_date` 生成单日分数是正式链路中的一个步骤
- 但不应该把它当成“脱离正式链路、任意对某个文件打补丁”的自由入口

## 四、建议的实践规范

建议以后统一遵守下面几条：

1. 首次建立 latest_active 数据源时，必须走完整正式链路。
2. 后续扩展日期时，默认也走正式链路，只允许做“按缺口日期增量执行”的轻量化。
3. 不直接手工编辑 `premarket_scores_*_latest_active.parquet`。
4. 如果确实要增量补齐，也必须以正式 factor snapshot 和正式 walk_forward as-of-date 参数为准。
5. `score_source_manifest.json` 中的元数据应与实际生成链路保持一致。

## 五、推荐入口

如果只是日常生产，优先使用：

```bash
./scripts/run_v1_1_premarket.sh --signal-date T --trade-date T+1
```

如果只是补一个或少量缺失交易日，推荐仍然按正式链路的步骤执行：

1. 更新 storage
2. 更新 factor snapshot
3. 逐缺口日期执行 `train-lgbm-walk-forward-as-of-date`
4. 把新增日期并入 latest_active 数据源

## 六、和 /paper 页补齐功能的关系

`/paper` 页上的“补齐到信号日”按钮，正确的产品语义应该是：

- 自动识别当前数据源缺失了哪些交易日
- 通过正式 active score source 生产链路补齐缺口日期
- 补齐完成后再允许生成快照

而不是：

- 随机用另一套参数临时生成一天分数
- 直接把结果写进数据源

所以 `/paper` 页的自动补齐，应被视为“正式链路的 UI 封装”，而不是独立于正式链路之外的另一套生产方式。
