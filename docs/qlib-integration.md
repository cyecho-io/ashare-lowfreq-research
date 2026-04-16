# Qlib 集成使用指南

这份文档对应当前仓库里的第一阶段 Qlib 集成实现。

目标不是把项目整体迁到 Qlib，而是先增加一条可选的 Qlib research path，并继续复用现有的下游能力：

- 分数文件导出
- 分数驱动策略
- 分数回测
- 分层分析

当前设计是双路径并存：

- native path：继续使用当前项目自己的因子构建和训练逻辑
- Qlib path：使用 Qlib 生成特征、标签、训练和预测

两条路径都必须产出兼容的 `scores.parquet`，并让下游直接消费。

## 当前范围

当前已经接入的 Qlib CLI 命令有：

- `ashare-backtest qlib-train-walk-forward`
- `ashare-backtest qlib-train-as-of-date`
- `ashare-backtest qlib-train-single-date`

当前阶段的特性边界：

- Qlib 仅用于 research 上游
- 回测和策略执行仍使用本项目原有逻辑
- Qlib 不是默认依赖
- 没有改写现有 native research 命令
- 默认先使用你本地已有的 `~/.qlib/qlib_data/cn_data`

当前阶段还没有做：

- 项目自有数据一键导出为 Qlib 数据目录
- native / qlib 结果的统一对比分析页

## 环境准备

### 1. Python 环境

需要 Python 3.11+。

建议先创建虚拟环境并安装项目：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

如果你要使用 Qlib 命令，再额外安装 Qlib 可选依赖：

```bash
python -m pip install -e ".[qlib]"
```

如果没有安装 `pyqlib`，运行 `qlib-*` 命令时会直接报错，并提示安装 `.[qlib]`。

### 2. 准备 Qlib 数据目录

当前实现默认使用：

```text
~/.qlib/qlib_data/cn_data
```

你也可以在命令里通过 `--provider-uri` 指向其他目录。

当前代码会在运行时调用：

- `qlib.init(provider_uri=..., region=...)`
- `D.instruments(...)`
- `D.features(...)`

所以你需要准备一套能被 Qlib 正常识别的中国市场日频数据目录。

### 3. 验证 Qlib 是否可导入

建议先在虚拟环境里确认：

```bash
python - <<'PY'
import qlib
print("qlib ok")
PY
```

如果这一步失败，先不要继续跑训练命令。

## 当前默认配置

当前代码里的 Qlib 路径默认值如下：

- `provider_uri`: `~/.qlib/qlib_data/cn_data`
- `region`: `cn`
- `market`: `csi300`
- `model_name`: `lgbm`
- `config_id`: `qlib_default`

当前默认特征是一个刻意收缩后的最小子集，主要为了先验证接入链路，而不是对齐 native path 的全部特征：

- 动量窗口
- 均线乖离
- 波动率窗口

默认标签是一个简单的 5 日前瞻收益表达式。

## 配置文件接入

如果你通过 Web 的 `/research` 页面触发 Qlib 研究任务，可以在研究配置 TOML 里额外提供一个可选的 `[qlib]` 段。

当前会读取这些字段：

- `provider_uri`
- `region`
- `market`
- `model_name`
- `config_id`

示例：

```toml
[storage]
root = "storage"

[factor_spec]
id = "industry_v4_v1_1"
universe_name = "tradable_core"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
validation_window_months = 1
test_start_month = "2026-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_scores.parquet"
metric_output_path = "research/models/walk_forward_metrics.json"

[analysis]
layer_output_path = "research/models/walk_forward_layers.json"

[qlib]
provider_uri = "~/.qlib/qlib_data/cn_data"
region = "cn"
market = "csi300"
model_name = "lgbm"
config_id = "qlib_smoke"

[model_backtest]
output_dir = "results/model_score_backtest_qlib_smoke"
top_k = 6
rebalance_every = 5
lookback_window = 20
```

说明：

- `[qlib]` 目前只在 Web 里的 `backend=qlib` 研究任务中生效
- 没有提供 `[qlib]` 时，会使用代码里的默认值
- Qlib 路径以 `[training].test_start_month` / `test_end_month` 为时间主线，自动推导训练取数窗口
- Qlib 不需要配置 `[factors].output_path`，也不需要手写 `factor_spec.start_date` 或 `research_snapshot.as_of_date`
- `[model_backtest].start_date` / `end_date` 可省略；省略时会按交易日历覆盖完整打分月份
- native research path 仍需要 factor panel 的 `start_date` / `as_of_date` 或 `[factors]` 配置

## Web 控制台使用方式

当前 `/research` 页面已经支持主动选择研究后端：

- `native`
- `qlib`

使用方式：

1. 准备一份标准研究配置
2. 如果要跑 Qlib，在 TOML 里可选地补上 `[qlib]`
3. 打开 `/research`
4. 在 `研究后端` 下拉框里选择 `qlib`
5. 提交任务

任务完成后，研究记录、分数文件列表和下游结果会带出这些 provenance 字段：

- `backend`
- `model`
- `config_id`

当前 Web 页面也已经支持按 `backend` 筛选研究记录和回测结果，所以 native 与 qlib 产物不会完全混在一起。

## 命令说明

### Walk Forward

按月滚动训练并导出多个月份分数：

```bash
ashare-backtest qlib-train-walk-forward \
  --provider-uri ~/.qlib/qlib_data/cn_data \
  --region cn \
  --market csi300 \
  --config-id qlib_demo \
  --test-start-month 2025-07 \
  --test-end-month 2026-02 \
  --output-scores-path research/models/walk_forward_scores_qlib.parquet \
  --output-metrics-path research/models/walk_forward_metrics_qlib.json
```

适用场景：

- 验证一段历史区间内 Qlib 分数是否可用
- 与 native walk-forward 结果做 A/B 比较
- 为现有回测链路提供一份可直接消费的分数文件

### As-Of Date

对某一个交易日做一次打分：

```bash
ashare-backtest qlib-train-as-of-date \
  --provider-uri ~/.qlib/qlib_data/cn_data \
  --region cn \
  --market csi300 \
  --config-id qlib_demo \
  --as-of-date 2026-03-31 \
  --output-scores-path research/models/walk_forward_scores_qlib_2026-03-31.parquet \
  --output-metrics-path research/models/walk_forward_metrics_qlib_2026-03-31.json
```

适用场景：

- 做某个交易日的最新候选分数
- 准备与盘前参考、策略状态生成联动

### Single Date

按指定 `test_month` 的训练窗口语义，对某个具体日期打分：

```bash
ashare-backtest qlib-train-single-date \
  --provider-uri ~/.qlib/qlib_data/cn_data \
  --region cn \
  --market csi300 \
  --config-id qlib_demo \
  --test-month 2026-03 \
  --as-of-date 2026-03-31 \
  --output-scores-path research/models/walk_forward_scores_qlib_single_2026-03-31.parquet \
  --output-metrics-path research/models/walk_forward_metrics_qlib_single_2026-03-31.json
```

适用场景：

- 你想显式指定“这一天属于哪个月度滚动窗口”
- 需要和历史 walk-forward 窗口做更严格对齐

## 输出约定

Qlib 路径最终不会把结果停留在 Qlib 自己的实验工件里，而是导出为项目兼容的 parquet 分数文件。

当前导出的关键字段包括：

- `trade_date`
- `symbol`
- `prediction`

当前实现还会附带这些元数据字段：

- `label`
- `train_end_date`
- `validation_end_date`
- `backend`
- `model`
- `config_id`

其中：

- `backend` 固定写入 `qlib`
- `model` 默认是 `lgbm`
- `config_id` 用于区分你自己的实验配置

## Symbol 约定

当前项目下游统一使用：

- `600000.SH`
- `000001.SZ`

Qlib 侧常见格式可能是：

- `sh600000`
- `sz000001`

当前导出层会自动把 Qlib 符号归一化到项目格式后再写出 parquet。

如果你自己的 Qlib 数据目录里使用了不同的 symbol 编码规则，需要先确认它能被当前归一化逻辑识别。

## 与现有回测链路联动

Qlib 命令跑完后，可以直接把导出的分数文件喂给现有回测命令：

```bash
ashare-backtest run-model-backtest \
  --scores-path research/models/walk_forward_scores_qlib.parquet \
  --storage-root storage \
  --start-date 2025-07-01 \
  --end-date 2026-02-28 \
  --output-dir results/model_score_backtest_qlib
```

这一步仍然依赖本项目自己的 `storage/` Parquet 数据，因为执行层、风控字段、行业信息、成交约束等仍由当前项目维护。

换句话说：

- research upstream 可以来自 Qlib
- execution downstream 仍然来自本项目数据层

## 推荐搭建顺序

如果你现在的目标是先把环境搭起来，再验证第一条命令，建议顺序如下：

1. 创建虚拟环境并安装 `.[dev]`
2. 安装 `.[qlib]`
3. 准备或确认 `~/.qlib/qlib_data/cn_data`
4. 用一行 Python 验证 `import qlib`
5. 先跑 `ashare-backtest qlib-train-as-of-date`
6. 再跑 `ashare-backtest qlib-train-walk-forward`
7. 最后把导出的分数文件交给 `run-model-backtest`

原因很直接：

- `as-of-date` 最容易确认环境是否通
- `walk-forward` 更适合确认一段历史区间是否稳定
- 回测应当放在分数产物已验证之后

## 常见问题

### 1. 报错提示缺少 `pyqlib`

说明你当前环境只安装了基础依赖，没有安装 Qlib 可选依赖。

执行：

```bash
python -m pip install -e ".[qlib]"
```

### 2. Qlib 数据目录存在，但命令仍然失败

优先检查三件事：

- `--provider-uri` 是否指向正确目录
- 这套数据是否真的是 Qlib 能识别的标准格式
- `region` 与数据内容是否匹配

### 3. 训练成功，但下游回测失败

优先看这几类问题：

- 分数文件里的 symbol 是否都被正确归一化
- 分数日期是否落在项目 `storage/` 数据覆盖范围内
- 回测区间是否与分数区间匹配

### 4. 为什么不能直接用 Qlib 做回测

这是当前项目的刻意边界。

当前目标是先替换 research 上游，而不是同时迁移执行层。这样可以让你先验证：

- Qlib 的特征和训练基础设施是否有价值
- 现有策略与回测层是否还能继续复用

## 当前已知限制

- 本机未安装 Qlib 时，当前仓库无法替你验证真实训练
- 当前默认特征集仍然很小
- 当前还没有项目数据导出到 Qlib 目录的统一脚本

## 下一阶段建议

等你把环境搭起来后，建议直接做三件事：

1. 先跑一次 `qlib-train-as-of-date`
2. 再跑一段短区间 `qlib-train-walk-forward`
3. 把生成的 `scores.parquet` 交给现有 `run-model-backtest`

如果这三步都通，下一轮就可以继续补：

- native / qlib 结果统一对比视图
- README 的完整双路径说明收口
- 项目数据导出为 Qlib 数据目录的脚本
