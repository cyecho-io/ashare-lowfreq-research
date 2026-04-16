# A 股低频研究与回测指南

这个文档站是 `ashare-lowfreq-research` 的使用入口，重点帮助新用户先通过 Web 页面跑通 demo，再理解 native 和 qlib 两条模型研究链路。

## 推荐入口

新用户建议先走 Web 控制台：

```bash
bash scripts/bootstrap_demo.sh
source .venv/bin/activate
ashare-backtest-web
```

然后打开：

```text
http://127.0.0.1:8888
```

## 你可以体验什么

- 在首页查看 demo 数据和最近运行记录
- 在模型研究台运行 native / qlib 研究配置
- 在回测控制台选择模型分数并执行回测
- 在模拟成交台查看后续盘前和模拟执行入口

## 当前状态说明

仓库已经内置 `storage/demo/` 小型数据源，可用于无 Tushare token、无私有行情库的 demo 体验。

这份 demo 数据同时包含：

- 项目标准 Parquet 数据：给 native 链路和下游回测使用
- 小型 qlib provider：给 qlib 链路训练使用

因此用户第一次 clone 项目后，可以用同一份 demo 数据体验 native 和 qlib 两条链路。

## 文档地图

- [快速体验](quickstart.md)：从 clone 后启动 Web，到跑研究和回测
- [Web 控制台](web-console.md)：每个页面的使用方式
- [CLI 对照](cli-reference.md)：Web 核心功能对应的命令行
- [Demo 数据](demo-data.md)：内置 demo 数据源的范围和约定
- [接入真实数据](real-data.md)：从 demo 数据切换到自己的 Tushare / qlib 数据源

## 本地预览文档

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```
