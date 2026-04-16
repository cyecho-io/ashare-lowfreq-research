# Web 控制台

Web 控制台是推荐使用入口。启动后访问 `http://127.0.0.1:8888`。

```bash
ashare-backtest-web
```

## Workspace

页面顶部有 workspace 切换器：

- `native`：项目内置 factor、LightGBM 训练和分析链路
- `qlib`：Qlib 表达式和训练链路，输出兼容项目下游的 scores

研究记录、分数文件、回测结果和模拟执行结果都会跟随 workspace 隔离展示。

## 首页

首页用于确认数据和运行状态：

- 查看本地数据准备情况
- 查看最近研究和回测记录
- 快速进入模型研究台、回测控制台和模拟成交台

## 模型研究台

路径：`/research`

主要用途：

- 选择或编辑研究配置
- 运行 native / qlib 模型研究
- 查看任务日志
- 查看 scores、metrics、layer analysis 等产物路径

研究任务完成后，生成的 score parquet 会出现在回测控制台的分数文件下拉框里。

## 回测控制台

路径：`/backtest`

主要用途：

- 选择模型分数文件
- 自动关联策略配置
- 调整回测区间和初始资金
- 运行分数驱动回测
- 查看收益曲线、交易明细和 summary

## 模拟成交台

路径：`/simulation`

主要用途：

- 创建模拟账户
- 基于模型分数生成盘前计划
- 查看执行记录和账户状态

demo 阶段建议先跑通 `/research` 和 `/backtest`，再探索模拟成交流程。
