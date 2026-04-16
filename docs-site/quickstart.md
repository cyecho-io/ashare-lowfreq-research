# 快速体验

这条路径面向第一次 clone 项目的用户。目标是先打开 Web 页面，再用内置 demo 数据体验模型研究和分数回测。

## 1. 初始化环境

```bash
bash scripts/bootstrap_demo.sh
source .venv/bin/activate
```

如果你要体验 qlib 链路，还需要安装 qlib 可选依赖。仓库已经内置小型 qlib provider，不需要再下载外部 qlib 数据：

```bash
python -m pip install -e ".[qlib]"
```

## 2. 启动 Web 控制台

```bash
ashare-backtest-web
```

打开：

```text
http://127.0.0.1:8888
```

## 3. 体验 qlib 链路

在页面顶部把 workspace 切到 `qlib`。

进入 `/research` 模型研究台：

1. 选择 `demo strategy` 策略
2. 点击运行研究
3. 等待任务完成，确认生成 qlib scores、metrics 和 layer analysis

![alt text](research_run_page.png)

进入 `/backtest` 回测控制台：

1. 选择 qlib demo 分数文件
2. 确认回测区间
3. 点击开始回测
4. 查看回测结果

![alt text](qlib_back_test.png)

## 4. 体验 native 链路

在页面顶部把 workspace 切到 `native`。

进入 `/research` 模型研究台：

1. 选择 native demo 配置
2. 点击运行研究
3. 等待任务完成，确认生成 scores、metrics 和 layer analysis

进入 `/backtest` 回测控制台：

1. 选择刚生成的 native demo 分数文件
2. 确认回测区间
3. 点击开始回测
4. 查看收益曲线、交易明细和 summary

!!! note
    快速体验默认使用 `storage/demo/` 小型数据源。它只用于验证流程，不代表真实研究数据质量。

## 5. 命令行用户

如果你更习惯命令行，可以直接看 [CLI 对照](cli-reference.md)。文档里的 CLI 命令和 Web 页面核心功能一一对应。
