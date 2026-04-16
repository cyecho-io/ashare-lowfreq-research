const state = {
  strategies: [],
  scoreFiles: [],
  runs: [],
  activeRun: null,
  activeTrades: [],
  currentJobId: null,
};

const els = {
  scoreFileSelect: document.getElementById("score-file-select"),
  strategyConfigDisplay: document.getElementById("strategy-config-display"),
  startDate: document.getElementById("start-date"),
  endDate: document.getElementById("end-date"),
  initialCash: document.getElementById("initial-cash"),
  label: document.getElementById("label"),
  presetMeta: document.getElementById("preset-meta"),
  runForm: document.getElementById("run-form"),
  runButton: document.getElementById("run-button"),
  jobStatus: document.getElementById("job-status"),
  runsList: document.getElementById("runs-list"),
  resultTitle: document.getElementById("result-title"),
  summaryCards: document.getElementById("summary-cards"),
  equityChart: document.getElementById("equity-chart"),
  equityPoints: document.getElementById("equity-points"),
  benchmarkLabel: document.getElementById("benchmark-label"),
  tradesBody: document.getElementById("trades-body"),
  tradeFilter: document.getElementById("trade-filter"),
  tradeCount: document.getElementById("trade-count"),
};

function logUi(message, details = null) {
  if (details === null) {
    console.info(`[Backtest UI] ${message}`);
    return;
  }
  console.info(`[Backtest UI] ${message}`, details);
}

async function fetchJson(url, options = {}) {
  const method = options.method || "GET";
  logUi(`Request ${method} ${url}`, options.body ? JSON.parse(options.body) : null);
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    logUi(`Request failed ${method} ${url}`, payload);
    throw new Error(payload.error || "request_failed");
  }
  logUi(`Request ok ${method} ${url}`, payload);
  return payload;
}

function formatNumber(value, digits = 2) {
  return new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(Number(value || 0));
}

function formatPercent(value) {
  return `${formatNumber((Number(value || 0) * 100), 2)}%`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[character]);
}

function presetForConfigPath(configPath) {
  const normalized = String(configPath || "").trim();
  return state.strategies.find((item) => item.config_path === normalized) || null;
}

function activeScoreFile() {
  return state.scoreFiles.find((item) => item.path === els.scoreFileSelect.value) || null;
}

function activePreset() {
  const selectedScore = activeScoreFile();
  return presetForConfigPath(selectedScore?.config_path || "");
}

function scoreFileLabel(item) {
  const backend = item.backend || item.workspace || "native";
  const model = item.model ? `/${item.model}` : "";
  const configId = item.config_id || item.factor_spec_id || "";
  const range = item.start_date && item.end_date ? ` · ${item.start_date}~${item.end_date}` : "";
  return `[${backend}${model}] ${configId ? `${configId} · ` : ""}${item.path}${range}`;
}

function renderScoreFileOptions(selectedPath = "") {
  els.scoreFileSelect.innerHTML = "";
  if (!state.scoreFiles.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "未找到 score parquet";
    option.disabled = true;
    option.selected = true;
    els.scoreFileSelect.appendChild(option);
    return;
  }

  for (const item of state.scoreFiles) {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = scoreFileLabel(item);
    els.scoreFileSelect.appendChild(option);
  }
  const selectedExists = selectedPath && state.scoreFiles.some((item) => item.path === selectedPath);
  els.scoreFileSelect.value = selectedExists ? selectedPath : state.scoreFiles[0].path;
}

function applyQueryPrefill() {
  const params = new URLSearchParams(window.location.search);
  const configPath = params.get("config_path") || "";
  const scoresPath = params.get("scores_path") || "";

  let selectedPath = "";
  if (scoresPath && state.scoreFiles.some((item) => item.path === scoresPath)) {
    selectedPath = scoresPath;
  }
  if (!selectedPath && configPath) {
    selectedPath = state.scoreFiles.find((item) => item.config_path === configPath)?.path || "";
  }
  renderScoreFileOptions(selectedPath);
  applyScoreFileDates();
}

function renderPresetMeta() {
  const selectedScore = activeScoreFile();
  const preset = activePreset();
  const configPath = selectedScore?.config_path || "";
  els.strategyConfigDisplay.textContent = configPath || "未能自动关联";
  if (!selectedScore) {
    els.presetMeta.textContent = "未找到可用于回测的分数文件。";
    return;
  }
  const lines = [
    `当前选择分数文件: <strong>${escapeHtml(selectedScore.path || "-")}</strong>`,
    `关联策略配置: <strong>${escapeHtml(configPath || "未能自动关联")}</strong>`,
    `当前研究后端: <strong>${escapeHtml(selectedScore.backend || "-")}</strong> / 模型 <strong>${escapeHtml(selectedScore.model || "-")}</strong>`,
    `当前分数区间: <strong>${escapeHtml(selectedScore.start_date || "-")}</strong> 至 <strong>${escapeHtml(selectedScore.end_date || "-")}</strong>`,
  ];
  if (preset) {
    lines.push(
      `默认参数: top_k ${escapeHtml(preset.top_k)}, rebalance_every ${escapeHtml(preset.rebalance_every)}, min_hold_bars ${escapeHtml(preset.min_hold_bars)}`,
    );
  } else {
    lines.push("未自动匹配策略配置，不能直接提交回测。");
  }
  els.presetMeta.innerHTML = lines.join("<br />");
}

function applyScoreFileDates() {
  const preset = activePreset();
  const selectedScore = activeScoreFile();
  if (!selectedScore) {
    els.startDate.value = "";
    els.endDate.value = "";
    renderPresetMeta();
    return;
  }
  els.startDate.value = selectedScore.start_date || preset?.default_start_date || "";
  els.endDate.value = selectedScore.end_date || preset?.default_end_date || "";
  if (preset?.initial_cash != null) {
    els.initialCash.value = preset.initial_cash;
  } else if (!els.initialCash.value) {
    els.initialCash.value = 1000000;
  }
  renderPresetMeta();
}

function renderRuns() {
  els.runsList.innerHTML = "";
  const visibleRuns = state.runs;
  if (!visibleRuns.length) {
    els.runsList.innerHTML = `<div class="muted">还没有可展示的回测结果。</div>`;
    return;
  }
  for (const run of visibleRuns) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `run-card${state.activeRun?.id === run.id ? " active" : ""}`;
    card.innerHTML = `
      <h3>${run.name}</h3>
      <div class="run-meta">
        <span>收益 ${formatPercent(run.summary.total_return)}</span>
        <span>Sharpe ${formatNumber(run.summary.sharpe_ratio, 2)}</span>
      </div>
      <div class="run-meta">
        <span>${run.backend || "-"}/${run.model || "-"}</span>
        <span>${run.updated_at}</span>
      </div>
    `;
    card.addEventListener("click", () => loadRun(run.id));
    els.runsList.appendChild(card);
  }
}

function renderSummary(run) {
  els.resultTitle.textContent = run.name;
  const summary = run.summary;
  const backtestRange = run.backtest_start_date && run.backtest_end_date
    ? `${run.backtest_start_date} 至 ${run.backtest_end_date}`
    : "-";
  const scoreRange = run.score_start_date && run.score_end_date
    ? `${run.score_start_date} 至 ${run.score_end_date}`
    : "-";
  const initialCash = run.strategy_state?.strategy_config?.initial_cash;
  const items = [
    ["分数文件", run.scores_path || "-"],
    ["研究后端", run.backend || "-"],
    ["模型", run.model || "-"],
    ["回测区间", backtestRange],
    ["分数区间", scoreRange],
    ["初始资金", initialCash == null ? "-" : formatNumber(initialCash, 2)],
    ["总收益", formatPercent(summary.total_return)],
    ["年化收益", formatPercent(summary.annual_return)],
    ["最大回撤", formatPercent(summary.max_drawdown)],
    ["Sharpe", formatNumber(summary.sharpe_ratio, 2)],
    ["已成交/拒单", `${summary.filled_trade_count} / ${summary.rejected_trade_count}`],
  ];
  els.summaryCards.innerHTML = items
    .map(([label, value]) => {
      const valueTag = label === "分数文件" ? "small" : "strong";
      return `<div class="summary-card"><span>${label}</span><${valueTag}>${value}</${valueTag}></div>`;
    })
    .join("");
}

function renderChart(points, benchmarkPoints = [], benchmarkLabel = "") {
  const width = 760;
  const height = 260;
  els.equityPoints.textContent = `${points.length} 个交易日`;
  els.benchmarkLabel.textContent = benchmarkLabel ? `对比: ${benchmarkLabel}` : "";
  if (!points.length) {
    els.equityChart.innerHTML = "";
    return;
  }
  const values = points
    .map((item) => Number(item.equity))
    .concat(benchmarkPoints.map((item) => Number(item.equity)));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const coords = points.map((point, index) => {
    const x = (index / Math.max(points.length - 1, 1)) * width;
    const y = height - ((Number(point.equity) - min) / span) * (height - 28) - 14;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  const benchmarkCoords = benchmarkPoints.map((point, index) => {
    const x = (index / Math.max(benchmarkPoints.length - 1, 1)) * width;
    const y = height - ((Number(point.equity) - min) / span) * (height - 28) - 14;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  const area = `${coords[0]} ${coords.join(" ")} ${width},${height} 0,${height}`;
  els.equityChart.innerHTML = `
    <defs>
      <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="rgba(37, 58, 82, 0.32)"></stop>
        <stop offset="100%" stop-color="rgba(37, 58, 82, 0.03)"></stop>
      </linearGradient>
    </defs>
    <line x1="0" y1="${height - 14}" x2="${width}" y2="${height - 14}" stroke="rgba(23,33,33,0.12)" />
    <polygon points="${area}" fill="url(#equityFill)"></polygon>
    ${
      benchmarkCoords.length
        ? `<polyline points="${benchmarkCoords.join(" ")}" fill="none" stroke="#b48a56" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="7 7"></polyline>`
        : ""
    }
    <polyline points="${coords.join(" ")}" fill="none" stroke="#253a52" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></polyline>
  `;
}

function renderTrades() {
  const keyword = els.tradeFilter.value.trim().toLowerCase();
  const filtered = !keyword
    ? state.activeTrades
    : state.activeTrades.filter((trade) =>
        [trade.symbol, trade.reason, trade.status, trade.side].join(" ").toLowerCase().includes(keyword),
      );
  els.tradeCount.textContent = `${filtered.length} / ${state.activeTrades.length} 笔`;
  els.tradesBody.innerHTML = filtered
    .map(
      (trade) => `
        <tr>
          <td>${trade.trade_date}</td>
          <td>${trade.symbol}</td>
          <td class="side-${trade.side.toLowerCase()}">${trade.side}</td>
          <td>${formatNumber(trade.quantity, 0)}</td>
          <td>${formatNumber(trade.price, 4)}</td>
          <td>${formatNumber(trade.amount, 2)}</td>
          <td class="status-${trade.status.toLowerCase()}">${trade.status}</td>
          <td>${trade.reason}</td>
        </tr>
      `,
    )
    .join("");
}

async function loadStrategies() {
  const payload = await fetchJson(window.AshareWorkspace.withWorkspaceUrl("/api/strategies"));
  state.strategies = payload.strategies;
  state.scoreFiles = payload.score_files || [];
  renderScoreFileOptions();
  applyScoreFileDates();
}

async function loadRuns(selectFirst = true) {
  const payload = await fetchJson(window.AshareWorkspace.withWorkspaceUrl("/api/runs"));
  state.runs = payload.runs;
  renderRuns();
  if (selectFirst && state.runs.length) {
    await loadRun(state.runs[0].id);
  }
}

async function loadRun(runId) {
  const payload = await fetchJson(window.AshareWorkspace.withWorkspaceUrl(`/api/runs/${encodeURIComponent(runId)}`));
  state.activeRun = payload;
  state.activeTrades = payload.trades;
  renderRuns();
  renderSummary(payload);
  renderChart(payload.equity_curve, payload.benchmark_curve || [], payload.benchmark_label || "");
  renderTrades();
}

async function submitRun(event) {
  event.preventDefault();
  els.runButton.disabled = true;
  els.jobStatus.textContent = "任务已提交，等待启动。";
  const selectedScore = activeScoreFile();
  const configPath = selectedScore?.config_path || "";
  if (!selectedScore?.path || !configPath) {
    els.jobStatus.textContent = "提交失败: 当前分数文件无法关联策略配置。请重新生成 score 或检查 config_id/factor_spec.id。";
    els.runButton.disabled = false;
    return;
  }
  const body = {
    config_path: configPath,
    scores_path: selectedScore.path,
    start_date: els.startDate.value,
    end_date: els.endDate.value,
    initial_cash: Number(els.initialCash.value),
    label: els.label.value,
  };
  logUi("点击 [运行回测]，对应 submit_backtest -> /api/backtests -> BacktestWebApp.submit_backtest", body);
  try {
    const payload = await fetchJson("/api/backtests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(window.AshareWorkspace.withWorkspaceBody(body)),
    });
    state.currentJobId = payload.job.id;
    pollJob();
  } catch (error) {
    els.jobStatus.textContent = `提交失败: ${error.message}`;
    els.runButton.disabled = false;
  }
}

async function pollJob() {
  if (!state.currentJobId) return;
  try {
    const payload = await fetchJson(window.AshareWorkspace.withWorkspaceUrl(`/api/jobs/${encodeURIComponent(state.currentJobId)}`));
    const { job, run } = payload;
    logUi("轮询普通回测任务状态", job);
    if (job.status === "queued") {
      els.jobStatus.textContent = "任务排队中。";
      window.setTimeout(pollJob, 1000);
      return;
    }
    if (job.status === "running") {
      els.jobStatus.textContent = `正在回测: ${job.start_date} 到 ${job.end_date}`;
      window.setTimeout(pollJob, 1500);
      return;
    }
    if (job.status === "failed") {
      els.jobStatus.textContent = `回测失败: ${job.error}`;
      els.runButton.disabled = false;
      return;
    }
    els.jobStatus.textContent = `回测完成，结果目录: ${job.result_dir}`;
    els.runButton.disabled = false;
    await loadRuns(false);
    if (run) {
      state.activeRun = run;
      state.activeTrades = run.trades;
      renderRuns();
      renderSummary(run);
      renderChart(run.equity_curve, run.benchmark_curve || [], run.benchmark_label || "");
      renderTrades();
    } else {
      await loadRun(job.id);
    }
  } catch (error) {
    els.jobStatus.textContent = `状态查询失败: ${error.message}`;
    els.runButton.disabled = false;
  }
}

function bindEvents() {
  els.scoreFileSelect.addEventListener("change", applyScoreFileDates);
  els.runForm.addEventListener("submit", submitRun);
  els.tradeFilter.addEventListener("input", renderTrades);
  window.addEventListener("workspacechange", async () => {
    state.currentJobId = null;
    state.activeRun = null;
    state.activeTrades = [];
    await loadStrategies();
    applyQueryPrefill();
    await loadRuns(true);
  });
}

async function init() {
  window.AshareWorkspace.initWorkspaceControls();
  bindEvents();
  await loadStrategies();
  applyQueryPrefill();
  await loadRuns(true);
}

init().catch((error) => {
  els.jobStatus.textContent = `初始化失败: ${error.message}`;
});
