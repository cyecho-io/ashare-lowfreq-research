const state = {
  strategies: [],
  runs: [],
  activeRun: null,
  activeHistory: null,
  activeHistoryTrades: [],
  activeLineage: null,
  activeTab: "latest",
  currentJobId: null,
};

const els = {
  strategySelect: document.getElementById("paper-strategy-select"),
  tradeDate: document.getElementById("paper-trade-date"),
  backtestStartDate: document.getElementById("paper-backtest-start-date"),
  initialCash: document.getElementById("paper-initial-cash"),
  label: document.getElementById("paper-label"),
  latestMeta: document.getElementById("paper-latest-meta"),
  presetMeta: document.getElementById("paper-preset-meta"),
  form: document.getElementById("paper-form"),
  runButton: document.getElementById("paper-run-button"),
  backtestButton: document.getElementById("paper-backtest-button"),
  jobStatus: document.getElementById("paper-job-status"),
  runsList: document.getElementById("paper-runs-list"),
  resultTitle: document.getElementById("paper-result-title"),
  tabLatest: document.getElementById("paper-tab-latest"),
  tabHistory: document.getElementById("paper-tab-history"),
  tabLineage: document.getElementById("paper-tab-lineage"),
  panelLatest: document.getElementById("paper-panel-latest"),
  panelHistory: document.getElementById("paper-panel-history"),
  panelLineage: document.getElementById("paper-panel-lineage"),
  summaryCards: document.getElementById("paper-summary-cards"),
  selectionCount: document.getElementById("paper-selection-count"),
  selectionList: document.getElementById("paper-selection-list"),
  preopenCount: document.getElementById("paper-preopen-count"),
  preopenList: document.getElementById("paper-preopen-list"),
  actionCount: document.getElementById("paper-action-count"),
  actionList: document.getElementById("paper-action-list"),
  riskCount: document.getElementById("paper-risk-count"),
  riskList: document.getElementById("paper-risk-list"),
  nextState: document.getElementById("paper-next-state"),
  historySummary: document.getElementById("paper-history-summary"),
  historyFilter: document.getElementById("paper-history-filter"),
  historyCount: document.getElementById("paper-history-count"),
  historyBody: document.getElementById("paper-history-body"),
  lineageFilter: document.getElementById("paper-lineage-filter"),
  lineageCount: document.getElementById("paper-lineage-count"),
  lineageBody: document.getElementById("paper-lineage-body"),
};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "request_failed");
  }
  return payload;
}

function formatNumber(value, digits = 2) {
  return new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(Number(value || 0));
}

function formatPercent(value) {
  return `${formatNumber(Number(value || 0) * 100, 2)}%`;
}

function formatDateInput(date = new Date()) {
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function activePreset() {
  return state.strategies.find((item) => item.config_path === els.strategySelect.value) || null;
}

function renderPresetMeta() {
  const preset = activePreset();
  if (!preset) {
    els.presetMeta.textContent = "未找到策略配置。";
    els.latestMeta.textContent = "";
    return;
  }
  const sourceLabel =
    preset.paper_source_kind === "merged_history"
      ? "merged-history"
      : preset.paper_source_kind === "latest_manifest_source"
      ? "latest-source"
      : preset.paper_source_kind === "latest_manifest"
        ? "latest"
        : "config";
  const hasLatestManifest =
    preset.paper_source_kind === "merged_history" ||
    preset.paper_source_kind === "latest_manifest_source" ||
    preset.paper_source_kind === "latest_manifest";
  const latestLine =
    hasLatestManifest
      ? `最新信号: <strong>${preset.latest_signal_date || "-"}</strong> / 执行日 <strong>${preset.latest_execution_date || "-"}</strong>`
      : "未发现 latest manifest，当前回退到历史 score_output_path";
  els.presetMeta.innerHTML = [
    `回测分数: <strong>${preset.score_output_path}</strong>`,
    `策略快照分数: <strong>${preset.paper_score_output_path}</strong> <span class="muted">(${sourceLabel})</span>`,
    `可回放区间: <strong>${preset.paper_score_start_date || "-"}</strong> 至 <strong>${preset.paper_score_end_date || "-"}</strong>`,
    latestLine,
    `标准参数: top_k ${preset.top_k}, rebalance_every ${preset.rebalance_every}, min_hold_bars ${preset.min_hold_bars}`,
  ].join("<br />");
  els.latestMeta.innerHTML =
    hasLatestManifest
      ? [
          `当前 latest 信号日: <strong>${preset.latest_signal_date || "-"}</strong>`,
          `对应执行日: <strong>${preset.latest_execution_date || "-"}</strong>`,
        ].join(" · ")
      : `当前还没有 latest manifest，页面会回退到配置里的历史分数文件。`;
}

function applyPresetDefaults() {
  const preset = activePreset();
  if (!preset) return;
  els.initialCash.value = preset.initial_cash;
  els.tradeDate.value = preset.latest_execution_date || formatDateInput();
  els.backtestStartDate.value = preset.paper_score_start_date || preset.default_start_date || formatDateInput();
  renderPresetMeta();
}

function setJobButtonsDisabled(disabled) {
  els.runButton.disabled = disabled;
  els.backtestButton.disabled = disabled;
}

function renderRuns() {
  els.runsList.innerHTML = "";
  if (!state.runs.length) {
    els.runsList.innerHTML = `<div class="muted">还没有可展示的策略快照。</div>`;
    return;
  }
  for (const run of state.runs) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `run-card${state.activeRun?.id === run.id ? " active" : ""}`;
    card.innerHTML = `
      <h3>${run.name}</h3>
      <div class="run-meta">
        <span>执行日 ${run.trade_date || "-"}</span>
        <span>${run.updated_at}</span>
      </div>
      <div class="run-meta">
        <span>当前持仓 ${run.summary.current_position_count ?? 0}</span>
        <span>目标持仓 ${run.summary.target_position_count ?? 0}</span>
      </div>
    `;
    card.addEventListener("click", () => loadRun(run.id));
    els.runsList.appendChild(card);
  }
}

function renderSummary(strategyState, title) {
  const summary = strategyState?.summary || {};
  els.resultTitle.textContent = title;
  const items = [
    ["执行日", summary.execution_date || "-"],
    ["信号日", summary.signal_date || "-"],
    ["决策原因", summary.decision_reason || "-"],
    ["组合市值", formatNumber(summary.portfolio_value_pre_open, 2)],
    ["策略现金", formatNumber(summary.model_cash_pre_open, 2)],
  ];
  els.summaryCards.innerHTML = items
    .map(([label, value]) => `<div class="summary-card"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderSourceInfo(run) {
  const sourceLine =
    run?.paper_source_kind === "merged_history"
      ? "merged historical scores"
      : run?.paper_source_kind === "latest_manifest_source"
      ? "latest manifest source_scores_path"
      : run?.paper_source_kind === "latest_manifest"
        ? "latest manifest"
        : "config score_output_path";
  const scorePath = run?.scores_path || run?.strategy_state?.strategy_config?.scores_path || "-";
  const existing = els.summaryCards.innerHTML;
  els.summaryCards.innerHTML =
    existing +
    `<div class="summary-card"><span>快照分数源</span><strong>${sourceLine}</strong><small>${scorePath}</small></div>`;
}

function renderHistorySummary(history) {
  const summary = history?.summary || {};
  const items =
    history?.source_kind === "latest_trade_log"
      ? [
          ["来源", "latest 策略流水"],
          ["信号日", summary.signal_date || "-"],
          ["执行日", summary.execution_date || "-"],
          ["决策原因", summary.decision_reason || "-"],
          ["交易数", `${summary.trade_count ?? 0}`],
        ]
      : [
          ["总收益", formatPercent(summary.total_return)],
          ["年化收益", formatPercent(summary.annual_return)],
          ["最大回撤", formatPercent(summary.max_drawdown)],
          ["Sharpe", formatNumber(summary.sharpe_ratio, 2)],
          ["交易数", `${summary.trade_count ?? 0}`],
        ];
  els.historySummary.innerHTML = items
    .map(([label, value]) => `<div class="summary-card"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderHistoryTrades() {
  const keyword = els.historyFilter.value.trim().toLowerCase();
  const ordered = [...state.activeHistoryTrades].sort((left, right) => {
    const leftKey = `${left.trade_date || ""} ${left.symbol || ""} ${left.reason || ""}`;
    const rightKey = `${right.trade_date || ""} ${right.symbol || ""} ${right.reason || ""}`;
    return rightKey.localeCompare(leftKey);
  });
  const filtered = !keyword
    ? ordered
    : ordered.filter((trade) =>
        [trade.symbol, trade.reason, trade.status, trade.side].join(" ").toLowerCase().includes(keyword),
      );
  const limited = filtered.slice(0, 300);
  els.historyCount.textContent = `${limited.length} / ${filtered.length} 笔（最多显示最新 300 条）`;
  els.historyBody.innerHTML = limited
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

function setActiveTab(tab) {
  state.activeTab = tab;
  const isLatest = tab === "latest";
  const isHistory = tab === "history";
  const isLineage = tab === "lineage";
  els.tabLatest.classList.toggle("active", isLatest);
  els.tabHistory.classList.toggle("active", isHistory);
  els.tabLineage.classList.toggle("active", isLineage);
  els.tabLatest.setAttribute("aria-selected", String(isLatest));
  els.tabHistory.setAttribute("aria-selected", String(isHistory));
  els.tabLineage.setAttribute("aria-selected", String(isLineage));
  els.panelLatest.classList.toggle("active", isLatest);
  els.panelHistory.classList.toggle("active", isHistory);
  els.panelLineage.classList.toggle("active", isLineage);
  els.panelLatest.hidden = !isLatest;
  els.panelHistory.hidden = !isHistory;
  els.panelLineage.hidden = !isLineage;
}

function renderLatestStateBanner(run) {
  const sourceLine =
    run?.paper_source_kind === "latest_manifest" || run?.paper_source_kind === "latest_manifest_source"
      ? "当前展示的是 latest 快照"
      : run?.paper_source_kind === "merged_history"
        ? "当前展示的是合并历史快照"
        : "当前展示的是回退快照";
  const signalDate = run?.latest_signal_date || run?.strategy_state?.summary?.signal_date || "-";
  const executionDate = run?.latest_execution_date || run?.strategy_state?.summary?.execution_date || "-";
  els.latestMeta.innerHTML = `${sourceLine} · 信号日 <strong>${signalDate}</strong> · 执行日 <strong>${executionDate}</strong>`;
}

function renderHoldings(strategyState) {
  const selected = strategyState?.plan?.selected_symbols || [];
  const preopen = strategyState?.pre_open?.positions || [];
  els.selectionCount.textContent = selected.length ? `${selected.length} 只` : "暂无";
  els.selectionList.innerHTML = selected.length
    ? selected.map((symbol) => `<span class="chip">${symbol}</span>`).join("")
    : `<span class="muted">当前快照没有目标持仓。</span>`;
  els.preopenCount.textContent = preopen.length ? `${preopen.length} 只` : "空仓";
  els.preopenList.innerHTML = preopen.length
    ? preopen
        .map(
          (item) => `
            <div class="holding-row">
              <strong>${item.symbol}</strong>
              <span>${formatPercent(item.weight)}</span>
              <span>${formatNumber(item.market_value, 2)}</span>
            </div>
          `,
        )
        .join("")
    : `<span class="muted">执行日前策略账为空仓。</span>`;
}

function renderActions(strategyState) {
  const actions = strategyState?.plan?.actions || [];
  els.actionCount.textContent = actions.length ? `${actions.length} 条` : "暂无";
  els.actionList.innerHTML = actions.length
    ? actions
        .map(
          (item) => `
            <div class="action-row">
              <div class="action-main">
                <strong>${item.symbol}</strong>
                <span class="action-badge action-${String(item.action || "").toLowerCase()}">${item.action}</span>
              </div>
              <div class="action-meta">
                <span>当前 ${formatPercent(item.current_weight)}</span>
                <span>目标 ${formatPercent(item.target_weight)}</span>
                <span>变动 ${formatPercent(item.delta_weight)}</span>
              </div>
            </div>
          `,
        )
        .join("")
    : `<div class="muted">当前快照没有动作建议。</div>`;
}

function renderRiskAndNextState(strategyState) {
  const riskFlags = strategyState?.plan?.risk_flags || [];
  const nextState = strategyState?.next_state || {};
  els.riskCount.textContent = riskFlags.length ? `${riskFlags.length} 项` : "无";
  els.riskList.innerHTML = riskFlags.length
    ? riskFlags
        .map(
          (item) => `
            <div class="detail-row">
              <strong>${item.flag}</strong>
              <span>${item.symbol ? `${item.symbol} · ` : ""}${item.detail || ""}</span>
            </div>
          `,
        )
        .join("")
    : `<div class="muted">当前快照没有额外风险提示。</div>`;

  const lines = [
    ["状态日期", nextState.as_of_trade_date || "-"],
    ["执行待完成", nextState.execution_pending ? "是" : "否"],
    ["组合市值", formatNumber(nextState.portfolio_value, 2)],
    ["现金", formatNumber(nextState.cash, 2)],
    ["持仓数", Array.isArray(nextState.positions) ? `${nextState.positions.length} 只` : "0 只"],
  ];
  els.nextState.innerHTML = lines
    .map(
      ([label, value]) => `
        <div class="detail-row">
          <strong>${label}</strong>
          <span>${value}</span>
        </div>
      `,
    )
    .join("");
}

async function loadStrategies() {
  const payload = await fetchJson("/api/paper/strategies");
  state.strategies = payload.strategies;
  els.strategySelect.innerHTML = state.strategies
    .map((strategy) => `<option value="${strategy.config_path}">${strategy.name}</option>`)
    .join("");
  applyPresetDefaults();
}

async function loadRuns(selectFirst = true) {
  const payload = await fetchJson("/api/paper/runs");
  state.runs = payload.runs;
  renderRuns();
  if (selectFirst && state.runs.length) {
    await loadRun(state.runs[0].id);
  }
}

async function loadLatestSnapshot() {
  const preset = activePreset();
  if (!preset) return;
  try {
    const payload = await fetchJson(`/api/paper/latest?strategy_id=${encodeURIComponent(preset.id)}`);
    state.activeRun = payload;
    renderSummary(payload.strategy_state, `${preset.name} · latest`);
    renderSourceInfo(payload);
    renderLatestStateBanner(payload);
    renderHoldings(payload.strategy_state);
    renderActions(payload.strategy_state);
    renderRiskAndNextState(payload.strategy_state);
    renderRuns();
  } catch (error) {
    els.resultTitle.textContent = "当前没有 latest 快照";
    els.summaryCards.innerHTML = "";
    els.selectionCount.textContent = "暂无";
    els.selectionList.innerHTML = `<span class="muted">先运行盘前脚本生成 latest 快照。</span>`;
    els.preopenCount.textContent = "暂无";
    els.preopenList.innerHTML = `<span class="muted">当前未找到 latest 策略状态。</span>`;
    els.actionCount.textContent = "暂无";
    els.actionList.innerHTML = `<div class="muted">${error.message}</div>`;
    els.riskCount.textContent = "暂无";
    els.riskList.innerHTML = `<div class="muted">当前没有可展示的 latest 风险信息。</div>`;
    els.nextState.innerHTML = `<div class="muted">运行盘前流程后，这里会展示下一状态摘要。</div>`;
  }
}

async function loadHistory() {
  const preset = activePreset();
  if (!preset) return;
  try {
    const payload = await fetchJson(`/api/paper/history?strategy_id=${encodeURIComponent(preset.id)}`);
    state.activeHistory = payload;
    state.activeHistoryTrades = payload.trades || [];
    renderHistorySummary(payload);
    renderHistoryTrades();
  } catch (error) {
    state.activeHistory = null;
    state.activeHistoryTrades = [];
    els.historySummary.innerHTML = "";
    els.historyCount.textContent = "暂无";
    els.historyBody.innerHTML = `<tr><td colspan="8" class="muted">未找到该策略的历史交易详情: ${error.message}</td></tr>`;
  }
}

function renderLineage() {
  const keyword = els.lineageFilter.value.trim().toLowerCase();
  const decisions = state.activeLineage?.decision_log || [];
  const ordered = [...decisions].sort((left, right) => right.trade_date.localeCompare(left.trade_date));
  const filtered = !keyword
    ? ordered
    : ordered.filter((item) =>
        [
          item.trade_date,
          item.signal_date,
          item.decision_reason,
          item.selected_symbols,
        ]
          .join(" ")
          .toLowerCase()
          .includes(keyword),
      );
  els.lineageCount.textContent = `${filtered.length} 条`;
  els.lineageBody.innerHTML = filtered
    .map(
      (item) => `
        <tr>
          <td>${item.trade_date}</td>
          <td>${item.signal_date}</td>
          <td>${item.decision_reason}</td>
          <td>${item.should_rebalance ? "是" : "否"}</td>
          <td>${item.current_position_count}</td>
          <td>${item.target_position_count}</td>
          <td>${item.selected_symbols || "-"}</td>
          <td>${formatNumber(item.cash_pre_decision, 2)}</td>
        </tr>
      `,
    )
    .join("");
}

async function loadLineage() {
  const preset = activePreset();
  if (!preset) return;
  try {
    const payload = await fetchJson(`/api/paper/lineage?strategy_id=${encodeURIComponent(preset.id)}`);
    state.activeLineage = payload;
    renderLineage();
  } catch (error) {
    state.activeLineage = null;
    els.lineageCount.textContent = "暂无";
    els.lineageBody.innerHTML = `<tr><td colspan="8" class="muted">未找到 latest 快照演化记录: ${error.message}</td></tr>`;
  }
}

async function loadRun(runId) {
  const payload = await fetchJson(`/api/paper/runs/${encodeURIComponent(runId)}`);
  if (payload.config_path && els.strategySelect.value !== payload.config_path) {
    els.strategySelect.value = payload.config_path;
    applyPresetDefaults();
    state.activeHistory = null;
    state.activeLineage = null;
    await loadHistory();
    await loadLineage();
  }
  state.activeRun = payload;
  renderRuns();
  renderSummary(payload.strategy_state, payload.name);
  renderSourceInfo(payload);
  renderHoldings(payload.strategy_state);
  renderActions(payload.strategy_state);
  renderRiskAndNextState(payload.strategy_state);
}

async function submitPaper(event) {
  event.preventDefault();
  setJobButtonsDisabled(true);
  els.jobStatus.textContent = "任务已提交，等待生成策略快照。";
  const body = {
    config_path: els.strategySelect.value,
    trade_date: els.tradeDate.value,
    initial_cash: Number(els.initialCash.value),
    label: els.label.value,
  };
  try {
    const payload = await fetchJson("/api/paper/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.currentJobId = payload.job.id;
    pollJob();
  } catch (error) {
    els.jobStatus.textContent = `提交失败: ${error.message}`;
    setJobButtonsDisabled(false);
  }
}

function historyPayloadFromRun(run) {
  return {
    strategy_id: activePreset()?.id || "",
    run_id: run?.id || "",
    result_dir: run?.result_dir || "",
    summary: run?.summary || {},
    equity_curve: run?.equity_curve || [],
    benchmark_label: run?.benchmark_label || "",
    benchmark_curve: run?.benchmark_curve || [],
    trades: run?.trades || [],
    strategy_state: run?.strategy_state || null,
    source_kind: "paper_backtest_run",
  };
}

async function submitPaperBacktest() {
  setJobButtonsDisabled(true);
  els.jobStatus.textContent = "任务已提交，等待生成完整回测。";
  const body = {
    config_path: els.strategySelect.value,
    start_date: els.backtestStartDate.value,
    initial_cash: Number(els.initialCash.value),
    label: els.label.value,
  };
  try {
    const payload = await fetchJson("/api/paper/backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.currentJobId = payload.job.id;
    pollJob();
  } catch (error) {
    els.jobStatus.textContent = `提交失败: ${error.message}`;
    setJobButtonsDisabled(false);
  }
}

async function pollJob() {
  if (!state.currentJobId) return;
  try {
    const payload = await fetchJson(`/api/paper/jobs/${encodeURIComponent(state.currentJobId)}`);
    const { job, run } = payload;
    if (job.status === "queued") {
      els.jobStatus.textContent = "任务排队中。";
      window.setTimeout(pollJob, 1000);
      return;
    }
    if (job.status === "running") {
      els.jobStatus.textContent =
        job.type === "paper_backtest"
          ? `正在回测 ${job.start_date} 到 ${job.end_date}。`
          : `正在生成 ${job.trade_date} 的策略快照。`;
      window.setTimeout(pollJob, 1500);
      return;
    }
    if (job.status === "failed") {
      els.jobStatus.textContent = `生成失败: ${job.error}`;
      setJobButtonsDisabled(false);
      return;
    }
    els.jobStatus.textContent =
      job.type === "paper_backtest"
        ? `完整回测已生成，目录: ${job.result_dir}`
        : `策略快照已生成，目录: ${job.result_dir}`;
    setJobButtonsDisabled(false);
    if (job.type === "paper_backtest" && run) {
      state.activeHistory = historyPayloadFromRun(run);
      state.activeHistoryTrades = run.trades || [];
      renderHistorySummary(state.activeHistory);
      renderHistoryTrades();
      setActiveTab("history");
      return;
    }
    await loadRuns(false);
    if (run) {
      state.activeRun = run;
      renderRuns();
      renderSummary(run.strategy_state, run.name);
      renderSourceInfo(run);
      renderHoldings(run.strategy_state);
      renderActions(run.strategy_state);
      renderRiskAndNextState(run.strategy_state);
    }
  } catch (error) {
    els.jobStatus.textContent = `状态查询失败: ${error.message}`;
    setJobButtonsDisabled(false);
  }
}

function bindEvents() {
  els.strategySelect.addEventListener("change", async () => {
    applyPresetDefaults();
    await loadLatestSnapshot();
    await loadHistory();
    await loadLineage();
  });
  els.form.addEventListener("submit", submitPaper);
  els.backtestButton.addEventListener("click", submitPaperBacktest);
  els.historyFilter.addEventListener("input", renderHistoryTrades);
  els.lineageFilter.addEventListener("input", renderLineage);
  els.tabLatest.addEventListener("click", () => setActiveTab("latest"));
  els.tabHistory.addEventListener("click", async () => {
    setActiveTab("history");
    if (!state.activeHistory) {
      await loadHistory();
    }
  });
  els.tabLineage.addEventListener("click", async () => {
    setActiveTab("lineage");
    if (!state.activeLineage) {
      await loadLineage();
    }
  });
}

async function init() {
  bindEvents();
  els.tradeDate.value = formatDateInput();
  await loadStrategies();
  await loadRuns(true);
  await loadLatestSnapshot();
  await loadHistory();
  await loadLineage();
  setActiveTab("latest");
}

init().catch((error) => {
  els.jobStatus.textContent = `初始化失败: ${error.message}`;
});
