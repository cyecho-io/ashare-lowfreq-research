const state = {
  strategies: [],
  runs: [],
  activeRun: null,
  activeHistory: null,
  activeHistoryTrades: [],
  activeLineage: null,
  activeTab: "account",
  currentJobId: null,
  primaryActionLoading: false,
  readiness: null,
};

const els = {
  strategySelect: document.getElementById("simulation-strategy-select"),
  tradeDate: document.getElementById("simulation-trade-date"),
  initialCash: document.getElementById("simulation-initial-cash"),
  label: document.getElementById("simulation-label"),
  latestMeta: document.getElementById("simulation-latest-meta"),
  presetMeta: document.getElementById("simulation-preset-meta"),
  form: document.getElementById("simulation-form"),
  runButton: document.getElementById("simulation-run-button"),
  primaryActionButton: document.getElementById("simulation-primary-action-button"),
  jobStatus: document.getElementById("simulation-job-status"),
  runsList: document.getElementById("simulation-runs-list"),
  resultTitle: document.getElementById("simulation-result-title"),
  tabAccount: document.getElementById("simulation-tab-account"),
  tabHistory: document.getElementById("simulation-tab-history"),
  tabLineage: document.getElementById("simulation-tab-lineage"),
  panelAccount: document.getElementById("simulation-panel-account"),
  panelHistory: document.getElementById("simulation-panel-history"),
  panelLineage: document.getElementById("simulation-panel-lineage"),
  summaryCards: document.getElementById("simulation-summary-cards"),
  preopenCount: document.getElementById("simulation-preopen-count"),
  preopenList: document.getElementById("simulation-preopen-list"),
  actionCount: document.getElementById("simulation-action-count"),
  actionList: document.getElementById("simulation-action-list"),
  riskCount: document.getElementById("simulation-risk-count"),
  riskList: document.getElementById("simulation-risk-list"),
  nextState: document.getElementById("simulation-next-state"),
  historySummary: document.getElementById("simulation-history-summary"),
  historyFilter: document.getElementById("simulation-history-filter"),
  historyCount: document.getElementById("simulation-history-count"),
  historyBody: document.getElementById("simulation-history-body"),
  lineageFilter: document.getElementById("simulation-lineage-filter"),
  lineageCount: document.getElementById("simulation-lineage-count"),
  lineageBody: document.getElementById("simulation-lineage-body"),
};

function logUi(message, details = null) {
  if (details === null) {
    console.info(`[Simulation UI] ${message}`);
    return;
  }
  console.info(`[Simulation UI] ${message}`, details);
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
  const readiness = state.readiness;
  let readinessLine = "任务状态: <strong>-</strong>";
  if (readiness?.error) {
    readinessLine = `任务状态: <strong>状态查询失败</strong> <span class="muted">${readiness.message || readiness.error}</span>`;
  } else if (readiness) {
    if (readiness.is_ready) {
      readinessLine = `当前状态: <strong>可以生成计划</strong> <span class="muted">${readiness.message}</span>`;
    } else {
      readinessLine = `当前状态: <strong>暂时不能生成</strong> <span class="muted">${readiness.message || "当前日期不可用"}</span>`;
    }
  }
  els.presetMeta.innerHTML = [
    `策略规格: <strong>${preset.factor_spec_id}</strong>`,
    `计划使用分数文件: <strong>${readiness?.scores_path || "-"}</strong>`,
    `计划使用 factor 快照: <strong>${readiness?.factor_panel_path || "-"}</strong>`,
    `下一交易日: <strong>${readiness?.execution_date || "-"}</strong>`,
    readinessLine,
    `计划参数: top_k <strong>${preset.top_k}</strong>, rebalance_every <strong>${preset.rebalance_every}</strong>, min_hold_bars <strong>${preset.min_hold_bars}</strong>, keep_buffer <strong>${preset.keep_buffer}</strong>`,
  ].join("<br />");
  els.latestMeta.innerHTML = `最近一次可用信号日 <strong>${preset.latest_signal_date || "-"}</strong> · 对应执行日 <strong>${preset.latest_execution_date || "-"}</strong>`;
}

function applyPresetDefaults() {
  const preset = activePreset();
  if (!preset) return;
  els.initialCash.value = preset.initial_cash;
  const candidateDate = preset.latest_signal_date || preset.paper_score_end_date || preset.default_end_date || formatDateInput();
  els.tradeDate.value = candidateDate;
  renderPresetMeta();
}

function renderReadiness() {
  const readiness = state.readiness;
  if (!readiness) return;
  const prefix = readiness.is_ready ? "可创建模拟账户" : "数据未就绪";
  els.jobStatus.textContent = `${prefix}。${readiness.message}`;
}

function activeHistoryRunId() {
  return String(state.activeRun?.executed_run_id || state.activeRun?.previous_run_id || "").trim();
}

function accountNodeType(run) {
  return String(run?.current_node_type || (run?.executed_run_id ? "executed" : "planned")).trim();
}

function accountStatusMessage(run) {
  return String(run?.account_status_message || run?.execution_message || "").trim();
}

async function refreshReadiness() {
  const preset = activePreset();
  if (!preset || !els.tradeDate.value) {
    state.readiness = null;
    return;
  }
  try {
    const query = new URLSearchParams({
      config_path: els.strategySelect.value,
      signal_date: els.tradeDate.value,
    });
    state.readiness = await fetchJson(window.AshareWorkspace.withWorkspaceUrl(`/api/simulation/readiness?${query.toString()}`));
  } catch (error) {
    state.readiness = {
      is_ready: false,
      error: "readiness_request_failed",
      message: error.message,
    };
  }
  renderPresetMeta();
  renderReadiness();
  els.tradeDate.removeAttribute("max");
}

function setJobButtonsDisabled(disabled) {
  els.runButton.disabled = disabled;
  updatePrimaryAction(disabled);
}

function setPrimaryActionLoading(loading) {
  state.primaryActionLoading = loading;
  updatePrimaryAction(els.runButton.disabled);
}

function primaryActionState() {
  const run = state.activeRun;
  if (!run?.id) {
    return {
      label: "生成盘前计划",
      disabled: true,
      reason: "请先选择一个模拟账户。",
      action: null,
    };
  }
  if (run.execution_ready && !run.executed_run_id) {
    return {
      label: "模拟下单",
      disabled: false,
      reason: accountStatusMessage(run),
      action: "execute",
    };
  }
  if (run.next_plan_ready) {
    return {
      label: "生成盘前计划",
      disabled: false,
      reason: String(run.next_plan_message || accountStatusMessage(run)),
      action: "next_plan",
    };
  }
  return {
    label: run.executed_run_id ? "生成盘前计划" : "模拟下单",
    disabled: true,
    reason: run.executed_run_id
      ? (run.next_plan_message || accountStatusMessage(run) || "当前账户还不能继续生成下一交易日盘前计划。")
      : (accountStatusMessage(run) || "当前账户还不能模拟下单。"),
    action: null,
  };
}

function updatePrimaryAction(forceDisabled = false) {
  const actionState = primaryActionState();
  const isLoading = state.primaryActionLoading;
  els.primaryActionButton.textContent = isLoading ? `${actionState.label}中...` : actionState.label;
  els.primaryActionButton.disabled = isLoading || forceDisabled || actionState.disabled;
  els.primaryActionButton.title = actionState.reason || "";
  els.primaryActionButton.classList.toggle("is-loading", isLoading);
  els.primaryActionButton.setAttribute("aria-busy", isLoading ? "true" : "false");
}

function renderRuns() {
  els.runsList.innerHTML = "";
  if (!state.runs.length) {
    els.runsList.innerHTML = `<div class="muted">还没有可展示的盘前计划。</div>`;
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
        <span>${run.execution_ready ? "执行行情已就绪" : "等待执行日行情"}</span>
        <span>${accountNodeType(run) === "executed" ? "已更新账户状态" : "待更新账户状态"}</span>
      </div>
    `;
    card.addEventListener("click", () => loadRun(run.id));
    els.runsList.appendChild(card);
  }
}

function renderSummary(run, title) {
  const summary = run?.strategy_state?.summary || {};
  els.resultTitle.textContent = title;
  const scorePath = run?.scores_path || run?.strategy_state?.strategy_config?.scores_path || "-";
  const nodeType = accountNodeType(run) === "executed" ? "已更新账户状态" : "待执行计划";
  const currentSnapshot = run?.strategy_state?.next_state || {};
  const items = [
    ["账户 ID", run?.account_id || "-"],
    ["当前节点", nodeType],
    ["最新状态日", currentSnapshot.as_of_trade_date || summary.execution_date || "-"],
    ["信号日", summary.signal_date || "-"],
    ["账户状态", accountStatusMessage(run) || (run?.executed_run_id ? "已完成模拟下单" : "等待执行日行情")],
    ["当前组合市值", formatNumber(currentSnapshot.portfolio_value ?? summary.portfolio_value_pre_open, 2)],
    ["可用现金", formatNumber(currentSnapshot.cash ?? summary.model_cash_pre_open, 2)],
    ["分数文件", scorePath],
  ];
  els.summaryCards.innerHTML = items
    .map(([label, value]) => {
      const valueTag = label === "分数文件" ? "small" : "strong";
      return `<div class="summary-card"><span>${label}</span><${valueTag}>${value}</${valueTag}></div>`;
    })
    .join("");
}

function renderHoldings(strategyState) {
  const currentPositions = strategyState?.next_state?.positions || strategyState?.pre_open?.positions || [];
  els.preopenCount.textContent = currentPositions.length ? `${currentPositions.length} 只` : "空仓";
  els.preopenList.innerHTML = currentPositions.length
    ? currentPositions
        .map(
          (item) => {
            const pnl = Number(item.market_value || 0) - (Number(item.cost_basis || 0) * Number(item.quantity || 0));
            const pnlClass = pnl > 0 ? "value-gain" : (pnl < 0 ? "value-loss" : "muted");
            return `
            <div class="holding-row">
              <strong>${item.symbol}</strong>
              <span>${formatPercent(item.weight)}</span>
              <span>${formatNumber(item.market_value, 2)}</span>
              <span class="${pnlClass}">${pnl >= 0 ? "+" : ""}${formatNumber(pnl, 2)}</span>
            </div>
          `;
          },
        )
        .join("")
    : `<span class="muted">当前账户为空仓。</span>`;
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
                <span>变化 ${formatPercent(item.delta_weight)}</span>
              </div>
              <div class="action-meta">
                <span>计划 ${formatNumber(item.planned_quantity || 0, 0)} 股</span>
                <span>实际 ${formatNumber(item.executed_quantity || 0, 0)} 股</span>
                <span>${renderExecutionStatus(item)}</span>
              </div>
            </div>
          `,
        )
        .join("")
    : `<div class="muted">当前没有订单建议。</div>`;
}

function renderExecutionStatus(item) {
  const status = String(item.execution_status || "");
  const reason = String(item.execution_reason || "");
  if (status === "executed") return "已执行";
  if (status === "partially_executed") return "部分执行";
  if (status === "below_round_lot") return "不足 100 股整手，暂不下单";
  if (status === "rejected") return `未执行: ${reason || "条件不满足"}`;
  if (status === "already_at_target") return "无需下单";
  if (status === "not_rebalanced") return "未触发调仓";
  if (status === "not_executed") return "未成交";
  return reason || "-";
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
    : `<div class="muted">当前没有额外风险提示。</div>`;
  const lines = [
    ["状态日期", nextState.as_of_trade_date || "-"],
    ["账户状态", nextState.execution_pending ? "待执行计划" : "已更新到账户当前状态"],
    ["当前动作说明", accountStatusMessage(state.activeRun) || state.activeRun?.next_plan_message || "-"],
    ["组合市值", formatNumber(nextState.portfolio_value, 2)],
    ["可用现金", formatNumber(nextState.cash, 2)],
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

function renderHistorySummary(history) {
  const summary = history?.summary || {};
  const items = [
    ["来源", history?.account_id ? `账户 ${history.account_id} 累计成交` : "当前执行记录"],
    ["信号日", summary.signal_date || "-"],
    ["执行日", summary.execution_date || "-"],
    ["决策原因", summary.decision_reason || "-"],
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

function renderLineage() {
  const keyword = els.lineageFilter.value.trim().toLowerCase();
  const decisions = state.activeLineage?.decision_log || [];
  const ordered = [...decisions].sort((left, right) => right.trade_date.localeCompare(left.trade_date));
  const filtered = !keyword
    ? ordered
    : ordered.filter((item) =>
        [item.trade_date, item.signal_date, item.decision_reason, item.selected_symbols].join(" ").toLowerCase().includes(keyword),
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

function setActiveTab(tab) {
  state.activeTab = tab;
  const isAccount = tab === "account";
  const isHistory = tab === "history";
  const isLineage = tab === "lineage";
  els.tabAccount.classList.toggle("active", isAccount);
  els.tabHistory.classList.toggle("active", isHistory);
  els.tabLineage.classList.toggle("active", isLineage);
  els.tabAccount.setAttribute("aria-selected", String(isAccount));
  els.tabHistory.setAttribute("aria-selected", String(isHistory));
  els.tabLineage.setAttribute("aria-selected", String(isLineage));
  els.panelAccount.classList.toggle("active", isAccount);
  els.panelHistory.classList.toggle("active", isHistory);
  els.panelLineage.classList.toggle("active", isLineage);
  els.panelAccount.hidden = !isAccount;
  els.panelHistory.hidden = !isHistory;
  els.panelLineage.hidden = !isLineage;
}

async function loadStrategies() {
  const payload = await fetchJson(window.AshareWorkspace.withWorkspaceUrl("/api/simulation/strategies"));
  state.strategies = payload.strategies;
  els.strategySelect.innerHTML = state.strategies
    .map((strategy) => `<option value="${strategy.config_path}">${strategy.name}</option>`)
    .join("");
  applyPresetDefaults();
  await refreshReadiness();
}

async function reloadStrategiesPreservingSelection() {
  const selectedConfig = els.strategySelect.value;
  const payload = await fetchJson(window.AshareWorkspace.withWorkspaceUrl("/api/simulation/strategies"));
  state.strategies = payload.strategies;
  els.strategySelect.innerHTML = state.strategies
    .map((strategy) => `<option value="${strategy.config_path}">${strategy.name}</option>`)
    .join("");
  if (selectedConfig && state.strategies.some((item) => item.config_path === selectedConfig)) {
    els.strategySelect.value = selectedConfig;
  }
  applyPresetDefaults();
}

async function loadRuns(selectFirst = true) {
  const payload = await fetchJson(window.AshareWorkspace.withWorkspaceUrl("/api/simulation/plans"));
  state.runs = payload.plans;
  renderRuns();
  if (selectFirst && state.runs.length) {
    await loadRun(state.runs[0].id);
  }
}

async function loadHistory() {
  const run = state.activeRun;
  const runId = activeHistoryRunId();
  if (!run || !runId) {
    state.activeHistory = null;
    state.activeHistoryTrades = [];
    els.historySummary.innerHTML = "";
    els.historyCount.textContent = "暂无";
    els.historyBody.innerHTML = `<tr><td colspan="8" class="muted">当前计划尚未执行，暂无执行历史。</td></tr>`;
    return;
  }
  try {
    const payload = await fetchJson(
      window.AshareWorkspace.withWorkspaceUrl(
        `/api/simulation/history?strategy_id=${encodeURIComponent(run.strategy_id || "")}&run_id=${encodeURIComponent(runId)}`,
      ),
    );
    state.activeHistory = payload;
    state.activeHistoryTrades = payload.trades || [];
    renderHistorySummary(payload);
    renderHistoryTrades();
  } catch (error) {
    state.activeHistory = null;
    state.activeHistoryTrades = [];
    els.historySummary.innerHTML = "";
    els.historyCount.textContent = "暂无";
    els.historyBody.innerHTML = `<tr><td colspan="8" class="muted">暂时还没有可展示的执行成交: ${error.message}</td></tr>`;
  }
}

async function loadLineage() {
  const run = state.activeRun;
  const runId = activeHistoryRunId();
  if (!run || !runId) {
    state.activeLineage = null;
    els.lineageCount.textContent = "暂无";
    els.lineageBody.innerHTML = `<tr><td colspan="8" class="muted">当前计划尚未执行，暂无状态演化记录。</td></tr>`;
    return;
  }
  try {
    const payload = await fetchJson(
      window.AshareWorkspace.withWorkspaceUrl(
        `/api/simulation/lineage?strategy_id=${encodeURIComponent(run.strategy_id || "")}&run_id=${encodeURIComponent(runId)}`,
      ),
    );
    state.activeLineage = payload;
    renderLineage();
  } catch (error) {
    state.activeLineage = null;
    els.lineageCount.textContent = "暂无";
    els.lineageBody.innerHTML = `<tr><td colspan="8" class="muted">暂时还没有可展示的状态演化记录: ${error.message}</td></tr>`;
  }
}

async function loadRun(runId) {
  const payload = await fetchJson(window.AshareWorkspace.withWorkspaceUrl(`/api/simulation/plans/${encodeURIComponent(runId)}`));
  if (payload.config_path && els.strategySelect.value !== payload.config_path) {
    els.strategySelect.value = payload.config_path;
    applyPresetDefaults();
  }
  state.activeRun = payload;
  state.activeHistory = null;
  state.activeLineage = null;
  renderRuns();
  renderSummary(payload, `${payload.name} · 账户状态`);
  renderHoldings(payload.strategy_state);
  renderActions(payload.strategy_state);
  renderRiskAndNextState(payload.strategy_state);
  await loadHistory();
  await loadLineage();
  setJobButtonsDisabled(false);
}

async function submitSimulation(event) {
  event.preventDefault();
  setJobButtonsDisabled(true);
  els.jobStatus.textContent = "已提交，正在创建新的模拟账户。";
  const body = {
    config_path: els.strategySelect.value,
    signal_date: els.tradeDate.value,
    initial_cash: Number(els.initialCash.value),
    label: els.label.value,
  };
  if (!state.readiness?.is_ready) {
    els.jobStatus.textContent = `现在还不能生成计划：${state.readiness?.message || "数据未就绪"}`;
    setJobButtonsDisabled(false);
    return;
  }
  try {
    const payload = await fetchJson("/api/simulation/plans", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(window.AshareWorkspace.withWorkspaceBody(body)),
    });
    state.currentJobId = payload.job.id;
    pollJob();
  } catch (error) {
    els.jobStatus.textContent = `提交失败：${error.message}`;
    setJobButtonsDisabled(false);
  }
}

async function submitNextPlan() {
  if (!state.activeRun?.id) {
    els.jobStatus.textContent = "请先选择一个模拟账户的盘前计划。";
    return;
  }
  if (!state.activeRun.next_plan_ready) {
    els.jobStatus.textContent = state.activeRun.next_plan_message || "当前账户还不能继续生成下一交易日盘前计划。";
    return;
  }
  setPrimaryActionLoading(true);
  setJobButtonsDisabled(true);
  els.jobStatus.textContent = state.activeRun.next_plan_message || "正在基于当前账户执行结果生成下一交易日盘前计划。";
  try {
    const payload = await fetchJson(`/api/simulation/plans/${encodeURIComponent(state.activeRun.id)}/next`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(window.AshareWorkspace.withWorkspaceBody({})),
    });
    state.currentJobId = payload.job.id;
    pollJob();
  } catch (error) {
    els.jobStatus.textContent = `提交失败：${error.message}`;
    setPrimaryActionLoading(false);
    setJobButtonsDisabled(false);
  }
}

async function submitRollForward() {
  if (!state.activeRun?.id) {
    els.jobStatus.textContent = "请先选择一份盘前计划。";
    return;
  }
  if (!state.activeRun.execution_ready) {
    els.jobStatus.textContent = accountStatusMessage(state.activeRun) || "执行日行情尚未落地，当前无法模拟下单。";
    return;
  }
  if (state.activeRun.executed_run_id) {
    els.jobStatus.textContent = `这份计划已经完成模拟下单，对应结果 run: ${state.activeRun.executed_run_id}`;
    return;
  }
  setPrimaryActionLoading(true);
  setJobButtonsDisabled(true);
  const executionDate = state.activeRun?.strategy_state?.summary?.execution_date || "-";
  els.jobStatus.textContent = `正在根据这份盘前计划模拟下单，执行日 ${executionDate}。`;
  try {
    const payload = await fetchJson(`/api/simulation/plans/${encodeURIComponent(state.activeRun.id)}/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(window.AshareWorkspace.withWorkspaceBody({
        label: state.activeRun.name || "",
      })),
    });
    state.currentJobId = payload.job.id;
    pollJob();
  } catch (error) {
    els.jobStatus.textContent = `提交失败：${error.message}`;
    setPrimaryActionLoading(false);
    setJobButtonsDisabled(false);
  }
}

async function handlePrimaryAction() {
  const actionState = primaryActionState();
  if (!actionState.action) {
    els.jobStatus.textContent = actionState.reason;
    return;
  }
  if (actionState.action === "execute") {
    await submitRollForward();
    return;
  }
  await submitNextPlan();
}

async function pollJob() {
  if (!state.currentJobId) return;
  try {
    const payload = await fetchJson(
      window.AshareWorkspace.withWorkspaceUrl(`/api/simulation/jobs/${encodeURIComponent(state.currentJobId)}`),
    );
    const { job, run, plan } = payload;
    if (job.status === "queued") {
      els.jobStatus.textContent = "任务排队中，请稍候。";
      window.setTimeout(pollJob, 1000);
      return;
    }
    if (job.status === "running") {
      if (job.type === "simulation_execute") {
        els.jobStatus.textContent = `正在按 ${job.signal_date || "-"} 的收盘信号，模拟 ${job.trade_date || "-"} 的下单结果。`;
      } else {
        els.jobStatus.textContent = `正在根据 ${job.signal_date || "-"} 的收盘信号，生成 ${job.trade_date || "-"} 的盘前计划。`;
      }
      window.setTimeout(pollJob, 1500);
      return;
    }
    if (job.status === "failed") {
      els.jobStatus.textContent = `处理失败：${job.error}`;
      state.currentJobId = null;
      setPrimaryActionLoading(false);
      setJobButtonsDisabled(false);
      return;
    }
    if (job.type === "simulation_execute") {
      els.jobStatus.textContent = `模拟下单已完成。账户 ${job.account_id || "-"} 已更新到执行节点 ${job.id || "-"}`;
    } else {
      els.jobStatus.textContent = `盘前计划已生成。账户 ${job.account_id || "-"} 当前计划节点为 ${job.id || "-"}`;
    }
    state.currentJobId = null;
    setPrimaryActionLoading(false);
    setJobButtonsDisabled(false);
    await loadRuns(false);
    if (plan) {
      state.activeRun = plan;
      renderRuns();
      renderSummary(plan, `${plan.name} · 账户状态`);
      renderHoldings(plan.strategy_state);
      renderActions(plan.strategy_state);
      renderRiskAndNextState(plan.strategy_state);
      await loadHistory();
      await loadLineage();
      setJobButtonsDisabled(false);
      return;
    }
    if (run) {
      if (job.plan_id) {
        await loadRun(job.plan_id);
        return;
      }
      state.activeRun = run;
      renderRuns();
      renderSummary(run, `${run.name} · 账户状态`);
      renderHoldings(run.strategy_state);
      renderActions(run.strategy_state);
      renderRiskAndNextState(run.strategy_state);
      await loadHistory();
      await loadLineage();
      setJobButtonsDisabled(false);
    }
  } catch (error) {
    els.jobStatus.textContent = `状态查询失败：${error.message}`;
    state.currentJobId = null;
    setPrimaryActionLoading(false);
    setJobButtonsDisabled(false);
  }
}

function bindEvents() {
  els.strategySelect.addEventListener("change", async () => {
    applyPresetDefaults();
    await refreshReadiness();
    await loadRuns(true);
    await loadHistory();
    await loadLineage();
  });
  els.tradeDate.addEventListener("change", refreshReadiness);
  els.form.addEventListener("submit", submitSimulation);
  els.primaryActionButton.addEventListener("click", handlePrimaryAction);
  els.historyFilter.addEventListener("input", renderHistoryTrades);
  els.lineageFilter.addEventListener("input", renderLineage);
  els.tabAccount.addEventListener("click", () => setActiveTab("account"));
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
  window.addEventListener("workspacechange", async () => {
    state.currentJobId = null;
    state.activeRun = null;
    state.activeHistory = null;
    state.activeHistoryTrades = [];
    state.activeLineage = null;
    await reloadStrategiesPreservingSelection();
    await refreshReadiness();
    await loadRuns(true);
    await loadHistory();
    await loadLineage();
  });
}

async function init() {
  window.AshareWorkspace.initWorkspaceControls();
  bindEvents();
  els.tradeDate.value = formatDateInput();
  await loadStrategies();
  await loadRuns(true);
  await loadHistory();
  await loadLineage();
  setJobButtonsDisabled(false);
}

init().catch((error) => {
  els.jobStatus.textContent = `页面初始化失败：${error.message}`;
});
