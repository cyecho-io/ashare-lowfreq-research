const researchState = {
  configs: [],
  originalConfigText: "",
  runs: [],
  activeRun: null,
  currentJobId: null,
};

const researchEls = {
  configSelect: document.getElementById("research-config-select"),
  configFilename: document.getElementById("research-config-filename"),
  configEditor: document.getElementById("research-config-editor"),
  resetButton: document.getElementById("research-reset-button"),
  form: document.getElementById("research-form"),
  runButton: document.getElementById("research-run-button"),
  jobStatus: document.getElementById("research-job-status"),
  logPanel: document.getElementById("research-log-panel"),
  logStatus: document.getElementById("research-log-status"),
  logHint: document.getElementById("research-log-hint"),
  logStream: document.getElementById("research-log-stream"),
  runsList: document.getElementById("research-runs-list"),
  resultTitle: document.getElementById("research-result-title"),
  summaryCards: document.getElementById("research-summary-cards"),
  detailList: document.getElementById("research-detail-list"),
  openBacktest: document.getElementById("research-open-backtest"),
  configHeader: document.getElementById("research-config-header"),
  configToggle: document.getElementById("research-config-toggle"),
  configBody: document.getElementById("research-config-body"),
  configPanel: document.getElementById("research-config-panel"),
};

async function fetchResearchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "request_failed");
  }
  return payload;
}

function formatResearchNumber(value, digits = 4) {
  if (value === "n/a" || value == null || value === "") return "-";
  return new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(Number(value));
}

function activeResearchConfig() {
  return researchState.configs.find((item) => item.config_path === researchEls.configSelect.value) || null;
}

function renderLogOutput(content) {
  researchEls.logStream.textContent = content || "暂无日志输出。";
  researchEls.logStream.scrollTop = researchEls.logStream.scrollHeight;
}

function renderActiveConfigMeta(configPath) {
  const workspace = window.AshareWorkspace.getWorkspace();
  researchEls.configFilename.textContent = `${configPath || "configs/**/*.toml"} · ${workspace}`;
}

function renderResearchRuns() {
  researchEls.runsList.innerHTML = "";
  const visibleRuns = researchState.runs;
  if (!visibleRuns.length) {
    researchEls.runsList.innerHTML = `<div class="muted">还没有研究记录。</div>`;
    return;
  }
  for (const run of visibleRuns) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `run-card${researchState.activeRun?.id === run.id ? " active" : ""}`;
    card.innerHTML = `
      <h3>${run.name}</h3>
      <div class="run-meta">
        <span>${run.updated_at}</span>
        <span>${run.backend || "-"} / ${run.model || "-"}</span>
      </div>
      <div class="run-meta">
        <span>IC ${formatResearchNumber(run.metrics?.mean_spearman_ic, 4)}</span>
        <span>Spread ${formatResearchNumber(run.layer_summary?.mean_top_bottom_spread, 4)}</span>
      </div>
    `;
    card.addEventListener("click", () => loadResearchRun(run.id));
    researchEls.runsList.appendChild(card);
  }
}

function renderResearchSummary(run) {
  researchState.activeRun = run;
  researchEls.resultTitle.textContent = run.name || "查看运行结果";
  const metrics = run.metrics || {};
  const layerSummary = run.layer_summary || {};
  const items = [
    ["研究后端", run.backend || "-"],
    ["模型", run.model || "-"],
    ["配置标识", run.config_id || "-"],
    ["窗口数", metrics.window_count ?? "-"],
    ["评估 IC", formatResearchNumber(metrics.mean_spearman_ic, 4)],
    ["验证 IC", formatResearchNumber(metrics.mean_validation_spearman_ic, 4)],
    ["MAE", formatResearchNumber(metrics.mean_mae, 4)],
    ["RMSE", formatResearchNumber(metrics.mean_rmse, 4)],
    ["分层 Spread", formatResearchNumber(layerSummary.mean_top_bottom_spread, 4)],
    ["Spread 正收益占比", formatResearchNumber((layerSummary.positive_spread_ratio || 0) * 100, 2) + "%"],
    ["分数区间", run.score_start_date && run.score_end_date ? `${run.score_start_date} 至 ${run.score_end_date}` : "-"],
  ];
  researchEls.summaryCards.innerHTML = items
    .map(([label, value]) => `<div class="summary-card"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");

  const details = [
    ["原始配置", run.config_path || "-"],
    ["本次配置快照", run.config_snapshot_path || "-"],
    ["执行日志", run.logs_path || "-"],
    ["backend", run.backend || "-"],
    ["model", run.model || "-"],
    ["config_id", run.config_id || "-"],
    ["factor", run.factor_panel_path || "-"],
    ["scores", run.scores_path || "-"],
    ["metrics", run.metrics_path || "-"],
    ["创建时间", run.created_at || "-"],
  ];
  researchEls.detailList.innerHTML = details
    .map(([label, value]) => `<div><strong>${label}</strong><span>${value}</span></div>`)
    .join("");

  const backtestUrl = new URL("/backtest", window.location.origin);
  if (run.config_path) backtestUrl.searchParams.set("config_path", run.config_path);
  if (run.scores_path) backtestUrl.searchParams.set("scores_path", run.scores_path);
  backtestUrl.searchParams.set("workspace", window.AshareWorkspace.getWorkspace());
  researchEls.openBacktest.href = backtestUrl.toString();

  renderResearchRuns();
}

async function loadResearchConfigs() {
  const payload = await fetchResearchJson(window.AshareWorkspace.withWorkspaceUrl("/api/research/configs"));
  researchState.configs = payload.configs || [];
  researchEls.configSelect.innerHTML = researchState.configs
    .map((item) => `<option value="${item.config_path}">${item.name}</option>`)
    .join("");
  if (researchState.configs.length) {
    await loadResearchConfigText(researchState.configs[0].id);
  }
}

async function loadResearchConfigText(configId) {
  const payload = await fetchResearchJson(window.AshareWorkspace.withWorkspaceUrl(`/api/research/configs/${encodeURIComponent(configId)}`));
  researchEls.configSelect.value = payload.config_path;
  researchState.originalConfigText = payload.content || "";
  researchEls.configEditor.value = researchState.originalConfigText;
  renderActiveConfigMeta(payload.config_path);
}

async function loadResearchRuns(selectFirst = true) {
  const payload = await fetchResearchJson(window.AshareWorkspace.withWorkspaceUrl("/api/research/runs"));
  researchState.runs = payload.runs || [];
  renderResearchRuns();
  if (selectFirst && researchState.runs.length) {
    await loadResearchRun(researchState.runs[0].id);
  }
}

async function loadResearchRun(runId) {
  const payload = await fetchResearchJson(window.AshareWorkspace.withWorkspaceUrl(`/api/research/runs/${encodeURIComponent(runId)}`));
  renderResearchSummary(payload);
}

async function submitResearchJob(event) {
  event.preventDefault();
  researchEls.runButton.disabled = true;
  researchEls.jobStatus.textContent = "研究任务已提交，等待启动。";
  showLogPanel();
  researchEls.logStatus.textContent = "任务已提交，等待启动...";
  renderLogOutput("正在启动研究流程...\n");
  try {
    const payload = await fetchResearchJson("/api/research/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(window.AshareWorkspace.withWorkspaceBody({
        config_path: researchEls.configSelect.value,
        config_text: researchEls.configEditor.value,
        backend: window.AshareWorkspace.getWorkspace(),
      })),
    });
    researchState.currentJobId = payload.job.id;
    pollResearchJob();
  } catch (error) {
    researchEls.jobStatus.textContent = `提交失败: ${error.message}`;
    researchEls.logStatus.textContent = `提交失败: ${error.message}`;
    researchEls.runButton.disabled = false;
  }
}

async function pollResearchJob() {
  if (!researchState.currentJobId) return;
  try {
    const payload = await fetchResearchJson(
      window.AshareWorkspace.withWorkspaceUrl(`/api/research/jobs/${encodeURIComponent(researchState.currentJobId)}`),
    );
    const { job, run } = payload;
    renderLogOutput(job.logs);
    if (job.status === "queued") {
      researchEls.jobStatus.textContent = "任务排队中。";
      researchEls.logStatus.textContent = "任务排队中，等待执行...";
      window.setTimeout(pollResearchJob, 1000);
      return;
    }
    if (job.status === "running") {
      researchEls.jobStatus.textContent = `正在执行 ${job.backend || "native"} 研究流水线。`;
      researchEls.logStatus.textContent = `正在执行 ${job.backend || "native"} 研究流水线...`;
      window.setTimeout(pollResearchJob, 1500);
      return;
    }
    if (job.status === "failed") {
      researchEls.jobStatus.textContent = `执行失败: ${job.error}`;
      researchEls.logStatus.textContent = `执行失败: ${job.error}`;
      researchEls.runButton.disabled = false;
      return;
    }
    researchEls.jobStatus.textContent = `执行完成，结果目录: ${job.result_dir}`;
    researchEls.logStatus.textContent = "执行完成";
    researchEls.runButton.disabled = false;
    await loadResearchRuns(false);
    if (run) {
      renderResearchSummary(run);
    } else {
      await loadResearchRun(job.id);
    }
  } catch (error) {
    researchEls.jobStatus.textContent = `状态查询失败: ${error.message}`;
    researchEls.logStatus.textContent = `状态查询失败: ${error.message}`;
    researchEls.runButton.disabled = false;
  }
}

function toggleConfigPanel() {
  const isCollapsed = researchEls.configPanel.classList.toggle("collapsed");
  researchEls.configToggle.setAttribute("aria-expanded", !isCollapsed);
}

function showLogPanel() {
  researchEls.logPanel.style.display = "block";
  researchEls.logPanel.classList.add("visible");
}

function hideLogPanel() {
  researchEls.logPanel.style.display = "none";
  researchEls.logPanel.classList.remove("visible");
}

function bindResearchEvents() {
  researchEls.configSelect.addEventListener("change", async () => {
    const active = activeResearchConfig();
    if (active) {
      await loadResearchConfigText(active.id);
    }
  });
  researchEls.resetButton.addEventListener("click", () => {
    researchEls.configEditor.value = researchState.originalConfigText;
  });
  researchEls.form.addEventListener("submit", submitResearchJob);
  researchEls.configHeader.addEventListener("click", (e) => {
    if (e.target !== researchEls.configToggle && !e.target.closest('.collapse-toggle')) {
      toggleConfigPanel();
    }
  });
  researchEls.configToggle.addEventListener("click", toggleConfigPanel);
  window.addEventListener("workspacechange", async () => {
    researchState.currentJobId = null;
    researchState.activeRun = null;
    await loadResearchConfigs();
    await loadResearchRuns(true);
  });
}

async function initResearch() {
  window.AshareWorkspace.initWorkspaceControls();
  bindResearchEvents();
  await loadResearchConfigs();
  await loadResearchRuns(true);
  // Default collapse config panel
  toggleConfigPanel();
}

initResearch().catch((error) => {
  researchEls.jobStatus.textContent = `初始化失败: ${error.message}`;
});
