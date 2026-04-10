const els = {
  meta: document.getElementById("dashboard-meta"),
  summaryCards: document.getElementById("dashboard-summary-cards"),
  calendarRange: document.getElementById("dashboard-calendar-range"),
  calendarCounts: document.getElementById("dashboard-calendar-counts"),
  calendarGrid: document.getElementById("dashboard-calendar-grid"),
  detailList: document.getElementById("dashboard-detail-list"),
};

function formatNumber(value, digits = 0) {
  return new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(Number(value || 0));
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "request_failed");
  }
  return payload;
}

function renderSummary(payload) {
  const cards = [
    ["最新开市日", payload.calendar.latest_open_date || "-"],
    ["策略数量", formatNumber(payload.strategies.count)],
    ["SQLite 股票数", formatNumber(payload.sqlite.equity_symbol_count)],
    ["股票时间范围", `${payload.sqlite.date_min || "-"} 至 ${payload.sqlite.date_max || "-"}`],
  ];
  els.summaryCards.innerHTML = cards
    .map(([label, value]) => `<div class="summary-card"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderCalendar(payload) {
  const cells = payload.calendar.cells || [];
  if (!cells.length) {
    els.calendarRange.textContent = "未找到交易日历数据。";
    els.calendarCounts.textContent = "";
    els.calendarGrid.innerHTML = `<div class="muted">当前没有可展示的交易日历。</div>`;
    return;
  }
  els.calendarRange.textContent = `${cells[0].date} 至 ${cells[cells.length - 1].date}`;
  els.calendarCounts.textContent = `开市 ${formatNumber(payload.calendar.open_days)} 天 · 休市 ${formatNumber(payload.calendar.closed_days)} 天`;
  els.calendarGrid.innerHTML = cells
    .map(
      (cell) => `
        <div
          class="calendar-cell ${cell.is_open ? "is-open" : "is-closed"}"
          title="${cell.date} · ${cell.is_open ? "开市" : "休市"}"
          aria-label="${cell.date} ${cell.is_open ? "开市" : "休市"}"
        ></div>
      `,
    )
    .join("");
}

function renderDetails(payload) {
  const details = [
    ["SQLite 源库", payload.sqlite.path || "-"],
    ["导入时间", payload.catalog.imported_at || "-"],
    ["Bars 数据区间", `${payload.catalog.bars_daily.min_date || "-"} 至 ${payload.catalog.bars_daily.max_date || "-"}`],
    ["Calendar 自然日区间", `${payload.catalog.calendar_ashare.min_date || "-"} 至 ${payload.catalog.calendar_ashare.max_date || "-"}`],
    ["Instrument 区间", `${payload.catalog.instruments_ashare.min_date || "-"} 至 ${payload.catalog.instruments_ashare.max_date || "-"}`],
    ["Instrument 数量", formatNumber(payload.sqlite.instrument_count)],
  ];
  els.detailList.innerHTML = details
    .map(
      ([label, value]) => `
        <div class="detail-card">
          <strong>${label}</strong>
          <span>${value}</span>
        </div>
      `,
    )
    .join("");
}

async function init() {
  window.AshareWorkspace.initWorkspaceControls();
  try {
    const payload = await fetchJson("/api/dashboard/summary");
    els.meta.textContent = `当前读取源库 ${payload.sqlite.path || "-"}，最近导入时间 ${payload.catalog.imported_at || "-"}`;
    renderSummary(payload);
    renderCalendar(payload);
    renderDetails(payload);
  } catch (error) {
    els.meta.textContent = `读取失败: ${error.message}`;
    els.summaryCards.innerHTML = "";
    els.calendarRange.textContent = "";
    els.calendarCounts.textContent = "";
    els.calendarGrid.innerHTML = `<div class="muted">无法加载交易日历。</div>`;
    els.detailList.innerHTML = `<div class="detail-card"><strong>错误</strong><span>${error.message}</span></div>`;
  }
}

init();
