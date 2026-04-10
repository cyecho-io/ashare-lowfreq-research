const WORKSPACE_STORAGE_KEY = "ashare-backtest-workspace";
const DEFAULT_WORKSPACE = "native";
const SUPPORTED_WORKSPACES = new Set(["native", "qlib"]);

function normalizeWorkspace(workspace) {
  const normalized = String(workspace || "").trim().toLowerCase() || DEFAULT_WORKSPACE;
  return SUPPORTED_WORKSPACES.has(normalized) ? normalized : DEFAULT_WORKSPACE;
}

function getWorkspace() {
  return normalizeWorkspace(window.localStorage.getItem(WORKSPACE_STORAGE_KEY));
}

function applyWorkspaceTheme(workspace = getWorkspace()) {
  const normalized = normalizeWorkspace(workspace);
  document.body.dataset.workspace = normalized;
  document.querySelectorAll("[data-workspace-label]").forEach((node) => {
    node.textContent = normalized;
  });
  document.querySelectorAll("[data-workspace-switch]").forEach((node) => {
    node.value = normalized;
  });
  return normalized;
}

function setWorkspace(workspace) {
  const normalized = normalizeWorkspace(workspace);
  window.localStorage.setItem(WORKSPACE_STORAGE_KEY, normalized);
  applyWorkspaceTheme(normalized);
  window.dispatchEvent(new CustomEvent("workspacechange", { detail: { workspace: normalized } }));
  return normalized;
}

function withWorkspaceUrl(url) {
  const nextUrl = new URL(url, window.location.origin);
  nextUrl.searchParams.set("workspace", getWorkspace());
  if (nextUrl.origin === window.location.origin) {
    return `${nextUrl.pathname}${nextUrl.search}${nextUrl.hash}`;
  }
  return nextUrl.toString();
}

function withWorkspaceBody(body = {}) {
  return {
    ...body,
    workspace: getWorkspace(),
  };
}

function initWorkspaceControls() {
  applyWorkspaceTheme();
  document.querySelectorAll("[data-workspace-switch]").forEach((node) => {
    node.addEventListener("change", (event) => {
      setWorkspace(event.target.value);
    });
  });
  window.addEventListener("storage", (event) => {
    if (event.key === WORKSPACE_STORAGE_KEY) {
      applyWorkspaceTheme(event.newValue || DEFAULT_WORKSPACE);
      window.dispatchEvent(new CustomEvent("workspacechange", { detail: { workspace: getWorkspace() } }));
    }
  });
}

window.AshareWorkspace = {
  DEFAULT_WORKSPACE,
  getWorkspace,
  setWorkspace,
  withWorkspaceUrl,
  withWorkspaceBody,
  applyWorkspaceTheme,
  initWorkspaceControls,
};
