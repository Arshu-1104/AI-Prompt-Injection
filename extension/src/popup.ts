type Stored = { enabled?: boolean; endpoint?: string };
type LastResult = { label?: string; risk_score?: number };

const statusEl = document.getElementById("status") as HTMLDivElement;
const lastResult = document.getElementById("lastResult") as HTMLDivElement;
const toggle = document.getElementById("enableToggle") as HTMLInputElement;
const settingsLink = document.getElementById("settingsLink") as HTMLAnchorElement;

void chrome.storage.sync.get({ enabled: true, endpoint: "" }).then((data: Stored) => {
  toggle.checked = data.enabled !== false;
  statusEl.textContent = data.endpoint ? (toggle.checked ? "Active" : "Disabled") : "Configure in Settings";
});

void chrome.storage.local.get({ lastResult: null }).then((data: { lastResult?: LastResult | null }) => {
  if (data.lastResult?.label) lastResult.textContent = `${data.lastResult.label} - risk ${data.lastResult.risk_score?.toFixed(1) ?? "n/a"}`;
});

toggle.addEventListener("change", () => { void chrome.storage.sync.set({ enabled: toggle.checked }); });
settingsLink.addEventListener("click", (event) => { event.preventDefault(); chrome.runtime.openOptionsPage(); });
