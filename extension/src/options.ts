import { DEFAULT_SLOW_TIMEOUT_SECONDS, loadSettings, saveSettings } from "./storage";

type HealthResponse = { status: "ok" | "error" | "timeout"; message?: string };

const endpoint = document.getElementById("endpoint") as HTMLInputElement;
const apiKey = document.getElementById("apiKey") as HTMLInputElement;
const enabled = document.getElementById("enabled") as HTMLInputElement;
const model = document.getElementById("model") as HTMLSelectElement;
const slowNote = document.getElementById("slowNote") as HTMLDivElement;
const slowTimeout = document.getElementById("slowTimeout") as HTMLInputElement;

function updateSlowNote(): void {
  slowNote.hidden = model.value === "classical";
}
model.addEventListener("change", updateSlowNote);
const testBtn = document.getElementById("testBtn") as HTMLButtonElement;
const saveBtn = document.getElementById("saveBtn") as HTMLButtonElement;
const testStatus = document.getElementById("testStatus") as HTMLSpanElement;
const msg = document.getElementById("msg") as HTMLDivElement;

void loadSettings().then((data) => {
  endpoint.value = data.endpoint;
  apiKey.value = data.apiKey;
  enabled.checked = data.enabled;
  model.value = data.model;
  slowTimeout.value = String(data.slowTimeoutSeconds || DEFAULT_SLOW_TIMEOUT_SECONDS);
  updateSlowNote();
});

testBtn.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "health_check", payload: { endpoint: endpoint.value } }, (response: HealthResponse) => {
    testStatus.textContent = response.status === "ok" ? "Connected OK" : `Failed: ${response.message ?? response.status}`;
  });
});

saveBtn.addEventListener("click", () => {
  try {
    const url = new URL(endpoint.value);
    void chrome.permissions.request({ origins: [`${url.origin}/*`] }, (granted) => {
      if (!granted) { msg.textContent = "Permission denied"; return; }
      const seconds = Math.min(60, Math.max(1, Number(slowTimeout.value) || DEFAULT_SLOW_TIMEOUT_SECONDS));
      void saveSettings({ endpoint: endpoint.value, apiKey: apiKey.value, enabled: enabled.checked, model: model.value, slowTimeoutSeconds: seconds }).then(() => { msg.textContent = "Saved"; });
    });
  } catch {
    msg.textContent = "Invalid URL";
  }
});

export {};
