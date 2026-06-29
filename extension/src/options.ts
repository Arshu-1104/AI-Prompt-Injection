type HealthResponse = { status: "ok" | "error" | "timeout"; message?: string };
type Stored = { endpoint?: string; apiKey?: string; enabled?: boolean };

const endpoint = document.getElementById("endpoint") as HTMLInputElement;
const apiKey = document.getElementById("apiKey") as HTMLInputElement;
const enabled = document.getElementById("enabled") as HTMLInputElement;
const testBtn = document.getElementById("testBtn") as HTMLButtonElement;
const saveBtn = document.getElementById("saveBtn") as HTMLButtonElement;
const testStatus = document.getElementById("testStatus") as HTMLSpanElement;
const msg = document.getElementById("msg") as HTMLDivElement;

void chrome.storage.sync.get({ endpoint: "", apiKey: "", enabled: true }).then((data: Stored) => {
  endpoint.value = data.endpoint ?? "";
  apiKey.value = data.apiKey ?? "";
  enabled.checked = data.enabled !== false;
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
      void chrome.storage.sync.set({ endpoint: endpoint.value, apiKey: apiKey.value, enabled: enabled.checked }).then(() => { msg.textContent = "Saved"; });
    });
  } catch {
    msg.textContent = "Invalid URL";
  }
});
