type PredictRequest = { type: "predict"; payload: { text: string } };
type HealthRequest = { type: "health_check"; payload: { endpoint: string } };
type MessageRequest = PredictRequest | HealthRequest;
type MessageResponse =
  | { status: "unconfigured" }
  | { status: "ok"; label?: string; confidence?: number; risk_score?: number; attack_patterns?: string[] }
  | { status: "timeout" }
  | { status: "error"; message: string };

type StoredConfig = { endpoint?: string; apiKey?: string; enabled?: boolean };

function withTimeout(ms: number): { signal: AbortSignal; cancel: () => void } {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), ms);
  return { signal: controller.signal, cancel: () => window.clearTimeout(timer) };
}

async function handlePredict(text: string): Promise<MessageResponse> {
  const config = await chrome.storage.sync.get({ enabled: true, endpoint: "", apiKey: "" }) as StoredConfig;
  if (config.enabled === false || !config.endpoint || !config.apiKey) return { status: "unconfigured" };
  const timeout = withTimeout(3000);
  try {
    const response = await fetch(`${config.endpoint.replace(/\/$/, "")}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${config.apiKey}` },
      body: JSON.stringify({ text, model: "classical" }),
      signal: timeout.signal,
    });
    timeout.cancel();
    if (!response.ok) return { status: "error", message: `HTTP ${response.status}` };
    const data = await response.json() as { label: string; confidence: number; risk_score: number; attack_patterns: string[] };
    await chrome.storage.local.set({ lastResult: data });
    return { status: "ok", label: data.label, confidence: data.confidence, risk_score: data.risk_score, attack_patterns: data.attack_patterns };
  } catch (error) {
    timeout.cancel();
    if (error instanceof DOMException && error.name === "AbortError") return { status: "timeout" };
    return { status: "error", message: error instanceof Error ? error.message : "Unknown error" };
  }
}

async function handleHealth(endpoint: string): Promise<MessageResponse> {
  const timeout = withTimeout(5000);
  try {
    const response = await fetch(`${endpoint.replace(/\/$/, "")}/health`, { signal: timeout.signal });
    timeout.cancel();
    return response.ok ? { status: "ok" } : { status: "error", message: `HTTP ${response.status}` };
  } catch (error) {
    timeout.cancel();
    if (error instanceof DOMException && error.name === "AbortError") return { status: "timeout" };
    return { status: "error", message: error instanceof Error ? error.message : "Unknown error" };
  }
}

chrome.runtime.onMessage.addListener((request: MessageRequest, _sender, sendResponse: (response: MessageResponse) => void) => {
  if (request.type === "predict") {
    void handlePredict(request.payload.text).then(sendResponse);
    return true;
  }
  if (request.type === "health_check") {
    void handleHealth(request.payload.endpoint).then(sendResponse);
    return true;
  }
  return false;
});
