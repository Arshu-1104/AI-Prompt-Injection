type PredictRequest = { type: "predict"; payload: { text: string } };
type HealthRequest = { type: "health_check"; payload: { endpoint: string } };
type StatusRequest = { type: "status" };
type MessageRequest = PredictRequest | HealthRequest | StatusRequest;
type MessageResponse =
  | { status: "unconfigured" }
  | { status: "ok"; label?: string; confidence?: number; risk_score?: number; attack_patterns?: string[] }
  | { status: "timeout" }
  | { status: "error"; message: string };

import { DEFAULT_SLOW_TIMEOUT_SECONDS, isActive, loadSettings, migrateApiKey } from "./storage";

const MODELS = ["classical", "bert", "guard"];
const CLASSICAL_TIMEOUT_MS = 3000;

// Move an apiKey saved by versions before 1.1.0 out of synced storage (see storage.ts).
chrome.runtime.onInstalled.addListener(() => { void migrateApiKey(); });
chrome.runtime.onStartup.addListener(() => { void migrateApiKey(); });

function withTimeout(ms: number): { signal: AbortSignal; cancel: () => void } {
  const controller = new AbortController();
  const timer = self.setTimeout(() => controller.abort(), ms);
  return { signal: controller.signal, cancel: () => self.clearTimeout(timer) };
}

async function handlePredict(text: string): Promise<MessageResponse> {
  const config = await loadSettings();
  const model = MODELS.includes(config.model) ? config.model : "classical";
  if (!isActive(config)) return { status: "unconfigured" };
  // bert / guard are transformer models and much slower on a CPU server.
  const timeoutMs = model === "classical" ? CLASSICAL_TIMEOUT_MS : (config.slowTimeoutSeconds || DEFAULT_SLOW_TIMEOUT_SECONDS) * 1000;
  const timeout = withTimeout(timeoutMs);
  try {
    const response = await fetch(`${config.endpoint.replace(/\/$/, "")}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${config.apiKey}` },
      body: JSON.stringify({ text, model }),
      signal: timeout.signal,
    });
    timeout.cancel();
    if (!response.ok) {
      // 503 carries the reason, e.g. the selected model is not loaded on the backend.
      const body = await response.json().catch(() => null) as { detail?: unknown } | null;
      const detail = typeof body?.detail === "string" ? `: ${body.detail}` : "";
      return { status: "error", message: `HTTP ${response.status}${detail}` };
    }
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
  const timeout = withTimeout(15000);
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

chrome.runtime.onMessage.addListener((request: MessageRequest, _sender, sendResponse: (response: MessageResponse | { active: boolean }) => void) => {
  if (request.type === "status") {
    // Lets the content script decide whether to read the prompt box at all, without
    // ever handing it the API key.
    void loadSettings().then((settings) => sendResponse({ active: isActive(settings) }));
    return true;
  }
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
