// Where settings live:
// - apiKey: chrome.storage.local only (this device; never synced).
// - endpoint, enabled, model, slowTimeoutSeconds: chrome.storage.sync.
// - lastResult: chrome.storage.local.

export const DEFAULT_SLOW_TIMEOUT_SECONDS = 10;

export type Settings = { endpoint: string; apiKey: string; enabled: boolean; model: string; slowTimeoutSeconds: number };

const SYNC_DEFAULTS = { endpoint: "", enabled: true, model: "classical", slowTimeoutSeconds: DEFAULT_SLOW_TIMEOUT_SECONDS };

/** One-time move of apiKey from chrome.storage.sync (versions before 1.1.0) to
 * chrome.storage.local. Idempotent: an existing local key wins, and the synced
 * copy is always removed. */
export async function migrateApiKey(): Promise<void> {
  const synced = await chrome.storage.sync.get("apiKey");
  if (!("apiKey" in synced)) return;
  const local = await chrome.storage.local.get("apiKey");
  if (typeof synced.apiKey === "string" && synced.apiKey && !local.apiKey) {
    await chrome.storage.local.set({ apiKey: synced.apiKey });
  }
  await chrome.storage.sync.remove("apiKey");
}

export async function loadSettings(): Promise<Settings> {
  await migrateApiKey();
  const synced = await chrome.storage.sync.get(SYNC_DEFAULTS);
  const local = await chrome.storage.local.get({ apiKey: "" });
  return {
    endpoint: String(synced.endpoint ?? ""),
    apiKey: String(local.apiKey ?? ""),
    enabled: synced.enabled !== false,
    model: String(synced.model ?? "classical"),
    slowTimeoutSeconds: Number(synced.slowTimeoutSeconds ?? DEFAULT_SLOW_TIMEOUT_SECONDS),
  };
}

export async function saveSettings(settings: Settings): Promise<void> {
  const { apiKey, ...synced } = settings;
  await chrome.storage.sync.set(synced);
  await chrome.storage.local.set({ apiKey });
  await chrome.storage.sync.remove("apiKey");
}

/** True only when checking is switched on and an endpoint and a key are set. */
export function isActive(settings: Settings): boolean {
  return settings.enabled && Boolean(settings.endpoint) && Boolean(settings.apiKey);
}
