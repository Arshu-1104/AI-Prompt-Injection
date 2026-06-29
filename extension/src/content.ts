type PredictResponse = { status: "ok" | "unconfigured" | "timeout" | "error"; label?: string; confidence?: number; risk_score?: number; attack_patterns?: string[]; message?: string };

type BadgeParts = { root: HTMLDivElement; badge: HTMLDivElement; tooltip: HTMLDivElement };

// Site selectors reviewed on 2026-06-20. These services change markup often; generic textarea/contenteditable detection remains the primary fallback.
const SITE_SELECTORS = [
  "#prompt-textarea",
  "textarea[data-testid='textbox']",
  "div[contenteditable='true'][role='textbox']",
  "rich-textarea div[contenteditable='true']",
];

const decorated = "pgBadge";
const debounceTimers = new WeakMap<Element, number>();

function getText(element: Element): string {
  if (element instanceof HTMLTextAreaElement || element instanceof HTMLInputElement) return element.value;
  return element.textContent ?? "";
}

function findInputs(): Element[] {
  const found = new Set<Element>();
  document.querySelectorAll("textarea").forEach((el) => { if (el instanceof HTMLElement && el.offsetHeight > 100) found.add(el); });
  document.querySelectorAll('[contenteditable="true"][role="textbox"]').forEach((el) => { if (el instanceof HTMLElement && el.offsetHeight > 100) found.add(el); });
  SITE_SELECTORS.forEach((selector) => document.querySelectorAll(selector).forEach((el) => found.add(el)));
  return [...found].filter((el) => el instanceof HTMLElement && !el.dataset[decorated]);
}

function positionBadge(input: HTMLElement, root: HTMLDivElement): void {
  root.style.position = "fixed";
  root.style.zIndex = "2147483647";
  document.body.appendChild(root);
  updateBadgePosition(input, root);
}

function updateBadgePosition(input: HTMLElement, root: HTMLDivElement): void {
  const rect = input.getBoundingClientRect();
  root.style.left = `${Math.max(0, rect.right - 90)}px`;
  root.style.top = `${rect.top + 8}px`;
}

function createBadge(input: HTMLElement): BadgeParts {
  const root = document.createElement("div");
  const shadow = root.attachShadow({ mode: "open" });
  const style = document.createElement("style");
  style.textContent = `.badge{height:28px;min-width:80px;border-radius:14px;font-size:11px;font-weight:500;display:flex;align-items:center;justify-content:center;cursor:default;transition:background .2s}.safe{background:#E8F4FD;color:#0D3D63;border:1px solid #1B6CA8}.suspicious{background:#FEF3C7;color:#6B3A07;border:1px solid #B45309}.malicious{background:#FEE2E2;color:#6B0F0F;border:1px solid #B91C1C}.loading{background:#F1F5F9;color:#64748B;border:1px solid #CBD5E1}.hidden{display:none}.tip{display:none;position:absolute;top:34px;right:0;max-width:260px;border:1px solid #CBD5E1;background:white;color:#0F172A;border-radius:6px;padding:8px;font-size:11px;box-shadow:0 8px 24px rgba(15,23,42,.16)}.malicious+.tip{display:block}`;
  const badge = document.createElement("div");
  badge.className = "badge hidden";
  const tooltip = document.createElement("div");
  tooltip.className = "tip";
  shadow.append(style, badge, tooltip);
  positionBadge(input, root);
  window.addEventListener("scroll", () => updateBadgePosition(input, root), true);
  window.addEventListener("resize", () => updateBadgePosition(input, root));
  return { root, badge, tooltip };
}

function updateBadge(parts: BadgeParts, response: PredictResponse): void {
  if (response.status === "unconfigured") {
    parts.badge.className = "badge hidden";
    return;
  }
  if (response.status !== "ok") {
    parts.badge.className = "badge loading";
    parts.badge.textContent = response.status === "timeout" ? "Timeout" : "Offline";
    return;
  }
  const label = response.label ?? "SAFE";
  const state = label.toLowerCase();
  parts.badge.className = `badge ${state}`;
  parts.badge.textContent = label === "SAFE" ? "\u2713 Safe" : label === "SUSPICIOUS" ? "\u26A0 Review" : "\u2717 Danger";
  parts.tooltip.textContent = (response.attack_patterns ?? []).join(", ") || "No patterns listed";
}

function decorate(input: Element): void {
  if (!(input instanceof HTMLElement)) return;
  input.dataset[decorated] = "1";
  const parts = createBadge(input);
  const listener = (): void => {
    const existing = debounceTimers.get(input);
    if (existing) window.clearTimeout(existing);
    debounceTimers.set(input, window.setTimeout(() => {
      const text = getText(input).trim();
      if (!text) { parts.badge.className = "badge hidden"; return; }
      parts.badge.className = "badge loading";
      parts.badge.textContent = "Checking";
      chrome.runtime.sendMessage({ type: "predict", payload: { text } }, (response: PredictResponse) => updateBadge(parts, response));
    }, 500));
  };
  input.addEventListener("input", listener);
  input.addEventListener("keyup", listener);
}

function scan(): void { findInputs().forEach(decorate); }

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", scan);
} else {
  scan();
}

let _scanning = false;
const observer = new MutationObserver(() => {
  if (_scanning) return;
  _scanning = true;
  scan();
  _scanning = false;
});
observer.observe(document.body, { subtree: true, childList: true });
