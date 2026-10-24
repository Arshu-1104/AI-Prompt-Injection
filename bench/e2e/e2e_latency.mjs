// End-to-end latency of the PromptGuard extension: last keystroke -> final badge.
//
// For each model, the built extension (extension/dist) runs in Playwright's Chromium against
// the API on 127.0.0.1 (throwaway SQLite database, heavy models on so bert and guard load).
// Each trial clears a plain local test page's text box, types one test prompt character by
// character (so the 500 ms debounce in extension/src/content.ts resets as for a real user), and
// records, inside the page, the time of the last input event and the time the badge shows its
// final label. The API's own logged inference time (predictions.latency_ms) splits the total
// into debounce + inference + everything else (network, extension messaging, rendering).
//
// A separate scripted typing session counts how many /predict requests the debounce produces
// per minute, against the API's 60 requests/minute per-key limit.
//
// The extension copy used here also matches http://127.0.0.1/* (throwaway manifest in a temp
// folder); the committed manifest is not changed. Optionally, with PROMPTGUARD_HOSTED_URL and
// PROMPTGUARD_HOSTED_KEY in the environment, the same trials run against a hosted API: once
// after HOSTED_IDLE_MINUTES of idle time (cold) and then warm. Secrets stay in memory.
//
//   cd bench/e2e && npm ci && npx playwright install chromium && node e2e_latency.mjs
//   Output: artifacts/e2e_latency.json (tables: python -m src.e2e_report)
import { chromium } from "playwright";
import { spawn } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..", "..");
const PYTHON = process.env.PROMPTGUARD_PYTHON || "python";
const API_PORT = 8000;
const PAGE_PORT = 5510;
const API = `http://127.0.0.1:${API_PORT}`;
const MODELS = (process.env.E2E_MODELS || "classical,bert,guard").split(",");
const TRIALS = Number(process.env.E2E_TRIALS || 30);
const TYPING_SESSION_SECONDS = Number(process.env.E2E_TYPING_SECONDS || 120);
const HOSTED_IDLE_MINUTES = Number(process.env.HOSTED_IDLE_MINUTES || 20);
// Read from the extension source so the reported value can never drift from the code.
const DEBOUNCE_MS = (() => {
  const src = fs.readFileSync(path.join(REPO, "extension", "src", "content.ts"), "utf8");
  const m = src.match(/debounceTimers\.set\([\s\S]*?\},\s*(\d+)\)\);/);
  if (!m) throw new Error("could not find the debounce delay in extension/src/content.ts");
  return Number(m[1]);
})();
const SEED = 2026;
const OUT = path.join(REPO, "artifacts", "e2e_latency.json");

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pg-e2e-"));
const children = [];

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function waitFor(url, ms = 600000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    try { const r = await fetch(url); if (r.ok) return; } catch {}
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error(`timeout waiting for ${url}`);
}

function startApi(adminKey, db) {
  const env = { ...process.env, ADMIN_SECRET_KEY: adminKey, ENABLE_HEAVY_MODELS: "true", WEB_CONCURRENCY: "1",
                DATABASE_URL: `sqlite+aiosqlite:///${db.replace(/\\/g, "/")}`,
                BERT_MODEL_ID: process.env.BERT_MODEL_ID || path.join(REPO, "artifacts/paper_runs/bert/seed_42"),
                GUARD_MODEL_ID: process.env.GUARD_MODEL_ID || path.join(REPO, "artifacts/paper_runs/guard/seed_42"),
                CUDA_VISIBLE_DEVICES: "" }; // CPU inference, as in the paper's latency tables
  delete env.HF_TOKEN;
  const child = spawn(PYTHON, ["-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", String(API_PORT)],
                      { cwd: REPO, env, stdio: ["ignore", "pipe", "pipe"] });
  const log = fs.createWriteStream(path.join(tmp, "api.log"));
  child.stdout.pipe(log); child.stderr.pipe(log);
  children.push(child);
  return child;
}

function testPrompts() {
  // A fixed sample of short test-split rows, read through Python so the split checks apply.
  const script = "import json\nfrom src.preprocess import get_splits\n_,_,t,_=get_splits()\n" +
    "o=t[(t['derivation']=='original')&(t['raw_text'].str.len().between(20,200))]\n" +
    `print(json.dumps(o.sample(n=${TRIALS},random_state=${SEED})['raw_text'].tolist()))`;
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON, ["-X", "utf8", "-c", script], { cwd: REPO, env: process.env });
    let out = ""; let err = "";
    child.stdout.on("data", (d) => (out += d)); child.stderr.on("data", (d) => (err += d));
    child.on("close", (code) => (code === 0 ? resolve(JSON.parse(out.trim().split("\n").pop())) : reject(new Error(err))));
  });
}

const PAGE = `<!doctype html><html><head><meta charset="utf-8"><title>e2e</title></head><body>
<textarea id="box" style="width:800px;height:300px"></textarea>
<script>
  window.__lastInput = 0;
  document.getElementById("box").addEventListener("input", () => { window.__lastInput = performance.now(); });
</script></body></html>`;

function serverInferenceMs(db) {
  // Logged per request by the API (api/main.py _log_prediction), read back from the throwaway database.
  const script = "import sqlite3,sys,json\nc=sqlite3.connect(sys.argv[1])\n" +
    "print(json.dumps([r[0] for r in c.execute('select latency_ms from predictions order by timestamp')]))";
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON, ["-c", script, db], { cwd: REPO });
    let out = ""; child.stdout.on("data", (d) => (out += d));
    child.on("close", (code) => (code === 0 ? resolve(JSON.parse(out)) : reject(new Error("sqlite read failed"))));
  });
}

async function configure(context, extUrl, endpoint, apiKey, model) {
  const page = await context.newPage();
  await page.goto(extUrl("options.html"));
  await page.evaluate(async ([endpoint, key, model]) => {
    await chrome.storage.sync.set({ endpoint, enabled: true, model, slowTimeoutSeconds: 60 });
    await chrome.storage.local.set({ apiKey: key });
  }, [endpoint, apiKey, model]);
  await page.close();
}

async function trial(page, text) {
  // Clear (badge hides), let any pending request settle, then type and time the badge.
  await page.evaluate(() => { const b = document.getElementById("box"); b.value = ""; b.dispatchEvent(new Event("input")); });
  await page.waitForTimeout(DEBOUNCE_MS + 1500);
  await page.evaluate(() => {
    window.__badgeAt = null;
    const host = [...document.body.children].find((el) => el.shadowRoot);
    const badge = host.shadowRoot.querySelector(".badge");
    window.__obs?.disconnect();
    window.__obs = new MutationObserver(() => {
      if (/\b(safe|suspicious|malicious)\b/.test(badge.className) && window.__badgeAt === null) window.__badgeAt = performance.now();
      if (/Timeout|Offline/.test(badge.textContent) && window.__badgeAt === null) window.__badgeAt = -performance.now();
    });
    window.__obs.observe(badge, { attributes: true, childList: true, characterData: true, subtree: true });
  });
  await page.locator("#box").pressSequentially(text, { delay: 20 });
  await page.waitForFunction(() => window.__badgeAt !== null, null, { timeout: 120000 });
  const { lastInput, badgeAt } = await page.evaluate(() => ({ lastInput: window.__lastInput, badgeAt: window.__badgeAt }));
  return badgeAt < 0 ? { ok: false, ms: -badgeAt - lastInput } : { ok: true, ms: badgeAt - lastInput };
}

async function runTrials(page, prompts) {
  const out = [];
  for (const text of prompts) out.push(await trial(page, text));
  return out;
}

async function typingSession(page, seconds) {
  // Synthetic typing: inter-key gaps ~ 120-260 ms, with a 600-1600 ms pause after 12% of words.
  const rand = mulberry32(SEED);
  const words = "the quick brown fox jumps over the lazy dog while the user drafts a long message to the assistant".split(" ");
  await page.evaluate(() => { const b = document.getElementById("box"); b.value = ""; b.dispatchEvent(new Event("input")); });
  const box = page.locator("#box");
  await box.click();
  const end = Date.now() + seconds * 1000;
  let i = 0; let pauses = 0;
  while (Date.now() < end) {
    for (const ch of words[i++ % words.length] + " ") {
      await page.keyboard.type(ch);
      await page.waitForTimeout(120 + rand() * 140);
    }
    if (rand() < 0.12) { pauses += 1; await page.waitForTimeout(600 + rand() * 1000); }
  }
  await page.waitForTimeout(DEBOUNCE_MS + 2000);
  return { seconds, pauses_over_debounce: pauses };
}

const summary = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
  return s.length ? { n: s.length, mean: s.reduce((a, b) => a + b, 0) / s.length, median: q(0.5), p95: q(0.95), min: s[0], max: s[s.length - 1] } : { n: 0 };
};

const pageServer = http.createServer((_q, s) => { s.writeHead(200, { "content-type": "text/html" }); s.end(PAGE); });
await new Promise((r) => pageServer.listen(PAGE_PORT, "127.0.0.1", r));
const result = { generated_at: new Date().toISOString(), config: { trials: TRIALS, debounce_ms: DEBOUNCE_MS, typing_delay_ms: 20,
  typing_session_seconds: TYPING_SESSION_SECONDS, rate_limit_per_minute: 60, models: MODELS, inference_device: "cpu",
  machine: { cpu: os.cpus()[0]?.model, logical_cpus: os.cpus().length, platform: `${os.platform()} ${os.release()}` } },
  local: {}, hosted: null };

let context;
try {
  const prompts = await testPrompts();
  const adminKey = crypto.randomBytes(32).toString("base64url");
  const db = path.join(tmp, "e2e.db");
  startApi(adminKey, db);
  await waitFor(`${API}/health`);
  const health = await (await fetch(`${API}/health`)).json();
  result.local.health = health.models;
  const keyResp = await fetch(`${API}/admin/api-keys`, { method: "POST", headers: { "X-Admin-Key": adminKey, "Content-Type": "application/json" },
    body: JSON.stringify({ org_name: "e2e", rate_limit_per_minute: 10000 }) });
  const apiKey = (await keyResp.json()).key;

  const ext = path.join(tmp, "ext");
  fs.cpSync(path.join(REPO, "extension", "dist"), ext, { recursive: true });
  const manifest = JSON.parse(fs.readFileSync(path.join(ext, "manifest.json"), "utf8"));
  manifest.content_scripts[0].matches.push("http://127.0.0.1/*");
  manifest.host_permissions.push("http://127.0.0.1/*");
  fs.writeFileSync(path.join(ext, "manifest.json"), JSON.stringify(manifest));
  context = await chromium.launchPersistentContext(path.join(tmp, "profile"), { channel: "chromium", headless: true,
    args: [`--disable-extensions-except=${ext}`, `--load-extension=${ext}`] });
  let [sw] = context.serviceWorkers();
  if (!sw) sw = await context.waitForEvent("serviceworker");
  const extUrl = (p) => `chrome-extension://${new URL(sw.url()).host}/${p}`;
  const page = await context.newPage();
  await page.goto(`http://127.0.0.1:${PAGE_PORT}/`);

  for (const model of MODELS) {
    if (!health.models?.[model]?.available) { result.local[model] = { skipped: `not loaded: ${JSON.stringify(health.models?.[model])}` }; continue; }
    await configure(context, extUrl, API, apiKey, model);
    const before = (await serverInferenceMs(db)).length;
    await runTrials(page, prompts.slice(0, 3)); // warm-up, discarded
    const warmupRequests = (await serverInferenceMs(db)).length - before;
    const startIdx = before + warmupRequests;
    const trials = await runTrials(page, prompts);
    const inference = (await serverInferenceMs(db)).slice(startIdx);
    const ok = trials.filter((t) => t.ok).map((t) => t.ms);
    result.local[model] = { end_to_end_ms: summary(ok), failed_trials: trials.length - ok.length,
      server_inference_ms: summary(inference), requests_per_trial: inference.length / trials.length };
    console.log(model, JSON.stringify(result.local[model].end_to_end_ms));
  }

  await configure(context, extUrl, API, apiKey, "classical");
  const before = (await serverInferenceMs(db)).length;
  const session = await typingSession(page, TYPING_SESSION_SECONDS);
  const requests = (await serverInferenceMs(db)).length - before;
  result.local.typing_session = { ...session, requests, requests_per_minute: requests / (session.seconds / 60) };
  console.log("typing session", JSON.stringify(result.local.typing_session));

  if (process.env.PROMPTGUARD_HOSTED_URL && process.env.PROMPTGUARD_HOSTED_KEY) {
    const url = process.env.PROMPTGUARD_HOSTED_URL.replace(/\/+$/, "");
    await configure(context, extUrl, url, process.env.PROMPTGUARD_HOSTED_KEY, "classical");
    console.log(`hosted: idling ${HOSTED_IDLE_MINUTES} min for a cold start`);
    await page.waitForTimeout(HOSTED_IDLE_MINUTES * 60000);
    const cold = await trial(page, prompts[0]);
    const warm = await runTrials(page, prompts);
    result.hosted = { url_host: new URL(url).host, idle_minutes_before_cold: HOSTED_IDLE_MINUTES, cold_ms: cold,
      warm_end_to_end_ms: summary(warm.filter((t) => t.ok).map((t) => t.ms)), warm_failed: warm.filter((t) => !t.ok).length };
  }
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(result, null, 2));
  console.log(`wrote ${OUT}`);
} finally {
  await context?.close().catch(() => {});
  pageServer.close();
  for (const c of children) c.kill();
}
