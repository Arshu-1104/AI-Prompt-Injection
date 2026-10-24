"use client";

import { useEffect, useState } from "react";
import { RiskBadge, type RiskLabel } from "@/components/RiskBadge";
import { TokenHighlight } from "@/components/TokenHighlight";

type RiskComponents = { base: number; pattern_boost: number; confidence_term: number };
type Result = { label: RiskLabel; confidence: number; risk_score: number; risk_components?: RiskComponents; attack_patterns: string[]; explanation: string; token_highlights?: Record<string, number> };

function formatComponents(c: RiskComponents): string {
  const signed = (v: number) => (v < 0 ? `- ${Math.abs(v).toFixed(1)}` : `+ ${v.toFixed(1)}`);
  return `= base ${c.base.toFixed(0)} (label) ${signed(c.pattern_boost)} (patterns) ${signed(c.confidence_term)} (confidence), clipped to 0-100`;
}
type ModelStatus = { available: boolean; reason?: string | null };
type Health = { models?: Record<string, ModelStatus> };

const MODEL_IDS = ["classical", "bert", "guard"] as const;

// Reason codes from GET /health; details are only in the backend logs.
const REASON_TEXT: Record<string, string> = {
  heavy_disabled: "Heavy models are disabled on the backend (ENABLE_HEAVY_MODELS).",
  runtime_unavailable: "The backend has no torch/transformers installed.",
  weights_missing: "No weights configured (BERT_MODEL_ID / GUARD_MODEL_ID).",
  lfs_stub: "Only Git LFS pointer files are present, not real weights.",
  download_failed: "Weights could not be downloaded from the Hugging Face Hub.",
  load_failed: "The model failed to load; see the backend logs.",
};

export default function PlaygroundPage() {
  const [text, setText] = useState("");
  const [model, setModel] = useState("classical");
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  // null = still checking /health
  const [models, setModels] = useState<Record<string, ModelStatus> | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/proxy/health")
      .then(async (response) => (response.ok ? ((await response.json()) as Health) : null))
      .catch(() => null)
      .then((health) => {
        if (cancelled) return;
        // Older backends have no per-model status; assume only classical is served.
        setModels(health?.models ?? { classical: { available: true } });
      });
    return () => { cancelled = true; };
  }, []);

  async function analyze() {
    setError("");
    setResult(null);
    const response = await fetch("/api/proxy/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null) as { detail?: unknown } | null;
      const detail = typeof body?.detail === "string" ? `: ${body.detail}` : "";
      setError(`Request failed (${response.status})${detail}`);
      return;
    }
    setResult(await response.json() as Result);
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Playground</h1>
      <textarea className="min-h-40 w-full rounded-md border bg-white p-3" value={text} onChange={(event) => setText(event.target.value)} />
      <div className="flex gap-4">
        {MODEL_IDS.map((id) => {
          const status = models?.[id];
          const available = status?.available === true;
          const note = models === null ? " (checking...)" : available ? "" : " (unavailable)";
          return (
            <label key={id} title={available || !status?.reason ? undefined : REASON_TEXT[status.reason] ?? status.reason} className={`flex items-center gap-2 text-sm ${available ? "" : "text-slate-400"}`}>
              <input type="radio" checked={model === id} disabled={!available} onChange={() => available && setModel(id)} />
              {id}{note}
            </label>
          );
        })}
      </div>
      <button className="rounded-md bg-slate-950 px-4 py-2 text-white" onClick={analyze}>Analyze</button>
      {error ? <p className="text-sm text-malicious">{error}</p> : null}
      {result ? <div className="space-y-3 rounded-md border bg-white p-4"><RiskBadge label={result.label} /><div className="h-2 rounded bg-slate-200"><div className="h-2 rounded bg-slate-950" style={{ width: `${result.risk_score}%` }} /></div><p>Risk score: {result.risk_score.toFixed(1)}</p>{result.risk_components ? <p className="text-sm text-slate-600">{formatComponents(result.risk_components)}</p> : null}<p>Confidence: {result.confidence.toFixed(2)}</p><p>{result.explanation}</p><p className="text-sm text-slate-600">Patterns: {result.attack_patterns.join(", ") || "none"}</p>{result.token_highlights && Object.keys(result.token_highlights).length > 0 ? <div><p className="text-sm text-slate-600">Words that contributed most to this label:</p><TokenHighlight text={text} scores={result.token_highlights} /></div> : null}</div> : null}
    </div>
  );
}