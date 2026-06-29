"use client";

import { useState } from "react";
import { RiskBadge, type RiskLabel } from "@/components/RiskBadge";
import { TokenHighlight } from "@/components/TokenHighlight";

type Result = { label: RiskLabel; confidence: number; risk_score: number; attack_patterns: string[]; explanation: string; token_highlights?: Record<string, number> };

export default function PlaygroundPage() {
  const [text, setText] = useState("");
  const [model, setModel] = useState("classical");
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");

  async function analyze() {
    setError("");
    setResult(null);
    const response = await fetch("/api/proxy/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model }),
    });
    if (!response.ok) {
      setError(`Request failed: ${response.status}`);
      return;
    }
    setResult(await response.json() as Result);
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Playground</h1>
      <textarea className="min-h-40 w-full rounded-md border bg-white p-3" value={text} onChange={(event) => setText(event.target.value)} />
      <div className="flex gap-4">
        {["classical", "bert", "guard"].map((item) => <label key={item} className="flex items-center gap-2 text-sm"><input type="radio" checked={model === item} onChange={() => setModel(item)} />{item}</label>)}
      </div>
      <button className="rounded-md bg-slate-950 px-4 py-2 text-white" onClick={analyze}>Analyze</button>
      {error ? <p className="text-sm text-malicious">{error}</p> : null}
      {result ? <div className="space-y-3 rounded-md border bg-white p-4"><RiskBadge label={result.label} /><div className="h-2 rounded bg-slate-200"><div className="h-2 rounded bg-slate-950" style={{ width: `${result.risk_score}%` }} /></div><p>Risk score: {result.risk_score.toFixed(1)}</p><p>Confidence: {result.confidence.toFixed(2)}</p><p>{result.explanation}</p><p className="text-sm text-slate-600">Patterns: {result.attack_patterns.join(", ") || "none"}</p>{result.token_highlights && Object.keys(result.token_highlights).length > 0 ? <TokenHighlight text={text} scores={result.token_highlights} /> : null}</div> : null}
    </div>
  );
}
