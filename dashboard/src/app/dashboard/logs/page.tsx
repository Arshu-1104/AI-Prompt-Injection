"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useSession } from "next-auth/react";
import { RiskBadge, type RiskLabel } from "@/components/RiskBadge";

type LogItem = { id: string; timestamp: string; label: RiskLabel; confidence: number; risk_score: number; model_used: string; attack_patterns: string[]; explanation: string | null };
type LogsResponse = { total: number; page: number; per_page: number; items: LogItem[] };

function LogsContent() {
  const router = useRouter();
  const params = useSearchParams();
  const { status } = useSession();
  const [data, setData] = useState<LogsResponse | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [label, setLabel] = useState(params.get("label") ?? "");
  const [model, setModel] = useState(params.get("model") ?? "");
  const page = Number(params.get("page") ?? "1");

  useEffect(() => {
    if (status !== "authenticated") return;
    const query = new URLSearchParams(params.toString());
    fetch(`/api/proxy/api/logs?${query.toString()}`)
      .then((res) => res.json() as Promise<LogsResponse>)
      .then(setData);
  }, [params, status]);

  function applyFilters(nextPage = 1) {
    const query = new URLSearchParams();
    if (label) query.set("label", label);
    if (model) query.set("model", model);
    query.set("page", String(nextPage));
    router.push(`/dashboard/logs?${query.toString()}`);
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Logs</h1>
      <div className="flex flex-wrap gap-2">
        <select className="rounded-md border bg-white px-3 py-2" value={label} onChange={(event) => setLabel(event.target.value)}>
          <option value="">All labels</option>
          <option>SAFE</option>
          <option>SUSPICIOUS</option>
          <option>MALICIOUS</option>
        </select>
        <select className="rounded-md border bg-white px-3 py-2" value={model} onChange={(event) => setModel(event.target.value)}>
          <option value="">All models</option>
          <option>classical</option>
          <option>bert</option>
          <option>guard</option>
        </select>
        <button className="rounded-md bg-slate-950 px-3 py-2 text-white" onClick={() => applyFilters()}>Apply</button>
      </div>
      <div className="overflow-hidden rounded-md border bg-white">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-left">
            <tr>
              <th className="p-3">Time</th>
              <th>Label</th>
              <th>Risk</th>
              <th>Confidence</th>
              <th>Model</th>
              <th>Patterns</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map((item) => (
              <tr key={item.id} className="cursor-pointer border-t" onClick={() => setOpen(open === item.id ? null : item.id)}>
                <td className="p-3">{new Date(item.timestamp).toLocaleString()}</td>
                <td><RiskBadge label={item.label} /></td>
                <td>{item.risk_score.toFixed(1)}</td>
                <td>{item.confidence.toFixed(2)}</td>
                <td>{item.model_used}</td>
                <td>
                  {item.attack_patterns.length}
                  {open === item.id ? (
                    <div className="mt-2 text-slate-600">
                      {item.attack_patterns.join(", ") || "none"}
                      <br />
                      {item.explanation ?? ""}
                    </div>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex gap-2">
        <button className="rounded-md border px-3 py-2" disabled={page <= 1} onClick={() => applyFilters(page - 1)}>Previous</button>
        <button className="rounded-md border px-3 py-2" onClick={() => applyFilters(page + 1)}>Next</button>
      </div>
    </div>
  );
}

export default function LogsPage() {
  return (
    <Suspense fallback={<div className="rounded-md border bg-white p-6">Loading logs...</div>}>
      <LogsContent />
    </Suspense>
  );
}
