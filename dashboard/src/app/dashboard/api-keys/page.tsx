"use client";

import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";

type ApiKeyRow = { id: string; org_name: string; created_at: string; rate_limit_per_minute: number; is_active: boolean };

export default function ApiKeysPage() {
  const { status } = useSession();
  const [rows, setRows] = useState<ApiKeyRow[]>([]);
  const [org, setOrg] = useState("");
  const [limit, setLimit] = useState(60);
  const [rawKey, setRawKey] = useState("");

  async function load() {
    if (status !== "authenticated") return;
    const response = await fetch("/api/proxy/admin/api-keys");
    if (response.ok) setRows(await response.json() as ApiKeyRow[]);
  }
  useEffect(() => { void load(); }, [status]);

  async function createKey() {
    const response = await fetch("/api/proxy/admin/api-keys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ org_name: org, rate_limit_per_minute: limit }),
    });
    const data = await response.json() as { key?: string };
    setRawKey(data.key ?? "");
    await load();
  }
  async function revoke(id: string) {
    if (!confirm("Revoke this key?")) return;
    await fetch(`/api/proxy/admin/api-keys/${id}`, { method: "DELETE" });
    await load();
  }

  return <div className="space-y-4"><h1 className="text-2xl font-semibold">API Keys</h1><div className="rounded-md border bg-white p-4"><div className="flex flex-wrap gap-2"><input className="rounded-md border px-3 py-2" placeholder="Organization" value={org} onChange={(event) => setOrg(event.target.value)} /><input className="rounded-md border px-3 py-2" type="number" value={limit} onChange={(event) => setLimit(Number(event.target.value))} /><button className="rounded-md bg-slate-950 px-3 py-2 text-white" onClick={createKey}>Create Key</button></div>{rawKey ? <div className="mt-4 rounded-md border border-suspicious bg-suspicious-light p-3 text-sm">This key will not be shown again.<pre className="mt-2 overflow-auto">{rawKey}</pre></div> : null}</div><div className="overflow-hidden rounded-md border bg-white"><table className="w-full text-sm"><thead className="bg-slate-50 text-left"><tr><th className="p-3">Org</th><th>Created</th><th>Rate limit</th><th>Status</th><th></th></tr></thead><tbody>{rows.map((row) => <tr key={row.id} className="border-t"><td className="p-3">{row.org_name}</td><td>{new Date(row.created_at).toLocaleString()}</td><td>{row.rate_limit_per_minute}/min</td><td>{row.is_active ? "Active" : "Revoked"}</td><td><button className="rounded-md border px-2 py-1" onClick={() => revoke(row.id)}>Revoke</button></td></tr>)}</tbody></table></div></div>;
}
