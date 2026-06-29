"use client";

import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";

type Settings = { store_raw_text: boolean; webhook_url: string | null; risk_threshold: number };

export default function SettingsPage() {
  const { status } = useSession();
  const [settings, setSettings] = useState<Settings>({ store_raw_text: false, webhook_url: "", risk_threshold: 85 });
  const [message, setMessage] = useState("");
  useEffect(() => {
    if (status !== "authenticated") return;
    fetch("/api/proxy/api/settings").then((res) => res.json() as Promise<Settings>).then(setSettings);
  }, [status]);
  async function save() {
    if (settings.store_raw_text && !confirm("Raw prompt storage may contain sensitive user data. Continue?")) return;
    const response = await fetch("/api/proxy/api/settings", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });
    setMessage(response.ok ? "Saved" : "Save failed");
  }
  return <div className="max-w-2xl space-y-4"><h1 className="text-2xl font-semibold">Settings</h1><label className="flex items-center gap-2 rounded-md border bg-white p-4"><input type="checkbox" checked={settings.store_raw_text} onChange={(event) => setSettings({ ...settings, store_raw_text: event.target.checked })} />Store raw text</label><label className="block rounded-md border bg-white p-4"><span className="text-sm text-slate-500">Webhook URL</span><input className="mt-2 w-full rounded-md border px-3 py-2" value={settings.webhook_url ?? ""} onChange={(event) => setSettings({ ...settings, webhook_url: event.target.value })} /></label><label className="block rounded-md border bg-white p-4"><span className="text-sm text-slate-500">Risk threshold: {settings.risk_threshold}</span><input className="mt-2 w-full" type="range" min="0" max="100" value={settings.risk_threshold} onChange={(event) => setSettings({ ...settings, risk_threshold: Number(event.target.value) })} /></label><button className="rounded-md bg-slate-950 px-4 py-2 text-white" onClick={save}>Save</button>{message ? <p>{message}</p> : null}</div>;
}
