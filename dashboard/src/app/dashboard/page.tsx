"use client";

import { Suspense } from "react";
import useSWR from "swr";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useSession } from "next-auth/react";
import { StatCard } from "@/components/StatCard";

type Daily = { date: string; SAFE: number; SUSPICIOUS: number; MALICIOUS: number };
type Pattern = { pattern: string; count: number };
type Stats = { daily: Daily[]; top_patterns: Pattern[] };

function DashboardCharts() {
  const { status } = useSession();
  const fetcher = (url: string) => fetch(url).then((res) => res.json() as Promise<Stats>);
  const { data } = useSWR(status === "authenticated" ? "/api/proxy/api/stats?days=7" : null, fetcher, { refreshInterval: 60000 });
  const daily = data?.daily ?? [];
  const patterns = data?.top_patterns ?? [];
  const totals = daily.reduce((acc, row) => ({ safe: acc.safe + row.SAFE, suspicious: acc.suspicious + row.SUSPICIOUS, malicious: acc.malicious + row.MALICIOUS }), { safe: 0, suspicious: 0, malicious: 0 });
  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-4">
        <StatCard title="Total requests" value={totals.safe + totals.suspicious + totals.malicious} subtitle="Last 7 days" />
        <StatCard title="Safe" value={totals.safe} />
        <StatCard title="Review" value={totals.suspicious} />
        <StatCard title="Danger" value={totals.malicious} />
      </div>
      <div className="h-80 rounded-md border bg-white p-4">
        <ResponsiveContainer width="100%" height="100%"><LineChart data={daily}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="date" /><YAxis allowDecimals={false} /><Tooltip /><Legend /><Line dataKey="SAFE" stroke="#1B6CA8" /><Line dataKey="SUSPICIOUS" stroke="#B45309" /><Line dataKey="MALICIOUS" stroke="#B91C1C" /></LineChart></ResponsiveContainer>
      </div>
      <div className="h-80 rounded-md border bg-white p-4">
        <ResponsiveContainer width="100%" height="100%"><BarChart data={patterns.map((item) => ({ ...item, pattern: item.pattern.length > 30 ? `${item.pattern.slice(0, 30)}...` : item.pattern }))}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="pattern" /><YAxis allowDecimals={false} /><Tooltip /><Bar dataKey="count" fill="#B91C1C" /></BarChart></ResponsiveContainer>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  return <Suspense fallback={<div className="rounded-md border bg-white p-6">Loading...</div>}><DashboardCharts /></Suspense>;
}
