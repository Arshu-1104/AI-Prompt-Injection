"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { signOut } from "next-auth/react";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const links = [
  ["Overview", "/dashboard"],
  ["Playground", "/dashboard/playground"],
  ["Logs", "/dashboard/logs"],
  ["API Keys", "/dashboard/api-keys"],
  ["Settings", "/dashboard/settings"],
];

export default function DashboardLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="min-h-screen bg-slate-100 md:flex">
      <aside className="border-b bg-white p-4 md:min-h-screen md:w-60 md:border-b-0 md:border-r">
        <div className="text-lg font-semibold">PromptGuard</div>
        <nav className="mt-6 flex gap-2 md:flex-col">
          {links.map(([label, href]) => (
            <Link key={href} href={href} className={cn("rounded-md px-3 py-2 text-sm", pathname === href ? "bg-slate-950 text-white" : "text-slate-700 hover:bg-slate-100")}>
              {label}
            </Link>
          ))}
        </nav>
        <button className="mt-6 rounded-md border px-3 py-2 text-sm" onClick={() => signOut({ callbackUrl: "/login" })}>Sign out</button>
      </aside>
      <main className="flex-1 p-6">{children}</main>
    </div>
  );
}
