import { auth } from "../../../../../auth";
import { NextRequest, NextResponse } from "next/server";

async function proxyRequest(req: any, pathSegments: string[]) {
  const session = req.auth;
  // TEMP DIAGNOSTIC — remove once the 401 is resolved. Logs cookie NAMES
  // only (never values) — safe to leave in Render's log stream.
  const cookieHeader: string = req.headers.get("cookie") ?? "";
  const cookieNames = cookieHeader
    .split(";")
    .map((c) => c.trim().split("=")[0])
    .filter(Boolean);
  console.log("[proxy debug]", {
    hasSession: Boolean(session),
    path: pathSegments.join("/"),
    hasApiUrl: Boolean(process.env.API_URL),
    hasAdminKey: Boolean(process.env.ADMIN_SECRET_KEY),
    hasPromptguardKey: Boolean(process.env.PROMPTGUARD_API_KEY),
    cookieNames,
  });
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const path = pathSegments.join("/");
  const url = `${process.env.API_URL ?? "http://localhost:8000"}/${path}${req.nextUrl.search}`;
  const headers: Record<string, string> = {
    Authorization: `Bearer ${process.env.PROMPTGUARD_API_KEY ?? ""}`,
    "X-Admin-Key": process.env.ADMIN_SECRET_KEY ?? "",
  };
  const contentType = req.headers.get("content-type");
  if (contentType) headers["Content-Type"] = contentType;

  const init: RequestInit = { method: req.method, headers };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.text();
  }

  const res = await fetch(url, init);
  const text = await res.text();
  try {
    return NextResponse.json(JSON.parse(text), { status: res.status });
  } catch {
    return new NextResponse(text, { status: res.status });
  }
}

type RouteContext = { params: Promise<{ path: string[] }> };

export const GET = auth(async function GET(req: any, context: any) {
  const params = await (context as RouteContext).params;
  return proxyRequest(req, params.path);
});

export const POST = auth(async function POST(req: any, context: any) {
  const params = await (context as RouteContext).params;
  return proxyRequest(req, params.path);
});

export const PATCH = auth(async function PATCH(req: any, context: any) {
  const params = await (context as RouteContext).params;
  return proxyRequest(req, params.path);
});

export const DELETE = auth(async function DELETE(req: any, context: any) {
  const params = await (context as RouteContext).params;
  return proxyRequest(req, params.path);
});