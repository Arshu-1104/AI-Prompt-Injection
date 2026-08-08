import { auth } from "../../../../../auth";
import { NextRequest, NextResponse } from "next/server";

type RouteContext = { params: Promise<{ path: string[] }> };

async function proxyRequest(req: NextRequest, pathSegments: string[]) {
  const session = await auth();
  if (!session?.user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const path = pathSegments.join("/");
  const url = `${process.env.API_URL ?? "http://localhost:8000"}/${path}${req.nextUrl.search}`;
  const headers: Record<string, string> = {
    "X-Admin-Key": process.env.ADMIN_SECRET_KEY ?? "",
  };
  const serviceKey = process.env.PROMPTGUARD_API_KEY;
  if (serviceKey) headers.Authorization = `Bearer ${serviceKey}`;
  const contentType = req.headers.get("content-type");
  if (contentType) headers["Content-Type"] = contentType;

  const init: RequestInit = { method: req.method, headers };
  if (req.method !== "GET" && req.method !== "HEAD") init.body = await req.text();

  try {
    const response = await fetch(url, init);
    const body = await response.text();
    try {
      return NextResponse.json(JSON.parse(body), { status: response.status });
    } catch {
      return new NextResponse(body, { status: response.status });
    }
  } catch {
    return NextResponse.json({ error: "Backend service unavailable" }, { status: 502 });
  }
}

export async function GET(req: NextRequest, context: RouteContext) {
  return proxyRequest(req, (await context.params).path);
}

export async function POST(req: NextRequest, context: RouteContext) {
  return proxyRequest(req, (await context.params).path);
}

export async function PATCH(req: NextRequest, context: RouteContext) {
  return proxyRequest(req, (await context.params).path);
}

export async function DELETE(req: NextRequest, context: RouteContext) {
  return proxyRequest(req, (await context.params).path);
}