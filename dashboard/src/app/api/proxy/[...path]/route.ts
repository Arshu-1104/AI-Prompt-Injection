import { auth } from "../../../../../auth";
import { NextRequest, NextResponse } from "next/server";

async function proxyRequest(req: NextRequest, pathSegments: string[]) {
  const session = await auth();
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
