import NextAuthImport from "next-auth";
import { authConfig } from "../auth.config";

// Edge-safe instance (no bcrypt): it only checks the session cookie. Login itself runs
// in the Node.js route handler built from ../auth.ts.
export const { auth: middleware } = (NextAuthImport as CallableFunction)(authConfig);

// Only the dashboard needs a login; /, /login and /privacy stay public.
export const config = {
  matcher: ["/dashboard/:path*"],
};
