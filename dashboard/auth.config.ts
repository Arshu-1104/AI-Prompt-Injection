// Edge-safe Auth.js settings shared by the middleware and auth.ts. No providers and no
// bcrypt here: the middleware runs in the Edge runtime, which bcryptjs does not support.
// Secret and URL come from AUTH_SECRET / AUTH_URL, falling back to NEXTAUTH_SECRET /
// NEXTAUTH_URL (next-auth v5 reads both), so NEXTAUTH_SECRET alone is enough.
export const authConfig = {
  session: { strategy: "jwt" as const, maxAge: 8 * 60 * 60 },
  pages: { signIn: "/login" },
  // Vercel and Render sit behind a reverse proxy. Auth.js v5 rejects requests whose Host
  // header it doesn't recognize unless explicitly told to trust it.
  trustHost: true,
  providers: [],
  callbacks: {
    authorized({ auth: session, request }: { auth: { user?: unknown } | null; request: { nextUrl: { pathname: string } } }) {
      if (request.nextUrl.pathname.startsWith("/dashboard")) return Boolean(session?.user);
      return true;
    },
  },
};
