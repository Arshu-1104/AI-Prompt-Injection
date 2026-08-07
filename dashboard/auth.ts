import NextAuthImport from "next-auth";
import Credentials from "next-auth/providers/credentials";
import bcrypt from "bcryptjs";

export const { handlers, auth, signIn, signOut } = (NextAuthImport as CallableFunction)({
  session: { strategy: "jwt", maxAge: 8 * 60 * 60 },
  pages: { signIn: "/login" },
  // Render (like any non-Vercel host) sits behind a reverse proxy. Auth.js v5
  // rejects requests whose Host header it doesn't recognize unless explicitly
  // told to trust it — without this, session verification silently fails on
  // every deployment platform except Vercel.
  trustHost: true,
  providers: [
    Credentials({
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        const email = String(credentials?.email ?? "");
        const password = String(credentials?.password ?? "");
        const adminEmail = process.env.ADMIN_EMAIL ?? "";
        const passwordHash = process.env.ADMIN_PASSWORD_HASH ?? "";
        if (!adminEmail || !passwordHash || email !== adminEmail) return null;
        const ok = await bcrypt.compare(password, passwordHash);
        if (!ok) return null;
        return { id: "admin", email };
      },
    }),
  ],
  callbacks: {
    authorized({ auth: session, request }: { auth: { user?: unknown } | null; request: { nextUrl: { pathname: string } } }) {
      if (request.nextUrl.pathname.startsWith("/dashboard")) return Boolean(session?.user);
      return true;
    },
  },
});