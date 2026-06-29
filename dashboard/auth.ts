import NextAuthImport from "next-auth";
import Credentials from "next-auth/providers/credentials";
import bcrypt from "bcryptjs";

export const { handlers, auth, signIn, signOut } = (NextAuthImport as CallableFunction)({
  session: { strategy: "jwt", maxAge: 8 * 60 * 60 },
  pages: { signIn: "/login" },
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
