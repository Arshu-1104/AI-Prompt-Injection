import NextAuthImport from "next-auth";
import Credentials from "next-auth/providers/credentials";
import bcrypt from "bcryptjs";
import { authConfig } from "./auth.config";

export const { handlers, auth, signIn, signOut } = (NextAuthImport as CallableFunction)({
  ...authConfig,
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
});
