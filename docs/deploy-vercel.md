# Deploying the dashboard to Vercel

The dashboard (`dashboard/`, Next.js 15 + Auth.js v5) is a front end for the PromptGuard API. The API
itself stays on Render (`https://ai-prompt-injection.onrender.com`); only the dashboard goes to
Vercel. Nothing here changes the Render service.

## Before you start

Have these ready:

| value | where it comes from |
|---|---|
| `ADMIN_SECRET_KEY` | Copy it from the Render dashboard: service `promptguard-api` → Environment. It must be identical on Render and Vercel; the dashboard sends it to the API as `X-Admin-Key`. |
| `NEXTAUTH_SECRET` | A fresh random secret (32 random bytes, base64), e.g. `node -e "console.log(require('crypto').randomBytes(32).toString('base64'))"`. |
| `ADMIN_PASSWORD_HASH` | A bcrypt hash of a new admin password, e.g. `cd dashboard && node -e "console.log(require('bcryptjs').hashSync(process.argv[1], 12))" "<new password>"`. Paste it **as is**: Vercel stores literal values, so do **not** escape the `$` signs (escaping is only needed in a local `.env.local` file). |

Never commit these values.

## 1. Import the project

1. Sign in at https://vercel.com with the GitHub account that can see `Arshu-1104/AI-Prompt-Injection`.
2. **Add New… → Project**, choose `Arshu-1104/AI-Prompt-Injection` and click **Import**.
3. **Root Directory**: click **Edit** and select `dashboard`. Framework preset: **Next.js** (detected
   automatically). Leave the build and install commands at their defaults (`npm install` /
   `next build`).

## 2. Production environment variables

Under **Environment Variables**, add these for the **Production** environment:

| name | value |
|---|---|
| `API_URL` | `https://ai-prompt-injection.onrender.com` (no trailing slash) |
| `ADMIN_SECRET_KEY` | the value copied from Render |
| `ADMIN_EMAIL` | the admin login email |
| `ADMIN_PASSWORD_HASH` | the bcrypt hash, unescaped |
| `NEXTAUTH_SECRET` | the fresh secret (Auth.js v5 also accepts it as `AUTH_SECRET`; set one of the two) |

Leave `PROMPTGUARD_API_KEY` unset unless you want the proxy to also send a bearer API key.
`API_URL` is required in production: without it the dashboard's API proxy answers
`500 Dashboard misconfigured: API_URL is not set` instead of falling back to localhost.

Click **Deploy**.

## 3. Set NEXTAUTH_URL after the first deploy

The production URL is known only after the first deploy (for example
`https://<project>.vercel.app`, or your own domain).

1. **Settings → Environment Variables → Add**: `NEXTAUTH_URL` = that URL, e.g.
   `https://<project>.vercel.app` (no trailing slash), environment **Production**.
2. **Deployments** → the latest deployment → **⋯ → Redeploy**, so the new variable takes effect.

## 4. Check the deployment

1. `https://<your-url>/privacy` loads without logging in and shows the privacy policy.
2. `https://<your-url>/dashboard` redirects to `/login`.
3. Log in with `ADMIN_EMAIL` and the new password; the dashboard loads.
4. **Playground**: classical is selectable (bert and guard show as unavailable while Render runs
   without heavy models). Analyze a prompt and get a label back.
5. **API Keys**: the list loads (this proves `ADMIN_SECRET_KEY` matches Render).

If step 4 or 5 shows "Backend service unavailable" or a timeout, open
`https://ai-prompt-injection.onrender.com/health` in another tab first and retry after it answers.
The Render service sleeps when idle and a cold start took about 112 s when measured (warm requests
take under a second), which can exceed a Vercel function's time limit.

## Troubleshooting

| symptom | cause |
|---|---|
| Login always fails | `ADMIN_PASSWORD_HASH` was pasted with `\$` escapes, or `ADMIN_EMAIL` differs from what you type. |
| Redirect loop or login fails right after the first deploy | `NEXTAUTH_URL` is missing or does not match the URL in the browser; set it and redeploy. |
| `MissingSecret` error in the logs | neither `NEXTAUTH_SECRET` nor `AUTH_SECRET` is set. |
| API Keys / Logs pages return 403 | `ADMIN_SECRET_KEY` differs between Vercel and Render. |
| `API_URL is not set` | add `API_URL` and redeploy. |
