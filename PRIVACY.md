# PromptGuard Privacy Policy

Effective date: 15 October 2026

This policy covers the PromptGuard Chrome extension and the PromptGuard API it talks to. PromptGuard
checks text you type into AI chat prompt boxes for prompt-injection patterns.

## What the extension reads

- The text you type into prompt boxes on chatgpt.com, chat.openai.com, claude.ai and
  gemini.google.com. It runs on no other websites.
- It reads that text only while the extension is enabled and both an API endpoint and an API key are
  configured. While it is disabled or not configured, it does not read or send anything.
- It does not read other page content, your chat history, cookies or browsing history.

## Where the text is sent

- Each checked text is sent over the network to one place only: the API endpoint you configure in the
  extension settings, together with your API key and the model you selected.
- The endpoint may be a server you run yourself or the project's hosted API
  (ai-prompt-injection.onrender.com). Whoever operates that server receives the text.
- The settings page also sends a connection check to that endpoint's /health address when you click
  "Test Connection".
- Nothing is sent to any other server.

## What the extension stores on your device

- Your API key, in Chrome's local extension storage on this device (chrome.storage.local). It is not
  synced to your other devices.
- Your endpoint URL, the on/off switch, the selected model and the timeout, in Chrome's synced
  extension storage (chrome.storage.sync). If you use Chrome sync, these settings sync to your other
  Chrome browsers.
- The result of the most recent check (label, confidence, risk score and matched attack patterns), in
  local extension storage, to show it in the popup.

## What the PromptGuard API stores

For each checked text, the API stores:

- a SHA-256 hash of the text (identical texts give identical hashes, so a short or common text
  could be guessed from its hash);
- the result: label, confidence and risk score;
- the matched attack patterns;
- the processing time (latency);
- the model used;
- a timestamp;
- which API key made the request.

The full text is stored only if the "store raw text" setting is turned on for that API key. It is
off by default.

## What we do not do

- We do not sell or share your data with third parties.
- We do not show ads or use your data for advertising.
- The extension contains no analytics or tracking.
- The extension does not load or run remote code; all of its code ships in the extension package.

## How to delete your data

- On your device: remove the extension, which deletes everything it stored. Or clear the API key and
  endpoint on its settings page.
- On the server: there is currently no self-service delete. Revoking an API key stops its use but
  does not delete stored results. If you run your own server, delete the rows from its database. For
  data held by the project's hosted API, open an issue (see Contact) and the maintainer will delete it.
- Stored results are kept until they are deleted. There is no automatic expiry.

## Contact

Questions and deletion requests: open an issue at
https://github.com/Arshu-1104/AI-Prompt-Injection/issues

## Changes

If this policy changes, the effective date above changes with it.
