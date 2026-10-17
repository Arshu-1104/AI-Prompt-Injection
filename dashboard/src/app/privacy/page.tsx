import type { Metadata } from "next";
import type { ReactNode } from "react";
import { POLICY_MARKDOWN } from "./policy";

export const metadata: Metadata = { title: "Privacy Policy - PromptGuard" };

// Public page: src/middleware.ts only protects /dashboard/*, so no login is needed here.

// Renders bare URLs in a line as links.
function inline(text: string): ReactNode[] {
  return text.split(/(https?:\/\/\S+)/g).map((part, i) =>
    /^https?:\/\//.test(part) ? (
      <a key={i} href={part} className="text-blue-700 underline dark:text-blue-400">
        {part}
      </a>
    ) : (
      part
    ),
  );
}

// The policy uses only "# ", "## ", "- " bullets (with indented continuation lines) and paragraphs.
function render(markdown: string): ReactNode[] {
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];
  let paragraph: string[] = [];
  const flush = () => {
    if (paragraph.length) blocks.push(<p key={blocks.length}>{inline(paragraph.join(" "))}</p>);
    if (bullets.length)
      blocks.push(
        <ul key={blocks.length} className="list-disc space-y-1 pl-6">
          {bullets.map((b, i) => <li key={i}>{inline(b)}</li>)}
        </ul>,
      );
    paragraph = [];
    bullets = [];
  };
  for (const line of markdown.split("\n")) {
    if (line.startsWith("# ")) {
      flush();
      blocks.push(<h1 key={blocks.length} className="text-2xl font-semibold">{line.slice(2)}</h1>);
    } else if (line.startsWith("## ")) {
      flush();
      blocks.push(<h2 key={blocks.length} className="pt-2 text-lg font-semibold">{line.slice(3)}</h2>);
    } else if (line.startsWith("- ")) {
      if (paragraph.length) flush();
      bullets.push(line.slice(2));
    } else if (line.startsWith("  ") && bullets.length) {
      bullets[bullets.length - 1] += ` ${line.trim()}`;
    } else if (!line.trim()) {
      flush();
    } else {
      if (bullets.length) flush();
      paragraph.push(line.trim());
    }
  }
  flush();
  return blocks;
}

export default function PrivacyPage() {
  return <main className="mx-auto max-w-3xl space-y-3 px-4 py-10 text-sm leading-6">{render(POLICY_MARKDOWN)}</main>;
}
