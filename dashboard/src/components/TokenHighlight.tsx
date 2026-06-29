export function TokenHighlight({ text, scores }: { text: string; scores: Record<string, number> }) {
  return (
    <p className="leading-8">
      {text.split(/\s+/).filter(Boolean).map((word, index) => {
        const normalized = word.toLowerCase().replace(/[^a-z0-9_-]/g, "");
        const score = scores[word] ?? scores[normalized] ?? 0;
        const background = score >= 0.66 ? "#FEE2E2" : score >= 0.33 ? "#FEF3C7" : "#E8F4FD";
        return (
          <span key={`${word}-${index}`} className="mr-1 inline-block rounded px-1" style={{ background }}>
            {word}
          </span>
        );
      })}
    </p>
  );
}
