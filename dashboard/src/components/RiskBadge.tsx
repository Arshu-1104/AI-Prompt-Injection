import { cn } from "@/lib/utils";

export type RiskLabel = "SAFE" | "SUSPICIOUS" | "MALICIOUS";

const styles: Record<RiskLabel, string> = {
  SAFE: "bg-safe-light text-safe-dark border-safe",
  SUSPICIOUS: "bg-suspicious-light text-suspicious-dark border-suspicious",
  MALICIOUS: "bg-malicious-light text-malicious-dark border-malicious",
};

export function RiskBadge({ label }: { label: RiskLabel }) {
  return <span className={cn("inline-flex rounded-full border px-2 py-1 text-xs font-medium", styles[label])}>{label}</span>;
}
