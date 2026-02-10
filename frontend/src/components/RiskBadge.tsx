import "./RiskBadge.css";

export type RiskLevel = "low" | "medium" | "high" | "critical" | "none" | string;

interface RiskBadgeProps {
  level: RiskLevel | null | undefined;
}

export function RiskBadge({ level }: RiskBadgeProps) {
  if (!level) return <span className="risk-badge risk-badge-none">N/A</span>;

  const normalized = String(level).toLowerCase() as RiskLevel;

  return (
    <span className={`risk-badge risk-badge-${normalized}`}>
      {normalized.toUpperCase()}
    </span>
  );
}

