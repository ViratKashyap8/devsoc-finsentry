import { RiskBadge } from "./RiskBadge";
import type { FinanceAnalysisResponse } from "../api/client";
import "./ResultCards.css";

interface ResultCardsProps {
  analysis: FinanceAnalysisResponse | null;
}

export function ResultCards({ analysis }: ResultCardsProps) {
  if (!analysis) {
    return (
      <div className="result-cards empty">
        <p className="result-empty-text">
          Run an analysis to see call-level insights here.
        </p>
      </div>
    );
  }

  const cm = analysis.call_metrics;
  const intent = cm.dominant_intent ?? "N/A";
  const riskLevel = cm.overall_risk_level ?? "none";

  // Simple heuristics from entities for amount, merchant, payment method
  const amounts = new Set<string>();
  const merchants = new Set<string>();
  const paymentMethods = new Set<string>();

  for (const ent of analysis.all_entities ?? []) {
    const label = (ent.entity_type || "").toUpperCase();
    if (label.includes("AMOUNT") || label.includes("MONEY") || label.includes("CURRENCY")) {
      amounts.add(ent.text);
    } else if (label.includes("MERCHANT") || label.includes("COMPANY")) {
      merchants.add(ent.text);
    } else if (
      label.includes("CARD") ||
      label.includes("UPI") ||
      label.includes("PAYMENT_METHOD") ||
      label.includes("WALLET")
    ) {
      paymentMethods.add(ent.text);
    }
  }

  const amountDisplay =
    amounts.size > 0 ? Array.from(amounts).sort().join(", ") : "N/A";
  const merchantDisplay =
    merchants.size > 0 ? Array.from(merchants).sort().join(", ") : "N/A";
  const paymentDisplay =
    paymentMethods.size > 0
      ? Array.from(paymentMethods).sort().join(", ")
      : "N/A";

  return (
    <div className="result-cards">
      <div className="result-card">
        <h3>Detected Language</h3>
        <p>
          {analysis.detected_language
            ? analysis.language_probability
              ? `${analysis.detected_language} (${analysis.language_probability.toFixed(2)})`
              : analysis.detected_language
            : "N/A"}
        </p>
        {typeof analysis.avg_logprob === "number" && (
          <p className="result-subtext">
            STT avg logprob: {analysis.avg_logprob.toFixed(3)}
          </p>
        )}
      </div>

      <div className="result-card">
        <h3>Intent</h3>
        <p className="result-intent">{intent}</p>
      </div>

      <div className="result-card">
        <h3>Risk Level</h3>
        <RiskBadge level={riskLevel} />
      </div>

      <div className="result-card">
        <h3>Amount</h3>
        <p>{amountDisplay}</p>
      </div>

      <div className="result-card">
        <h3>Merchant</h3>
        <p>{merchantDisplay}</p>
      </div>

      <div className="result-card">
        <h3>Payment Method</h3>
        <p>{paymentDisplay}</p>
      </div>
    </div>
  );
}

