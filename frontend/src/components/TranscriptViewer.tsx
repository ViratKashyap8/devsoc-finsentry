import type { FinanceAnalysisResponse, FinancialEntity } from "../api/client";
import "./TranscriptViewer.css";

interface TranscriptViewerProps {
  analysis: FinanceAnalysisResponse | null;
}

function highlightText(text: string, entities: FinancialEntity[]) {
  if (!entities.length || !text) return <span>{text}</span>;

  const sorted = [...entities].sort(
    (a, b) => a.span_start - b.span_start || a.span_end - b.span_end,
  );

  const parts: React.ReactNode[] = [];
  let cursor = 0;

  for (const ent of sorted) {
    const start = ent.span_start ?? 0;
    const end = ent.span_end ?? start;
    if (start > cursor) {
      parts.push(<span key={`t-${cursor}`}>{text.slice(cursor, start)}</span>);
    }
    const label = (ent.entity_type || "").toUpperCase();
    parts.push(
      <mark
        key={`e-${start}-${end}`}
        className="entity-highlight"
        data-entity={label}
      >
        {text.slice(start, end)}
      </mark>,
    );
    cursor = end;
  }

  if (cursor < text.length) {
    parts.push(<span key={`t-end`}>{text.slice(cursor)}</span>);
  }

  return <>{parts}</>;
}

export function TranscriptViewer({ analysis }: TranscriptViewerProps) {
  if (!analysis) {
    return (
      <div className="transcript-viewer empty">
        <p>Run an analysis to see the transcript here.</p>
      </div>
    );
  }

  const segments = analysis.segments;

  return (
    <div className="transcript-viewer">
      <h3>Transcript</h3>
      <div className="transcript-scroll">
        {segments.map((seg, index) => {
          const entities = seg.entities ?? [];
          const start = seg.start ?? 0;
          const end = seg.end ?? 0;
          return (
            <div key={index} className="transcript-segment">
              <div className="transcript-meta">
                <span className="timestamp">
                  {start.toFixed(1)}s – {end.toFixed(1)}s
                </span>
              </div>
              <p className="transcript-text">
                {highlightText(seg.text ?? "", entities)}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

