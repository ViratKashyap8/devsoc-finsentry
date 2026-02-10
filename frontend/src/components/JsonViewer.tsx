import { useState } from "react";
import "./JsonViewer.css";

interface JsonViewerProps {
  data: unknown;
  title?: string;
}

export function JsonViewer({ data, title = "JSON Report" }: JsonViewerProps) {
  const [open, setOpen] = useState(false);

  if (!data) return null;

  return (
    <div className="json-viewer">
      <button
        type="button"
        className="json-toggle"
        onClick={() => setOpen((prev) => !prev)}
      >
        {open ? "Hide" : "Show"} {title}
      </button>
      {open && (
        <pre className="json-body">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

