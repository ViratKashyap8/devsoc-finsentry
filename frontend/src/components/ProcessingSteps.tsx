import "./ProcessingSteps.css";

export type ProcessingStage =
  | "idle"
  | "uploading"
  | "transcribing"
  | "analyzing"
  | "done"
  | "error";

interface ProcessingStepsProps {
  stage: ProcessingStage;
}

const STEPS: { id: ProcessingStage; label: string }[] = [
  { id: "uploading", label: "Uploading" },
  { id: "transcribing", label: "Transcribing" },
  { id: "analyzing", label: "Analyzing" },
  { id: "done", label: "Done" },
];

export function ProcessingSteps({ stage }: ProcessingStepsProps) {
  if (stage === "idle") return null;

  const currentIndex =
    stage === "error"
      ? 0
      : STEPS.findIndex((s) => s.id === stage) === -1
        ? 0
        : STEPS.findIndex((s) => s.id === stage);

  return (
    <div className="steps-container" aria-label="Processing steps">
      {STEPS.map((step, index) => {
        const isActive = index === currentIndex;
        const isComplete = index < currentIndex;
        return (
          <div key={step.id} className="step-item">
            <div
              className={[
                "step-circle",
                isComplete ? "complete" : "",
                isActive ? "active" : "",
              ]
                .filter(Boolean)
                .join(" ")}
            >
              {isComplete ? "✓" : index + 1}
            </div>
            <div className="step-label">{step.label}</div>
            {index < STEPS.length - 1 && (
              <div className="step-connector" aria-hidden="true" />
            )}
          </div>
        );
      })}
    </div>
  );
}

