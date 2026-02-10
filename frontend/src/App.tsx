import "./App.css";
import { useState } from "react";
import {
  apiClient,
  ApiError,
  type FinanceAnalysisResponse,
} from "./api/client";
import { UploadBox } from "./components/UploadBox";
import {
  ProcessingSteps,
  type ProcessingStage,
} from "./components/ProcessingSteps";
import { ResultCards } from "./components/ResultCards";
import { TranscriptViewer } from "./components/TranscriptViewer";
import { JsonViewer } from "./components/JsonViewer";
import { ErrorBanner } from "./components/ErrorBanner";

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<FinanceAnalysisResponse | null>(null);
  const [rawReport, setRawReport] = useState<unknown>(null);
  const [stage, setStage] = useState<ProcessingStage>("idle");
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    setAnalysis(null);
    setRawReport(null);
    setError(null);
    setStage("idle");
  };

  const runAnalysis = async () => {
    if (!selectedFile || isRunning) return;
    setIsRunning(true);
    setError(null);
    setStage("uploading");

    try {
      // Backend health check
      await apiClient.health();

      // End-to-end audio + finance analysis (upload + pipeline + finance)
      setStage("transcribing");
      const analysisRes = await apiClient.analyzeAudio(selectedFile, "medium");

      setAnalysis(analysisRes);
      setRawReport(analysisRes);
      setStage("done");
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : "Unexpected error";
      setError(message);
      setStage("error");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="app-root">
      <main className="app-shell">
        <header className="app-header">
          <div>
            <h1>FinSentry Live Judge Demo</h1>
            <p>Upload a call recording and see instant financial risk insights.</p>
          </div>
        </header>

        <ErrorBanner message={error} onClose={() => setError(null)} />

        <section className="app-layout">
          <div className="left-column">
            <UploadBox onFileSelected={handleFileSelected} disabled={isRunning} />
            <button
              type="button"
              className="analyze-button"
              disabled={!selectedFile || isRunning}
              onClick={() => {
                void runAnalysis();
              }}
            >
              {isRunning ? "Running analysis..." : "Run Analysis"}
            </button>
            <ProcessingSteps stage={stage} />
          </div>

          <div className="right-column">
            <ResultCards analysis={analysis} />
            <TranscriptViewer analysis={analysis} />
            <JsonViewer data={rawReport} />
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;

