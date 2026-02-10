const DEFAULT_TIMEOUT_MS = 60000;

const baseUrl =
  (import.meta.env.VITE_BACKEND_URL as string | undefined) ??
  window.location.origin;

export interface HealthResponse {
  status: string;
}

export interface AudioUploadResponse {
  file_id: string;
}

export interface FinancialEntity {
  text: string;
  entity_type: string;
  span_start: number;
  span_end: number;
  normalized_value?: string | null;
}

export interface SegmentAnalysis {
  start: number;
  end: number;
  text: string;
  intent?: string | null;
  intent_confidence?: number;
  entities: FinancialEntity[];
  obligations?: unknown[];
  regulatory_phrases?: unknown[];
  emotion?: string | null;
  emotion_confidence?: number;
  stress_score?: number;
}

export interface CallMetrics {
  dominant_intent?: string | null;
  overall_risk_level: string;
  risk_score: number;
  risk_factors: string[];
  total_obligations: number;
  obligation_summary: Array<{ type: string; text: string }>;
  stress_trend?: string | null;
  regulatory_compliance_score: number;
}

export interface FinanceAnalysisResponse {
  call_id: string;
  full_transcript: string;
  segments: SegmentAnalysis[];
  call_metrics: CallMetrics;
  all_entities: FinancialEntity[];
  all_obligations: unknown[];
  all_regulatory: unknown[];
  processing_time_sec: number;
  // Optional STT metadata if backend propagates it
  detected_language?: string | null;
  language_probability?: number | null;
  avg_logprob?: number | null;
}

export interface AnalyzeRequest {
  full_transcript: string;
  segments?: Array<{ start: number; end: number; text: string }> | null;
  call_id: string;
  use_llm_extraction?: boolean;
}

class ApiError extends Error {
  status: number;
  details?: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

async function request<T>(
  path: string,
  init: RequestInit & { timeoutMs?: number } = {},
): Promise<T> {
  const controller = new AbortController();
  const timeout = init.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timeoutId = window.setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(new URL(path, baseUrl).toString(), {
      ...init,
      signal: init.signal ?? controller.signal,
    });

    const text = await response.text();
    const json = text ? JSON.parse(text) : null;

    if (!response.ok) {
      const message =
        (json && (json.detail as string | undefined)) ??
        `Request failed with status ${response.status}`;
      throw new ApiError(message, response.status, json ?? undefined);
    }

    return json as T;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError("Request timed out", 408);
    }
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(
      err instanceof Error ? err.message : "Unknown error",
      0,
    );
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export const apiClient = {
  async health(signal?: AbortSignal): Promise<HealthResponse> {
    return request<HealthResponse>("/api/health", { method: "GET", signal });
  },

  async uploadAudio(
    file: File,
    signal?: AbortSignal,
  ): Promise<AudioUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);

    // Simple retry once for transient failures.
    try {
      return await request<AudioUploadResponse>("/api/audio/upload", {
        method: "POST",
        body: formData,
        signal,
        timeoutMs: DEFAULT_TIMEOUT_MS * 5,
      });
    } catch (err) {
      if (err instanceof ApiError && err.status >= 500) {
        return request<AudioUploadResponse>("/api/audio/upload", {
          method: "POST",
          body: formData,
          signal,
          timeoutMs: DEFAULT_TIMEOUT_MS * 5,
        });
      }
      throw err;
    }
  },

  async analyzeTranscript(
    body: AnalyzeRequest,
    signal?: AbortSignal,
  ): Promise<FinanceAnalysisResponse> {
    return request<FinanceAnalysisResponse>("/api/finance/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal,
      timeoutMs: DEFAULT_TIMEOUT_MS * 10,
    });
  },

  async analyzeAudio(
    file: File,
    modelSize = "medium",
    signal?: AbortSignal,
  ): Promise<FinanceAnalysisResponse> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_size", modelSize);
    formData.append("use_llm_extraction", "false");

    return request<FinanceAnalysisResponse>("/api/audio/analyze", {
      method: "POST",
      body: formData,
      signal,
      timeoutMs: DEFAULT_TIMEOUT_MS * 10,
    });
  },
};

export { ApiError };

