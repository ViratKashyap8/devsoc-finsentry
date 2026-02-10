import { useCallback, useRef, useState } from "react";
import "./UploadBox.css";

interface UploadBoxProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

const ACCEPTED_TYPES = ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/x-m4a"];
const MAX_FILE_MB = 50;

export function UploadBox({ onFileSelected, disabled }: UploadBoxProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const validateFile = useCallback((candidate: File): string | null => {
    if (candidate.size > MAX_FILE_MB * 1024 * 1024) {
      return `File is too large. Max ${MAX_FILE_MB} MB allowed.`;
    }
    if (!ACCEPTED_TYPES.includes(candidate.type)) {
      return "Unsupported format. Please upload WAV, MP3, or M4A audio.";
    }
    return null;
  }, []);

  const handleFile = useCallback(
    (candidate: File) => {
      const validationError = validateFile(candidate);
      if (validationError) {
        setError(validationError);
        setFile(null);
        return;
      }
      setError(null);
      setFile(candidate);
      onFileSelected(candidate);
    },
    [onFileSelected, validateFile],
  );

  const onChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    if (selected) {
      void handleFile(selected);
    }
  };

  const onDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (disabled) return;
    setIsDragging(true);
  };

  const onDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (disabled) return;
    setIsDragging(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) {
      void handleFile(dropped);
    }
  };

  const label = file ? "Change file" : "Browse files";

  return (
    <div className="upload-box">
      <div
        className={[
          "upload-dropzone",
          isDragging ? "dragging" : "",
          disabled ? "disabled" : "",
        ]
          .filter(Boolean)
          .join(" ")}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => {
          if (!disabled) {
            inputRef.current?.click();
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".wav,.mp3,.m4a,audio/*"
          className="upload-input"
          onChange={onChange}
          disabled={disabled}
        />
        <div className="upload-content">
          <div className="upload-icon">🎵</div>
          <div className="upload-text">
            <p className="upload-title">
              Drag &amp; drop a call recording here
            </p>
            <p className="upload-subtitle">
              Supported: WAV, MP3, M4A · Max {MAX_FILE_MB} MB
            </p>
          </div>
          <button
            type="button"
            className="upload-button"
            disabled={disabled}
          >
            {label}
          </button>
        </div>
        {file && (
          <div className="upload-file-meta">
            <span className="upload-file-name">{file.name}</span>
            <span className="upload-file-size">
              {(file.size / (1024 * 1024)).toFixed(2)} MB
            </span>
          </div>
        )}
      </div>
      {error && <div className="upload-error">{error}</div>}
    </div>
  );
}

