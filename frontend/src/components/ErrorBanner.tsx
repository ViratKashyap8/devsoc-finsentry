import "./ErrorBanner.css";

interface ErrorBannerProps {
  message: string | null;
  onClose?: () => void;
}

export function ErrorBanner({ message, onClose }: ErrorBannerProps) {
  if (!message) return null;

  return (
    <div className="error-banner">
      <div className="error-banner-content">
        <span className="error-banner-icon">⚠️</span>
        <span className="error-banner-text">{message}</span>
      </div>
      {onClose && (
        <button
          type="button"
          className="error-banner-close"
          onClick={onClose}
          aria-label="Dismiss error"
        >
          ×
        </button>
      )}
    </div>
  );
}

