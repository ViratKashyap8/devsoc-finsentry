import { useState, useRef } from 'react';
import './FileStats.css';

interface AudioResponse {
    file_id: string;
    status: string;
}

export default function FileStats() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [fileId, setFileId] = useState<string | null>(null);
    const [audioData, setAudioData] = useState<AudioResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [message, setMessage] = useState<string | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            processFile(e.target.files[0]);
        }
    };

    const processFile = (file: File) => {
        if (!file.type.startsWith('audio/')) {
            setError('Please select an audio file.');
            setSelectedFile(null);
            return;
        }
        setSelectedFile(file);
        setError(null);
        setMessage(null);
        setAudioData(null);
        setFileId(null);
    };

    const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const onDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            processFile(e.dataTransfer.files[0]);
        }
    };

    const uploadFile = async () => {
        if (!selectedFile) return;

        setLoading(true);
        setError(null);
        setMessage(null);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/api/audio/upload', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || 'Upload failed');
            }

            const data = await response.json();
            setFileId(data.file_id);
            setMessage('File uploaded successfully!');
        } catch (err: any) {
            setError(err.message || 'Failed to upload file. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const getStatus = async () => {
        if (!fileId) return;

        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`/api/audio/${fileId}`);
            if (!response.ok) throw new Error('Failed to fetch status');
            const data = await response.json();
            setAudioData(data);
        } catch (err: any) {
            setError(err.message || 'Failed to fetch status.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <h2>Audio Analysis Dashboard</h2>
            </header>

            <div className="main-content">
                {/* Left Side: Upload Section */}
                <div className="upload-section-wrapper">
                    <div
                        className={`drop-zone ${isDragging ? 'dragging' : ''} ${selectedFile ? 'has-file' : ''}`}
                        onDragOver={onDragOver}
                        onDragLeave={onDragLeave}
                        onDrop={onDrop}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                            className="hidden-input"
                            accept="audio/*"
                        />

                        <div className="drop-zone-content">
                            {selectedFile ? (
                                <div className="file-info">
                                    <span className="icon">🎵</span>
                                    <span className="filename">{selectedFile.name}</span>
                                    <span className="filesize">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
                                    <button
                                        className="btn-change-file"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            fileInputRef.current?.click();
                                        }}
                                    >
                                        Change File
                                    </button>
                                </div>
                            ) : (
                                <div className="placeholder">
                                    <span className="icon-large">☁️</span>
                                    <p>Drag & Drop Audio here</p>
                                    <span className="browse-text">or click to browse</span>
                                </div>
                            )}
                        </div>
                    </div>

                    <button
                        onClick={uploadFile}
                        disabled={!selectedFile || loading}
                        className={`btn-primary full-width ${loading ? 'loading' : ''}`}
                    >
                        {loading ? 'Processing...' : 'Upload File'}
                    </button>

                    {error && <div className="status-message error">{error}</div>}
                    {message && <div className="status-message success">{message}</div>}
                </div>

                {/* Right Side: Status/Results Card */}
                <div className="status-card">
                    <div className="card-header">
                        <h3>File Status</h3>
                        {fileId ? <p className="status-active">Ready to check</p> : <p className="status-idle">No file uploaded yet</p>}
                    </div>

                    <div className="card-content">
                        <div className="status-actions">
                            <button
                                onClick={getStatus}
                                disabled={!fileId || loading}
                                className="btn-secondary full-width"
                            >
                                Check Current Status
                            </button>
                        </div>

                        {audioData ? (
                            <div className="results-panel">
                                <div className="result-row">
                                    <span className="label">File ID</span>
                                    <code className="value-code">{audioData.file_id}</code>
                                </div>
                                <div className="result-row">
                                    <span className="label">Status</span>
                                    <span className={`status-badge ${audioData.status.toLowerCase()}`}>
                                        {audioData.status}
                                    </span>
                                </div>
                            </div>
                        ) : (
                            <div className="empty-state">
                                <span className="empty-icon">📊</span>
                                <p>Upload a file to see details here</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
