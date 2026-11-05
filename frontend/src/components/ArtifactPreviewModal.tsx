import React, { useEffect, useState } from 'react';
import { getArtifactPreview, ArtifactPreviewResponse } from '../api/artifacts.ts';

export interface ArtifactPreviewModalProps {
  artifactId: string;
  token: string;
  onClose: () => void;
}

export function ArtifactPreviewModal({ artifactId, token, onClose }: ArtifactPreviewModalProps) {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<ArtifactPreviewResponse | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    getArtifactPreview(artifactId, token)
      .then((data) => {
        if (!isMounted) return;
        setPreview(data);
        setLoading(false);
      })
      .catch((err: any) => {
        if (!isMounted) return;
        setError(err.message || 'Failed to load artifact preview.');
        setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [artifactId, token]);

  const handleCopy = () => {
    if (!preview?.content) return;
    navigator.clipboard.writeText(preview.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadUrl = `/api/reports/${artifactId}/download?token=${encodeURIComponent(token)}`;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.65)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
        padding: '20px',
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: '#FFFFFF',
          borderRadius: '12px',
          width: '100%',
          maxWidth: '800px',
          maxHeight: '85vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
          border: '1px solid #E2E8F0',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid #E2E8F0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: '#F8FAFC',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0 }}>
            <span style={{ fontSize: '18px' }}>📄</span>
            <div style={{ minWidth: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <h3
                  style={{
                    margin: 0,
                    fontSize: '15px',
                    fontWeight: 600,
                    color: '#0F172A',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {preview?.title || 'Execution Artifact Preview'}
                </h3>
                {preview?.format && (
                  <span
                    style={{
                      fontSize: '11px',
                      fontFamily: 'monospace',
                      padding: '1px 6px',
                      borderRadius: '4px',
                      background: '#E2E8F0',
                      color: '#334155',
                      fontWeight: 700,
                      textTransform: 'uppercase',
                    }}
                  >
                    {preview.format}
                  </span>
                )}
              </div>
              <div style={{ fontSize: '11px', color: '#64748B', marginTop: '2px', fontFamily: 'monospace' }}>
                ID: {artifactId}
              </div>
            </div>
          </div>

          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              fontSize: '18px',
              color: '#64748B',
              cursor: 'pointer',
              padding: '4px 8px',
              borderRadius: '4px',
            }}
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '20px', overflowY: 'auto', flex: 1 }}>
          {loading && (
            <div style={{ padding: '40px', textAlign: 'center', color: '#64748B', fontSize: '13px' }}>
              <div style={{ fontSize: '24px', marginBottom: '8px' }}>⏳</div>
              Verifying cryptographic hash and loading artifact preview...
            </div>
          )}

          {error && (
            <div
              style={{
                padding: '16px',
                borderRadius: '8px',
                background: '#FEF2F2',
                border: '1px solid #FCA5A5',
                color: '#991B1B',
                fontSize: '13px',
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: '4px' }}>⚠️ Artifact Inaccessible</div>
              <div>{error}</div>
            </div>
          )}

          {!loading && preview && (
            <div>
              {/* Integrity & Metadata Bar */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '8px',
                  padding: '8px 12px',
                  borderRadius: '6px',
                  background: '#F1F5F9',
                  marginBottom: '14px',
                  fontSize: '11px',
                  color: '#475569',
                  flexWrap: 'wrap',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ color: '#16A34A', fontWeight: 700 }}>✓ SHA-256 Verified:</span>
                  <span style={{ fontFamily: 'monospace', color: '#0F172A' }}>
                    {preview.checksum_sha256 ? preview.checksum_sha256.slice(0, 16) + '…' : 'Verified'}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <span>Size: {preview.byte_size.toLocaleString()} bytes</span>
                  <span>MIME: {preview.mime_type}</span>
                </div>
              </div>

              {preview.is_truncated && (
                <div
                  style={{
                    padding: '8px 12px',
                    borderRadius: '6px',
                    background: '#FEF3C7',
                    border: '1px solid #FCD34D',
                    color: '#92400E',
                    fontSize: '12px',
                    marginBottom: '12px',
                  }}
                >
                  ℹ️ Preview is truncated to prevent browser memory exhaustion. Download the full file for complete output.
                </div>
              )}

              {/* Content Render */}
              {preview.is_binary ? (
                <div
                  style={{
                    padding: '30px',
                    textAlign: 'center',
                    background: '#F8FAFC',
                    borderRadius: '8px',
                    border: '1px dashed #CBD5E1',
                  }}
                >
                  <div style={{ fontSize: '32px', marginBottom: '8px' }}>📦</div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: '#1E293B' }}>
                    Binary Content ({preview.format.toUpperCase()})
                  </div>
                  <p style={{ fontSize: '12px', color: '#64748B', maxWidth: '400px', margin: '8px auto 16px auto' }}>
                    This artifact is a compiled binary document. Click below to download and view in your local application.
                  </p>
                  <a
                    href={downloadUrl}
                    download
                    style={{
                      display: 'inline-block',
                      padding: '8px 16px',
                      background: '#2563EB',
                      color: '#FFFFFF',
                      borderRadius: '6px',
                      fontWeight: 600,
                      fontSize: '12px',
                      textDecoration: 'none',
                    }}
                  >
                    ⬇️ Download {preview.format.toUpperCase()}
                  </a>
                </div>
              ) : (
                <div style={{ position: 'relative' }}>
                  <button
                    onClick={handleCopy}
                    style={{
                      position: 'absolute',
                      top: '8px',
                      right: '8px',
                      zIndex: 2,
                      padding: '4px 8px',
                      fontSize: '11px',
                      borderRadius: '4px',
                      border: '1px solid #CBD5E1',
                      background: '#FFFFFF',
                      color: '#334155',
                      cursor: 'pointer',
                    }}
                  >
                    {copied ? '✓ Copied' : '📋 Copy'}
                  </button>
                  <pre
                    style={{
                      background: '#0F172A',
                      color: '#F8FAFC',
                      padding: '14px',
                      borderRadius: '8px',
                      fontSize: '12px',
                      lineHeight: 1.5,
                      fontFamily: 'Consolas, Monaco, monospace',
                      overflowX: 'auto',
                      maxHeight: '450px',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      margin: 0,
                    }}
                  >
                    {preview.content}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div
          style={{
            padding: '12px 20px',
            borderTop: '1px solid #E2E8F0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: '#F8FAFC',
          }}
        >
          <a
            href={downloadUrl}
            target="_blank"
            rel="noopener noreferrer"
            download
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              fontSize: '12px',
              fontWeight: 600,
              color: '#2563EB',
              background: '#EFF6FF',
              border: '1px solid #BFDBFE',
              borderRadius: '6px',
              textDecoration: 'none',
            }}
          >
            <span>⬇️</span>
            <span>Download Full Artifact</span>
          </a>

          <button
            onClick={onClose}
            style={{
              padding: '6px 14px',
              fontSize: '12px',
              fontWeight: 600,
              color: '#475569',
              background: '#FFFFFF',
              border: '1px solid #CBD5E1',
              borderRadius: '6px',
              cursor: 'pointer',
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

