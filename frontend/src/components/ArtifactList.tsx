import React, { useState } from 'react';
import { ArtifactMetadata, CoverageReport } from '../api/chat';

/**
 * Clean SVG outline icons matching user's explicit design reference:
 * 1. PDF: Document with folded corner and horizontal lines
 * 2. CSV: 2x2 Table / Grid icon
 * 3. DOCX: Document with folded corner and letter 'T' (Text document)
 * 4. JSON: Document with folded corner and code brackets '{}'
 */
export function PdfDocumentIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <line x1="10" y1="9" x2="8" y2="9" />
    </svg>
  );
}

export function CsvTableIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <rect x="3" y="3" width="18" height="18" rx="4" />
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="10" y1="3" x2="10" y2="21" />
    </svg>
  );
}

export function DocxTextIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="8.5" y1="13" x2="15.5" y2="13" strokeWidth="2.2" />
      <line x1="12" y1="13" x2="12" y2="19" strokeWidth="2.2" />
    </svg>
  );
}

export function JsonCodeIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <path d="M9.5 13.5c-.7 0-1 .3-1 1v.5c0 .7-.4 1-1.2 1 .8 0 1.2.3 1.2 1v.5c0 .7.3 1 1 1" strokeWidth="1.8" />
      <path d="M14.5 13.5c.7 0 1 .3 1 1v.5c0 .7.4 1 1.2 1-.8 0-1.2.3-1.2 1v.5c0 .7-.3 1-1 1" strokeWidth="1.8" />
    </svg>
  );
}

export function DownloadButtonIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

export function PreviewButtonIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

/**
 * Robustly extracts the auth token from props, localStorage, or sessionStorage.
 * Strips any redundant "Bearer " prefix and surrounding quotes.
 */
export function getAuthToken(token?: string): string {
  if (token && typeof token === 'string' && token.trim()) {
    return token.replace(/^Bearer\s+/i, '').replace(/^['"]+|['"]+$/g, '').trim();
  }
  if (typeof window !== 'undefined') {
    for (const key of ['token', 'access_token', 'jwt', 'auth_token', 'rag_token']) {
      const val = localStorage.getItem(key) || sessionStorage.getItem(key);
      if (val && typeof val === 'string' && val.trim()) {
        try {
          const parsed = JSON.parse(val);
          if (typeof parsed === 'string') return parsed.replace(/^Bearer\s+/i, '').trim();
          if (parsed && typeof parsed.token === 'string') return parsed.token.replace(/^Bearer\s+/i, '').trim();
          if (parsed && typeof parsed.access_token === 'string') return parsed.access_token.replace(/^Bearer\s+/i, '').trim();
        } catch {
          return val.replace(/^Bearer\s+/i, '').replace(/^['"]+|['"]+$/g, '').trim();
        }
      }
    }
  }
  return '';
}

export async function downloadArtifactFile(
  artifact: ArtifactMetadata,
  token?: string,
  onStatusChange?: (downloading: boolean, error?: string | null) => void
): Promise<void> {
  if (artifact.error) {
    onStatusChange?.(false, artifact.error);
    return;
  }
  onStatusChange?.(true, null);
  try {
    const authToken = getAuthToken(token);

    let url = artifact.download_url;
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      if (!url.startsWith('/')) {
        url = `/${url}`;
      }
    }

    // Attach token query param as universal fallback for direct downloads
    if (authToken && !url.includes('token=')) {
      url += (url.includes('?') ? '&' : '?') + `token=${encodeURIComponent(authToken)}`;
    }

    const response = await fetch(url, {
      method: 'GET',
      headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
    });

    if (!response.ok) {
      if (response.status === 410) {
        throw new Error('This artifact has expired. Please click "Regenerate".');
      } else if (response.status === 403) {
        throw new Error('Access denied: You are not authorized to download this artifact.');
      } else if (response.status === 404) {
        throw new Error('Artifact file was not found on the server.');
      } else if (response.status === 401) {
        throw new Error('Authentication required: Your session may have expired. Please re-login.');
      } else {
        throw new Error(`Download failed with server status ${response.status}`);
      }
    }

    const blob = await response.blob();
    const blobUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = blobUrl;
    a.download = artifact.filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(blobUrl);
    onStatusChange?.(false, null);
  } catch (err: any) {
    const msg = err.message || 'Download failed';
    onStatusChange?.(false, msg);
    throw err;
  }
}

export function formatBytes(bytes: number): string {
  if (!bytes) return '0 B';
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

export function getArtifactDescription(format: string): string {
  const f = (format || '').toLowerCase();
  switch (f) {
    case 'pdf':
      return 'Official formatted PDF report with executive summary, tables, and verified citations';
    case 'csv':
      return 'Structured raw tabular records in comma-separated spreadsheet format';
    case 'docx':
      return 'Formatted Microsoft Word document report with narrative sections and structured tables';
    case 'json':
      return 'Structured canonical JSON dataset with field-level provenance and verification metadata';
    default:
      return 'Generated downloadable artifact from verified evidence';
  }
}

interface PreviewModalProps {
  artifact: ArtifactMetadata;
  token?: string;
  onClose: () => void;
  onDownload: () => void;
}

export function ArtifactPreviewModal({ artifact, token, onClose, onDownload }: PreviewModalProps) {
  const [loading, setLoading] = useState(true);
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    let active = true;
    const fetchContent = async () => {
      try {
        const authToken = getAuthToken(token);
        let url = artifact.download_url;
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
          if (!url.startsWith('/')) url = `/${url}`;
        }
        if (authToken && !url.includes('token=')) {
          url += (url.includes('?') ? '&' : '?') + `token=${encodeURIComponent(authToken)}`;
        }
        const resp = await fetch(url, {
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        });
        if (!resp.ok) {
          throw new Error(`Failed to load preview (${resp.status})`);
        }
        const text = await resp.text();
        if (active) {
          setContent(text);
          setLoading(false);
        }
      } catch (err: any) {
        if (active) {
          setError(err.message || 'Could not load preview');
          setLoading(false);
        }
      }
    };
    fetchContent();
    return () => {
      active = false;
    };
  }, [artifact.download_url, token]);

  const fmt = (artifact.format || '').toLowerCase();

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        background: 'rgba(15, 23, 42, 0.55)',
        backdropFilter: 'blur(3px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: '#FFFFFF',
          color: '#0F172A',
          border: '1px solid #E2E8F0',
          borderRadius: '12px',
          width: '100%',
          maxWidth: '850px',
          maxHeight: '85vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.05)',
          overflow: 'hidden',
          fontFamily: "var(--font-sans, 'Outfit', system-ui, sans-serif)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '14px 20px',
            borderBottom: '1px solid #E2E8F0',
            background: '#F8FAFC',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span
              style={{
                fontSize: '11px',
                fontWeight: 700,
                padding: '2px 8px',
                borderRadius: '4px',
                background: '#E0F2FE',
                color: '#0369A1',
                textTransform: 'uppercase',
                border: '1px solid #BAE6FD',
              }}
            >
              {artifact.format}
            </span>
            <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--color-ink, #0F172A)' }}>
              {artifact.filename}
            </span>
            <span style={{ fontSize: '11.5px', color: '#64748B' }}>
              ({formatBytes(artifact.size_bytes)})
            </span>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#64748B',
              cursor: 'pointer',
              fontSize: '18px',
              lineHeight: 1,
              padding: '4px 8px',
              borderRadius: '4px',
            }}
            title="Close preview"
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: '20px', overflowY: 'auto', flex: 1, background: '#F8FAFC' }}>
          {loading && (
            <div style={{ textAlign: 'center', padding: '40px', color: '#64748B', fontSize: '13px' }}>
              Loading preview content…
            </div>
          )}
          {error && (
            <div
              style={{
                color: '#B91C1C',
                background: '#FEF2F2',
                border: '1px solid #FCA5A5',
                padding: '12px',
                borderRadius: '6px',
                fontSize: '13px',
              }}
            >
              {error}
            </div>
          )}
          {!loading && !error && content && fmt === 'json' && (
            <pre
              style={{
                margin: 0,
                fontFamily: 'var(--font-mono, monospace)',
                fontSize: '12px',
                lineHeight: 1.55,
                color: '#F8FAFC',
                background: '#0F172A',
                padding: '14px',
                borderRadius: '8px',
                border: '1px solid #334155',
                overflowX: 'auto',
                whiteSpace: 'pre',
              }}
            >
              {(() => {
                try {
                  return JSON.stringify(JSON.parse(content), null, 2);
                } catch {
                  return content;
                }
              })()}
            </pre>
          )}
          {!loading && !error && content && fmt === 'csv' && (
            <div style={{ overflowX: 'auto', background: '#FFFFFF', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
              <table
                style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontSize: '12px',
                  color: '#334155',
                }}
              >
                <tbody>
                  {content.split('\n').filter((l) => l.trim()).map((line, rIdx) => {
                    const cells = line.split(',');
                    return (
                      <tr
                        key={rIdx}
                        style={{
                          background: rIdx === 0 ? '#F1F5F9' : rIdx % 2 === 0 ? '#FFFFFF' : '#FBFCFD',
                          borderBottom: '1px solid #E2E8F0',
                        }}
                      >
                        {cells.map((cell, cIdx) => (
                          <td
                            key={cIdx}
                            style={{
                              padding: '8px 12px',
                              fontWeight: rIdx === 0 ? 600 : 400,
                              color: rIdx === 0 ? '#0F172A' : '#334155',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {cell.trim().replace(/^"|"$/g, '')}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          style={{
            padding: '12px 20px',
            borderTop: '1px solid #E2E8F0',
            background: '#F8FAFC',
            display: 'flex',
            justifyContent: 'flex-end',
            gap: '10px',
          }}
        >
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
              fontFamily: "var(--font-sans, 'Outfit', system-ui, sans-serif)",
            }}
          >
            Close
          </button>
          <button
            onClick={onDownload}
            style={{
              padding: '6px 16px',
              fontSize: '12px',
              fontWeight: 600,
              color: '#FFFFFF',
              background: '#0284C7',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              fontFamily: "var(--font-sans, 'Outfit', system-ui, sans-serif)",
            }}
          >
            <DownloadButtonIcon />
            <span>Download {artifact.format.toUpperCase()}</span>
          </button>
        </div>
      </div>
    </div>
  );
}

export function ArtifactCard({
  artifact,
  token,
  onRegenerate,
}: {
  artifact: ArtifactMetadata;
  token?: string;
  onRegenerate?: (updated: ArtifactMetadata) => void;
}) {
  const [downloading, setDownloading] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const [cardError, setCardError] = useState<string | null>(artifact.error || null);
  const [showPreviewModal, setShowPreviewModal] = useState(false);

  // Soft, beautiful format themes coherent with the website's clean light background & custom icons
  const formatThemes: Record<
    string,
    { badgeBg: string; badgeBorder: string; badgeText: string; iconComponent: React.ReactNode }
  > = {
    pdf: {
      badgeBg: '#FEF2F2',
      badgeBorder: '#FECACA',
      badgeText: '#DC2626',
      iconComponent: <PdfDocumentIcon />,
    },
    csv: {
      badgeBg: '#ECFDF5',
      badgeBorder: '#A7F3D0',
      badgeText: '#059669',
      iconComponent: <CsvTableIcon />,
    },
    docx: {
      badgeBg: '#EFF6FF',
      badgeBorder: '#BFDBFE',
      badgeText: '#2563EB',
      iconComponent: <DocxTextIcon />,
    },
    json: {
      badgeBg: '#FFFBEB',
      badgeBorder: '#FDE68A',
      badgeText: '#D97706',
      iconComponent: <JsonCodeIcon />,
    },
  };

  const fmt = (artifact.format || 'file').toLowerCase();
  const theme = formatThemes[fmt] || {
    badgeBg: '#F1F5F9',
    badgeBorder: '#E2E8F0',
    badgeText: '#475569',
    iconComponent: <PdfDocumentIcon />,
  };

  const description = getArtifactDescription(fmt);
  const canPreview = ['json', 'csv', 'pdf'].includes(fmt);

  const handleDownload = async () => {
    setCardError(null);
    try {
      await downloadArtifactFile(artifact, token, (isDl, err) => {
        setDownloading(isDl);
        if (err) setCardError(err);
      });
    } catch {
      // Handled via callback
    }
  };

  const handlePreviewOrOpen = async () => {
    if (fmt === 'pdf') {
      try {
        const authToken = getAuthToken(token);
        let url = artifact.download_url;
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
          if (!url.startsWith('/')) url = `/${url}`;
        }
        if (authToken && !url.includes('token=')) {
          url += (url.includes('?') ? '&' : '?') + `token=${encodeURIComponent(authToken)}`;
        }
        const resp = await fetch(url, {
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        });
        if (resp.ok) {
          const blob = await resp.blob();
          const blobUrl = window.URL.createObjectURL(blob);
          window.open(blobUrl, '_blank');
        } else {
          handleDownload();
        }
      } catch {
        handleDownload();
      }
    } else {
      setShowPreviewModal(true);
    }
  };

  const handleRegenerate = async () => {
    setRegenerating(true);
    setCardError(null);
    try {
      const authToken = getAuthToken(token);
      let regUrl = `/api/reports/${artifact.id}/regenerate`;
      if (authToken) {
        regUrl += `?token=${encodeURIComponent(authToken)}`;
      }
      const response = await fetch(regUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Regeneration failed: ${response.statusText}`);
      }
      const updated: ArtifactMetadata = await response.json();
      if (onRegenerate) {
        onRegenerate(updated);
      }
    } catch (err: any) {
      setCardError(err.message || 'Regeneration failed');
    } finally {
      setRegenerating(false);
    }
  };

  return (
    <>
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          padding: '13px 16px',
          background: '#FFFFFF',
          border: '1px solid #E2E8F0',
          borderRadius: '10px',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02)',
          gap: '8px',
          transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
          fontFamily: "var(--font-sans, 'Outfit', system-ui, -apple-system, sans-serif)",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.borderColor = '#CBD5E1';
          e.currentTarget.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -2px rgba(0, 0, 0, 0.04)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.borderColor = '#E2E8F0';
          e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02)';
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '14px' }}>
          {/* File details */}
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', minWidth: 0 }}>
            {/* Format pill badge with custom SVG icon */}
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '4px 9px',
                borderRadius: '6px',
                fontSize: '11px',
                fontWeight: 700,
                letterSpacing: '0.04em',
                textTransform: 'uppercase',
                background: theme.badgeBg,
                color: theme.badgeText,
                border: `1px solid ${theme.badgeBorder}`,
                flexShrink: 0,
                marginTop: '1px',
              }}
            >
              {theme.iconComponent}
              <span>{artifact.format}</span>
            </span>

            {/* Filename, description & metadata - Matching website font-sans */}
            <div style={{ minWidth: 0, display: 'flex', flexDirection: 'column', gap: '2px' }}>
              <div
                style={{
                  fontSize: '14px',
                  fontWeight: 600,
                  color: 'var(--color-ink, #1A1A18)',
                  fontFamily: "var(--font-sans, 'Outfit', system-ui, -apple-system, sans-serif)",
                  letterSpacing: '-0.01em',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
                title={artifact.filename}
              >
                {artifact.filename}
              </div>
              <div
                style={{
                  fontSize: '11.5px',
                  color: 'var(--color-ink-muted, #6B6B64)',
                  lineHeight: 1.4,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  fontFamily: "var(--font-sans, 'Outfit', system-ui, -apple-system, sans-serif)",
                }}
              >
                {description}
              </div>
              <div
                style={{
                  fontSize: '11px',
                  color: 'var(--color-ink-muted, #6B6B64)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  marginTop: '1px',
                  fontFamily: "var(--font-sans, 'Outfit', system-ui, -apple-system, sans-serif)",
                }}
              >
                {artifact.size_bytes > 0 && (
                  <span style={{ fontWeight: 500, color: 'var(--color-ink, #374151)' }}>
                    {formatBytes(artifact.size_bytes)}
                  </span>
                )}
                {!artifact.error && !artifact.is_expired && (
                  <span style={{ color: '#16A34A', display: 'inline-flex', alignItems: 'center', gap: '4px', fontWeight: 500 }}>
                    <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#16A34A', display: 'inline-block' }} />
                    Ready for download
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Action buttons */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
            {artifact.error ? (
              <span
                style={{
                  fontSize: '11px',
                  fontWeight: 600,
                  color: '#DC2626',
                  background: '#FEF2F2',
                  border: '1px solid #FECACA',
                  padding: '4px 10px',
                  borderRadius: '6px',
                  fontFamily: "var(--font-sans, 'Outfit', system-ui, -apple-system, sans-serif)",
                }}
              >
                Generation Failed
              </span>
            ) : artifact.is_expired ? (
              <button
                type="button"
                onClick={handleRegenerate}
                disabled={regenerating}
                style={{
                  padding: '5px 12px',
                  fontSize: '11.5px',
                  fontWeight: 600,
                  background: '#F8FAFC',
                  border: '1px solid #CBD5E1',
                  borderRadius: '6px',
                  cursor: regenerating ? 'default' : 'pointer',
                  color: '#1E293B',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '5px',
                  fontFamily: "var(--font-sans, 'Outfit', system-ui, -apple-system, sans-serif)",
                }}
              >
                <span>🔄</span>
                <span>{regenerating ? 'Regenerating…' : 'Regenerate'}</span>
              </button>
            ) : (
              <>
                {canPreview && (
                  <button
                    type="button"
                    onClick={handlePreviewOrOpen}
                    title={`Preview ${artifact.filename}`}
                    style={{
                      padding: '5px 12px',
                      fontSize: '11.5px',
                      fontWeight: 600,
                      background: '#F8FAFC',
                      border: '1px solid #CBD5E1',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      color: '#334155',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '5px',
                      transition: 'all 0.15s ease',
                      fontFamily: "var(--font-sans, 'Outfit', system-ui, -apple-system, sans-serif)",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = '#F1F5F9';
                      e.currentTarget.style.borderColor = '#94A3B8';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = '#F8FAFC';
                      e.currentTarget.style.borderColor = '#CBD5E1';
                    }}
                  >
                    <PreviewButtonIcon />
                    <span>{fmt === 'pdf' ? 'Open' : 'Preview'}</span>
                  </button>
                )}
                <button
                  type="button"
                  onClick={handleDownload}
                  disabled={downloading}
                  style={{
                    padding: '5px 14px',
                    fontSize: '11.5px',
                    fontWeight: 600,
                    background: downloading ? '#0284C7' : '#0284C7',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: downloading ? 'default' : 'pointer',
                    color: '#FFFFFF',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '5px',
                    boxShadow: '0 1px 2px rgba(2, 132, 199, 0.25)',
                    transition: 'background 0.15s ease',
                    fontFamily: "var(--font-sans, 'Outfit', system-ui, -apple-system, sans-serif)",
                  }}
                  onMouseEnter={(e) => {
                    if (!downloading) e.currentTarget.style.background = '#0369A1';
                  }}
                  onMouseLeave={(e) => {
                    if (!downloading) e.currentTarget.style.background = '#0284C7';
                  }}
                >
                  <DownloadButtonIcon />
                  <span>{downloading ? 'Downloading…' : 'Download'}</span>
                </button>
              </>
            )}
          </div>
        </div>

        {cardError && (
          <div
            style={{
              fontSize: '11.5px',
              color: '#B91C1C',
              background: '#FEF2F2',
              border: '1px solid #FCA5A5',
              padding: '6px 10px',
              borderRadius: '6px',
              marginTop: '4px',
              fontFamily: "var(--font-sans, 'Outfit', system-ui, -apple-system, sans-serif)",
            }}
          >
            {cardError}
          </div>
        )}
      </div>

      {showPreviewModal && (
        <ArtifactPreviewModal
          artifact={artifact}
          token={token}
          onClose={() => setShowPreviewModal(false)}
          onDownload={() => {
            setShowPreviewModal(false);
            handleDownload();
          }}
        />
      )}
    </>
  );
}

export function ArtifactList({
  artifacts,
  token,
  onRegenerate,
}: {
  artifacts: ArtifactMetadata[];
  token?: string;
  onRegenerate?: (updated: ArtifactMetadata) => void;
}) {
  if (!artifacts || artifacts.length === 0) return null;

  return (
    <div style={{ marginTop: '16px', marginBottom: '10px' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontFamily: "var(--font-sans, 'Outfit', system-ui, sans-serif)",
          fontSize: '11px',
          letterSpacing: '0.06em',
          color: 'var(--color-ink-muted, #6B6B64)',
          textTransform: 'uppercase',
          fontWeight: 700,
          marginBottom: '10px',
        }}
      >
        <span>📁</span>
        <span>Downloadable Artifacts ({artifacts.length})</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {artifacts.map((art) => (
          <ArtifactCard key={art.id} artifact={art} token={token} onRegenerate={onRegenerate} />
        ))}
      </div>
    </div>
  );
}

export function CoverageNotice({ coverage }: { coverage: CoverageReport }) {
  if (!coverage) return null;
  const isSampled = coverage.retrieval_coverage === 'sampled_evidence';
  return (
    <div
      style={{
        margin: '10px 0',
        padding: '10px 14px',
        borderRadius: '8px',
        background: isSampled ? '#FFFBEB' : '#ECFDF5',
        border: `1px solid ${isSampled ? '#FDE68A' : '#A7F3D0'}`,
        fontSize: '12px',
        color: isSampled ? '#92400E' : '#065F46',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '10px',
        lineHeight: 1.5,
        fontFamily: "var(--font-sans, 'Outfit', system-ui, sans-serif)",
      }}
    >
      <span style={{ fontSize: '14px', marginTop: '1px' }}>{isSampled ? 'ℹ️' : '✓'}</span>
      <div>
        <strong style={{ fontWeight: 600 }}>
          {isSampled ? 'Sample Evidence Notice: ' : 'Census Coverage: '}
        </strong>
        {coverage.disclaimer}
      </div>
    </div>
  );
}
