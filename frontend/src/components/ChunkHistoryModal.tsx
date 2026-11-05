import React, { useState, useEffect, useCallback } from 'react';
import {
  fetchChunkDetail,
  fetchChunkVersions,
  fetchChunkDiff,
  fetchChunkEvidence,
  reportChunkIssue,
  editChunk,
  rollbackChunk,
  ChunkDetailResponse,
  ChunkVersionResponse,
  ChunkDiffResponse,
  ChunkEvidenceResponse,
  QualityCheckFlag,
  QualityReportItem,
} from '../api/chunkCuration';
import {
  parseUnifiedDiff,
  checkCurationPermissions,
  formatChunkHash,
  formatAuditTimestamp,
  isOptimisticConflictError,
  DiffLine,
} from './chunkDiffDefinitions';

export interface ChunkHistoryModalProps {
  isOpen?: boolean;
  onClose: () => void;
  documentId: string;
  chunkId: string;
  token?: string;
  fileName?: string;
  userRole?: string;
  workspaceRole?: string;
  onChunkUpdated?: (newChunk: ChunkDetailResponse) => void;
}

type ActiveTab = 'history' | 'diff' | 'evidence' | 'edit';

export const ChunkHistoryModal: React.FC<ChunkHistoryModalProps> = ({
  isOpen = true,
  onClose,
  documentId,
  chunkId,
  token,
  fileName,
  userRole,
  workspaceRole,
  onChunkUpdated,
}) => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('history');

  // Core Data States
  const [chunk, setChunk] = useState<ChunkDetailResponse | null>(null);
  const [versions, setVersions] = useState<ChunkVersionResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isPermissionDenied, setIsPermissionDenied] = useState(false);

  // Edit Tab State
  const [editText, setEditText] = useState('');
  const [editReason, setEditReason] = useState('');
  const [editLoading, setEditLoading] = useState(false);
  const [editConflictError, setEditConflictError] = useState<string | null>(null);
  const [editSuccessMessage, setEditSuccessMessage] = useState<string | null>(null);

  // Diff Tab State
  const [diffV1, setDiffV1] = useState<number>(1);
  const [diffV2, setDiffV2] = useState<number>(1);
  const [diffData, setDiffData] = useState<ChunkDiffResponse | null>(null);
  const [diffLoading, setDiffLoading] = useState(false);
  const [diffError, setDiffError] = useState<string | null>(null);

  // Rollback Confirmation Dialog State
  const [rollbackTargetVersion, setRollbackTargetVersion] = useState<number | null>(null);
  const [rollbackReason, setRollbackReason] = useState('');
  const [rollbackLoading, setRollbackLoading] = useState(false);
  const [rollbackError, setRollbackError] = useState<string | null>(null);

  // Evidence & Quality State (Milestone 7A.3)
  const [evidence, setEvidence] = useState<ChunkEvidenceResponse | null>(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);

  // Issue Reporting Form State
  const [issueType, setIssueType] = useState('EXTRACT_DEFECT');
  const [issueDescription, setIssueDescription] = useState('');
  const [suggestedRemediation, setSuggestedRemediation] = useState<'MANUAL_CURATE' | 'REINGEST_DOCUMENT' | 'SOURCE_CORRECTION'>('MANUAL_CURATE');
  const [reportingLoading, setReportingLoading] = useState(false);
  const [reportSuccess, setReportSuccess] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  // Permissions evaluation
  const perm = checkCurationPermissions(userRole, workspaceRole, chunk?.is_managed_external);

  // Load Chunk and Versions
  const loadChunkData = useCallback(async () => {
    if (!documentId || !chunkId) return;
    setLoading(true);
    setError(null);
    setIsPermissionDenied(false);
    setEditConflictError(null);

    try {
      const [detailRes, versionsRes] = await Promise.all([
        fetchChunkDetail(documentId, chunkId, token),
        fetchChunkVersions(documentId, chunkId, token),
      ]);
      setChunk(detailRes);
      setVersions(versionsRes);
      setEditText(detailRes.text);

      // Default diff comparison: compare current version with previous if available
      if (detailRes.current_version > 1) {
        setDiffV1(detailRes.current_version - 1);
        setDiffV2(detailRes.current_version);
      } else {
        setDiffV1(1);
        setDiffV2(1);
      }
    } catch (err: any) {
      if (err?.status === 403) {
        setIsPermissionDenied(true);
        setError('Forbidden: You do not have permission to view this knowledge chunk.');
      } else {
        setError(err?.message || 'Failed to load chunk version history.');
      }
    } finally {
      setLoading(false);
    }
  }, [documentId, chunkId, token]);

  useEffect(() => {
    if (isOpen) {
      loadChunkData();
    }
  }, [isOpen, loadChunkData]);

  // Load Evidence & Quality
  const loadEvidence = useCallback(async () => {
    if (!documentId || !chunkId) return;
    setEvidenceLoading(true);
    setEvidenceError(null);
    try {
      const data = await fetchChunkEvidence(documentId, chunkId, token);
      setEvidence(data);
    } catch (err: any) {
      setEvidenceError(err?.message || 'Failed to fetch chunk source evidence.');
    } finally {
      setEvidenceLoading(false);
    }
  }, [documentId, chunkId, token]);

  useEffect(() => {
    if (isOpen && activeTab === 'evidence') {
      loadEvidence();
    }
  }, [isOpen, activeTab, loadEvidence]);

  const handleReportIssue = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!issueDescription.trim()) {
      setReportError('Please provide a description of the defect.');
      return;
    }
    setReportingLoading(true);
    setReportError(null);
    setReportSuccess(null);
    try {
      const res = await reportChunkIssue(
        documentId,
        chunkId,
        {
          issue_type: issueType,
          description: issueDescription.trim(),
          suggested_remediation: suggestedRemediation,
        },
        token
      );
      setReportSuccess(`Quality issue logged successfully (${res.report.report_id}). Total reports: ${res.total_reports}`);
      setIssueDescription('');
      await loadEvidence();
    } catch (err: any) {
      setReportError(err?.message || 'Failed to submit quality issue report.');
    } finally {
      setReportingLoading(false);
    }
  };

  // Load Diff when Diff tab is active or version selectors change
  const loadDiff = useCallback(async () => {
    if (!documentId || !chunkId || diffV1 < 1 || diffV2 < 1) return;
    setDiffLoading(true);
    setDiffError(null);
    try {
      const res = await fetchChunkDiff(documentId, chunkId, diffV1, diffV2, token);
      setDiffData(res);
    } catch (err: any) {
      setDiffError(err?.message || 'Failed to compute diff between selected versions.');
    } finally {
      setDiffLoading(false);
    }
  }, [documentId, chunkId, diffV1, diffV2, token]);

  useEffect(() => {
    if (isOpen && activeTab === 'diff') {
      loadDiff();
    }
  }, [isOpen, activeTab, loadDiff]);

  // Handle Edit Submit
  const handleEditSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chunk || !perm.canEdit) return;

    if (!editText.trim()) {
      alert('Chunk text cannot be empty.');
      return;
    }

    setEditLoading(true);
    setEditConflictError(null);
    setEditSuccessMessage(null);

    try {
      const res = await editChunk(
        documentId,
        chunkId,
        {
          new_text: editText.trim(),
          base_version: chunk.current_version,
          edit_reason: editReason.trim() || undefined,
        },
        token
      );

      setChunk(res.chunk);
      setEditSuccessMessage(`Successfully updated to version ${res.chunk.current_version}!`);
      setEditReason('');
      if (onChunkUpdated) onChunkUpdated(res.chunk);

      // Refresh versions
      const updatedVersions = await fetchChunkVersions(documentId, chunkId, token);
      setVersions(updatedVersions);
      setDiffV1(res.chunk.current_version - 1);
      setDiffV2(res.chunk.current_version);
    } catch (err: any) {
      if (isOptimisticConflictError(err)) {
        setEditConflictError(
          err.message || 'Conflict: Chunk has been modified by another user. Please reload the latest version.'
        );
      } else {
        setEditConflictError(err.message || 'Failed to update chunk.');
      }
    } finally {
      setEditLoading(false);
    }
  };

  // Handle Rollback
  const handleConfirmRollback = async () => {
    if (!chunk || rollbackTargetVersion === null || !perm.canRollback) return;

    setRollbackLoading(true);
    setRollbackError(null);

    try {
      const res = await rollbackChunk(
        documentId,
        chunkId,
        {
          target_version: rollbackTargetVersion,
          rollback_reason: rollbackReason.trim() || `Restored version ${rollbackTargetVersion}`,
        },
        token
      );

      setChunk(res.chunk);
      setEditText(res.chunk.text);
      if (onChunkUpdated) onChunkUpdated(res.chunk);

      // Close confirmation dialog
      setRollbackTargetVersion(null);
      setRollbackReason('');

      // Refresh versions
      const updatedVersions = await fetchChunkVersions(documentId, chunkId, token);
      setVersions(updatedVersions);

      // Switch to history tab and show success
      setActiveTab('history');
    } catch (err: any) {
      setRollbackError(err?.message || 'Rollback failed.');
    } finally {
      setRollbackLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.65)',
        backdropFilter: 'blur(4px)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '16px',
        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget && !rollbackTargetVersion) onClose();
      }}
    >
      <div
        style={{
          backgroundColor: '#FFFFFF',
          borderRadius: '12px',
          width: '100%',
          maxWidth: '920px',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          border: '1px solid #E2E8F0',
          overflow: 'hidden',
        }}
      >
        {/* MODAL HEADER */}
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid #E2E8F0',
            backgroundColor: '#F8FAFC',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              <span
                style={{
                  fontSize: '11px',
                  fontFamily: 'ui-monospace, monospace',
                  padding: '2px 8px',
                  borderRadius: '4px',
                  backgroundColor: '#E2E8F0',
                  color: '#334155',
                  fontWeight: 600,
                }}
              >
                Chunk #{chunk ? chunk.chunk_index : '—'}
              </span>

              {chunk && (
                <span
                  style={{
                    fontSize: '11px',
                    fontFamily: 'ui-monospace, monospace',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    backgroundColor: '#DCFCE7',
                    color: '#166534',
                    fontWeight: 700,
                  }}
                >
                  v{chunk.current_version} (Active)
                </span>
              )}

              {chunk?.is_managed_external && (
                <span
                  style={{
                    fontSize: '11px',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    backgroundColor: '#FEF3C7',
                    color: '#92400E',
                    fontWeight: 600,
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                  }}
                >
                  🛡️ Twenty CRM Managed
                </span>
              )}
            </div>

            <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#0F172A', margin: '4px 0 0 0' }}>
              {fileName || 'Knowledge Chunk Curation & Provenance'}
            </h3>
            <div style={{ fontSize: '11.5px', color: '#64748B', marginTop: '2px' }}>
              ID: <code style={{ color: '#475569' }}>{chunkId}</code>
            </div>
          </div>

          <button
            onClick={onClose}
            aria-label="Close modal"
            style={{
              background: 'transparent',
              border: 'none',
              fontSize: '18px',
              color: '#64748B',
              cursor: 'pointer',
              padding: '6px 10px',
              borderRadius: '6px',
            }}
          >
            ✕
          </button>
        </div>

        {/* TAB NAVIGATION */}
        <div
          style={{
            display: 'flex',
            borderBottom: '1px solid #E2E8F0',
            backgroundColor: '#FFFFFF',
            padding: '0 20px',
          }}
        >
          <button
            onClick={() => setActiveTab('history')}
            style={{
              padding: '12px 16px',
              fontSize: '13px',
              fontWeight: 600,
              color: activeTab === 'history' ? '#0F172A' : '#64748B',
              borderBottom: activeTab === 'history' ? '2px solid #0F172A' : '2px solid transparent',
              background: 'none',
              borderTop: 'none',
              borderLeft: 'none',
              borderRight: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            🕒 Version History ({versions.length || 1})
          </button>

          <button
            onClick={() => setActiveTab('diff')}
            style={{
              padding: '12px 16px',
              fontSize: '13px',
              fontWeight: 600,
              color: activeTab === 'diff' ? '#0F172A' : '#64748B',
              borderBottom: activeTab === 'diff' ? '2px solid #0F172A' : '2px solid transparent',
              background: 'none',
              borderTop: 'none',
              borderLeft: 'none',
              borderRight: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            ⚖️ Compare & Diff
          </button>

          <button
            onClick={() => setActiveTab('evidence')}
            style={{
              padding: '12px 16px',
              fontSize: '13px',
              fontWeight: 600,
              color: activeTab === 'evidence' ? '#0F172A' : '#64748B',
              borderBottom: activeTab === 'evidence' ? '2px solid #0F172A' : '2px solid transparent',
              background: 'none',
              borderTop: 'none',
              borderLeft: 'none',
              borderRight: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            🔍 Source Evidence & Quality
          </button>

          <button
            onClick={() => setActiveTab('edit')}
            style={{
              padding: '12px 16px',
              fontSize: '13px',
              fontWeight: 600,
              color: activeTab === 'edit' ? '#0F172A' : '#64748B',
              borderBottom: activeTab === 'edit' ? '2px solid #0F172A' : '2px solid transparent',
              background: 'none',
              borderTop: 'none',
              borderLeft: 'none',
              borderRight: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            ✏️ Curate / Edit Chunk
          </button>
        </div>

        {/* MODAL BODY */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
          {/* LOADING STATE */}
          {loading && (
            <div style={{ textAlign: 'center', padding: '40px', color: '#64748B', fontSize: '13px' }}>
              <div
                style={{
                  display: 'inline-block',
                  width: '24px',
                  height: '24px',
                  border: '3px solid #E2E8F0',
                  borderTopColor: '#0F172A',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite',
                  marginBottom: '10px',
                }}
              />
              <div>Loading chunk history and audit trail…</div>
            </div>
          )}

          {/* ERROR / PERMISSION DENIED STATE */}
          {!loading && (error || isPermissionDenied) && (
            <div
              style={{
                padding: '16px',
                borderRadius: '8px',
                backgroundColor: isPermissionDenied ? '#FEF2F2' : '#FFF1F2',
                border: '1px solid #FECDD3',
                color: '#9F1239',
                fontSize: '13px',
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: '4px' }}>
                {isPermissionDenied ? 'Permission Denied' : 'Unable to Load Chunk Data'}
              </div>
              <div>{error}</div>
              {!isPermissionDenied && (
                <button
                  onClick={loadChunkData}
                  style={{
                    marginTop: '10px',
                    padding: '6px 12px',
                    backgroundColor: '#9F1239',
                    color: '#FFFFFF',
                    border: 'none',
                    borderRadius: '6px',
                    fontSize: '12px',
                    cursor: 'pointer',
                  }}
                >
                  Retry
                </button>
              )}
            </div>
          )}

          {/* TAB 1: VERSION HISTORY & ROLLBACK */}
          {!loading && !error && activeTab === 'history' && (
            <div>
              {chunk?.is_managed_external && (
                <div
                  style={{
                    padding: '10px 14px',
                    backgroundColor: '#FEF3C7',
                    border: '1px solid #FDE68A',
                    borderRadius: '6px',
                    color: '#92400E',
                    fontSize: '12px',
                    marginBottom: '16px',
                  }}
                >
                  ℹ️ <strong>External CRM Governance:</strong> This chunk is continuously synchronized from Twenty CRM. Manual rollback and edits are restricted to avoid sync overwrite.
                </div>
              )}

              {versions.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '30px', color: '#94A3B8', fontSize: '13px' }}>
                  No historical revisions recorded. Chunk is in its initial ingestion state (v1).
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {versions.slice().reverse().map((ver) => {
                    const isCurrent = chunk?.current_version === ver.version_number;

                    return (
                      <div
                        key={ver.id || ver.version_number}
                        style={{
                          border: isCurrent ? '2px solid #0F172A' : '1px solid #E2E8F0',
                          borderRadius: '8px',
                          padding: '14px 16px',
                          backgroundColor: isCurrent ? '#F8FAFC' : '#FFFFFF',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span
                              style={{
                                fontSize: '12px',
                                fontFamily: 'ui-monospace, monospace',
                                fontWeight: 700,
                                padding: '2px 8px',
                                borderRadius: '4px',
                                backgroundColor: isCurrent ? '#0F172A' : '#E2E8F0',
                                color: isCurrent ? '#FFFFFF' : '#334155',
                              }}
                            >
                              v{ver.version_number}
                            </span>

                            {isCurrent && (
                              <span style={{ fontSize: '11px', fontWeight: 600, color: '#166534', backgroundColor: '#DCFCE7', padding: '1px 6px', borderRadius: '4px' }}>
                                Active
                              </span>
                            )}

                            {ver.is_rollback && (
                              <span
                                style={{
                                  fontSize: '11px',
                                  fontWeight: 600,
                                  color: '#6D28D9',
                                  backgroundColor: '#EDE9FE',
                                  padding: '1px 6px',
                                  borderRadius: '4px',
                                }}
                              >
                                🔄 Rollback from v{ver.reverted_from_version}
                              </span>
                            )}

                            <span style={{ fontSize: '11.5px', color: '#64748B' }}>
                              {formatAuditTimestamp(ver.created_at)}
                            </span>
                          </div>

                          {/* Rollback Action Button */}
                          {!isCurrent && perm.canRollback && (
                            <button
                              onClick={() => {
                                setRollbackTargetVersion(ver.version_number);
                                setRollbackReason(`Restored content from version ${ver.version_number}`);
                              }}
                              style={{
                                padding: '4px 10px',
                                fontSize: '11.5px',
                                fontWeight: 600,
                                color: '#0F172A',
                                backgroundColor: '#F1F5F9',
                                border: '1px solid #CBD5E1',
                                borderRadius: '4px',
                                cursor: 'pointer',
                              }}
                            >
                              Restore to v{ver.version_number}
                            </button>
                          )}
                        </div>

                        {/* Audit Reason & Metadata */}
                        <div style={{ marginTop: '8px', fontSize: '12.5px', color: '#1E293B' }}>
                          {ver.edit_reason ? (
                            <span><strong>Reason:</strong> {ver.edit_reason}</span>
                          ) : (
                            <span style={{ color: '#94A3B8', fontStyle: 'italic' }}>No edit reason recorded</span>
                          )}
                        </div>

                        {/* Snapshot Technical Metadata */}
                        <div
                          style={{
                            marginTop: '8px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '12px',
                            fontSize: '10.5px',
                            color: '#64748B',
                            fontFamily: 'ui-monospace, monospace',
                          }}
                        >
                          <span>Hash: {formatChunkHash(ver.chunk_hash)}</span>
                          <span>Model: {ver.embedding_model} ({ver.embedding_dim}d)</span>
                          {ver.edited_by && <span>Editor: {ver.edited_by.slice(0, 8)}…</span>}
                        </div>

                        {/* Text Preview */}
                        <div
                          style={{
                            marginTop: '10px',
                            padding: '8px 12px',
                            backgroundColor: '#FFFFFF',
                            border: '1px solid #E2E8F0',
                            borderRadius: '6px',
                            fontSize: '12px',
                            lineHeight: 1.5,
                            maxHeight: '120px',
                            overflowY: 'auto',
                            whiteSpace: 'pre-wrap',
                            color: '#334155',
                          }}
                        >
                          {ver.text_after}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* TAB 2: DIFF VIEWER */}
          {!loading && !error && activeTab === 'diff' && (
            <div>
              {/* Version Selectors */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '12px 16px',
                  backgroundColor: '#F8FAFC',
                  borderRadius: '8px',
                  border: '1px solid #E2E8F0',
                  marginBottom: '16px',
                  flexWrap: 'wrap',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <label htmlFor="diff-base" style={{ fontSize: '12px', fontWeight: 600, color: '#334155' }}>
                    Base Version:
                  </label>
                  <select
                    id="diff-base"
                    value={diffV1}
                    onChange={(e) => setDiffV1(parseInt(e.target.value, 10))}
                    style={{
                      padding: '4px 8px',
                      borderRadius: '4px',
                      border: '1px solid #CBD5E1',
                      fontSize: '12px',
                    }}
                  >
                    {versions.map((v) => (
                      <option key={v.version_number} value={v.version_number}>
                        v{v.version_number} {chunk?.current_version === v.version_number ? '(Active)' : ''}
                      </option>
                    ))}
                  </select>
                </div>

                <span style={{ color: '#94A3B8', fontWeight: 600 }}>→</span>

                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <label htmlFor="diff-target" style={{ fontSize: '12px', fontWeight: 600, color: '#334155' }}>
                    Compared Version:
                  </label>
                  <select
                    id="diff-target"
                    value={diffV2}
                    onChange={(e) => setDiffV2(parseInt(e.target.value, 10))}
                    style={{
                      padding: '4px 8px',
                      borderRadius: '4px',
                      border: '1px solid #CBD5E1',
                      fontSize: '12px',
                    }}
                  >
                    {versions.map((v) => (
                      <option key={v.version_number} value={v.version_number}>
                        v{v.version_number} {chunk?.current_version === v.version_number ? '(Active)' : ''}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Diff Metric Badges */}
                {diffData && (
                  <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', fontSize: '11px', fontWeight: 600 }}>
                    <span style={{ color: '#166534', backgroundColor: '#DCFCE7', padding: '2px 8px', borderRadius: '4px' }}>
                      +{diffData.lines_added} lines
                    </span>
                    <span style={{ color: '#991B1B', backgroundColor: '#FEE2E2', padding: '2px 8px', borderRadius: '4px' }}>
                      -{diffData.lines_removed} lines
                    </span>
                    <span style={{ color: '#334155', backgroundColor: '#E2E8F0', padding: '2px 8px', borderRadius: '4px' }}>
                      {diffData.characters_changed} chars changed
                    </span>
                  </div>
                )}
              </div>

              {diffLoading && (
                <div style={{ textAlign: 'center', padding: '30px', color: '#64748B', fontSize: '12px' }}>
                  Computing diff between v{diffV1} and v{diffV2}…
                </div>
              )}

              {diffError && (
                <div style={{ padding: '12px', backgroundColor: '#FEF2F2', border: '1px solid #FECDD3', borderRadius: '6px', color: '#991B1B', fontSize: '12px' }}>
                  {diffError}
                </div>
              )}

              {/* Render Diff Lines */}
              {!diffLoading && !diffError && diffData && (
                <div>
                  {diffV1 === diffV2 || !diffData.diff_patch.trim() ? (
                    <div style={{ textAlign: 'center', padding: '30px', color: '#64748B', fontSize: '13px' }}>
                      Identical content. No text differences between version {diffV1} and version {diffV2}.
                    </div>
                  ) : (
                    <div
                      style={{
                        backgroundColor: '#F8FAFC',
                        border: '1px solid #E2E8F0',
                        borderRadius: '6px',
                        overflowX: 'auto',
                        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                        fontSize: '12px',
                        lineHeight: 1.6,
                      }}
                    >
                      {parseUnifiedDiff(diffData.diff_patch).map((line, idx) => {
                        let bgColor = 'transparent';
                        let textColor = '#334155';
                        let prefix = ' ';

                        if (line.type === 'addition') {
                          bgColor = '#DCFCE7';
                          textColor = '#14532D';
                          prefix = '+';
                        } else if (line.type === 'deletion') {
                          bgColor = '#FEE2E2';
                          textColor = '#7F1D1D';
                          prefix = '-';
                        } else if (line.type === 'header') {
                          bgColor = '#E2E8F0';
                          textColor = '#475569';
                          prefix = '@';
                        }

                        return (
                          <div
                            key={idx}
                            style={{
                              display: 'flex',
                              backgroundColor: bgColor,
                              color: textColor,
                              padding: '2px 8px',
                              borderBottom: '1px solid rgba(226, 232, 240, 0.4)',
                            }}
                          >
                            <span
                              style={{
                                width: '36px',
                                userSelect: 'none',
                                color: '#94A3B8',
                                textAlign: 'right',
                                paddingRight: '8px',
                              }}
                            >
                              {line.oldLineNumber || ''}
                            </span>
                            <span
                              style={{
                                width: '36px',
                                userSelect: 'none',
                                color: '#94A3B8',
                                textAlign: 'right',
                                paddingRight: '12px',
                              }}
                            >
                              {line.newLineNumber || ''}
                            </span>
                            <span style={{ width: '16px', userSelect: 'none', fontWeight: 700 }}>
                              {prefix}
                            </span>
                            <span style={{ flex: 1, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                              {line.content}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* TAB 3: CURATE / EDIT CHUNK */}
          {!loading && !error && activeTab === 'edit' && (
            <div>
              {/* Permission Alert Banner */}
              {!perm.canEdit && (
                <div
                  style={{
                    padding: '12px 16px',
                    backgroundColor: '#FEF3C7',
                    border: '1px solid #FDE68A',
                    borderRadius: '8px',
                    color: '#92400E',
                    fontSize: '12.5px',
                    marginBottom: '16px',
                  }}
                >
                  🔒 <strong>Edit Restricted:</strong> {perm.reason}
                </div>
              )}

              {/* Optimistic Concurrency 409 Conflict Banner */}
              {editConflictError && (
                <div
                  style={{
                    padding: '14px 16px',
                    backgroundColor: '#FFFBEB',
                    border: '1px solid #FCD34D',
                    borderRadius: '8px',
                    color: '#B45309',
                    fontSize: '13px',
                    marginBottom: '16px',
                  }}
                >
                  <div style={{ fontWeight: 700, marginBottom: '4px' }}>
                    ⚠️ Concurrency Conflict (HTTP 409)
                  </div>
                  <div>{editConflictError}</div>
                  <button
                    type="button"
                    onClick={loadChunkData}
                    style={{
                      marginTop: '8px',
                      padding: '6px 12px',
                      backgroundColor: '#B45309',
                      color: '#FFFFFF',
                      border: 'none',
                      borderRadius: '4px',
                      fontSize: '12px',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                  >
                    Reload Latest Version
                  </button>
                </div>
              )}

              {/* Success Banner */}
              {editSuccessMessage && (
                <div
                  style={{
                    padding: '12px 16px',
                    backgroundColor: '#ECFDF5',
                    border: '1px solid #A7F3D0',
                    borderRadius: '8px',
                    color: '#065F46',
                    fontSize: '12.5px',
                    marginBottom: '16px',
                  }}
                >
                  ✓ {editSuccessMessage}
                </div>
              )}

              <form onSubmit={handleEditSubmit}>
                <div style={{ marginBottom: '16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <label htmlFor="edit-chunk-text" style={{ fontSize: '13px', fontWeight: 600, color: '#0F172A' }}>
                      Chunk Text Content
                    </label>
                    <span style={{ fontSize: '11px', color: '#64748B', fontFamily: 'ui-monospace, monospace' }}>
                      Base Version Lock: v{chunk?.current_version} · {editText.length} characters
                    </span>
                  </div>
                  <textarea
                    id="edit-chunk-text"
                    disabled={!perm.canEdit || editLoading}
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    rows={8}
                    style={{
                      width: '100%',
                      padding: '12px',
                      borderRadius: '6px',
                      border: '1px solid #CBD5E1',
                      fontSize: '13px',
                      lineHeight: 1.5,
                      fontFamily: 'Inter, system-ui, sans-serif',
                      boxSizing: 'border-box',
                      backgroundColor: !perm.canEdit ? '#F8FAFC' : '#FFFFFF',
                      color: !perm.canEdit ? '#64748B' : '#0F172A',
                    }}
                    placeholder="Enter revised chunk text…"
                  />
                </div>

                <div style={{ marginBottom: '20px' }}>
                  <label htmlFor="edit-reason" style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#0F172A', marginBottom: '6px' }}>
                    Reason for Edit (Audit Log)
                  </label>
                  <input
                    id="edit-reason"
                    type="text"
                    disabled={!perm.canEdit || editLoading}
                    value={editReason}
                    onChange={(e) => setEditReason(e.target.value)}
                    maxLength={500}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      borderRadius: '6px',
                      border: '1px solid #CBD5E1',
                      fontSize: '13px',
                      boxSizing: 'border-box',
                      backgroundColor: !perm.canEdit ? '#F8FAFC' : '#FFFFFF',
                    }}
                    placeholder="e.g. Updated liability indemnity clause per Q4 legal contract review"
                  />
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '10px' }}>
                  <button
                    type="button"
                    onClick={onClose}
                    style={{
                      padding: '8px 16px',
                      borderRadius: '6px',
                      border: '1px solid #CBD5E1',
                      backgroundColor: '#FFFFFF',
                      color: '#475569',
                      fontSize: '13px',
                      fontWeight: 500,
                      cursor: 'pointer',
                    }}
                  >
                    Cancel
                  </button>

                  <button
                    type="submit"
                    disabled={!perm.canEdit || editLoading || editText.trim() === chunk?.text}
                    style={{
                      padding: '8px 18px',
                      borderRadius: '6px',
                      border: 'none',
                      backgroundColor: perm.canEdit && !editLoading && editText.trim() !== chunk?.text ? '#0F172A' : '#94A3B8',
                      color: '#FFFFFF',
                      fontSize: '13px',
                      fontWeight: 600,
                      cursor: perm.canEdit && !editLoading && editText.trim() !== chunk?.text ? 'pointer' : 'default',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                    }}
                  >
                    {editLoading ? 'Re-embedding & Saving…' : 'Save & Increment Version'}
                  </button>
                </div>
              </form>
            </div>
          )}

          {/* TAB 4: SOURCE EVIDENCE & QUALITY INSPECTOR (Milestone 7A.3) */}
          {activeTab === 'evidence' && !loading && !error && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {evidenceLoading && !evidence && (
                <div style={{ textAlign: 'center', padding: '30px', color: '#64748B' }}>
                  Loading source evidence and diagnostics…
                </div>
              )}

              {evidenceError && (
                <div style={{ padding: '12px', backgroundColor: '#FEF2F2', border: '1px solid #FECDD3', borderRadius: '8px', color: '#991B1B', fontSize: '13px' }}>
                  {evidenceError}
                </div>
              )}

              {evidence && (
                <>
                  {/* CARD 1: SOURCE PROVENANCE & AVAILABILITY */}
                  <div
                    style={{
                      border: '1px solid #E2E8F0',
                      borderRadius: '8px',
                      padding: '16px',
                      backgroundColor: '#F8FAFC',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
                      <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#0F172A', margin: 0 }}>
                        📂 Source Document & Ingestion Provenance
                      </h4>
                      {/* Honest Source Availability State */}
                      {evidence.source_available ? (
                        <span
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '4px',
                            backgroundColor: '#DCFCE7',
                            color: '#166534',
                            padding: '3px 10px',
                            borderRadius: '6px',
                            fontSize: '11.5px',
                            fontWeight: 600,
                          }}
                        >
                          ✓ Source file accessible on storage
                        </span>
                      ) : (
                        <span
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '4px',
                            backgroundColor: '#FEF2F2',
                            color: '#991B1B',
                            border: '1px solid #FECDD3',
                            padding: '3px 10px',
                            borderRadius: '6px',
                            fontSize: '11.5px',
                            fontWeight: 600,
                          }}
                        >
                          ⚠️ Source file unavailable ({evidence.source_availability_reason || 'Missing from host storage'})
                        </span>
                      )}
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px', fontSize: '12px' }}>
                      <div>
                        <div style={{ color: '#64748B', fontWeight: 500 }}>Original File Name</div>
                        <div style={{ fontWeight: 600, color: '#1E293B', wordBreak: 'break-all' }}>{evidence.file_name}</div>
                      </div>
                      <div>
                        <div style={{ color: '#64748B', fontWeight: 500 }}>MIME Type</div>
                        <div style={{ fontWeight: 600, color: '#1E293B' }}>{evidence.mime_type}</div>
                      </div>
                      <div>
                        <div style={{ color: '#64748B', fontWeight: 500 }}>Page Location</div>
                        <div style={{ fontWeight: 600, color: '#1E293B' }}>
                          {evidence.page_number !== null ? `Page ${evidence.page_number}` : 'Unpaginated / Not extracted'}
                        </div>
                      </div>
                      <div>
                        <div style={{ color: '#64748B', fontWeight: 500 }}>Section Heading</div>
                        <div style={{ fontWeight: 600, color: '#1E293B' }}>
                          {evidence.section_title ? `${evidence.section_ordinal ? evidence.section_ordinal + ' ' : ''}${evidence.section_title}` : 'Top-level / No Section'}
                        </div>
                      </div>
                    </div>

                    {evidence.file_path && (
                      <div style={{ marginTop: '10px', fontSize: '11.5px', color: '#64748B' }}>
                        Physical Path: <code style={{ color: '#334155', backgroundColor: '#EDE9FE', padding: '1px 5px', borderRadius: '3px' }}>{evidence.file_path}</code>
                      </div>
                    )}

                    {evidence.source_url && (
                      <div style={{ marginTop: '6px', fontSize: '11.5px' }}>
                        Source URL:{' '}
                        <a href={evidence.source_url} target="_blank" rel="noreferrer" style={{ color: '#2563EB', textDecoration: 'underline' }}>
                          {evidence.source_url}
                        </a>
                      </div>
                    )}
                  </div>

                  {/* CARD 2: ORIGINAL INGESTION VS CURATED TEXT */}
                  <div
                    style={{
                      border: '1px solid #E2E8F0',
                      borderRadius: '8px',
                      padding: '16px',
                      backgroundColor: '#FFFFFF',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                      <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#0F172A', margin: 0 }}>
                        📝 Content Provenance: Original Ingestion vs Active Curated State
                      </h4>
                      {evidence.is_curated && (
                        <button
                          type="button"
                          onClick={() => {
                            setDiffV1(1);
                            setDiffV2(evidence.current_version);
                            setActiveTab('diff');
                          }}
                          style={{
                            padding: '4px 10px',
                            borderRadius: '4px',
                            backgroundColor: '#F1F5F9',
                            border: '1px solid #CBD5E1',
                            fontSize: '11.5px',
                            color: '#334155',
                            cursor: 'pointer',
                            fontWeight: 600,
                          }}
                        >
                          ⚖️ Open in Diff Viewer
                        </button>
                      )}
                    </div>

                    {evidence.is_curated ? (
                      <div>
                        <div style={{ padding: '8px 12px', backgroundColor: '#EFF6FF', border: '1px solid #BFDBFE', borderRadius: '6px', fontSize: '12px', color: '#1E40AF', marginBottom: '12px' }}>
                          ⚡ <strong>Curated Content Drift</strong>: This chunk has been edited (Active Version is <strong>v{evidence.current_version}</strong>). The original parsed text from initial ingestion (v1) is preserved below for comparison.
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
                          <div>
                            <div style={{ fontSize: '11px', fontWeight: 700, color: '#64748B', textTransform: 'uppercase', marginBottom: '4px' }}>
                              Original Ingested Text (v1)
                            </div>
                            <div
                              style={{
                                padding: '10px',
                                backgroundColor: '#F8FAFC',
                                border: '1px solid #E2E8F0',
                                borderRadius: '6px',
                                fontSize: '12px',
                                maxHeight: '180px',
                                overflowY: 'auto',
                                whiteSpace: 'pre-wrap',
                                fontFamily: 'ui-monospace, monospace',
                                color: '#475569',
                              }}
                            >
                              {evidence.original_text}
                            </div>
                          </div>

                          <div>
                            <div style={{ fontSize: '11px', fontWeight: 700, color: '#166534', textTransform: 'uppercase', marginBottom: '4px' }}>
                              Active Curated Text (v{evidence.current_version})
                            </div>
                            <div
                              style={{
                                padding: '10px',
                                backgroundColor: '#F0FDF4',
                                border: '1px solid #BBF7D0',
                                borderRadius: '6px',
                                fontSize: '12px',
                                maxHeight: '180px',
                                overflowY: 'auto',
                                whiteSpace: 'pre-wrap',
                                fontFamily: 'ui-monospace, monospace',
                                color: '#14532D',
                              }}
                            >
                              {evidence.current_text}
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div style={{ padding: '8px 12px', backgroundColor: '#F0FDF4', border: '1px solid #BBF7D0', borderRadius: '6px', fontSize: '12px', color: '#166534', marginBottom: '10px' }}>
                          ✓ <strong>Unmodified Ingestion State</strong>: Active text is identical to initial parsed ingestion (v1).
                        </div>
                        <div
                          style={{
                            padding: '10px',
                            backgroundColor: '#F8FAFC',
                            border: '1px solid #E2E8F0',
                            borderRadius: '6px',
                            fontSize: '12px',
                            maxHeight: '140px',
                            overflowY: 'auto',
                            whiteSpace: 'pre-wrap',
                            fontFamily: 'ui-monospace, monospace',
                            color: '#334155',
                          }}
                        >
                          {evidence.current_text}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* CARD 3: DETERMINISTIC QUALITY DIAGNOSTICS */}
                  <div
                    style={{
                      border: '1px solid #E2E8F0',
                      borderRadius: '8px',
                      padding: '16px',
                      backgroundColor: '#FFFFFF',
                    }}
                  >
                    <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#0F172A', margin: '0 0 4px 0' }}>
                      🔬 Deterministic Knowledge-Quality Diagnostics
                    </h4>
                    <div style={{ fontSize: '11.5px', color: '#64748B', marginBottom: '12px' }}>
                      Fast, substantiated structural checks supported by ingestion metadata and text structure.
                    </div>

                    {evidence.quality_flags.length === 0 ? (
                      <div style={{ padding: '10px 14px', backgroundColor: '#F0FDF4', border: '1px solid #BBF7D0', borderRadius: '6px', fontSize: '12.5px', color: '#166534' }}>
                        ✓ No structural defects or parser anomalies detected on this chunk.
                      </div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {evidence.quality_flags.map((flag) => {
                          const isErr = flag.severity === 'ERROR';
                          const isWarn = flag.severity === 'WARNING';
                          const bg = isErr ? '#FEF2F2' : isWarn ? '#FFFBEB' : '#EFF6FF';
                          const border = isErr ? '#FECDD3' : isWarn ? '#FDE68A' : '#BFDBFE';
                          const textCol = isErr ? '#991B1B' : isWarn ? '#92400E' : '#1E40AF';

                          return (
                            <div
                              key={flag.flag_id}
                              style={{
                                padding: '10px 12px',
                                backgroundColor: bg,
                                border: `1px solid ${border}`,
                                borderRadius: '6px',
                                display: 'flex',
                                alignItems: 'flex-start',
                                justifyContent: 'space-between',
                                gap: '10px',
                                flexWrap: 'wrap',
                              }}
                            >
                              <div style={{ flex: 1, minWidth: '220px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
                                  <span style={{ fontSize: '10px', fontWeight: 700, padding: '1px 5px', borderRadius: '3px', backgroundColor: border, color: textCol }}>
                                    {flag.severity}
                                  </span>
                                  <strong style={{ fontSize: '12px', color: textCol }}>{flag.flag_id}</strong>
                                </div>
                                <div style={{ fontSize: '12px', color: '#334155', lineHeight: 1.4 }}>
                                  {flag.message}
                                </div>
                              </div>

                              <div style={{ alignSelf: 'center' }}>
                                <span
                                  style={{
                                    fontSize: '11px',
                                    padding: '3px 8px',
                                    borderRadius: '4px',
                                    backgroundColor: '#FFFFFF',
                                    border: '1px solid #CBD5E1',
                                    color: '#475569',
                                    fontWeight: 600,
                                  }}
                                >
                                  Recommended: {flag.suggested_remediation === 'MANUAL_CURATE' ? '✏️ Curate Chunk' : flag.suggested_remediation === 'REINGEST_DOCUMENT' ? '🔄 Re-ingest Doc' : '🌐 Source Correction'}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  {/* CARD 4: REPORT DEFECT & SUGGEST REMEDIATION */}
                  <div
                    style={{
                      border: '1px solid #E2E8F0',
                      borderRadius: '8px',
                      padding: '16px',
                      backgroundColor: '#F8FAFC',
                    }}
                  >
                    <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#0F172A', margin: '0 0 4px 0' }}>
                      🚩 Report Knowledge Quality Defect
                    </h4>
                    <div style={{ fontSize: '11.5px', color: '#64748B', marginBottom: '12px' }}>
                      Flag suspicious extractions, OCR noise, or factual anomalies for review without altering source files.
                    </div>

                    {reportSuccess && (
                      <div style={{ padding: '8px 12px', backgroundColor: '#F0FDF4', border: '1px solid #BBF7D0', borderRadius: '6px', color: '#166534', fontSize: '12px', marginBottom: '12px' }}>
                        {reportSuccess}
                      </div>
                    )}

                    {reportError && (
                      <div style={{ padding: '8px 12px', backgroundColor: '#FEF2F2', border: '1px solid #FECDD3', borderRadius: '6px', color: '#991B1B', fontSize: '12px', marginBottom: '12px' }}>
                        {reportError}
                      </div>
                    )}

                    {/* Form */}
                    <form onSubmit={handleReportIssue} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
                        <div>
                          <label htmlFor="issue-type-select" style={{ display: 'block', fontSize: '11.5px', fontWeight: 600, color: '#334155', marginBottom: '4px' }}>
                            Issue Category:
                          </label>
                          <select
                            id="issue-type-select"
                            value={issueType}
                            onChange={(e) => setIssueType(e.target.value)}
                            style={{
                              width: '100%',
                              padding: '6px 8px',
                              borderRadius: '6px',
                              border: '1px solid #CBD5E1',
                              fontSize: '12px',
                              backgroundColor: '#FFFFFF',
                            }}
                          >
                            <option value="EXTRACT_DEFECT">Malformed / Corrupted Extraction</option>
                            <option value="OCR_NOISE">OCR Noise / Garbled Characters</option>
                            <option value="METADATA_MISSING">Missing Page or Section Location</option>
                            <option value="FACTUAL_ERROR">Inaccurate / Contradictory Information</option>
                            <option value="OUTDATED">Stale or Superseded Knowledge</option>
                            <option value="OTHER">Other Knowledge Defect</option>
                          </select>
                        </div>

                        <div>
                          <label htmlFor="remediation-select" style={{ display: 'block', fontSize: '11.5px', fontWeight: 600, color: '#334155', marginBottom: '4px' }}>
                            Suggested Remediation Path:
                          </label>
                          <select
                            id="remediation-select"
                            value={suggestedRemediation}
                            onChange={(e) => setSuggestedRemediation(e.target.value as any)}
                            style={{
                              width: '100%',
                              padding: '6px 8px',
                              borderRadius: '6px',
                              border: '1px solid #CBD5E1',
                              fontSize: '12px',
                              backgroundColor: '#FFFFFF',
                            }}
                          >
                            <option value="MANUAL_CURATE">✏️ Manual Chunk Override (Chunk Curation)</option>
                            <option value="REINGEST_DOCUMENT">🔄 Re-ingest Document (Parser/OCR re-run)</option>
                            <option value="SOURCE_CORRECTION">🌐 Source Correction (Origin System / CRM)</option>
                          </select>
                        </div>
                      </div>

                      <div>
                        <label htmlFor="issue-desc" style={{ display: 'block', fontSize: '11.5px', fontWeight: 600, color: '#334155', marginBottom: '4px' }}>
                          Defect Description & Details:
                        </label>
                        <textarea
                          id="issue-desc"
                          rows={2}
                          value={issueDescription}
                          onChange={(e) => setIssueDescription(e.target.value)}
                          placeholder="e.g. Page 4 table header is shifted; columns do not match original PDF..."
                          style={{
                            width: '100%',
                            padding: '8px 10px',
                            borderRadius: '6px',
                            border: '1px solid #CBD5E1',
                            fontSize: '12px',
                            fontFamily: 'inherit',
                            boxSizing: 'border-box',
                          }}
                        />
                      </div>

                      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                        <button
                          type="submit"
                          disabled={reportingLoading || !issueDescription.trim()}
                          style={{
                            padding: '6px 14px',
                            borderRadius: '6px',
                            border: 'none',
                            backgroundColor: reportingLoading || !issueDescription.trim() ? '#94A3B8' : '#DC2626',
                            color: '#FFFFFF',
                            fontSize: '12px',
                            fontWeight: 600,
                            cursor: reportingLoading || !issueDescription.trim() ? 'default' : 'pointer',
                          }}
                        >
                          {reportingLoading ? 'Submitting…' : '🚩 Submit Quality Report'}
                        </button>
                      </div>
                    </form>

                    {/* Historical Reports on this chunk */}
                    {evidence.quality_reports && evidence.quality_reports.length > 0 && (
                      <div style={{ marginTop: '16px', borderTop: '1px solid #E2E8F0', paddingTop: '12px' }}>
                        <div style={{ fontSize: '12px', fontWeight: 600, color: '#475569', marginBottom: '8px' }}>
                          📋 Prior Quality Reports ({evidence.quality_reports.length})
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                          {evidence.quality_reports.map((rep) => (
                            <div
                              key={rep.report_id}
                              style={{
                                padding: '8px 10px',
                                backgroundColor: '#FFFFFF',
                                border: '1px solid #E2E8F0',
                                borderRadius: '6px',
                                fontSize: '11.5px',
                              }}
                            >
                              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#64748B', marginBottom: '3px' }}>
                                <span><strong>{rep.issue_type}</strong> reported by {rep.reported_by}</span>
                                <span>{new Date(rep.reported_at).toLocaleDateString()}</span>
                              </div>
                              <div style={{ color: '#1E293B' }}>{rep.description}</div>
                              <div style={{ marginTop: '4px', fontSize: '10.5px', color: '#475569' }}>
                                Remediation: <code>{rep.suggested_remediation}</code> | Status: <span style={{ fontWeight: 600, color: '#D97706' }}>{rep.status}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* ROLLBACK CONFIRMATION MODAL OVERLAY */}
        {rollbackTargetVersion !== null && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              backgroundColor: 'rgba(15, 23, 42, 0.75)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '20px',
              zIndex: 10000,
            }}
          >
            <div
              style={{
                backgroundColor: '#FFFFFF',
                borderRadius: '10px',
                padding: '24px',
                maxWidth: '460px',
                width: '100%',
                boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
                border: '1px solid #CBD5E1',
              }}
            >
              <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#0F172A', margin: '0 0 10px 0' }}>
                Confirm Version Rollback
              </h4>

              <p style={{ fontSize: '13px', color: '#475569', lineHeight: 1.5, margin: '0 0 16px 0' }}>
                You are restoring this chunk to the content of <strong>Version {rollbackTargetVersion}</strong>.
                This will create a new forward-moving <strong>Version {chunk ? chunk.current_version + 1 : ''}</strong> while preserving the full immutable audit trail.
              </p>

              {rollbackError && (
                <div style={{ padding: '8px 12px', backgroundColor: '#FEF2F2', border: '1px solid #FECDD3', borderRadius: '6px', color: '#991B1B', fontSize: '12px', marginBottom: '12px' }}>
                  {rollbackError}
                </div>
              )}

              <div style={{ marginBottom: '16px' }}>
                <label htmlFor="rollback-reason" style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#334155', marginBottom: '4px' }}>
                  Rollback Explanation:
                </label>
                <input
                  id="rollback-reason"
                  type="text"
                  value={rollbackReason}
                  onChange={(e) => setRollbackReason(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 10px',
                    borderRadius: '6px',
                    border: '1px solid #CBD5E1',
                    fontSize: '12.5px',
                    boxSizing: 'border-box',
                  }}
                  placeholder="e.g. Accidental change; reverting to approved version"
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                <button
                  type="button"
                  disabled={rollbackLoading}
                  onClick={() => setRollbackTargetVersion(null)}
                  style={{
                    padding: '6px 14px',
                    borderRadius: '6px',
                    border: '1px solid #CBD5E1',
                    backgroundColor: '#FFFFFF',
                    color: '#475569',
                    fontSize: '12.5px',
                    cursor: 'pointer',
                  }}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  disabled={rollbackLoading}
                  onClick={handleConfirmRollback}
                  style={{
                    padding: '6px 16px',
                    borderRadius: '6px',
                    border: 'none',
                    backgroundColor: '#DC2626',
                    color: '#FFFFFF',
                    fontSize: '12.5px',
                    fontWeight: 600,
                    cursor: rollbackLoading ? 'default' : 'pointer',
                  }}
                >
                  {rollbackLoading ? 'Restoring…' : 'Confirm & Restore'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChunkHistoryModal;
