import React, { useState, useEffect, useCallback } from 'react';
import {
  listDocumentVersions,
  getShadowReport,
  publishDocumentVersion,
  rollbackDocumentVersion,
  DocumentVersionItem,
  ShadowReportResponse,
} from '../api/documents';

export interface DocumentVersionHistoryModalProps {
  isOpen?: boolean;
  onClose: () => void;
  documentId: string;
  documentName?: string;
  token?: string;
  userRole?: string;
  onVersionChanged?: () => void;
}

export const DocumentVersionHistoryModal: React.FC<DocumentVersionHistoryModalProps> = ({
  isOpen = true,
  onClose,
  documentId,
  documentName,
  token = '',
  userRole = 'VIEWER',
  onVersionChanged,
}) => {
  const [versions, setVersions] = useState<DocumentVersionItem[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<DocumentVersionItem | null>(null);
  const [shadowReport, setShadowReport] = useState<ShadowReportResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);

  const canMutate = userRole === 'ADMIN' || userRole === 'EDITOR' || userRole === 'OWNER';

  const loadVersions = useCallback(async () => {
    if (!documentId || !token) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await listDocumentVersions(documentId, token);
      setVersions(resp.versions);
      if (resp.versions.length > 0) {
        const active = resp.versions.find((v) => v.is_active) || resp.versions[0];
        setSelectedVersion(active);
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to load document version history.');
    } finally {
      setLoading(false);
    }
  }, [documentId, token]);

  useEffect(() => {
    if (isOpen) {
      loadVersions();
    }
  }, [isOpen, loadVersions]);

  useEffect(() => {
    if (selectedVersion && token) {
      getShadowReport(documentId, selectedVersion.id, token)
        .then((rep) => setShadowReport(rep))
        .catch(() => setShadowReport(null));
    }
  }, [selectedVersion, documentId, token]);

  const handlePublish = async (versionId: string) => {
    if (!token) return;
    setLoading(true);
    setError(null);
    try {
      await publishDocumentVersion(documentId, versionId, token);
      setActionSuccess('Version published successfully.');
      await loadVersions();
      onVersionChanged?.();
    } catch (err: any) {
      setError(err?.message || 'Failed to publish version.');
    } finally {
      setLoading(false);
    }
  };

  const handleRollback = async (targetVersionId: string) => {
    if (!token || !selectedVersion) return;
    setLoading(true);
    setError(null);
    try {
      await rollbackDocumentVersion(documentId, selectedVersion.id, targetVersionId, token);
      setActionSuccess('Document rolled back successfully.');
      await loadVersions();
      onVersionChanged?.();
    } catch (err: any) {
      setError(err?.message || 'Failed to rollback version.');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-4xl max-h-[90vh] flex flex-col bg-slate-900 border border-slate-800 rounded-xl shadow-2xl text-slate-100 overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
          <div>
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <span>Living Article Version History</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 font-mono">
                Milestone 7B
              </span>
            </h2>
            <p className="text-sm text-slate-400 truncate max-w-xl">
              {documentName || documentId}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            ✕
          </button>
        </div>

        {/* Alerts */}
        {error && (
          <div className="px-6 py-3 bg-rose-500/10 border-b border-rose-500/20 text-rose-300 text-sm flex items-center justify-between">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="text-rose-400 hover:text-white text-xs">Dismiss</button>
          </div>
        )}
        {actionSuccess && (
          <div className="px-6 py-3 bg-emerald-500/10 border-b border-emerald-500/20 text-emerald-300 text-sm flex items-center justify-between">
            <span>{actionSuccess}</span>
            <button onClick={() => setActionSuccess(null)} className="text-emerald-400 hover:text-white text-xs">Dismiss</button>
          </div>
        )}

        {/* Content Body */}
        <div className="flex-1 grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-slate-800 overflow-hidden">
          {/* Version List Sidebar */}
          <div className="p-4 overflow-y-auto space-y-2 max-h-[70vh]">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              Document Versions ({versions.length})
            </div>
            {versions.map((ver) => {
              const isSelected = selectedVersion?.id === ver.id;
              return (
                <div
                  key={ver.id}
                  onClick={() => setSelectedVersion(ver)}
                  className={`p-3 rounded-lg border cursor-pointer transition ${
                    isSelected
                      ? 'bg-indigo-600/20 border-indigo-500 text-white'
                      : 'bg-slate-950/30 border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-sm">Version {ver.version_number}</span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                        ver.is_active
                          ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                          : ver.status === 'SUPERSEDED'
                          ? 'bg-slate-800 text-slate-400'
                          : ver.status === 'ROLLED_BACK'
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                          : 'bg-blue-500/20 text-blue-300'
                      }`}
                    >
                      {ver.is_active ? 'ACTIVE' : ver.status}
                    </span>
                  </div>
                  <div className="text-xs text-slate-400">
                    Created: {new Date(ver.created_at).toLocaleString()}
                  </div>
                  {ver.change_summary && (
                    <div className="mt-2 text-[11px] text-slate-400 flex items-center gap-2">
                      <span>Pages: {ver.change_summary.total_pages}</span>
                      <span>•</span>
                      <span className="text-emerald-400">Reused: {ver.change_summary.reused_chunks || 0}</span>
                      <span>•</span>
                      <span className="text-indigo-400">New: {ver.change_summary.created_chunks || 0}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Details & Metrics Area */}
          <div className="md:col-span-2 p-6 overflow-y-auto space-y-6 max-h-[70vh]">
            {selectedVersion ? (
              <>
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-base font-semibold text-white">
                      Version {selectedVersion.version_number} Overview
                    </h3>
                    <p className="text-xs text-slate-400 mt-1 font-mono">
                      Hash: {selectedVersion.source_content_hash.slice(0, 16)}...
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {/* Action buttons */}
                    {canMutate && selectedVersion.status === 'PENDING_APPROVAL' && (
                      <button
                        onClick={() => handlePublish(selectedVersion.id)}
                        disabled={loading}
                        className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium transition shadow disabled:opacity-50"
                      >
                        Publish Version
                      </button>
                    )}
                    {canMutate && !selectedVersion.is_active && selectedVersion.status !== 'ROLLED_BACK' && (
                      <button
                        onClick={() => handleRollback(selectedVersion.id)}
                        disabled={loading}
                        className="px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-xs font-medium transition shadow disabled:opacity-50"
                      >
                        Rollback to This
                      </button>
                    )}
                  </div>
                </div>

                {/* Incremental Metrics Grid */}
                {selectedVersion.change_summary && (
                  <div className="bg-slate-950/40 border border-slate-800 rounded-lg p-4 space-y-3">
                    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Incremental Processing Metrics
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
                      <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
                        <div className="text-lg font-bold text-white">
                          {selectedVersion.change_summary.total_pages || 0}
                        </div>
                        <div className="text-[11px] text-slate-400">Total Pages</div>
                      </div>
                      <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
                        <div className="text-lg font-bold text-emerald-400">
                          {selectedVersion.change_summary.unchanged_pages || 0}
                        </div>
                        <div className="text-[11px] text-slate-400">Unchanged Pages</div>
                      </div>
                      <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
                        <div className="text-lg font-bold text-indigo-400">
                          {selectedVersion.change_summary.modified_pages || 0}
                        </div>
                        <div className="text-[11px] text-slate-400">Modified Pages</div>
                      </div>
                      <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
                        <div className="text-lg font-bold text-cyan-400">
                          {selectedVersion.change_summary.embedding_calls_avoided || 0}
                        </div>
                        <div className="text-[11px] text-slate-400">Embeddings Avoided</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Shadow Comparison Report */}
                <div className="bg-slate-950/40 border border-slate-800 rounded-lg p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Isolated Shadow Comparison Report
                    </h4>
                    {shadowReport && (
                      <span
                        className={`text-xs px-2.5 py-0.5 rounded-full font-medium ${
                          shadowReport.comparison_status === 'EXACT_MATCH'
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                            : shadowReport.comparison_status === 'ACCEPTABLE_DIFFERENCE'
                            ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                            : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                        }`}
                      >
                        {shadowReport.comparison_status}
                      </span>
                    )}
                  </div>
                  {shadowReport ? (
                    <div className="space-y-2 text-xs text-slate-300">
                      <div className="flex items-center gap-2">
                        <span className="text-slate-400">Can Publish:</span>
                        <span className={shadowReport.can_publish ? 'text-emerald-400 font-medium' : 'text-rose-400 font-medium'}>
                          {shadowReport.can_publish ? 'Yes' : 'No'}
                        </span>
                      </div>
                      {shadowReport.reasons && shadowReport.reasons.length > 0 && (
                        <div className="p-2 bg-slate-900 rounded border border-slate-800">
                          <span className="text-slate-400 font-medium">Details: </span>
                          <span>{shadowReport.reasons.join(' • ')}</span>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-xs text-slate-500 italic">
                      No shadow validation report recorded for this version.
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="h-full flex items-center justify-center text-sm text-slate-500">
                Select a document version to view its diff and metrics.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

