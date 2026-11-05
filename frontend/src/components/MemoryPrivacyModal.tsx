import React, { useEffect, useState } from 'react';
import {
  getMemoryPrivacyStatus,
  pauseMemoryCollection,
  resumeMemoryCollection,
  updatePrivacyPolicy,
  deleteAllPersonalMemories,
  scrubProfileAttribute,
  exportUserMemories,
  MemoryPrivacyStatusResponse,
} from '../api/memory';

interface MemoryPrivacyModalProps {
  isOpen: boolean;
  onClose: () => void;
  token: string;
  workspaceId?: string;
}

export const MemoryPrivacyModal: React.FC<MemoryPrivacyModalProps> = ({
  isOpen,
  onClose,
  token,
  workspaceId,
}) => {
  const [status, setStatus] = useState<MemoryPrivacyStatusResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [selectedRetention, setSelectedRetention] = useState<number | null>(null);

  const fetchStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getMemoryPrivacyStatus(token, workspaceId);
      setStatus(data);
      setSelectedRetention(data.effective_policy.retention_days ?? null);
    } catch (err: any) {
      setError(err?.message || 'Failed to load memory privacy status');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchStatus();
      setMessage(null);
    }
  }, [isOpen, token, workspaceId]);

  if (!isOpen) return null;

  const handleToggleCollection = async () => {
    if (!status) return;
    setActionLoading(true);
    setError(null);
    setMessage(null);
    try {
      if (status.collection_state === 'ENABLED') {
        await pauseMemoryCollection(token);
        setMessage('Memory collection paused. Future conversational turns will not be remembered.');
      } else {
        await resumeMemoryCollection(token);
        setMessage('Memory collection resumed.');
      }
      await fetchStatus();
    } catch (err: any) {
      setError(err?.message || 'Failed to toggle memory collection');
    } finally {
      setActionLoading(false);
    }
  };

  const handleUpdateRetention = async (days: number | null) => {
    setActionLoading(true);
    setError(null);
    setMessage(null);
    try {
      await updatePrivacyPolicy(token, {
        scope_type: 'USER',
        retention_days: days,
      });
      setSelectedRetention(days);
      setMessage(`Retention policy updated to ${days ? `${days} days` : 'Indefinite'}.`);
      await fetchStatus();
    } catch (err: any) {
      setError(err?.message || 'Failed to update retention policy');
    } finally {
      setActionLoading(false);
    }
  };

  const handleScrub = async (section: string, key: string) => {
    if (!confirm(`Are you sure you want to scrub "${section}.${key}"? Supporting memories will be purged and cannot be resurrected.`)) {
      return;
    }
    setActionLoading(true);
    setError(null);
    setMessage(null);
    try {
      const res = await scrubProfileAttribute(token, section, key);
      setMessage(`Attribute "${key}" scrubbed successfully (${res.scrubbed_memory_records} supporting memories purged).`);
      await fetchStatus();
    } catch (err: any) {
      setError(err?.message || 'Failed to scrub attribute');
    } finally {
      setActionLoading(false);
    }
  };

  const handleExport = async () => {
    setActionLoading(true);
    setError(null);
    try {
      const exportData = await exportUserMemories(token, workspaceId);
      const jsonStr = JSON.stringify(exportData, null, 2);
      const blob = new Blob([jsonStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `user_memories_export_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setMessage(`Export downloaded successfully (${exportData.total_memories} memories).`);
    } catch (err: any) {
      setError(err?.message || 'Failed to export memories');
    } finally {
      setActionLoading(false);
    }
  };

  const handleDeleteAll = async () => {
    if (!confirm('CAUTION: This will purge ALL your personal memories, zero out vector embeddings, and reset your living profile. This cannot be undone. Proceed?')) {
      return;
    }
    setActionLoading(true);
    setError(null);
    try {
      const res = await deleteAllPersonalMemories(token, workspaceId);
      setMessage(`All personal memories purged (${res.deleted_count} records removed). Shared knowledge remains intact.`);
      await fetchStatus();
    } catch (err: any) {
      setError(err?.message || 'Failed to delete all memories');
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4 overflow-y-auto font-sans">
      <div className="bg-white border border-gray-200 text-gray-900 rounded-xl shadow-2xl max-w-2xl w-full p-6 relative max-h-[90vh] flex flex-col">
        {/* Header - Minimalist & Clean */}
        <div className="flex items-center justify-between pb-4 border-b border-gray-200">
          <div>
            <h2 className="text-base font-semibold text-gray-900 flex items-center gap-2">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-gray-700">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                <path d="M7 11V7a5 5 0 0 1 10 0v4" />
              </svg>
              <span>Memory Privacy &amp; Retention Controls</span>
            </h2>
            <p className="text-xs text-gray-500 mt-0.5">
              Control memory collection, automatic retention limits, living profile attributes, and export.
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-700 p-1.5 rounded-lg hover:bg-gray-100 transition"
          >
            ✕
          </button>
        </div>

        {/* Alerts */}
        {error && (
          <div className="mt-4 p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-800">
            {error}
          </div>
        )}
        {message && (
          <div className="mt-4 p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-xs text-emerald-800">
            {message}
          </div>
        )}

        {/* Content Body */}
        <div className="overflow-y-auto flex-1 mt-4 space-y-5 pr-1">
          {loading ? (
            <div className="py-12 text-center text-gray-400 text-xs">
              Loading privacy status...
            </div>
          ) : status ? (
            <>
              {/* Collection Status Card */}
              <div className="bg-gray-50/70 border border-gray-200 rounded-lg p-4 flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Collection State:</span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded font-medium ${
                        status.collection_state === 'ENABLED'
                          ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                          : status.collection_state === 'PAUSED'
                          ? 'bg-amber-50 text-amber-800 border border-amber-200'
                          : 'bg-rose-50 text-rose-800 border border-rose-200'
                      }`}
                    >
                      {status.collection_state}
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    {status.collection_state === 'ENABLED'
                      ? 'The system captures durable preferences and context during conversations.'
                      : 'Memory collection is paused. Conversational turns will not be memorized.'}
                  </p>
                  <div className="flex gap-4 mt-2.5 text-xs text-gray-600 font-mono">
                    <div>Active: <span className="text-emerald-700 font-semibold">{status.total_active_memories}</span></div>
                    <div>Expired: <span className="text-amber-700 font-semibold">{status.total_expired_memories}</span></div>
                    <div>Purged: <span className="text-gray-500 font-semibold">{status.total_purged_memories}</span></div>
                  </div>
                </div>

                <button
                  onClick={handleToggleCollection}
                  disabled={actionLoading}
                  className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition ${
                    status.collection_state === 'ENABLED'
                      ? 'bg-white hover:bg-gray-50 text-gray-700 border border-gray-300'
                      : 'bg-gray-900 hover:bg-gray-800 text-white'
                  }`}
                >
                  {status.collection_state === 'ENABLED' ? 'Pause Memory' : 'Resume Memory'}
                </button>
              </div>

              {/* Retention Policy Setting */}
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h3 className="text-xs font-semibold text-gray-900 mb-1">
                  Retention Limit (Auto-Expiration)
                </h3>
                <p className="text-xs text-gray-500 mb-3">
                  Memories older than this limit are automatically expired and vector embeddings cleared.
                </p>
                <div className="grid grid-cols-4 gap-2">
                  {[
                    { label: '30 Days', val: 30 },
                    { label: '90 Days', val: 90 },
                    { label: '365 Days', val: 365 },
                    { label: 'Indefinite', val: null },
                  ].map((opt) => (
                    <button
                      key={String(opt.val)}
                      onClick={() => handleUpdateRetention(opt.val)}
                      disabled={actionLoading}
                      className={`px-3 py-2 text-xs rounded-md border font-medium transition ${
                        selectedRetention === opt.val
                          ? 'bg-gray-900 border-gray-900 text-white'
                          : 'bg-white border-gray-200 text-gray-700 hover:bg-gray-50'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Granular Profile Scrubbing */}
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h3 className="text-xs font-semibold text-gray-900 mb-1">
                  Living Profile Attributes
                </h3>
                <p className="text-xs text-gray-500 mb-3">
                  Selectively remove specific attributes. Supporting memory records are purged with anti-resurrection guarantees.
                </p>
                {status.living_profile_summary &&
                Object.keys(status.living_profile_summary).some(
                  (k) =>
                    typeof status.living_profile_summary[k] === 'object' &&
                    Object.keys(status.living_profile_summary[k] || {}).length > 0
                ) ? (
                  <div className="space-y-3">
                    {['user_preferences', 'work_context', 'stable_technical_preferences'].map(
                      (sec) => {
                        const secObj = status.living_profile_summary[sec] || {};
                        const keys = Object.keys(secObj);
                        if (keys.length === 0) return null;
                        return (
                          <div key={sec} className="bg-gray-50/70 p-3 rounded-lg border border-gray-200">
                            <span className="text-[11px] font-mono text-gray-700 uppercase font-semibold">
                              {sec.replace(/_/g, ' ')}
                            </span>
                            <div className="mt-2 space-y-1.5">
                              {keys.map((k) => (
                                <div
                                  key={k}
                                  className="flex items-center justify-between text-xs py-1 border-b border-gray-200/80 last:border-0"
                                >
                                  <div>
                                    <span className="font-medium text-gray-800">{k}: </span>
                                    <span className="text-gray-500">
                                      {typeof secObj[k] === 'object'
                                        ? JSON.stringify(secObj[k])
                                        : String(secObj[k])}
                                    </span>
                                  </div>
                                  <button
                                    onClick={() => handleScrub(sec, k)}
                                    disabled={actionLoading}
                                    className="text-[10px] text-rose-700 hover:text-rose-800 px-2 py-0.5 rounded bg-rose-50 border border-rose-200 hover:bg-rose-100 transition"
                                  >
                                    Scrub
                                  </button>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      }
                    )}
                  </div>
                ) : (
                  <p className="text-xs text-gray-400 italic">No living profile attributes currently compiled.</p>
                )}
              </div>

              {/* Data Export & Danger Zone */}
              <div className="border-t border-gray-200 pt-4 flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-semibold text-gray-900">Data Portability &amp; Purge</h4>
                  <p className="text-[11px] text-gray-500">Export or permanently clear your personal memories.</p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={handleExport}
                    disabled={actionLoading}
                    className="px-3 py-1.5 text-xs font-medium rounded-md bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 transition"
                  >
                    Export (JSON)
                  </button>
                  <button
                    onClick={handleDeleteAll}
                    disabled={actionLoading}
                    className="px-3 py-1.5 text-xs font-medium rounded-md bg-rose-50 border border-rose-200 hover:bg-rose-100 text-rose-700 transition"
                  >
                    Purge All
                  </button>
                </div>
              </div>
            </>
          ) : null}
        </div>

        {/* Footer */}
        <div className="mt-4 pt-3 border-t border-gray-200 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-xs font-medium rounded-md text-gray-700 transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

