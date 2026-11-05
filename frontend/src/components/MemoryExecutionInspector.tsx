import React, { useState, useEffect } from 'react';
import {
  getMemoryOverview,
  listMemoryJobs,
  listMemoryRelationships,
  listDreamingRuns,
  triggerDreamingRun,
  getEvaluationMetrics,
  triggerEvaluationRun,
  MemoryOverview,
  MemoryProcessingJobRecord,
  MemoryRelationshipRecord,
  MemoryDreamingRunRecord,
  MemoryEvalMetricRecord,
} from '../api/memory';

interface MemoryExecutionInspectorProps {
  token: string;
  isOpen: boolean;
  onClose: () => void;
}

export const MemoryExecutionInspector: React.FC<MemoryExecutionInspectorProps> = ({
  token,
  isOpen,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'relationships' | 'dreaming' | 'jobs' | 'evaluation'>('overview');
  const [loading, setLoading] = useState<boolean>(false);
  const [actionLoading, setActionLoading] = useState<boolean>(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const [overview, setOverview] = useState<MemoryOverview | null>(null);
  const [jobs, setJobs] = useState<MemoryProcessingJobRecord[]>([]);
  const [relationships, setRelationships] = useState<MemoryRelationshipRecord[]>([]);
  const [dreamingRuns, setDreamingRuns] = useState<MemoryDreamingRunRecord[]>([]);
  const [evalMetrics, setEvalMetrics] = useState<MemoryEvalMetricRecord[]>([]);

  const [dreamThreshold, setDreamThreshold] = useState<number>(3);

  useEffect(() => {
    if (isOpen && token) {
      loadTabData(activeTab);
    }
  }, [isOpen, activeTab, token]);

  const loadTabData = async (tab: typeof activeTab) => {
    setLoading(true);
    setMessage(null);
    try {
      if (tab === 'overview') {
        const data = await getMemoryOverview(token);
        setOverview(data);
      } else if (tab === 'jobs') {
        const data = await listMemoryJobs(token, 50);
        setJobs(data);
      } else if (tab === 'relationships') {
        const data = await listMemoryRelationships(token, 100);
        setRelationships(data);
      } else if (tab === 'dreaming') {
        const data = await listDreamingRuns(token, 50);
        setDreamingRuns(data);
      } else if (tab === 'evaluation') {
        const data = await getEvaluationMetrics(token, 50);
        setEvalMetrics(data);
      }
    } catch (err: any) {
      setMessage({ type: 'error', text: err?.message || 'Failed to load inspector data.' });
    } finally {
      setLoading(false);
    }
  };

  const handleTriggerDreaming = async () => {
    setActionLoading(true);
    setMessage(null);
    try {
      const res = await triggerDreamingRun(token, dreamThreshold);
      setMessage({
        type: 'success',
        text: `Dreaming cycle completed (${res.run_id}): ${res.clusters_consolidated} cluster(s) consolidated, ${res.memories_consolidated} memories superseded.`,
      });
      loadTabData('dreaming');
    } catch (err: any) {
      setMessage({ type: 'error', text: err?.message || 'Failed to trigger dreaming cycle.' });
    } finally {
      setActionLoading(false);
    }
  };

  const handleTriggerEvaluation = async () => {
    setActionLoading(true);
    setMessage(null);
    try {
      const res = await triggerEvaluationRun(token);
      setMessage({
        type: 'success',
        text: `Evaluation complete (${res.eval_run_id}): ${res.passed_scenarios}/${res.total_scenarios} passed (${(res.pass_rate * 100).toFixed(1)}%) in ${res.duration_seconds}s.`,
      });
      loadTabData('evaluation');
    } catch (err: any) {
      setMessage({ type: 'error', text: err?.message || 'Failed to execute evaluation.' });
    } finally {
      setActionLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4 font-sans">
      <div className="flex h-[88vh] w-full max-w-5xl flex-col rounded-xl bg-white border border-gray-200 shadow-2xl text-gray-900 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4 bg-white">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gray-100 flex items-center justify-center text-gray-700 font-semibold border border-gray-200">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z" />
                <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z" />
              </svg>
            </div>
            <div>
              <h2 className="text-sm font-semibold text-gray-900 tracking-tight">Memory &amp; Autonomous Dreaming Inspector</h2>
              <p className="text-xs text-gray-500">Structured memory types, consolidation graph, dreaming ledger &amp; quality metrics</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-700 transition-colors"
            title="Close"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Tabs Bar */}
        <div className="flex border-b border-gray-200 px-6 bg-gray-50/70">
          {[
            { id: 'overview', label: 'Overview & Living Profile' },
            { id: 'relationships', label: 'Consolidation Graph' },
            { id: 'dreaming', label: 'Autonomous Dreaming Runs' },
            { id: 'jobs', label: 'Background Capture Jobs' },
            { id: 'evaluation', label: 'Quality & Evaluation' },
          ].map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as typeof activeTab)}
                className={`border-b-2 px-4 py-3 text-xs font-medium transition-colors ${
                  isActive
                    ? 'border-gray-900 text-gray-900 font-semibold'
                    : 'border-transparent text-gray-500 hover:text-gray-900 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Notifications */}
        {message && (
          <div
            className={`mx-6 mt-4 rounded-lg px-4 py-2.5 text-xs flex items-center justify-between ${
              message.type === 'success'
                ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                : 'bg-rose-50 text-rose-800 border border-rose-200'
            }`}
          >
            <div className="flex items-center gap-2">
              <span>{message.text}</span>
              {message.type === 'error' && (
                <button
                  type="button"
                  onClick={() => loadTabData(activeTab)}
                  className="underline font-semibold ml-2 hover:text-rose-900 cursor-pointer"
                >
                  Retry
                </button>
              )}
            </div>
            <button
              onClick={() => setMessage(null)}
              className="text-gray-400 hover:text-gray-600 text-xs ml-3"
            >
              ✕
            </button>
          </div>
        )}

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 bg-white">
          {loading ? (
            <div className="flex h-64 flex-col items-center justify-center text-gray-400 text-xs gap-2">
              <div className="w-5 h-5 border-2 border-gray-300 border-t-gray-700 rounded-full animate-spin" />
              <span>Loading telemetry...</span>
            </div>
          ) : (
            <>
              {/* TAB 1: OVERVIEW */}
              {activeTab === 'overview' && overview && (
                <div className="space-y-6">
                  {/* Stats Grid */}
                  <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                    <div className="rounded-lg bg-gray-50/70 p-4 border border-gray-200">
                      <div className="text-[11px] font-medium uppercase tracking-wider text-gray-500">Active Memories</div>
                      <div className="mt-1 text-2xl font-semibold text-gray-900">{overview.active_memories}</div>
                      <div className="text-[11px] text-gray-500 mt-0.5">{overview.total_memories} total lifetime</div>
                    </div>
                    <div className="rounded-lg bg-gray-50/70 p-4 border border-gray-200">
                      <div className="text-[11px] font-medium uppercase tracking-wider text-gray-500">Relationships</div>
                      <div className="mt-1 text-2xl font-semibold text-gray-900">{overview.total_relationships}</div>
                      <div className="text-[11px] text-gray-500 mt-0.5">dedup, conflict, temporal links</div>
                    </div>
                    <div className="rounded-lg bg-gray-50/70 p-4 border border-gray-200">
                      <div className="text-[11px] font-medium uppercase tracking-wider text-gray-500">Dreaming Runs</div>
                      <div className="mt-1 text-2xl font-semibold text-gray-900">{overview.total_dreaming_runs}</div>
                      <div className="text-[11px] text-gray-500 mt-0.5">{overview.last_dreaming_run ? 'Consolidated recently' : 'No runs yet'}</div>
                    </div>
                    <div className="rounded-lg bg-gray-50/70 p-4 border border-gray-200">
                      <div className="text-[11px] font-medium uppercase tracking-wider text-gray-500">Structured Types</div>
                      <div className="mt-1 flex flex-wrap gap-1 text-[11px]">
                        {Object.entries(overview.by_type || {}).length === 0 ? (
                          <span className="text-gray-400">None yet</span>
                        ) : (
                          Object.entries(overview.by_type || {}).map(([type, count]) => (
                            <span key={type} className="rounded bg-white px-2 py-0.5 text-gray-700 border border-gray-200 font-medium">
                              {type}: {count}
                            </span>
                          ))
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Living Profile Sections */}
                  <div className="rounded-xl border border-gray-200 bg-gray-50/40 p-5">
                    <h3 className="mb-3 text-xs font-semibold text-gray-900 flex items-center justify-between">
                      <span>Compiled Living Profile (Deterministic Ingestion Projection)</span>
                      <span className="text-[11px] text-gray-500 font-normal">
                        Memory version count: {overview.living_profile_summary?.memory_version_count || 0}
                      </span>
                    </h3>
                    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                      <div className="rounded-lg bg-white p-3 border border-gray-200 shadow-sm">
                        <div className="text-[11px] font-semibold text-gray-700 uppercase tracking-wider mb-2">User Preferences</div>
                        <pre className="text-xs text-gray-600 overflow-x-auto bg-gray-50 p-2.5 rounded border border-gray-100 font-mono">
                          {JSON.stringify(overview.living_profile_summary?.user_preferences || {}, null, 2)}
                        </pre>
                      </div>
                      <div className="rounded-lg bg-white p-3 border border-gray-200 shadow-sm">
                        <div className="text-[11px] font-semibold text-gray-700 uppercase tracking-wider mb-2">Work Context &amp; Milestones</div>
                        <pre className="text-xs text-gray-600 overflow-x-auto bg-gray-50 p-2.5 rounded border border-gray-100 font-mono">
                          {JSON.stringify(overview.living_profile_summary?.work_context || {}, null, 2)}
                        </pre>
                      </div>
                      <div className="rounded-lg bg-white p-3 border border-gray-200 shadow-sm">
                        <div className="text-[11px] font-semibold text-gray-700 uppercase tracking-wider mb-2">Stable Technical Preferences</div>
                        <pre className="text-xs text-gray-600 overflow-x-auto bg-gray-50 p-2.5 rounded border border-gray-100 font-mono">
                          {JSON.stringify(overview.living_profile_summary?.stable_technical_preferences || {}, null, 2)}
                        </pre>
                      </div>
                      <div className="rounded-lg bg-white p-3 border border-gray-200 shadow-sm">
                        <div className="text-[11px] font-semibold text-gray-700 uppercase tracking-wider mb-2">Procedural Conventions</div>
                        <pre className="text-xs text-gray-600 overflow-x-auto bg-gray-50 p-2.5 rounded border border-gray-100 font-mono">
                          {JSON.stringify(overview.living_profile_summary?.procedural_conventions || {}, null, 2)}
                        </pre>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 2: RELATIONSHIPS */}
              {activeTab === 'relationships' && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-gray-500">
                      Consolidation Graph: Discovered duplicate, complementary, conflicting, and temporal relationships.
                    </p>
                    <button
                      onClick={() => loadTabData('relationships')}
                      className="rounded border border-gray-300 bg-white px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 font-medium transition-colors"
                    >
                      Refresh Graph
                    </button>
                  </div>
                  {relationships.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-gray-300 p-12 text-center text-xs text-gray-500 bg-gray-50/40">
                      No relationship links recorded yet. Run an autonomous Dreaming cycle or chat to establish links.
                    </div>
                  ) : (
                    <div className="overflow-x-auto rounded-lg border border-gray-200">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-gray-50 text-gray-600 font-medium border-b border-gray-200">
                          <tr>
                            <th className="p-3">Type</th>
                            <th className="p-3">Source Memory</th>
                            <th className="p-3">Target Memory</th>
                            <th className="p-3">Confidence</th>
                            <th className="p-3">Reason / Details</th>
                            <th className="p-3">Recorded At</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {relationships.map((rel) => {
                            let badgeStyle = 'bg-gray-100 text-gray-700 border-gray-200';
                            if (rel.relationship_type === 'DUPLICATE') badgeStyle = 'bg-blue-50 text-blue-700 border-blue-200';
                            if (rel.relationship_type === 'SUPERSEDES') badgeStyle = 'bg-purple-50 text-purple-700 border-purple-200';
                            if (rel.relationship_type === 'CONTRADICTS') badgeStyle = 'bg-rose-50 text-rose-700 border-rose-200';
                            if (rel.relationship_type === 'COMPLEMENTS') badgeStyle = 'bg-emerald-50 text-emerald-700 border-emerald-200';

                            return (
                              <tr key={rel.id} className="hover:bg-gray-50/80 transition-colors">
                                <td className="p-3">
                                  <span className={`rounded border px-2 py-0.5 text-[10px] font-semibold ${badgeStyle}`}>
                                    {rel.relationship_type}
                                  </span>
                                </td>
                                <td className="p-3 font-mono text-[11px] text-gray-500">{rel.source_memory_id.slice(0, 8)}...</td>
                                <td className="p-3 font-mono text-[11px] text-gray-500">{rel.target_memory_id.slice(0, 8)}...</td>
                                <td className="p-3 text-gray-700 font-medium">{((rel.confidence || 1.0) * 100).toFixed(0)}%</td>
                                <td className="p-3 text-gray-600 max-w-xs truncate">{rel.reason || '-'}</td>
                                <td className="p-3 text-gray-500">{rel.created_at ? new Date(rel.created_at).toLocaleTimeString() : '-'}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* TAB 3: DREAMING RUNS */}
              {activeTab === 'dreaming' && (
                <div className="space-y-5">
                  {/* Trigger Box */}
                  <div className="rounded-lg border border-gray-200 bg-gray-50/60 p-4 flex items-center justify-between">
                    <div>
                      <h4 className="text-xs font-semibold text-gray-900">Trigger Autonomous Dreaming Cycle</h4>
                      <p className="text-xs text-gray-500 mt-0.5">
                        Consolidates active memories into cohesive summaries when clusters meet the threshold.
                      </p>
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-600 flex items-center gap-1.5">
                        Threshold:
                        <input
                          type="number"
                          min="2"
                          max="20"
                          value={dreamThreshold}
                          onChange={(e) => setDreamThreshold(Number(e.target.value))}
                          className="w-14 rounded border border-gray-300 bg-white px-2 py-1 text-xs text-gray-900 text-center font-medium"
                        />
                      </label>
                      <button
                        onClick={handleTriggerDreaming}
                        disabled={actionLoading}
                        className="rounded-lg bg-gray-900 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-gray-800 disabled:opacity-50 transition-colors shadow-sm"
                      >
                        {actionLoading ? 'Consolidating...' : 'Run Consolidation Now'}
                      </button>
                    </div>
                  </div>

                  {/* Runs Table */}
                  {dreamingRuns.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-gray-300 p-12 text-center text-xs text-gray-500 bg-gray-50/40">
                      No dreaming runs recorded yet. Click "Run Consolidation Now" to execute your first cycle.
                    </div>
                  ) : (
                    <div className="overflow-x-auto rounded-lg border border-gray-200">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-gray-50 text-gray-600 font-medium border-b border-gray-200">
                          <tr>
                            <th className="p-3">Run ID</th>
                            <th className="p-3">Status</th>
                            <th className="p-3">Trigger</th>
                            <th className="p-3">Inspected</th>
                            <th className="p-3">Duplicates</th>
                            <th className="p-3">Conflicts</th>
                            <th className="p-3">Created</th>
                            <th className="p-3">Superseded</th>
                            <th className="p-3">Duration</th>
                            <th className="p-3">Started</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {dreamingRuns.map((r) => (
                            <tr key={r.id} className="hover:bg-gray-50/80 transition-colors">
                              <td className="p-3 font-mono text-[11px] text-gray-700">{r.run_id}</td>
                              <td className="p-3">
                                <span
                                  className={`rounded border px-2 py-0.5 text-[10px] font-semibold ${
                                    r.status === 'COMPLETED'
                                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                                      : r.status === 'FAILED'
                                      ? 'bg-rose-50 text-rose-700 border-rose-200'
                                      : 'bg-amber-50 text-amber-700 border-amber-200'
                                  }`}
                                >
                                  {r.status}
                                </span>
                              </td>
                              <td className="p-3 text-gray-600 capitalize">{r.trigger_type}</td>
                              <td className="p-3 text-gray-700">{r.memories_inspected}</td>
                              <td className="p-3 text-gray-700">{r.duplicates_detected}</td>
                              <td className="p-3 text-gray-700">{r.conflicts_detected}</td>
                              <td className="p-3 font-semibold text-emerald-700">{r.new_memories_created}</td>
                              <td className="p-3 text-gray-700">{r.memories_superseded}</td>
                              <td className="p-3 text-gray-500">{r.duration_seconds ? `${r.duration_seconds.toFixed(2)}s` : '-'}</td>
                              <td className="p-3 text-gray-500">{new Date(r.started_at).toLocaleTimeString()}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* TAB 4: BACKGROUND JOBS */}
              {activeTab === 'jobs' && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-gray-500">
                      Durable background outbox jobs enqueued during normal chat turns.
                    </p>
                    <button
                      onClick={() => loadTabData('jobs')}
                      className="rounded border border-gray-300 bg-white px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 font-medium transition-colors"
                    >
                      Refresh Jobs
                    </button>
                  </div>
                  {jobs.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-gray-300 p-12 text-center text-xs text-gray-500 bg-gray-50/40">
                      No background jobs recorded yet. Normal user messages enqueue capture jobs automatically.
                    </div>
                  ) : (
                    <div className="overflow-x-auto rounded-lg border border-gray-200">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-gray-50 text-gray-600 font-medium border-b border-gray-200">
                          <tr>
                            <th className="p-3">Job ID</th>
                            <th className="p-3">Status</th>
                            <th className="p-3">Trigger</th>
                            <th className="p-3">Attempts</th>
                            <th className="p-3">Source Snippet</th>
                            <th className="p-3">Error / Note</th>
                            <th className="p-3">Enqueued At</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {jobs.map((j) => (
                            <tr key={j.id} className="hover:bg-gray-50/80 transition-colors">
                              <td className="p-3 font-mono text-[11px] text-gray-500">{j.id.slice(0, 8)}...</td>
                              <td className="p-3">
                                <span
                                  className={`rounded border px-2 py-0.5 text-[10px] font-semibold ${
                                    j.status === 'SUCCEEDED'
                                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                                      : j.status === 'PENDING'
                                      ? 'bg-amber-50 text-amber-700 border-amber-200'
                                      : j.status.startsWith('SKIPPED')
                                      ? 'bg-gray-100 text-gray-600 border-gray-200'
                                      : 'bg-rose-50 text-rose-700 border-rose-200'
                                  }`}
                                >
                                  {j.status}
                                </span>
                              </td>
                              <td className="p-3 text-gray-600">{j.trigger_kind}</td>
                              <td className="p-3 text-gray-700">{j.attempt_count} / {j.max_attempts}</td>
                              <td className="p-3 text-gray-700 max-w-xs truncate">{j.source_text_snapshot}</td>
                              <td className="p-3 text-gray-500 max-w-xs truncate">{j.error_message || '-'}</td>
                              <td className="p-3 text-gray-500">{new Date(j.created_at).toLocaleTimeString()}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* TAB 5: QUALITY & EVALUATION */}
              {activeTab === 'evaluation' && (
                <div className="space-y-5">
                  {/* Action Banner */}
                  <div className="rounded-lg border border-gray-200 bg-gray-50/60 p-4 flex items-center justify-between">
                    <div>
                      <h4 className="text-xs font-semibold text-gray-900">Run 20-Scenario Quality Evaluation Suite</h4>
                      <p className="text-xs text-gray-500 mt-0.5">
                        Verifies extraction accuracy, deduplication, conflict detection, temporal supersession, and isolation.
                      </p>
                    </div>
                    <button
                      onClick={handleTriggerEvaluation}
                      disabled={actionLoading}
                      className="rounded-lg bg-gray-900 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-gray-800 disabled:opacity-50 transition-colors shadow-sm"
                    >
                      {actionLoading ? 'Evaluating...' : 'Execute All 20 Scenarios'}
                    </button>
                  </div>

                  {/* Metrics Table */}
                  {evalMetrics.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-gray-300 p-12 text-center text-xs text-gray-500 bg-gray-50/40">
                      No evaluation benchmark runs recorded yet. Click "Execute All 20 Scenarios" to run the benchmark suite.
                    </div>
                  ) : (
                    <div className="overflow-x-auto rounded-lg border border-gray-200">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-gray-50 text-gray-600 font-medium border-b border-gray-200">
                          <tr>
                            <th className="p-3">Scenario</th>
                            <th className="p-3">Category</th>
                            <th className="p-3">Result</th>
                            <th className="p-3">Details / Evidence</th>
                            <th className="p-3">Run ID</th>
                            <th className="p-3">Timestamp</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {evalMetrics.map((m) => (
                            <tr key={m.id} className="hover:bg-gray-50/80 transition-colors">
                              <td className="p-3 font-medium text-gray-900">{m.scenario_name}</td>
                              <td className="p-3 text-gray-500">{m.metric_category}</td>
                              <td className="p-3">
                                <span
                                  className={`rounded border px-2 py-0.5 text-[10px] font-semibold ${
                                    m.passed
                                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                                      : 'bg-rose-50 text-rose-700 border-rose-200'
                                  }`}
                                >
                                  {m.passed ? 'PASSED' : 'FAILED'}
                                </span>
                              </td>
                              <td className="p-3 text-gray-600 max-w-sm truncate">{m.details || JSON.stringify(m.metrics_json)}</td>
                              <td className="p-3 font-mono text-[11px] text-gray-500">{m.eval_run_id}</td>
                              <td className="p-3 text-gray-500">{new Date(m.created_at).toLocaleTimeString()}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};
