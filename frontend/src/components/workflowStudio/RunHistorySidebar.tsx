import React from 'react';
import type { WorkflowRunResponse } from '../../api/workflows.ts';
import { formatDuration, getExecutionStatusBadge } from './executionUtils.ts';

interface RunHistorySidebarProps {
  runs: WorkflowRunResponse[];
  selectedRunId: string | null;
  onSelectRun: (runId: string) => void;
  onRefreshRuns: () => void;
  loading: boolean;
  isPolling: boolean;
  onOpenTestModal?: () => void;
}

export const RunHistorySidebar: React.FC<RunHistorySidebarProps> = ({
  runs,
  selectedRunId,
  onSelectRun,
  onRefreshRuns,
  loading,
  isPolling,
  onOpenTestModal,
}) => {
  return (
    <aside className="w-64 bg-[#fafaf7] border-r border-[#d4d4cc] flex flex-col h-full select-none shrink-0 min-h-0">
      {/* Header */}
      <div className="p-3 border-b border-[#d4d4cc] bg-[#f0f0eb] flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">
            Run History
          </span>
          <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-slate-200 text-slate-700">
            {runs.length}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {onOpenTestModal && (
            <button
              type="button"
              onClick={onOpenTestModal}
              title="Execute a draft test run"
              className="text-[11px] font-semibold px-2 py-0.5 bg-purple-600 hover:bg-purple-700 text-white rounded shadow-2xs flex items-center gap-1 transition-colors"
            >
              <span>+</span>
              <span>Test</span>
            </button>
          )}
          {isPolling && (
            <span
              className="flex h-2 w-2 relative"
              title="Actively polling live run state"
            >
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-teal-600"></span>
            </span>
          )}
          <button
            type="button"
            onClick={onRefreshRuns}
            disabled={loading}
            title="Refresh run list"
            className="text-xs text-slate-500 hover:text-slate-800 p-1 rounded hover:bg-slate-200 disabled:opacity-50"
          >
            ↻
          </button>
        </div>
      </div>

      {/* Runs List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1.5 min-h-0 overscroll-contain">
        {runs.length === 0 ? (
          <div className="p-4 text-center text-slate-400 text-xs">
            {loading ? 'Loading executions...' : 'No execution runs recorded yet.'}
          </div>
        ) : (
          runs.map((run, idx) => {
            const isSelected = run.id === selectedRunId;
            const badge = getExecutionStatusBadge(run.status);
            const duration = formatDuration(run.created_at, run.completed_at || run.failed_at);

            return (
              <button
                key={run.id}
                type="button"
                onClick={() => onSelectRun(run.id)}
                className={`w-full text-left p-2.5 rounded-lg border transition-all text-xs ${
                  isSelected
                    ? 'bg-white border-teal-600 shadow-xs ring-1 ring-teal-500'
                    : 'bg-white/80 hover:bg-white border-slate-200 hover:border-slate-300'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-1.5 truncate max-w-[120px]">
                    <span className="font-bold text-slate-800 text-[11px] truncate">
                      Run #{runs.length - idx}
                    </span>
                    {(run.is_test || (run.thread_id && run.thread_id.startsWith('test:'))) && (
                      <span className="text-[9px] font-bold px-1 rounded bg-purple-100 text-purple-700 border border-purple-200">
                        TEST
                      </span>
                    )}
                  </div>
                  <span
                    className={`text-[9px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider flex items-center gap-1 ${
                      badge.badgeBg
                    } ${badge.badgeText} ${badge.pulse ? 'animate-pulse' : ''}`}
                  >
                    <span>{badge.icon}</span>
                    <span>{badge.label}</span>
                  </span>
                </div>

                <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
                  <span className="font-mono text-slate-400 truncate max-w-[95px]" title={run.id}>
                    {run.id.slice(0, 8)}...
                  </span>
                  <span className="font-mono">{duration}</span>
                </div>

                <div className="flex items-center justify-between text-[9px] text-slate-400 border-t border-slate-100 pt-1 mt-1">
                  <span>{new Date(run.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
                  {run.workflow_version_id && (
                    <span className="bg-slate-100 px-1 py-0.2 rounded font-mono text-slate-600">
                      v-pinned
                    </span>
                  )}
                </div>
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
};

