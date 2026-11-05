import React, { useState } from 'react';
import type { WorkflowRunResponse, WorkflowApprovalResponse } from '../../api/workflows.ts';
import { submitWorkflowApproval } from '../../api/workflows.ts';
import { formatDuration, sanitizePayload, isApprovalExpired, canSubmitApproval, buildApprovalPayload } from './executionUtils.ts';

interface WorkflowApprovalModalProps {
  run: WorkflowRunResponse;
  workflowName?: string;
  versionNumber?: number;
  token: string;
  isOpen: boolean;
  onClose: () => void;
  onDecisionCompleted: (result: WorkflowApprovalResponse) => void;
}

export const WorkflowApprovalModal: React.FC<WorkflowApprovalModalProps> = ({
  run,
  workflowName = 'Workflow',
  versionNumber,
  token,
  isOpen,
  onClose,
  onDecisionCompleted,
}) => {
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [rejectionReason, setRejectionReason] = useState<string>('');
  const [showRejectInput, setShowRejectInput] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successResult, setSuccessResult] = useState<WorkflowApprovalResponse | null>(null);

  if (!isOpen) return null;

  const isExpired = isApprovalExpired(run.approval_expires_at);
  const canDecide = canSubmitApproval(run.status, run.approval_expires_at, submitting);

  const handleApprove = async () => {
    if (!canDecide) return;

    try {
      setSubmitting(true);
      setErrorMessage(null);

      const payload = buildApprovalPayload({
        run,
        decision: 'APPROVED',
      });

      // Submit approval using server-provided draft hash without frontend tampering
      const res = await submitWorkflowApproval(run.id, payload, token);

      setSuccessResult(res);
      setTimeout(() => {
        onDecisionCompleted(res);
      }, 1000);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to submit approval decision.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleReject = async () => {
    if (!canDecide) return;

    try {
      setSubmitting(true);
      setErrorMessage(null);

      const payload = buildApprovalPayload({
        run,
        decision: 'REJECTED',
        rejectionReason,
      });

      const res = await submitWorkflowApproval(run.id, payload, token);

      setSuccessResult(res);
      setTimeout(() => {
        onDecisionCompleted(res);
      }, 1000);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to submit rejection decision.');
    } finally {
      setSubmitting(false);
    }
  };

  // Find synthesis task draft
  const synthTask = run.tasks.find((t) => t.task_type === 'SYNTHESIS');
  const draftContent = run.draft_text || synthTask?.output_result?.draft_text || run.initial_query;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-xs flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl border border-slate-300 shadow-2xl w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-5 py-4 border-b border-slate-200 bg-amber-50/70 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-amber-500 text-white flex items-center justify-center text-base font-bold shadow-xs">
              ⏸
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-bold uppercase tracking-wider text-amber-800 bg-amber-200/80 px-2 py-0.5 rounded">
                  Human-in-the-Loop Review
                </span>
                <span className="text-xs font-mono text-slate-500">
                  Ticket #{run.approval_id ? String(run.approval_id).slice(0, 8) : 'N/A'}
                </span>
              </div>
              <h2 className="text-sm font-bold text-slate-900 mt-0.5">
                Approval Required: {run.intended_action || 'CRM Writeback'}
              </h2>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="text-slate-400 hover:text-slate-600 p-1.5 rounded-lg hover:bg-slate-100 disabled:opacity-50"
          >
            ✕
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {/* Success Banner */}
          {successResult && (
            <div className="p-3.5 bg-emerald-50 border border-emerald-300 rounded-xl text-emerald-800 text-xs">
              <div className="font-bold flex items-center gap-1.5 mb-1 text-emerald-900">
                <span>✓</span>
                <span>Decision Successfully Recorded ({successResult.status})</span>
              </div>
              <p className="text-[11px] text-emerald-700">
                {successResult.side_effect_executed
                  ? 'Side effect successfully executed in CRM target entity.'
                  : 'Rejection acknowledged. Execution terminated safely with zero remote mutations.'}
              </p>
            </div>
          )}

          {/* Error Banner */}
          {errorMessage && (
            <div className="p-3.5 bg-red-50 border border-red-300 rounded-xl text-red-800 text-xs">
              <div className="font-bold flex items-center gap-1.5 mb-1 text-red-900">
                <span>⚠</span>
                <span>Approval Request Failed</span>
              </div>
              <p className="text-[11px] text-red-700 font-mono break-words">{errorMessage}</p>
            </div>
          )}

          {/* Expiry Warning */}
          {isExpired && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-xs flex items-center gap-2">
              <span>⏰</span>
              <span>This approval ticket has expired. Decisions are blocked for safety.</span>
            </div>
          )}

          {/* Ticket Metadata Grid */}
          <div className="bg-slate-50 rounded-xl p-3.5 border border-slate-200 grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
            <div>
              <span className="text-[10px] font-semibold text-slate-400 uppercase block">Workflow</span>
              <span className="font-semibold text-slate-800 truncate block">
                {workflowName} {versionNumber ? `(v${versionNumber})` : ''}
              </span>
            </div>
            <div>
              <span className="text-[10px] font-semibold text-slate-400 uppercase block">Run ID</span>
              <span className="font-mono text-slate-700 truncate block" title={run.id}>
                {run.id}
              </span>
            </div>
            <div>
              <span className="text-[10px] font-semibold text-slate-400 uppercase block">Target Entity</span>
              <span className="font-mono text-slate-800">
                {run.target_object || 'company'}: <strong className="text-teal-700">{run.target_record_id || 'unassigned'}</strong>
              </span>
            </div>
            <div>
              <span className="text-[10px] font-semibold text-slate-400 uppercase block">Expires At</span>
              <span className="font-mono text-slate-700">
                {run.approval_expires_at ? new Date(run.approval_expires_at).toLocaleString() : '24h lease'}
              </span>
            </div>
          </div>

          {/* Cryptographic Hash Verification Block */}
          <div className="bg-slate-900 text-slate-100 p-3 rounded-xl border border-slate-800 space-y-1">
            <div className="flex items-center justify-between text-[10px]">
              <span className="text-slate-400 uppercase font-mono tracking-wider">
                Canonical SHA-256 Draft Hash
              </span>
              <span className="text-emerald-400 font-bold">🔒 Server Verified</span>
            </div>
            <div className="font-mono text-[11px] text-teal-300 break-all bg-slate-950/80 p-2 rounded border border-slate-800">
              {run.draft_hash || 'Draft hash pending computation'}
            </div>
            <p className="text-[9px] text-slate-400">
              The backend recalculates and asserts this hash before executing any side-effects. Altering the draft invalidates the ticket.
            </p>
          </div>

          {/* Proposed Draft Text Review */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-700 uppercase tracking-wide">
                Proposed Content for CRM Writeback
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                {draftContent.length} characters
              </span>
            </div>
            <div className="bg-white p-3.5 rounded-xl border border-slate-300 max-h-56 overflow-y-auto text-xs text-slate-800 leading-relaxed font-sans whitespace-pre-wrap shadow-inner">
              {draftContent}
            </div>
          </div>

          {/* Evidence Citations */}
          {run.evidence_items && run.evidence_items.length > 0 && (
            <div className="space-y-1.5">
              <span className="text-xs font-bold text-slate-700 uppercase tracking-wide block">
                Supporting Evidence ({run.evidence_items.length})
              </span>
              <div className="space-y-1.5 max-h-36 overflow-y-auto">
                {run.evidence_items.map((ev, idx) => (
                  <div key={idx} className="bg-slate-50 p-2.5 rounded-lg border border-slate-200 text-xs">
                    <div className="flex items-center justify-between text-slate-600 font-semibold mb-0.5">
                      <span>{ev.source_type || 'Internal Evidence'}</span>
                      {ev.confidence !== undefined && (
                        <span className="text-teal-700 font-mono">{(ev.confidence * 100).toFixed(0)}% conf</span>
                      )}
                    </div>
                    <p className="text-slate-600 text-[11px] line-clamp-2">{ev.snippet || JSON.stringify(ev)}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Optional Rejection Reason Field */}
          {showRejectInput && (
            <div className="space-y-1.5 bg-red-50/50 p-3 rounded-xl border border-red-200 animate-in fade-in duration-100">
              <label className="block text-[11px] font-bold text-red-900 uppercase">
                Rejection Reason (Audit Trail) *
              </label>
              <textarea
                rows={2}
                value={rejectionReason}
                onChange={(e) => setRejectionReason(e.target.value)}
                placeholder="Explain why this proposed action or draft is being rejected..."
                className="w-full text-xs p-2 rounded-lg border border-red-300 bg-white focus:outline-none focus:ring-1 focus:ring-red-500"
              />
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="px-5 py-3.5 border-t border-slate-200 bg-slate-50 flex items-center justify-between shrink-0">
          <div>
            {!showRejectInput && (
              <button
                type="button"
                onClick={() => setShowRejectInput(true)}
                disabled={!canDecide}
                className="text-xs text-red-600 hover:text-red-700 font-semibold hover:underline disabled:opacity-50"
              >
                + Add Rejection Reason
              </button>
            )}
          </div>

          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={onClose}
              disabled={submitting}
              className="px-3.5 py-1.5 text-xs text-slate-600 hover:bg-slate-200 rounded-lg border border-slate-300 transition-colors disabled:opacity-50 font-medium"
            >
              Cancel
            </button>

            <button
              type="button"
              onClick={handleReject}
              disabled={!canDecide}
              className="px-3.5 py-1.5 text-xs font-semibold bg-white hover:bg-red-50 text-red-600 border border-red-300 rounded-lg shadow-xs transition-colors disabled:opacity-50"
            >
              {submitting ? 'Rejecting...' : 'Reject Action'}
            </button>

            <button
              type="button"
              onClick={handleApprove}
              disabled={!canDecide}
              className="px-4 py-1.5 text-xs font-semibold bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg shadow-xs transition-colors disabled:opacity-50 flex items-center gap-1.5"
            >
              <span>{submitting ? '⟳' : '✓'}</span>
              <span>{submitting ? 'Verifying & Executing...' : 'Approve & Execute Side Effect'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

