import React, { useEffect, useState, useRef } from 'react';
import {
  getWorkflowRun,
  submitWorkflowApproval,
  WorkflowRunResponse,
  WorkflowStatus,
} from '../api/workflows';

interface WorkflowRunProgressCardProps {
  runId: string;
  token: string;
  initialStatus?: string | null;
  onWorkflowComplete?: (finalRun: WorkflowRunResponse) => void;
  onOpenCanvas?: (runId: string) => void;
}

const TERMINAL_STATUSES: Set<WorkflowStatus> = new Set([
  'COMPLETED',
  'REJECTED',
  'EXPIRED',
  'CANCELLED',
  'FAILED',
]);

export function WorkflowRunProgressCard({
  runId,
  token,
  initialStatus,
  onWorkflowComplete,
  onOpenCanvas,
}: WorkflowRunProgressCardProps) {
  const [run, setRun] = useState<WorkflowRunResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [rejectionReason, setRejectionReason] = useState<string>('');
  const [isSubmittingApproval, setIsSubmittingApproval] = useState<boolean>(false);
  const [approvalError, setApprovalError] = useState<string | null>(null);

  const isMountedRef = useRef<boolean>(true);
  const isPollingRef = useRef<boolean>(false);

  // Poll status while non-terminal
  useEffect(() => {
    isMountedRef.current = true;

    async function fetchRun() {
      if (isPollingRef.current) return;
      isPollingRef.current = true;
      try {
        const data = await getWorkflowRun(runId, token);
        if (!isMountedRef.current) return;
        setRun(data);
        setLoading(false);
        setError(null);

        if (TERMINAL_STATUSES.has(data.status)) {
          if (onWorkflowComplete) {
            onWorkflowComplete(data);
          }
        }
      } catch (err: any) {
        if (!isMountedRef.current) return;
        console.warn('Transient workflow polling error:', err);
        // Do not mark run as failed on transient network glitch
        if (!run) {
          setError(err.message || 'Unable to load workflow progress.');
        }
      } finally {
        isPollingRef.current = false;
      }
    }

    fetchRun();

    const intervalId = setInterval(() => {
      if (!isMountedRef.current) return;
      // Stop polling once terminal or awaiting approval
      if (run && (TERMINAL_STATUSES.has(run.status) || run.status === 'AWAITING_APPROVAL')) {
        clearInterval(intervalId);
        return;
      }
      fetchRun();
    }, 2500);

    return () => {
      isMountedRef.current = false;
      clearInterval(intervalId);
    };
  }, [runId, token, run?.status]);

  async function handleApproval(decision: 'APPROVED' | 'REJECTED') {
    if (!run || !run.draft_hash) return;
    setIsSubmittingApproval(true);
    setApprovalError(null);
    try {
      await submitWorkflowApproval(
        run.id,
        {
          decision,
          draft_hash: run.draft_hash,
          rejection_reason: decision === 'REJECTED' ? rejectionReason || 'Rejected by reviewer' : undefined,
          proposed_action: run.intended_action || 'CRM_CREATE_NOTE',
          proposed_target: run.target_record_id || undefined,
        },
        token
      );
      // Immediately refresh authoritative status
      const updated = await getWorkflowRun(run.id, token);
      if (isMountedRef.current) {
        setRun(updated);
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        setApprovalError(err.message || 'Failed to submit approval decision.');
      }
    } finally {
      if (isMountedRef.current) {
        setIsSubmittingApproval(false);
      }
    }
  }

  const currentStatus = run?.status || (initialStatus as WorkflowStatus) || 'RUNNING';

  const getStatusColor = (status: WorkflowStatus) => {
    switch (status) {
      case 'COMPLETED':
        return '#10b981';
      case 'AWAITING_APPROVAL':
        return '#f59e0b';
      case 'FAILED':
      case 'REJECTED':
      case 'EXPIRED':
        return '#ef4444';
      default:
        return '#3b82f6';
    }
  };

  return (
    <div
      style={{
        marginTop: '12px',
        padding: '16px',
        borderRadius: '10px',
        border: '1px solid var(--color-border)',
        background: 'var(--color-surface)',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.04)',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--color-ink)' }}>
            Adaptive Workflow Run
          </span>
          <span
            style={{
              fontSize: '11px',
              fontFamily: 'var(--font-mono)',
              padding: '2px 8px',
              borderRadius: '12px',
              backgroundColor: `${getStatusColor(currentStatus)}20`,
              color: getStatusColor(currentStatus),
              fontWeight: 600,
            }}
          >
            {currentStatus}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {onOpenCanvas && (
            <button
              onClick={() => onOpenCanvas(runId)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                padding: '2px 8px',
                fontSize: '11px',
                fontWeight: 600,
                color: '#0284c7',
                background: '#f0f9ff',
                border: '1px solid #bae6fd',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
              title="Open full interactive Execution Canvas with DAG visualization"
            >
              <span>📊</span>
              <span>View Canvas</span>
            </button>
          )}
          <span style={{ fontSize: '11px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)' }}>
            ID: {runId.slice(0, 8)}…
          </span>
        </div>
      </div>

      {error && (
        <div style={{ fontSize: '12px', color: '#ef4444', marginBottom: '8px' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Task DAG List */}
      {run && run.tasks && run.tasks.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '14px' }}>
          <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--color-ink-muted)' }}>
            Execution Tasks ({run.tasks.filter((t) => t.status === 'COMPLETED').length}/{run.tasks.length})
          </span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {run.tasks.map((task) => {
              const isTaskDone = task.status === 'COMPLETED';
              const isTaskRunning = task.status === 'RUNNING';
              return (
                <div
                  key={task.id}
                  title={`${task.task_key} (${task.task_type}): ${task.status}`}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    fontSize: '11px',
                    fontFamily: 'var(--font-mono)',
                    padding: '3px 8px',
                    borderRadius: '4px',
                    border: '1px solid var(--color-border)',
                    background: isTaskDone
                      ? 'rgba(16, 185, 129, 0.08)'
                      : isTaskRunning
                      ? 'rgba(59, 130, 246, 0.08)'
                      : 'var(--color-panel)',
                    color: isTaskDone ? '#10b981' : isTaskRunning ? '#3b82f6' : 'var(--color-ink-muted)',
                  }}
                >
                  <span>{isTaskDone ? '✓' : isTaskRunning ? '●' : '○'}</span>
                  <span>{task.task_key}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Human Approval Review Gate */}
      {currentStatus === 'AWAITING_APPROVAL' && (
        <div
          style={{
            marginTop: '12px',
            padding: '14px',
            borderRadius: '8px',
            border: '1px solid #f59e0b',
            background: 'rgba(245, 158, 11, 0.05)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#d97706' }}>
              🛡️ Human Approval Required
            </span>
            <span style={{ fontSize: '11px', color: '#b45309', fontFamily: 'var(--font-mono)' }}>
              Ticket: {run?.approval_id?.slice(0, 8) || 'pending'}…
            </span>
          </div>

          <p style={{ fontSize: '12px', color: 'var(--color-ink)', lineHeight: 1.5, margin: '0 0 10px 0' }}>
            The agent has prepared a proposed CRM write-back action. Review the target entity and exact mutation payload before authorizing execution.
          </p>

          {/* Action & Target Metadata Table */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '8px',
              padding: '10px',
              borderRadius: '6px',
              background: 'var(--color-panel)',
              border: '1px solid var(--color-border)',
              marginBottom: '10px',
              fontSize: '11px',
            }}
          >
            <div>
              <span style={{ color: 'var(--color-ink-muted)', display: 'block' }}>Target Operation</span>
              <strong style={{ color: 'var(--color-ink)', fontFamily: 'var(--font-mono)' }}>
                {run?.intended_action || 'create_one_note'}
              </strong>
            </div>
            <div>
              <span style={{ color: 'var(--color-ink-muted)', display: 'block' }}>Target Record / Company</span>
              <strong style={{ color: 'var(--color-ink)', fontFamily: 'var(--font-mono)' }}>
                {run?.target_record_id || 'company'}
              </strong>
            </div>
            <div>
              <span style={{ color: 'var(--color-ink-muted)', display: 'block' }}>Requesting Workspace</span>
              <span style={{ color: 'var(--color-ink)', fontFamily: 'var(--font-mono)' }}>
                {run?.workspace_id ? `${run.workspace_id.slice(0, 8)}…` : 'Current'}
              </span>
            </div>
            <div>
              <span style={{ color: 'var(--color-ink-muted)', display: 'block' }}>Approval Expiry</span>
              <span style={{ color: 'var(--color-ink)' }}>24 Hours</span>
            </div>
          </div>

          {/* Authoritative Server Draft Preview */}
          {run?.draft_text && (
            <div style={{ marginBottom: '10px' }}>
              <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-ink-muted)', display: 'block', marginBottom: '4px' }}>
                Proposed Mutation Content (Payload):
              </span>
              <div
                style={{
                  maxHeight: '180px',
                  overflowY: 'auto',
                  padding: '10px',
                  borderRadius: '6px',
                  background: 'var(--color-surface)',
                  border: '1px solid var(--color-border)',
                  fontSize: '12px',
                  fontFamily: 'var(--font-sans)',
                  color: 'var(--color-ink)',
                  whiteSpace: 'pre-wrap',
                }}
              >
                {run.draft_text}
              </div>
            </div>
          )}

          {/* Canonical Hash Badge */}
          {run?.draft_hash && (
            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)', marginBottom: '10px' }}>
              Cryptographic Payload Hash: <code>{run.draft_hash.slice(0, 16)}…{run.draft_hash.slice(-8)}</code>
            </div>
          )}

          {/* Citations Preview */}
          {run?.evidence_items && run.evidence_items.length > 0 && (
            <div style={{ marginBottom: '10px' }}>
              <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-ink-muted)' }}>
                Evidence Citations ({run.evidence_items.length}):
              </span>
              <ul style={{ margin: '4px 0 0 0', paddingLeft: '16px', fontSize: '11px', color: 'var(--color-ink-muted)' }}>
                {run.evidence_items.slice(0, 3).map((ev, idx) => (
                  <li key={idx}>
                    [{ev.source_type || 'evidence'}] {ev.snippet ? ev.snippet.slice(0, 80) : ev.source_id}…
                  </li>
                ))}
              </ul>
            </div>
          )}

          {approvalError && (
            <div style={{ fontSize: '12px', color: '#ef4444', marginBottom: '8px' }}>
              ⚠️ {approvalError}
            </div>
          )}

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '10px' }}>
            <button
              onClick={() => handleApproval('APPROVED')}
              disabled={isSubmittingApproval}
              style={{
                backgroundColor: '#10b981',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                padding: '7px 14px',
                fontSize: '12px',
                fontWeight: 600,
                cursor: isSubmittingApproval ? 'default' : 'pointer',
                opacity: isSubmittingApproval ? 0.6 : 1,
              }}
            >
              {isSubmittingApproval ? 'Executing…' : '✓ Authorize & Commit'}
            </button>

            <button
              onClick={() => handleApproval('REJECTED')}
              disabled={isSubmittingApproval}
              style={{
                backgroundColor: 'transparent',
                color: '#ef4444',
                border: '1px solid #ef4444',
                borderRadius: '6px',
                padding: '6px 12px',
                fontSize: '12px',
                fontWeight: 600,
                cursor: isSubmittingApproval ? 'default' : 'pointer',
                opacity: isSubmittingApproval ? 0.6 : 1,
              }}
            >
              ✗ Reject
            </button>
          </div>
        </div>
      )}

      {/* Outcome Presentation Banners */}
      {currentStatus === 'COMPLETED' && (
        <div
          style={{
            marginTop: '10px',
            padding: '10px',
            borderRadius: '6px',
            border: '1px solid #10b981',
            background: 'rgba(16, 185, 129, 0.06)',
            fontSize: '12px',
            color: '#065f46',
          }}
        >
          ✓ <strong>Execution Confirmed:</strong> Workflow completed successfully and side-effect was committed.
        </div>
      )}

      {(run?.failure_category === 'UNKNOWN' || run?.error_message?.includes('UNKNOWN') || run?.error_message?.includes('timed out')) && (
        <div
          style={{
            marginTop: '10px',
            padding: '10px',
            borderRadius: '6px',
            border: '1px solid #ef4444',
            background: 'rgba(239, 68, 68, 0.08)',
            fontSize: '12px',
            color: '#991b1b',
          }}
        >
          ⚠️ <strong>Outcome Ambiguous / Unresolved:</strong> The external MCP tool timed out after dispatch.
          The record may have been created in Twenty CRM. Blind retries are blocked to prevent duplicate notes.
        </div>
      )}
    </div>
  );
}
