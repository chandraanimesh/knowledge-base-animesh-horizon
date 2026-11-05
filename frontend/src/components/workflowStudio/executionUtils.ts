import type { WorkflowStatus, WorkflowTaskResponse, WorkflowRunResponse, WorkflowApprovalRequest } from '../../api/workflows.ts';

const SENSITIVE_KEY_REGEX = /^(.*_)?(token|secret|key|password|auth|authorization|cred|credential)(_.*)?$/i;

/**
 * Distinguishes missing or unavailable metrics from actual zero values.
 * Returns '—' for null or undefined, and formatted number for 0 or positive counts.
 */
export function formatMetric(val: number | null | undefined): string {
  if (val === null || val === undefined) {
    return '—';
  }
  return val.toLocaleString();
}

/**
 * Calculates human-readable duration between two ISO timestamp strings.
 */
export function formatDuration(
  startIso?: string | null,
  endIso?: string | null
): string {
  if (!startIso) return '—';
  const start = new Date(startIso).getTime();
  if (isNaN(start)) return '—';

  const end = endIso ? new Date(endIso).getTime() : Date.now();
  if (isNaN(end)) return '—';

  const diffMs = Math.max(0, end - start);
  if (diffMs < 1000) {
    return `${diffMs}ms`;
  }
  if (diffMs < 60000) {
    return `${(diffMs / 1000).toFixed(2)}s`;
  }
  const minutes = Math.floor(diffMs / 60000);
  const seconds = Math.floor((diffMs % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
}

/**
 * Recursively deep-clones an object or array and redacts sensitive keys.
 * Preserves safe payload data for debugging without exposing secrets.
 */
export function sanitizePayload(data: any): any {
  if (data === null || data === undefined) {
    return data;
  }

  if (typeof data !== 'object') {
    return data;
  }

  if (Array.isArray(data)) {
    return data.map((item) => sanitizePayload(item));
  }

  const sanitized: Record<string, any> = {};

  // Check if this object represents a name-value or key-value pair for headers/env
  const isSensitiveNamedPair =
    (typeof data.name === 'string' && SENSITIVE_KEY_REGEX.test(data.name)) ||
    (typeof data.key === 'string' && SENSITIVE_KEY_REGEX.test(data.key));

  for (const [key, val] of Object.entries(data)) {
    if (isSensitiveNamedPair && (key === 'value' || key === 'val')) {
      sanitized[key] = '***REDACTED***';
    } else if (typeof val === 'object' && val !== null) {
      sanitized[key] = sanitizePayload(val);
    } else if (SENSITIVE_KEY_REGEX.test(key)) {
      sanitized[key] = '***REDACTED***';
    } else {
      sanitized[key] = val;
    }
  }
  return sanitized;
}

/**
 * Checks if a workflow run or task is in a terminal lifecycle state.
 * Terminal runs should never be actively polled.
 */
export function isTerminalStatus(status: WorkflowStatus | string): boolean {
  const terminalStates = ['COMPLETED', 'FAILED', 'CANCELLED', 'REJECTED', 'EXPIRED'];
  return terminalStates.includes(status);
}

export interface ExecutionBadgeStyle {
  label: string;
  badgeBg: string;
  badgeText: string;
  borderColor: string;
  pulse: boolean;
  icon: string;
}

/**
 * Provides UI style and animation attributes for task and run execution states.
 */
export function getExecutionStatusBadge(
  status: string,
  isApprovalGate: boolean = false
): ExecutionBadgeStyle {
  const upper = (status || 'PENDING').toUpperCase();

  if (upper === 'AWAITING_APPROVAL' || (isApprovalGate && upper === 'RUNNING')) {
    return {
      label: 'Awaiting Approval',
      badgeBg: 'bg-amber-100',
      badgeText: 'text-amber-800',
      borderColor: 'border-amber-400 ring-2 ring-amber-200',
      pulse: true,
      icon: '⏸',
    };
  }

  switch (upper) {
    case 'COMPLETED':
      return {
        label: 'Completed',
        badgeBg: 'bg-emerald-100',
        badgeText: 'text-emerald-800',
        borderColor: 'border-emerald-500 shadow-xs',
        pulse: false,
        icon: '✓',
      };
    case 'RUNNING':
    case 'PLANNING':
    case 'EXECUTING':
      return {
        label: upper,
        badgeBg: 'bg-blue-100',
        badgeText: 'text-blue-800',
        borderColor: 'border-blue-500 ring-2 ring-blue-200',
        pulse: true,
        icon: '⟳',
      };
    case 'FAILED':
      return {
        label: 'Failed',
        badgeBg: 'bg-red-100',
        badgeText: 'text-red-800',
        borderColor: 'border-red-500 shadow-xs',
        pulse: false,
        icon: '✕',
      };
    case 'CANCELLED':
      return {
        label: 'Cancelled',
        badgeBg: 'bg-slate-200',
        badgeText: 'text-slate-700',
        borderColor: 'border-slate-300',
        pulse: false,
        icon: '⊘',
      };
    case 'SKIPPED':
      return {
        label: 'Skipped',
        badgeBg: 'bg-slate-100',
        badgeText: 'text-slate-500',
        borderColor: 'border-slate-300 border-dashed',
        pulse: false,
        icon: '↷',
      };
    case 'READY':
      return {
        label: 'Ready',
        badgeBg: 'bg-sky-50',
        badgeText: 'text-sky-700',
        borderColor: 'border-sky-300',
        pulse: false,
        icon: '◌',
      };
    case 'PENDING':
    default:
      return {
        label: 'Pending',
        badgeBg: 'bg-slate-100',
        badgeText: 'text-slate-600',
        borderColor: 'border-slate-200',
        pulse: false,
        icon: '◌',
      };
  }
}

/**
 * Checks whether an approval ticket has exceeded its expiration timestamp.
 */
export function isApprovalExpired(
  approvalExpiresAt?: string | null,
  referenceTimeMs?: number
): boolean {
  if (!approvalExpiresAt) return false;
  const expiresAtMs = new Date(approvalExpiresAt).getTime();
  if (isNaN(expiresAtMs)) return false;
  const now = referenceTimeMs ?? Date.now();
  return expiresAtMs <= now;
}

/**
 * Determines whether the current user is eligible to submit an approval/rejection decision.
 * Strict conditions: run must be in AWAITING_APPROVAL, not expired, and not already submitting.
 */
export function canSubmitApproval(
  status?: string | null,
  approvalExpiresAt?: string | null,
  isSubmitting: boolean = false,
  referenceTimeMs?: number
): boolean {
  if (isSubmitting) return false;
  if ((status || '').toUpperCase() !== 'AWAITING_APPROVAL') return false;
  return !isApprovalExpired(approvalExpiresAt, referenceTimeMs);
}

/**
 * Constructs a secure approval/rejection payload.
 * Invariant: Never recalculates or modifies the server-persisted draft hash.
 */
export function buildApprovalPayload(params: {
  run: Pick<WorkflowRunResponse, 'draft_hash' | 'intended_action' | 'target_record_id'>;
  decision: 'APPROVED' | 'REJECTED' | 'CANCELLED';
  rejectionReason?: string;
}): WorkflowApprovalRequest {
  const { run, decision, rejectionReason } = params;

  if (!run.draft_hash) {
    throw new Error('Security Error: Run lacks server-persisted canonical draft hash.');
  }

  const payload: WorkflowApprovalRequest = {
    decision,
    draft_hash: run.draft_hash,
    proposed_action: run.intended_action || 'CRM_CREATE_NOTE',
    proposed_target: run.target_record_id || undefined,
  };

  if (decision === 'REJECTED') {
    payload.rejection_reason = (rejectionReason || '').trim() || 'Rejected by human reviewer.';
  }

  return payload;
}

