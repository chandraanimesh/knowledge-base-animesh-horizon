import { apiClient } from './client';

export type WorkflowStatus =
  | 'PENDING'
  | 'PLANNING'
  | 'RUNNING'
  | 'AWAITING_APPROVAL'
  | 'EXECUTING'
  | 'COMPLETED'
  | 'REJECTED'
  | 'EXPIRED'
  | 'CANCELLED'
  | 'FAILED';

export interface WorkflowTaskResponse {
  id: string;
  workflow_run_id: string;
  task_key: string;
  task_type: string;
  status: 'PENDING' | 'READY' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED' | 'SKIPPED';
  dependencies: string[];
  input_payload?: Record<string, any>;
  output_result?: Record<string, any> | null;
  error_message?: string | null;
  tokens_spent: number;
  provider_calls: number;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
}

export interface WorkflowRunResponse {
  id: string;
  user_id: string;
  workspace_id?: string | null;
  department_id?: string | null;
  conversation_id?: string | null;
  scope_type: 'GLOBAL' | 'WORKSPACE';
  status: WorkflowStatus;
  idempotency_key?: string | null;
  request_hash: string;
  draft_hash?: string | null;
  max_tokens_budget: number;
  tokens_spent: number;
  tokens_reserved: number;
  max_provider_calls_budget: number;
  provider_calls_spent: number;
  provider_calls_reserved: number;
  wall_clock_timeout_seconds: number;
  failure_category?: string | null;
  error_message?: string | null;
  created_at: string;
  started_at?: string | null;
  awaiting_approval_at?: string | null;
  approved_at?: string | null;
  executing_at?: string | null;
  completed_at?: string | null;
  cancelled_at?: string | null;
  failed_at?: string | null;
  tasks: WorkflowTaskResponse[];
  draft_text?: string | null;
  evidence_items?: Array<{
    source_type?: string;
    source_id?: string;
    snippet?: string;
    confidence?: number;
    [key: string]: any;
  }>;
  approval_id?: string | null;
  intended_action?: string | null;
  target_record_id?: string | null;
}

export interface WorkflowApprovalRequest {
  decision: 'APPROVED' | 'REJECTED' | 'CANCELLED';
  draft_hash: string;
  rejection_reason?: string | null;
  proposed_action?: string;
  proposed_target?: string | null;
}

export interface WorkflowApprovalResponse {
  workflow_run_id: string;
  approval_id: string;
  status: WorkflowStatus;
  side_effect_executed: boolean;
  error_message?: string | null;
  writeback_result?: Record<string, any> | null;
}

export async function getWorkflowRun(runId: string, token: string): Promise<WorkflowRunResponse> {
  return apiClient<WorkflowRunResponse>(`/api/v1/workflows/runs/${runId}`, {
    method: 'GET',
  }, token);
}

export async function submitWorkflowApproval(
  runId: string,
  payload: WorkflowApprovalRequest,
  token: string
): Promise<WorkflowApprovalResponse> {
  return apiClient<WorkflowApprovalResponse>(`/api/v1/workflows/runs/${runId}/approval`, {
    method: 'POST',
    body: JSON.stringify(payload),
  }, token);
}
