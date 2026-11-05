import { apiClient } from './client.ts';

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
  retry_count?: number;
  max_retries?: number;
  locked_by?: string | null;
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
  workflow_id?: string | null;
  workflow_version_id?: string | null;
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
  approval_status?: string | null;
  approval_expires_at?: string | null;
  approval_requested_by?: string | null;
  target_object?: string | null;
  is_test?: boolean;
  execution_mode?: 'TEST' | 'PRODUCTION';
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

export async function listWorkflowRuns(
  token: string,
  params: {
    workflow_id?: string;
    workspace_id?: string;
    limit?: number;
    skip?: number;
  } = {}
): Promise<WorkflowRunResponse[]> {
  const query = new URLSearchParams();
  if (params.workflow_id) query.set('workflow_id', params.workflow_id);
  if (params.workspace_id) query.set('workspace_id', params.workspace_id);
  if (params.limit) query.set('limit', String(params.limit));
  if (params.skip) query.set('skip', String(params.skip));

  const qs = query.toString();
  return apiClient<WorkflowRunResponse[]>(
    `/api/v1/workflows/runs${qs ? `?${qs}` : ''}`,
    { method: 'GET' },
    token
  );
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

export interface WorkflowTestRunRequest {
  inputs: Record<string, any>;
  budget_limits?: {
    max_tokens?: number;
    max_provider_calls?: number;
    wall_clock_timeout_seconds?: number;
  };
  idempotency_key?: string;
}

export async function executeWorkflowTestRun(
  workspaceId: string,
  workflowId: string,
  versionId: string,
  payload: WorkflowTestRunRequest,
  token: string
): Promise<WorkflowRunResponse> {
  return apiClient<WorkflowRunResponse>(
    `/api/v1/workflows/definitions/${workflowId}/versions/${versionId}/test?workspace_id=${workspaceId}`,
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    token
  );
}

