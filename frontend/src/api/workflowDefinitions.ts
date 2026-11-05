import { apiClient } from './client.ts';

export type WorkflowVersionStatus = 'DRAFT' | 'ACTIVE' | 'ARCHIVED';

export type WorkflowActionType =
  | 'RETRIEVAL'
  | 'WEB_SEARCH'
  | 'EXTRACTION'
  | 'GROUNDING'
  | 'SYNTHESIS'
  | 'EXECUTION'
  | 'IF_ELSE'
  | 'APPROVAL_GATE';

export type WorkflowTriggerType = 'MANUAL' | 'SCHEDULE' | 'WEBHOOK' | 'EVENT';

export interface WorkflowNodeDefinition {
  id: string;
  name: string;
  node_type: WorkflowActionType;
  inputs: Record<string, any>;
  dependencies: string[];
}

export interface WorkflowTriggerSpec {
  trigger_type: WorkflowTriggerType;
  config: Record<string, any>;
}

export interface CanvasPosition {
  x: number;
  y: number;
}

export interface CanvasNodeLayout {
  id: string;
  position: CanvasPosition;
  data?: Record<string, any>;
}

export interface CanvasEdgeLayout {
  id: string;
  source: string;
  target: string;
}

export interface WorkflowCanvasLayout {
  nodes: CanvasNodeLayout[];
  edges: CanvasEdgeLayout[];
  viewport?: { x: number; y: number; zoom: number };
}

export interface WorkflowDefinitionResponse {
  id: string;
  workspace_id: string;
  name: string;
  description?: string | null;
  is_active: boolean;
  active_version_id?: string | null;
  created_by_id: string;
  created_at: string;
  updated_at: string;
}

export interface WorkflowVersionResponse {
  id: string;
  workflow_id: string;
  version_number: number;
  status: WorkflowVersionStatus;
  trigger_spec: WorkflowTriggerSpec;
  steps_spec: WorkflowNodeDefinition[];
  canvas_layout: WorkflowCanvasLayout;
  schema_version: string;
  created_by_id: string;
  created_at: string;
  published_at?: string | null;
}

export interface WorkflowDAGValidationResponse {
  is_valid: boolean;
  node_count: number;
  errors: string[];
}

export interface WorkflowDefinitionCreate {
  workspace_id: string;
  name: string;
  description?: string | null;
  initial_trigger?: WorkflowTriggerSpec;
  initial_steps?: WorkflowNodeDefinition[];
  initial_canvas_layout?: WorkflowCanvasLayout;
}

export interface WorkflowDefinitionUpdate {
  name?: string;
  description?: string | null;
  is_active?: boolean;
}

export interface WorkflowVersionCreateDraft {
  fork_from_version_id?: string | null;
  trigger_spec?: WorkflowTriggerSpec;
  steps_spec?: WorkflowNodeDefinition[];
  canvas_layout?: WorkflowCanvasLayout;
}

export interface WorkflowVersionUpdate {
  trigger_spec?: WorkflowTriggerSpec;
  steps_spec?: WorkflowNodeDefinition[];
  canvas_layout?: WorkflowCanvasLayout;
}

// ---------------------------------------------------------------------------
// API Client Functions
// ---------------------------------------------------------------------------

export async function listWorkflowDefinitions(
  workspaceId: string,
  token: string,
  skip = 0,
  limit = 50
): Promise<WorkflowDefinitionResponse[]> {
  return apiClient<WorkflowDefinitionResponse[]>(
    `/api/v1/workflows/definitions?workspace_id=${encodeURIComponent(workspaceId)}&skip=${skip}&limit=${limit}`,
    { method: 'GET' },
    token
  );
}

export async function getWorkflowDefinition(
  workspaceId: string,
  workflowId: string,
  token: string
): Promise<WorkflowDefinitionResponse> {
  return apiClient<WorkflowDefinitionResponse>(
    `/api/v1/workflows/definitions/${encodeURIComponent(workflowId)}?workspace_id=${encodeURIComponent(workspaceId)}`,
    { method: 'GET' },
    token
  );
}

export async function createWorkflowDefinition(
  payload: WorkflowDefinitionCreate,
  token: string
): Promise<WorkflowDefinitionResponse> {
  return apiClient<WorkflowDefinitionResponse>(
    '/api/v1/workflows/definitions',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    token
  );
}

export async function updateWorkflowDefinition(
  workspaceId: string,
  workflowId: string,
  payload: WorkflowDefinitionUpdate,
  token: string
): Promise<WorkflowDefinitionResponse> {
  return apiClient<WorkflowDefinitionResponse>(
    `/api/v1/workflows/definitions/${encodeURIComponent(workflowId)}?workspace_id=${encodeURIComponent(workspaceId)}`,
    {
      method: 'PUT',
      body: JSON.stringify(payload),
    },
    token
  );
}

export async function deleteWorkflowDefinition(
  workspaceId: string,
  workflowId: string,
  token: string
): Promise<void> {
  return apiClient<void>(
    `/api/v1/workflows/definitions/${encodeURIComponent(workflowId)}?workspace_id=${encodeURIComponent(workspaceId)}`,
    { method: 'DELETE' },
    token
  );
}

export async function listWorkflowVersions(
  workspaceId: string,
  workflowId: string,
  token: string
): Promise<WorkflowVersionResponse[]> {
  return apiClient<WorkflowVersionResponse[]>(
    `/api/v1/workflows/definitions/${encodeURIComponent(workflowId)}/versions?workspace_id=${encodeURIComponent(workspaceId)}`,
    { method: 'GET' },
    token
  );
}

export async function getWorkflowVersion(
  workspaceId: string,
  workflowId: string,
  versionId: string,
  token: string
): Promise<WorkflowVersionResponse> {
  return apiClient<WorkflowVersionResponse>(
    `/api/v1/workflows/definitions/${encodeURIComponent(workflowId)}/versions/${encodeURIComponent(versionId)}?workspace_id=${encodeURIComponent(workspaceId)}`,
    { method: 'GET' },
    token
  );
}

export async function createDraftVersion(
  workspaceId: string,
  workflowId: string,
  payload: WorkflowVersionCreateDraft,
  token: string
): Promise<WorkflowVersionResponse> {
  return apiClient<WorkflowVersionResponse>(
    `/api/v1/workflows/definitions/${encodeURIComponent(workflowId)}/versions?workspace_id=${encodeURIComponent(workspaceId)}`,
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    token
  );
}

export async function updateDraftVersion(
  workspaceId: string,
  workflowId: string,
  versionId: string,
  payload: WorkflowVersionUpdate,
  token: string
): Promise<WorkflowVersionResponse> {
  return apiClient<WorkflowVersionResponse>(
    `/api/v1/workflows/definitions/${encodeURIComponent(workflowId)}/versions/${encodeURIComponent(versionId)}?workspace_id=${encodeURIComponent(workspaceId)}`,
    {
      method: 'PUT',
      body: JSON.stringify(payload),
    },
    token
  );
}

export async function validateWorkflowDag(
  workspaceId: string,
  workflowId: string,
  versionId: string,
  token: string
): Promise<WorkflowDAGValidationResponse> {
  return apiClient<WorkflowDAGValidationResponse>(
    `/api/v1/workflows/definitions/${encodeURIComponent(workflowId)}/versions/${encodeURIComponent(versionId)}/validate?workspace_id=${encodeURIComponent(workspaceId)}`,
    { method: 'POST' },
    token
  );
}

export async function publishWorkflowVersion(
  workspaceId: string,
  workflowId: string,
  versionId: string,
  token: string
): Promise<WorkflowVersionResponse> {
  return apiClient<WorkflowVersionResponse>(
    `/api/v1/workflows/definitions/${encodeURIComponent(workflowId)}/versions/${encodeURIComponent(versionId)}/publish?workspace_id=${encodeURIComponent(workspaceId)}`,
    { method: 'POST' },
    token
  );
}

export type { WorkflowTestRunRequest } from './workflows.ts';
export { executeWorkflowTestRun } from './workflows.ts';

