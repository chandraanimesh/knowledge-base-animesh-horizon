import { apiClient } from './client';

export type WorkspaceRole = 'OWNER' | 'EDITOR' | 'VIEWER';

export interface WorkspaceItem {
  id: string;
  name: string;
  code: string;
  description?: string | null;
  department_id: string;
  department_name?: string | null;
  department_code?: string | null;
  is_active: boolean;
  created_by?: string | null;
  created_at: string;
  updated_at: string;
  deleted_at?: string | null;
}

export interface WorkspaceMemberItem {
  workspace_id: string;
  user_id: string;
  user_email?: string | null;
  role: WorkspaceRole;
  created_at: string;
  updated_at: string;
}

export async function listWorkspaces(
  token: string,
  departmentId?: string,
  includeInactive = false
): Promise<WorkspaceItem[]> {
  const params = new URLSearchParams();
  if (departmentId) params.append('department_id', departmentId);
  if (includeInactive) params.append('include_inactive', 'true');
  const qs = params.toString() ? `?${params.toString()}` : '';
  return apiClient<WorkspaceItem[]>(`/workspaces${qs}`, { method: 'GET' }, token);
}

export async function getWorkspace(token: string, workspaceId: string): Promise<WorkspaceItem> {
  return apiClient<WorkspaceItem>(`/workspaces/${encodeURIComponent(workspaceId)}`, { method: 'GET' }, token);
}

export async function createWorkspace(
  token: string,
  data: { name: string; code: string; department_id: string; description?: string }
): Promise<WorkspaceItem> {
  return apiClient<WorkspaceItem>('/workspaces', {
    method: 'POST',
    body: JSON.stringify(data),
  }, token);
}

export async function updateWorkspace(
  token: string,
  workspaceId: string,
  data: { name?: string; description?: string; is_active?: boolean }
): Promise<WorkspaceItem> {
  return apiClient<WorkspaceItem>(`/workspaces/${encodeURIComponent(workspaceId)}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  }, token);
}

export async function deleteWorkspace(token: string, workspaceId: string): Promise<void> {
  return apiClient<void>(`/workspaces/${encodeURIComponent(workspaceId)}`, {
    method: 'DELETE',
  }, token);
}

export async function listWorkspaceMembers(
  token: string,
  workspaceId: string
): Promise<WorkspaceMemberItem[]> {
  return apiClient<WorkspaceMemberItem[]>(`/workspaces/${encodeURIComponent(workspaceId)}/members`, {
    method: 'GET',
  }, token);
}

export async function addWorkspaceMember(
  token: string,
  workspaceId: string,
  data: { user_id: string; role: WorkspaceRole }
): Promise<WorkspaceMemberItem> {
  return apiClient<WorkspaceMemberItem>(`/workspaces/${encodeURIComponent(workspaceId)}/members`, {
    method: 'POST',
    body: JSON.stringify(data),
  }, token);
}

export async function updateWorkspaceMemberRole(
  token: string,
  workspaceId: string,
  userId: string,
  data: { role: WorkspaceRole }
): Promise<WorkspaceMemberItem> {
  return apiClient<WorkspaceMemberItem>(`/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  }, token);
}

export async function removeWorkspaceMember(
  token: string,
  workspaceId: string,
  userId: string
): Promise<void> {
  return apiClient<void>(`/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}`, {
    method: 'DELETE',
  }, token);
}

export interface WorkspaceProviderProfile {
  id?: string;
  workspace_id: string;
  name: string;
  is_active: boolean;
  llm_provider: string | null;
  llm_model: string | null;
  llm_credential_ref: string | null;
  embedding_provider: string | null;
  embedding_model: string | null;
  document_reader_provider: string | null;
  reranker_provider: string | null;
  reranker_model: string | null;
  ocr_provider: string | null;
  web_provider: string | null;
  web_credential_ref: string | null;
  configuration?: Record<string, any>;
  created_at?: string;
  updated_at?: string;
}

export interface ProviderCapabilitiesSummary {
  llm: string[];
  embedding: string[];
  document_reader: string[];
  reranker: string[];
  ocr: string[];
  web: string[];
}

export async function getWorkspaceProviderProfile(
  token: string,
  workspaceId: string
): Promise<WorkspaceProviderProfile | null> {
  return apiClient<WorkspaceProviderProfile | null>(
    `/workspaces/${encodeURIComponent(workspaceId)}/provider-profile`,
    { method: 'GET' },
    token
  );
}

export async function upsertWorkspaceProviderProfile(
  token: string,
  workspaceId: string,
  data: Partial<WorkspaceProviderProfile>
): Promise<WorkspaceProviderProfile> {
  return apiClient<WorkspaceProviderProfile>(
    `/workspaces/${encodeURIComponent(workspaceId)}/provider-profile`,
    {
      method: 'PUT',
      body: JSON.stringify(data),
    },
    token
  );
}

export async function getProviderCapabilities(
  token: string
): Promise<ProviderCapabilitiesSummary> {
  return apiClient<ProviderCapabilitiesSummary>(
    '/workspaces/registry/capabilities',
    { method: 'GET' },
    token
  );
}

export const workspaceApi = {
  listWorkspaces,
  getWorkspace,
  createWorkspace,
  updateWorkspace,
  deleteWorkspace,
  listWorkspaceMembers,
  addWorkspaceMember,
  updateWorkspaceMemberRole,
  removeWorkspaceMember,
  getWorkspaceProviderProfile,
  upsertWorkspaceProviderProfile,
  getProviderCapabilities,
};

