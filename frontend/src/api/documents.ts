import { apiClient } from './client';

export interface IngestResponse {
  file_name: string;
  skipped: boolean;
  document_id?: string;
  chunks_created: number;
  reason?: string;
}

export async function uploadDocument(
  file: File,
  departmentCode: string,
  token: string,
  workspaceId?: string
): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  let url = `/documents/upload?department_code=${encodeURIComponent(departmentCode)}`;
  if (workspaceId) {
    url += `&workspace_id=${encodeURIComponent(workspaceId)}`;
  }
  
  return apiClient<IngestResponse>(url, {
    method: 'POST',
    body: formData,
  }, token);
}

export interface DocumentAdminResponse {
  id: string;
  file_name: string;
  department_id?: string;
  department_code?: string;
  department_name?: string;
  workspace_id?: string;
  workspace_code?: string;
  workspace_name?: string;
  status: string;
  created_at: string;
}

export async function updateDocumentDepartment(
  documentId: string,
  departmentCode: string,
  token: string
): Promise<DocumentAdminResponse> {
  return apiClient<DocumentAdminResponse>(`/admin/documents/${encodeURIComponent(documentId)}/department`, {
    method: 'PATCH',
    body: JSON.stringify({ department_code: departmentCode }),
  }, token);
}

export interface DocumentListItem {
  id: string;
  file_name: string;
  department_id?: string;
  department_code?: string;
  department_name?: string;
  workspace_id?: string;
  workspace_code?: string;
  workspace_name?: string;
  uploaded_by_id?: string;
  uploaded_by_email?: string;
  uploaded_by_name?: string;
  mime_type?: string;
  status: string;
  created_at: string;
  preview_url?: string;
}

export async function listDocuments(token: string): Promise<DocumentListItem[]> {
  return apiClient<DocumentListItem[]>('/documents', {
    method: 'GET',
  }, token);
}

export interface DocUserAccessItem {
  user_id: string;
  email: string;
  full_name?: string;
  department_permissions: string[];
  document_permissions: string[];
  effective_permissions: string[];
}

export interface DocumentAccessListResponse {
  document_id: string;
  file_name: string;
  department_id: string;
  department_code: string;
  department_name: string;
  users: DocUserAccessItem[];
}

export async function getDocumentPermissions(
  documentId: string,
  token: string
): Promise<DocumentAccessListResponse> {
  return apiClient<DocumentAccessListResponse>(`/admin/documents/${encodeURIComponent(documentId)}/permissions`, {
    method: 'GET',
  }, token);
}

export async function grantDocumentPermission(
  documentId: string,
  userId: string,
  permissionCode: string,
  token: string
): Promise<{ message: string }> {
  return apiClient<{ message: string }>(`/admin/documents/${encodeURIComponent(documentId)}/permissions/grant`, {
    method: 'POST',
    body: JSON.stringify({ user_id: userId, permission_code: permissionCode }),
  }, token);
}

export async function revokeDocumentPermission(
  documentId: string,
  userId: string,
  permissionCode: string,
  token: string
): Promise<{ message: string }> {
  return apiClient<{ message: string }>(`/admin/documents/${encodeURIComponent(documentId)}/permissions/revoke`, {
    method: 'POST',
    body: JSON.stringify({ user_id: userId, permission_code: permissionCode }),
  }, token);
}

export async function setDocumentPermissions(
  documentId: string,
  userId: string,
  permissions: string[],
  token: string
): Promise<{ message: string }> {
  return apiClient<{ message: string }>(`/admin/documents/${encodeURIComponent(documentId)}/permissions`, {
    method: 'PUT',
    body: JSON.stringify({ user_id: userId, permissions }),
  }, token);
}

export const documentApi = {
  uploadDocument,
  updateDocumentDepartment,
  listDocuments,
  getDocumentPermissions,
  grantDocumentPermission,
  revokeDocumentPermission,
  setDocumentPermissions,
};


