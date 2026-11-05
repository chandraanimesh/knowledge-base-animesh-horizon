import { apiClient } from './client';

export interface IngestResponse {
  file_name: string;
  skipped: boolean;
  document_id?: string;
  chunks_created: number;
  reason?: string;
  failed_chunks?: number;
  toxic_chunks_isolated?: Array<{
    chunk_index: number;
    text_preview: string;
    classification: string;
    error: string;
  }>;
}

export interface CheckHashResponse {
  exists: boolean;
  document_id?: string;
  file_name?: string;
  scope?: string;
}

export async function computeFileSHA256(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

export async function checkDocumentHash(
  hash: string,
  departmentCode: string,
  token: string,
  workspaceId?: string
): Promise<CheckHashResponse> {
  let url = `/documents/check-hash?hash=${encodeURIComponent(hash)}&department_code=${encodeURIComponent(departmentCode)}`;
  if (workspaceId) {
    url += `&workspace_id=${encodeURIComponent(workspaceId)}`;
  }
  return apiClient<CheckHashResponse>(url, { method: 'GET' }, token);
}

export interface DocumentMetadataPayload {
  doc_type?: string;
  tags?: string[];
  description?: string;
  classification?: string;
  visibility?: string;
}

export async function uploadDocument(
  file: File,
  departmentCode: string,
  token: string,
  workspaceId?: string,
  metadata?: DocumentMetadataPayload
): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append('file', file);

  if (metadata?.doc_type) {
    formData.append('doc_type', metadata.doc_type);
  }
  if (metadata?.tags && metadata.tags.length > 0) {
    formData.append('tags', JSON.stringify(metadata.tags));
  }
  if (metadata?.description) {
    formData.append('description', metadata.description);
  }
  if (metadata?.classification) {
    formData.append('classification', metadata.classification);
  }
  if (metadata?.visibility) {
    formData.append('visibility', metadata.visibility);
  }
  
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
  original_file_name?: string;
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
  original_file_name?: string;
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

export function getDocumentDisplayName(
  doc?: { file_name?: string | null; original_file_name?: string | null } | null
): string {
  if (!doc) return 'Authorized Document';
  return doc.original_file_name || doc.file_name || 'Authorized Document';
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

export interface DocumentVersionItem {
  id: string;
  document_id: string;
  version_number: number;
  parent_version_id?: string | null;
  source_content_hash: string;
  extraction_config_version: string;
  chunking_config_version: string;
  embedding_model: string;
  embedding_dim: number;
  status: 'DETECTED' | 'PROCESSING' | 'SHADOW_VALIDATION' | 'PENDING_APPROVAL' | 'PUBLISHED' | 'FAILED' | 'SUPERSEDED' | 'ROLLED_BACK';
  is_active: boolean;
  change_summary?: Record<string, any> | null;
  created_at: string;
  published_at?: string | null;
  rolled_back_at?: string | null;
}

export interface DocumentVersionListResponse {
  document_id: string;
  current_version_id?: string | null;
  versions: DocumentVersionItem[];
}

export interface ShadowReportResponse {
  id: string;
  document_version_id: string;
  comparison_status: 'EXACT_MATCH' | 'ACCEPTABLE_DIFFERENCE' | 'MATERIAL_DIFFERENCE' | 'FAILED';
  can_publish: boolean;
  reasons?: string[] | null;
  page_diff_summary?: Record<string, any> | null;
  chunk_diff_summary?: Record<string, any> | null;
  embedding_diff_summary?: Record<string, any> | null;
  knowledge_diff_summary?: Record<string, any> | null;
  retrieval_diff_summary?: Record<string, any> | null;
  created_at: string;
}

export async function listDocumentVersions(
  documentId: string,
  token: string
): Promise<DocumentVersionListResponse> {
  return apiClient<DocumentVersionListResponse>(
    `/api/documents/${encodeURIComponent(documentId)}/versions`,
    { method: 'GET' },
    token
  );
}

export async function getDocumentVersion(
  documentId: string,
  versionId: string,
  token: string
): Promise<DocumentVersionItem> {
  return apiClient<DocumentVersionItem>(
    `/api/documents/${encodeURIComponent(documentId)}/versions/${encodeURIComponent(versionId)}`,
    { method: 'GET' },
    token
  );
}

export async function getShadowReport(
  documentId: string,
  versionId: string,
  token: string
): Promise<ShadowReportResponse> {
  return apiClient<ShadowReportResponse>(
    `/api/documents/${encodeURIComponent(documentId)}/versions/${encodeURIComponent(versionId)}/shadow-report`,
    { method: 'GET' },
    token
  );
}

export async function publishDocumentVersion(
  documentId: string,
  versionId: string,
  token: string
): Promise<{ success: boolean; version_id: string; status: string; published_at?: string }> {
  return apiClient<{ success: boolean; version_id: string; status: string; published_at?: string }>(
    `/api/documents/${encodeURIComponent(documentId)}/versions/${encodeURIComponent(versionId)}/publish`,
    { method: 'POST' },
    token
  );
}

export async function rollbackDocumentVersion(
  documentId: string,
  versionId: string,
  targetVersionId: string,
  token: string
): Promise<{ success: boolean; document_id: string; rolled_back_version_id: string; active_version_id: string; status: string }> {
  return apiClient<{ success: boolean; document_id: string; rolled_back_version_id: string; active_version_id: string; status: string }>(
    `/api/documents/${encodeURIComponent(documentId)}/versions/${encodeURIComponent(versionId)}/rollback`,
    {
      method: 'POST',
      body: JSON.stringify({ target_version_id: targetVersionId }),
    },
    token
  );
}

export const documentApi = {
  uploadDocument,
  updateDocumentDepartment,
  listDocuments,
  getDocumentPermissions,
  grantDocumentPermission,
  revokeDocumentPermission,
  setDocumentPermissions,
  listDocumentVersions,
  getDocumentVersion,
  getShadowReport,
  publishDocumentVersion,
  rollbackDocumentVersion,
};


