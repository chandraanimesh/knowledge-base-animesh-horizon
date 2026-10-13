import { apiClient } from './client'

export interface CompanyNoteEvidenceItem {
  id: string
  source_type: 'DOCUMENT' | 'WEBSITE' | string
  document_chunk_id?: string | null
  document_id?: string | null
  file_name?: string | null
  original_file_name?: string | null
  page_number?: number | null
  section_title?: string | null
  section_ordinal?: string | null
  source_url?: string | null
  source_title?: string | null
  user_excerpt?: string | null
  verification_status?: 'NOT_VERIFIED' | 'VERIFIED' | 'FAILED' | string | null
  verified_at?: string | null
  excerpt: string
  extraction_method?: string | null
  confidence?: number | null
  created_at: string
}

export interface CompanyNote {
  id: string
  canonical_key: string
  company_name: string
  statement: string
  observation_type: 'SUPPORTED' | 'INFERRED' | 'UNVERIFIED' | string
  kind: string
  status: string
  visibility: 'PRIVATE' | 'WORKSPACE' | 'SHARED' | string
  approval_status: 'DRAFT' | 'PENDING' | 'APPROVED' | 'REJECTED' | string
  source_type: string
  workspace_id?: string | null
  workspace_name?: string | null
  department_id?: string | null
  department_code?: string | null
  author_id?: string | null
  author_email?: string | null
  approved_by_id?: string | null
  approved_by_email?: string | null
  approved_at?: string | null
  confidence?: number | null
  created_at: string
  updated_at: string
  evidence_count: number
  evidence: CompanyNoteEvidenceItem[]
}

export interface WebsiteSourceInput {
  url: string
  title?: string
  user_excerpt?: string
}

export interface CreateNotePayload {
  company_name: string
  statement: string
  observation_type: 'SUPPORTED' | 'INFERRED' | 'UNVERIFIED'
  workspace_id?: string | null
  department_id?: string | null
  visibility: 'PRIVATE' | 'WORKSPACE'
  document_chunk_ids?: string[]
  website_sources?: WebsiteSourceInput[]
  attributes?: Record<string, any>
}

export interface UpdateNotePayload {
  company_name?: string
  statement?: string
  observation_type?: 'SUPPORTED' | 'INFERRED' | 'UNVERIFIED'
  workspace_id?: string | null
  department_id?: string | null
  visibility?: 'PRIVATE' | 'WORKSPACE'
  attributes?: Record<string, any>
}

export interface AttachEvidencePayload {
  document_chunk_id?: string
  source_url?: string
  source_title?: string
  extraction_method?: string
  confidence?: number
  custom_excerpt?: string
}

export interface DocumentChunkInfo {
  id: string
  document_id: string
  chunk_index: number
  page_number?: number | null
  section_title?: string | null
  section_ordinal?: string | null
  excerpt: string
  file_name?: string | null
  original_file_name?: string | null
}

export async function fetchDocumentChunks(token: string, documentId: string): Promise<DocumentChunkInfo[]> {
  return apiClient<DocumentChunkInfo[]>(`/documents/${encodeURIComponent(documentId)}/chunks`, {
    method: 'GET',
  }, token)
}

export interface BatchPublishPayload {
  note_ids: string[]
  target_audience?: 'SHARED' | 'WORKSPACE'
  target_department_id?: string | null
}

export interface BatchPublishOutcome {
  note_id: string
  canonical_key?: string | null
  company_name?: string | null
  success: boolean
  status: string
  error?: string | null
}

export interface BatchPublishResponse {
  batch_id: string
  total_items: number
  published_count: number
  failed_count: number
  is_idempotent_retry: boolean
  outcomes: BatchPublishOutcome[]
}

export async function fetchCompanyNotes(
  token: string,
  params?: {
    company_name?: string
    visibility?: string
    approval_status?: string
    workspace_id?: string
    limit?: number
  }
): Promise<CompanyNote[]> {
  const query = new URLSearchParams()
  if (params?.company_name) query.append('company_name', params.company_name)
  if (params?.visibility) query.append('visibility', params.visibility)
  if (params?.approval_status) query.append('approval_status', params.approval_status)
  if (params?.workspace_id) query.append('workspace_id', params.workspace_id)
  if (params?.limit) query.append('limit', String(params.limit))

  const qs = query.toString() ? `?${query.toString()}` : ''
  return apiClient<CompanyNote[]>(`/knowledge/company-notes${qs}`, { method: 'GET' }, token)
}

export async function createCompanyNote(token: string, payload: CreateNotePayload): Promise<CompanyNote> {
  return apiClient<CompanyNote>('/knowledge/company-notes', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, token)
}

export async function updateCompanyNote(token: string, noteId: string, payload: UpdateNotePayload): Promise<CompanyNote> {
  return apiClient<CompanyNote>(`/knowledge/company-notes/${noteId}`, {
    method: 'PATCH',
    body: JSON.stringify(payload),
  }, token)
}

export async function attachNoteEvidence(token: string, noteId: string, payload: AttachEvidencePayload): Promise<CompanyNote> {
  return apiClient<CompanyNote>(`/knowledge/company-notes/${noteId}/evidence`, {
    method: 'POST',
    body: JSON.stringify(payload),
  }, token)
}

export async function removeNoteEvidence(token: string, noteId: string, evidenceId: string): Promise<CompanyNote> {
  return apiClient<CompanyNote>(`/knowledge/company-notes/${noteId}/evidence/${evidenceId}`, {
    method: 'DELETE',
  }, token)
}

export async function shareNoteToWorkspace(token: string, noteId: string, workspaceId: string): Promise<CompanyNote> {
  return apiClient<CompanyNote>(`/knowledge/company-notes/${noteId}/share?workspace_id=${workspaceId}`, {
    method: 'POST',
  }, token)
}

export async function submitCompanyNote(
  token: string,
  noteId: string,
  payload: { target_audience?: 'WORKSPACE' | 'SHARED'; target_workspace_id?: string; target_department_id?: string }
): Promise<CompanyNote> {
  return apiClient<CompanyNote>(`/knowledge/company-notes/${noteId}/submit`, {
    method: 'POST',
    body: JSON.stringify(payload),
  }, token)
}

export async function reviewCompanyNote(
  token: string,
  noteId: string,
  payload: { action: 'APPROVE' | 'REJECT' | 'REVISE'; reason?: string }
): Promise<CompanyNote> {
  return apiClient<CompanyNote>(`/knowledge/company-notes/${noteId}/review`, {
    method: 'POST',
    body: JSON.stringify(payload),
  }, token)
}

export async function batchPublishCompanyNotes(
  token: string,
  payload: BatchPublishPayload
): Promise<BatchPublishResponse> {
  return apiClient<BatchPublishResponse>('/knowledge/company-notes/batch-publish', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, token)
}

export async function archiveCompanyNote(
  token: string,
  noteId: string,
  reason?: string
): Promise<CompanyNote> {
  return apiClient<CompanyNote>(`/knowledge/company-notes/${noteId}/archive`, {
    method: 'POST',
    body: JSON.stringify(reason ? { reason } : {}),
  }, token)
}

export async function supersedeCompanyNote(
  token: string,
  noteId: string,
  replacementNoteId: string
): Promise<CompanyNote> {
  return apiClient<CompanyNote>(`/knowledge/company-notes/${noteId}/supersede`, {
    method: 'POST',
    body: JSON.stringify({ replacement_note_id: replacementNoteId }),
  }, token)
}
