/**
 * API client for Milestone 7A.1 / 7A.2: Chunk Curation & Rollback Lifecycle.
 */

import { apiClient } from './client';

export interface ChunkDetailResponse {
  id: string;
  document_id: string;
  chunk_index: number;
  text: string;
  chunk_hash: string;
  current_version: number;
  created_at: string;
  updated_at: string;
  page_number?: number | null;
  node_id?: string | null;
  is_managed_external: boolean;
}

export interface ChunkVersionResponse {
  id: string;
  chunk_id: string;
  version_number: number;
  text_before: string;
  text_after: string;
  diff_patch?: string | null;
  embedding_model: string;
  embedding_dim: number;
  chunk_hash: string;
  edited_by?: string | null;
  created_at: string;
  edit_reason?: string | null;
  is_rollback: boolean;
  reverted_from_version?: number | null;
}

export interface ChunkDiffResponse {
  chunk_id: string;
  v1: number;
  v2: number;
  text_v1: string;
  text_v2: string;
  diff_patch: string;
  lines_added: number;
  lines_removed: number;
  characters_changed: number;
}

export interface ChunkEditRequest {
  new_text: string;
  base_version: number;
  edit_reason?: string;
}

export interface ChunkRollbackRequest {
  target_version: number;
  rollback_reason?: string;
}

export interface ChunkEditMutationResponse {
  chunk: ChunkDetailResponse;
  version: ChunkVersionResponse;
}

export interface QualityCheckFlag {
  flag_id: string;
  severity: 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  suggested_remediation: 'MANUAL_CURATE' | 'REINGEST_DOCUMENT' | 'SOURCE_CORRECTION';
}

export interface QualityReportItem {
  report_id: string;
  reported_by: string;
  reported_at: string;
  issue_type: string;
  description: string;
  suggested_remediation: string;
  status: string;
}

export interface ChunkIssueReportRequest {
  issue_type: string;
  description: string;
  suggested_remediation: 'MANUAL_CURATE' | 'REINGEST_DOCUMENT' | 'SOURCE_CORRECTION';
}

export interface ChunkEvidenceResponse {
  chunk_id: string;
  document_id: string;
  chunk_index: number;
  file_name: string;
  file_path: string;
  mime_type: string;
  source_available: boolean;
  source_availability_reason?: string | null;
  source_url?: string | null;
  page_number?: number | null;
  node_id?: string | null;
  section_title?: string | null;
  section_ordinal?: string | null;
  section_abstract?: string | null;
  current_text: string;
  original_text: string;
  is_curated: boolean;
  current_version: number;
  created_at: string;
  updated_at: string;
  chunk_hash: string;
  is_managed_external: boolean;
  quality_flags: QualityCheckFlag[];
  quality_reports: QualityReportItem[];
}

export async function fetchChunkDetail(
  documentId: string,
  chunkId: string,
  token?: string
): Promise<ChunkDetailResponse> {
  return apiClient<ChunkDetailResponse>(
    `/documents/${encodeURIComponent(documentId)}/chunks/${encodeURIComponent(chunkId)}`,
    { method: 'GET' },
    token
  );
}

export async function fetchChunkVersions(
  documentId: string,
  chunkId: string,
  token?: string
): Promise<ChunkVersionResponse[]> {
  return apiClient<ChunkVersionResponse[]>(
    `/documents/${encodeURIComponent(documentId)}/chunks/${encodeURIComponent(chunkId)}/versions`,
    { method: 'GET' },
    token
  );
}

export async function fetchChunkDiff(
  documentId: string,
  chunkId: string,
  v1: number,
  v2: number,
  token?: string
): Promise<ChunkDiffResponse> {
  return apiClient<ChunkDiffResponse>(
    `/documents/${encodeURIComponent(documentId)}/chunks/${encodeURIComponent(chunkId)}/diff?v1=${v1}&v2=${v2}`,
    { method: 'GET' },
    token
  );
}

export async function editChunk(
  documentId: string,
  chunkId: string,
  payload: ChunkEditRequest,
  token?: string
): Promise<ChunkEditMutationResponse> {
  return apiClient<ChunkEditMutationResponse>(
    `/documents/${encodeURIComponent(documentId)}/chunks/${encodeURIComponent(chunkId)}`,
    {
      method: 'PATCH',
      body: JSON.stringify(payload),
    },
    token
  );
}

export async function rollbackChunk(
  documentId: string,
  chunkId: string,
  payload: ChunkRollbackRequest,
  token?: string
): Promise<ChunkEditMutationResponse> {
  return apiClient<ChunkEditMutationResponse>(
    `/documents/${encodeURIComponent(documentId)}/chunks/${encodeURIComponent(chunkId)}/rollback`,
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    token
  );
}

export async function fetchChunkEvidence(
  documentId: string,
  chunkId: string,
  token?: string
): Promise<ChunkEvidenceResponse> {
  return apiClient<ChunkEvidenceResponse>(
    `/documents/${encodeURIComponent(documentId)}/chunks/${encodeURIComponent(chunkId)}/evidence`,
    { method: 'GET' },
    token
  );
}

export async function reportChunkIssue(
  documentId: string,
  chunkId: string,
  payload: ChunkIssueReportRequest,
  token?: string
): Promise<{ message: string; report: QualityReportItem; total_reports: number }> {
  return apiClient<{ message: string; report: QualityReportItem; total_reports: number }>(
    `/documents/${encodeURIComponent(documentId)}/chunks/${encodeURIComponent(chunkId)}/report-issue`,
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    token
  );
}
