import { apiClient } from './client';

export interface PrivacyPolicyRecord {
  id?: string;
  scope_type: 'TENANT' | 'WORKSPACE' | 'USER';
  scope_id: string;
  memory_collection_enabled: boolean;
  memory_extraction_enabled: boolean;
  memory_dreaming_enabled: boolean;
  memory_retrieval_enabled: boolean;
  retention_days?: number | null;
  auto_expiration_enabled: boolean;
  user_deletion_enabled: boolean;
  profile_scrubbing_enabled: boolean;
  memory_export_enabled: boolean;
  user_can_pause: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface PrivacyPolicyPayload {
  scope_type?: 'TENANT' | 'WORKSPACE' | 'USER';
  scope_id?: string;
  memory_collection_enabled?: boolean;
  memory_extraction_enabled?: boolean;
  memory_dreaming_enabled?: boolean;
  memory_retrieval_enabled?: boolean;
  retention_days?: number | null;
  auto_expiration_enabled?: boolean;
  user_deletion_enabled?: boolean;
  profile_scrubbing_enabled?: boolean;
  memory_export_enabled?: boolean;
  user_can_pause?: boolean;
}

export interface MemoryPrivacyStatusResponse {
  user_id: string;
  collection_state: 'ENABLED' | 'PAUSED' | 'DISABLED';
  effective_policy: PrivacyPolicyRecord;
  total_active_memories: number;
  total_expired_memories: number;
  total_purged_memories: number;
  living_profile_summary: Record<string, any>;
  last_dreaming_run?: string | null;
}

export interface MemoryExportItem {
  id: string;
  canonical_key: string;
  memory_type: string;
  status: string;
  content: string;
  structured_data?: Record<string, any>;
  confidence?: number;
  source_type?: string;
  created_at: string;
  updated_at: string;
}

export interface MemoryExportResponse {
  user_id: string;
  workspace_id?: string | null;
  exported_at: string;
  total_memories: number;
  memories: MemoryExportItem[];
}

export async function getEffectivePolicy(token: string, workspaceId?: string): Promise<PrivacyPolicyRecord> {
  const url = workspaceId ? `/api/memory/privacy/policy?workspace_id=${workspaceId}` : '/api/memory/privacy/policy';
  return apiClient<PrivacyPolicyRecord>(url, { method: 'GET' }, token);
}

export async function updatePrivacyPolicy(token: string, payload: PrivacyPolicyPayload): Promise<PrivacyPolicyRecord> {
  return apiClient<PrivacyPolicyRecord>('/api/memory/privacy/policy', {
    method: 'PUT',
    body: JSON.stringify(payload),
  }, token);
}

export async function pauseMemoryCollection(token: string): Promise<PrivacyPolicyRecord> {
  return apiClient<PrivacyPolicyRecord>('/api/memory/privacy/pause', { method: 'POST' }, token);
}

export async function resumeMemoryCollection(token: string): Promise<PrivacyPolicyRecord> {
  return apiClient<PrivacyPolicyRecord>('/api/memory/privacy/resume', { method: 'POST' }, token);
}

export async function getMemoryPrivacyStatus(token: string, workspaceId?: string): Promise<MemoryPrivacyStatusResponse> {
  const url = workspaceId ? `/api/memory/privacy/status?workspace_id=${workspaceId}` : '/api/memory/privacy/status';
  return apiClient<MemoryPrivacyStatusResponse>(url, { method: 'GET' }, token);
}

export async function deleteMemoryRecord(token: string, memoryId: string): Promise<{ status: string; memory_id: string }> {
  return apiClient<{ status: string; memory_id: string }>(`/api/memory/${memoryId}`, { method: 'DELETE' }, token);
}

export async function deleteAllPersonalMemories(token: string, workspaceId?: string): Promise<{ status: string; deleted_count: number }> {
  const url = workspaceId ? `/api/memory?workspace_id=${workspaceId}` : '/api/memory';
  return apiClient<{ status: string; deleted_count: number }>(url, { method: 'DELETE' }, token);
}

export async function scrubProfileAttribute(
  token: string,
  section: string,
  key: string
): Promise<{ status: string; section: string; key: string; scrubbed_memory_records: number; updated_profile: any }> {
  return apiClient<{ status: string; section: string; key: string; scrubbed_memory_records: number; updated_profile: any }>(
    '/api/memory/profile/scrub',
    {
      method: 'POST',
      body: JSON.stringify({ section, key }),
    },
    token
  );
}

export async function exportUserMemories(token: string, workspaceId?: string): Promise<MemoryExportResponse> {
  const url = workspaceId ? `/api/memory/export?workspace_id=${workspaceId}` : '/api/memory/export';
  return apiClient<MemoryExportResponse>(url, { method: 'GET' }, token);
}

// ── Milestone 9: Intelligent Memory & Dreaming Engine API ──────────────────

export interface MemoryOverview {
  user_id: string;
  total_memories: number;
  active_memories: number;
  by_type: Record<string, number>;
  by_status: Record<string, number>;
  total_relationships: number;
  total_dreaming_runs: number;
  last_dreaming_run?: string | null;
  living_profile_summary: Record<string, any>;
}

export interface MemoryProcessingJobRecord {
  id: string;
  source_message_id?: string | null;
  conversation_id?: string | null;
  owner_user_id: string;
  workspace_id?: string | null;
  source_text_snapshot: string;
  trigger_kind: string;
  status: string;
  attempt_count: number;
  max_attempts: number;
  error_message?: string | null;
  created_at: string;
  processed_at?: string | null;
}

export interface MemoryRelationshipRecord {
  id: string;
  source_memory_id: string;
  target_memory_id: string;
  relationship_type: 'DUPLICATE' | 'COMPLEMENTS' | 'SUPERSEDES' | 'CONTRADICTS' | 'RELATED' | 'UNRESOLVED' | string;
  confidence: number;
  reason?: string | null;
  source_evidence?: Record<string, any> | null;
  created_at: string;
}

export interface MemoryDreamingRunRecord {
  id: string;
  run_id: string;
  user_id: string;
  workspace_id?: string | null;
  trigger_type: string;
  status: string;
  started_at: string;
  completed_at?: string | null;
  memories_inspected: number;
  memories_skipped: number;
  duplicates_detected: number;
  conflicts_detected: number;
  new_memories_created: number;
  memories_superseded: number;
  profile_updated: boolean;
  error_message?: string | null;
  duration_seconds?: number | null;
}

export interface MemoryEvalMetricRecord {
  id: string;
  eval_run_id: string;
  scenario_name: string;
  metric_category: string;
  metrics_json: Record<string, any>;
  passed: boolean;
  details?: string | null;
  created_at: string;
}

export async function getMemoryOverview(token: string): Promise<MemoryOverview> {
  return apiClient<MemoryOverview>('/api/memory/overview', { method: 'GET' }, token);
}

export async function listMemoryJobs(token: string, limit = 50): Promise<MemoryProcessingJobRecord[]> {
  return apiClient<MemoryProcessingJobRecord[]>(`/api/memory/jobs?limit=${limit}`, { method: 'GET' }, token);
}

export async function listMemoryRelationships(token: string, limit = 100): Promise<MemoryRelationshipRecord[]> {
  return apiClient<MemoryRelationshipRecord[]>(`/api/memory/relationships?limit=${limit}`, { method: 'GET' }, token);
}

export async function listDreamingRuns(token: string, limit = 50): Promise<MemoryDreamingRunRecord[]> {
  return apiClient<MemoryDreamingRunRecord[]>(`/api/memory/dreaming/runs?limit=${limit}`, { method: 'GET' }, token);
}

export async function triggerDreamingRun(
  token: string,
  threshold?: number,
  maxBatchSize = 50,
  workspaceId?: string
): Promise<any> {
  let url = `/api/memory/dreaming/run?max_batch_size=${maxBatchSize}`;
  if (threshold) url += `&threshold=${threshold}`;
  if (workspaceId) url += `&workspace_id=${workspaceId}`;
  return apiClient<any>(url, { method: 'POST' }, token);
}

export async function getEvaluationMetrics(token: string, limit = 50): Promise<MemoryEvalMetricRecord[]> {
  return apiClient<MemoryEvalMetricRecord[]>(`/api/memory/evaluation?limit=${limit}`, { method: 'GET' }, token);
}

export async function triggerEvaluationRun(token: string): Promise<any> {
  return apiClient<any>('/api/memory/evaluation/run', { method: 'POST' }, token);
}


