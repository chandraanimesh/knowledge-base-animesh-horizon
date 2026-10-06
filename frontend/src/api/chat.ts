import { apiClient } from './client';

export interface RetrievedChunk {
  chunk_id: string;
  document_id: string;
  department_id: string;
  file_name: string;
  chunk_index: number;
  text: string;
  distance?: number;
  page_number?: number;
}

export interface GenerateRequest {
  query: string;
  top_k?: number;
  rerank?: boolean;
  conversation_id?: string | null;
  workspace_id?: string | null;
  provider_overrides?: Record<string, string> | null;
  action?: string | null;
  target_message_id?: string | null;
}

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  reasoning_tokens: number;
  cache_read_tokens: number;
}

export interface SubStageMetric {
  name: string;
  latency_ms: number;
}

export interface StageMetrics {
  latency_ms: number;
  llm_calls: number;
  llm_usage: TokenUsage;
  sub_stages?: SubStageMetric[];
}

export interface ExecutionStages {
  retrieval?: StageMetrics | null;
  generation?: StageMetrics | null;
  research?: StageMetrics | null;
}

export interface ExecutionMetrics {
  request_id: string;
  trace_id?: string | null;
  workspace_id?: string | null;
  thread_id?: string | null;
  provider?: string | null;
  model?: string | null;
  total_latency_ms: number;
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  usage_available?: boolean;
  llm_usage: TokenUsage;

  retrieval_latency_ms?: number;
  generation_latency_ms?: number;
  grounding_latency_ms?: number;
  repair_latency_ms?: number;
  initial_generation_latency_ms?: number;
  regeneration_latency_ms?: number;

  planned_strategies?: string[];
  executed_strategies?: string[];
  vector_candidates?: number;
  keyword_candidates?: number;
  knowledge_candidates?: number;
  hierarchical_candidates?: number;
  candidate_count?: number;
  reranked_candidates?: number;
  accepted_results?: number;
  accepted_count?: number;
  final_context_count?: number;
  context_chunks?: number;
  context_tokens?: number;

  reranker_top_score?: number | null;
  reranker_bottom_score?: number | null;
  structured_evidence_count?: number;
  hierarchical_evidence_count?: number;

  sufficiency?: boolean | null;
  sufficiency_confidence?: number | null;
  sufficiency_coverage?: number | null;
  sufficiency_reason?: string | null;
  missing_requirements_count?: number;
  contradiction_count?: number;
  evidence_types?: string[];

  grounding?: boolean | null;
  grounding_confidence?: number | null;
  unsupported_claim_count?: number;
  evidence_ref_count?: number;
  repair_triggered?: boolean;
  regeneration_triggered?: boolean;
  safe_limitation_triggered?: boolean;

  visualization_requested?: boolean;
  visualization_generated?: boolean;
  visualization_validated?: boolean;
  visualization_repaired?: boolean;
  visualization_dropped?: boolean;
  visualization_failure_reason?: string | null;

  streaming?: boolean;
  failure_categories?: string[];
  stages: ExecutionStages;
}

import { VisualizationSpec } from '../components/visualizations/types';

export interface GenerateResponse {
  answer: string;
  visualization?: VisualizationSpec | null;
  citations: string[];
  retrieved_chunks: RetrievedChunk[];
  execution_metrics?: ExecutionMetrics | null;
  conversation_id?: string | null;
  message_id?: string | null;
  conversation_title?: string | null;
}

export interface ConversationItem {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  workspace_id?: string | null;
  workspace_name?: string | null;
}

export interface ChatMessageRecord {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  visualization?: VisualizationSpec | null;
  sources?: Array<{ title: string; excerpt: string }> | null;
  execution_metrics?: ExecutionMetrics | null;
  created_at: string;
}

export interface ConversationDetail {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  workspace_id?: string | null;
  messages: ChatMessageRecord[];
}

export async function askQuestion(
  request: GenerateRequest,
  token: string
): Promise<GenerateResponse> {
  return apiClient<GenerateResponse>('/chat/ask', {
    method: 'POST',
    body: JSON.stringify(request),
  }, token);
}

export async function listConversations(token: string, workspaceId?: string | null): Promise<ConversationItem[]> {
  const query = workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : '';
  return apiClient<ConversationItem[]>(`/conversations${query}`, {
    method: 'GET',
  }, token);
}

export async function getConversation(id: string, token: string): Promise<ConversationDetail> {
  return apiClient<ConversationDetail>(`/conversations/${id}`, {
    method: 'GET',
  }, token);
}

export async function createConversation(title?: string, token?: string, workspaceId?: string | null): Promise<ConversationItem> {
  const params = new URLSearchParams();
  if (title) params.append('title', title);
  if (workspaceId) params.append('workspace_id', workspaceId);
  const qs = params.toString() ? `?${params.toString()}` : '';
  return apiClient<ConversationItem>(`/conversations${qs}`, {
    method: 'POST',
  }, token);
}

export async function deleteConversation(id: string, token: string): Promise<{ status: string; id: string }> {
  return apiClient<{ status: string; id: string }>(`/conversations/${id}`, {
    method: 'DELETE',
  }, token);
}

