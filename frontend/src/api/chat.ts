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
  output_preferences?: OutputPreferences | null;
  execution_mode?: 'fast_path' | 'adaptive_workflow' | null;
  workflow_idempotency_key?: string | null;
}

export interface ArtifactMetadata {
  id: string;
  format: 'pdf' | 'csv' | 'docx' | 'json';
  filename: string;
  mime_type: string;
  size_bytes: number;
  download_url: string;
  created_at: string;
  expires_at?: string | null;
  is_expired?: boolean;
  error?: string | null;
}

export interface OutputPreferences {
  chat_summary?: boolean;
  include_inline_json?: boolean;
  include_inline_table?: boolean;
  download_formats?: Array<'pdf' | 'csv' | 'docx' | 'json'>;
}

export interface CoverageReport {
  retrieval_coverage: 'sampled_evidence' | 'exhaustive_census';
  export_truncation: {
    is_truncated: boolean;
    total_extracted_rows: number;
    exported_rows_count: number;
    truncation_reason?: string | null;
  };
  disclaimer: string;
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

export interface SourceItem {
  type?: 'document' | 'web_research' | 'shared_knowledge' | 'thread' | 'personal_memory' | string;
  chunk_id?: string;
  document_id?: string;
  department_id?: string;
  file_name?: string;
  chunk_index?: number;
  page_number?: number | null;
  title?: string;
  excerpt?: string;
  url?: string | null;
  thread_id?: string;
  thread_title?: string;
  workspace_name?: string;
  created_at?: string | null;
  [key: string]: any;
}

export interface GenerateResponse {
  answer: string;
  visualization?: VisualizationSpec | null;
  citations: string[];
  retrieved_chunks: RetrievedChunk[];
  sources?: SourceItem[] | null;
  execution_metrics?: ExecutionMetrics | null;
  conversation_id?: string | null;
  message_id?: string | null;
  conversation_title?: string | null;
  artifacts?: ArtifactMetadata[];
  coverage?: CoverageReport | null;
  workflow_run_id?: string | null;
  workflow_status?: string | null;
  routing_mode?: string | null;
  routing_reason?: string | null;
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
  sources?: SourceItem[] | null;
  execution_metrics?: ExecutionMetrics | null;
  created_at: string;
  artifacts?: ArtifactMetadata[];
  coverage?: CoverageReport | null;
  workflow_run_id?: string | null;
  workflow_status?: string | null;
  routing_mode?: string | null;
  routing_reason?: string | null;
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

export interface StreamChatHandlers {
  onStart?: (event: { request_id: string; conversation_id: string; query: string; routing_mode?: string; routing_reason?: string }) => void;
  onToken?: (delta: string) => void;
  onToolStart?: (event: { tool: string; input: any }) => void;
  onTrace?: (trace: any) => void;
  onDone?: (data: GenerateResponse) => void;
  onError?: (err: Error) => void;
}

export async function streamChat(
  request: GenerateRequest,
  token: string,
  handlers: StreamChatHandlers,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch('/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(errText || `Streaming failed with status ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('No readable stream available in response');

  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const block of lines) {
        if (!block.trim()) continue;
        const eventMatch = block.match(/^event:\s*(\w+)/m);
        const dataMatch = block.match(/^data:\s*(.+)$/m);

        const eventType = eventMatch ? eventMatch[1] : 'token';
        const rawData = dataMatch ? dataMatch[1] : '{}';

        try {
          const parsed = JSON.parse(rawData);
          if (eventType === 'start') {
            handlers.onStart?.(parsed);
          } else if (eventType === 'token' && parsed.delta) {
            handlers.onToken?.(parsed.delta);
          } else if (eventType === 'tool_start') {
            handlers.onToolStart?.(parsed);
          } else if (eventType === 'trace') {
            handlers.onTrace?.(parsed);
          } else if (eventType === 'done') {
            handlers.onDone?.(parsed);
          } else if (eventType === 'error') {
            handlers.onError?.(new Error(parsed.message || 'Stream error'));
          }
        } catch {
          // If parsing block fails, ignore malformed fragment
        }
      }
    }
  } catch (err: any) {
    if (signal?.aborted) return;
    handlers.onError?.(err);
    throw err;
  }
}

