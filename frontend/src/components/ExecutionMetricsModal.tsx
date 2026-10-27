import React, { useState } from 'react'

export interface TokenUsage {
  input_tokens?: number
  output_tokens?: number
  total_tokens?: number
  reasoning_tokens?: number
  cache_read_tokens?: number
}

export interface SubStageMetric {
  name: string
  latency_ms: number
  details?: Record<string, any>
}

export interface StageMetrics {
  latency_ms: number
  llm_calls: number
  llm_usage: TokenUsage
  sub_stages: SubStageMetric[]
}

export interface ExecutionStages {
  retrieval?: StageMetrics
  generation?: StageMetrics
  research?: StageMetrics
}

export interface ExecutionMetrics {
  request_id?: string
  trace_id?: string
  workspace_id?: string
  thread_id?: string
  provider?: string
  model?: string
  total_latency_ms?: number
  input_tokens?: number
  output_tokens?: number
  total_tokens?: number
  total_llm_calls?: number
  usage_available?: boolean
  llm_usage?: TokenUsage
  retrieval_latency_ms?: number
  generation_latency_ms?: number
  grounding_latency_ms?: number
  repair_latency_ms?: number
  initial_generation_latency_ms?: number
  regeneration_latency_ms?: number
  planned_strategies?: string[]
  executed_strategies?: string[]
  vector_candidates?: number
  keyword_candidates?: number
  knowledge_candidates?: number
  hierarchical_candidates?: number
  candidate_count?: number
  reranked_candidates?: number
  accepted_results?: number
  accepted_count?: number
  final_context_count?: number
  context_chunks?: number
  context_tokens?: number
  vector_top_k?: number
  keyword_top_k?: number
  hierarchical_count?: number
  reranker_top_score?: number
  reranker_bottom_score?: number
  structured_evidence_count?: number
  hierarchical_evidence_count?: number
  conversation_evidence_count?: number
  sufficiency?: boolean
  sufficiency_status?: string
  evidence_found?: boolean
  sufficiency_confidence?: number
  sufficiency_coverage?: number
  sufficiency_reason?: string
  grounding?: boolean
  grounding_confidence?: number
  unsupported_claim_count?: number
  evidence_ref_count?: number
  repair_triggered?: boolean
  regeneration_triggered?: boolean
  safe_limitation_triggered?: boolean
  stages?: ExecutionStages
}

interface Props {
  metrics: ExecutionMetrics
  onClose: () => void
}

function formatDuration(ms: number | undefined | null): string {
  if (ms === undefined || ms === null || isNaN(ms)) return '0 ms'
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

export const ExecutionMetricsModal: React.FC<Props> = ({ metrics, onClose }) => {
  const [showRaw, setShowRaw] = useState(false)

  const totalTokens = metrics.total_tokens || metrics.llm_usage?.total_tokens || 0
  const inputTokens = metrics.input_tokens || metrics.llm_usage?.input_tokens || 0
  const outputTokens = metrics.output_tokens || metrics.llm_usage?.output_tokens || 0
  const llmCalls = metrics.total_llm_calls ?? metrics.stages?.generation?.llm_calls ?? 0
  const totalLatency = metrics.total_latency_ms || 0
  const acceptedCount = metrics.accepted_count || metrics.accepted_results || 0
  const candidateCount = metrics.candidate_count || (
    (metrics.vector_candidates || 0) + (metrics.keyword_candidates || 0) + (metrics.knowledge_candidates || 0)
  )

  const retrieval = metrics.stages?.retrieval
  const generation = metrics.stages?.generation

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        background: 'rgba(15, 23, 42, 0.65)',
        backdropFilter: 'blur(5px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
        animation: 'fadeIn 0.15s ease-out',
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: '#FFFFFF',
          borderRadius: '16px',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          border: '1px solid #E2E8F0',
          width: '100%',
          maxWidth: '780px',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
          color: '#1E293B',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: '16px 24px',
            borderBottom: '1px solid #E2E8F0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '10px',
                background: 'linear-gradient(135deg, #0284C7 0%, #0369A1 100%)',
                color: '#FFFFFF',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '18px',
                boxShadow: '0 2px 4px rgba(2, 132, 199, 0.25)',
              }}
            >
              ⚡
            </div>
            <div>
              <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 700, color: '#0F172A' }}>
                Query Execution Inspector
              </h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '2px' }}>
                {metrics.provider && (
                  <span style={{ fontSize: '11px', fontWeight: 600, color: '#0284C7', background: '#E0F2FE', padding: '1px 6px', borderRadius: '4px' }}>
                    {metrics.provider} {metrics.model ? `• ${metrics.model}` : ''}
                  </span>
                )}
                {metrics.request_id && (
                  <span style={{ fontSize: '10px', color: '#94A3B8', fontFamily: 'monospace' }}>
                    ID: {metrics.request_id.slice(0, 8)}...
                  </span>
                )}
              </div>
            </div>
          </div>

          <button
            onClick={onClose}
            title="Close"
            style={{
              background: '#F1F5F9',
              border: 'none',
              borderRadius: '8px',
              width: '32px',
              height: '32px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              color: '#64748B',
              fontSize: '16px',
              fontWeight: 700,
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#E2E8F0'
              e.currentTarget.style.color = '#0F172A'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = '#F1F5F9'
              e.currentTarget.style.color = '#64748B'
            }}
          >
            ✕
          </button>
        </div>

        {/* Scrollable Content */}
        <div style={{ padding: '20px 24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Row 1: Key Metrics Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '12px' }}>
            {/* LLM Calls */}
            <div style={{ background: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: '12px', padding: '12px 14px' }}>
              <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                🤖 LLM Calls
              </div>
              <div style={{ fontSize: '22px', fontWeight: 800, color: '#0F172A', marginTop: '4px' }}>
                {llmCalls} <span style={{ fontSize: '12px', fontWeight: 500, color: '#64748B' }}>call{llmCalls !== 1 ? 's' : ''}</span>
              </div>
              <div style={{ fontSize: '11px', color: '#0284C7', marginTop: '2px' }}>
                {llmCalls === 0 ? 'No LLM calls invoked' : (metrics.stages?.research ? 'Multi-hop graph research' : 'Standard Generation')}
              </div>
            </div>

            {/* Total Tokens */}
            <div style={{ background: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: '12px', padding: '12px 14px' }}>
              <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                ⚡ Total Tokens
              </div>
              <div style={{ fontSize: '22px', fontWeight: 800, color: '#0F172A', marginTop: '4px' }}>
                {totalTokens.toLocaleString()}
              </div>
              <div style={{ fontSize: '11px', color: '#64748B', marginTop: '2px' }}>
                In: <strong>{inputTokens.toLocaleString()}</strong> • Out: <strong>{outputTokens.toLocaleString()}</strong>
              </div>
            </div>

            {/* Total Latency */}
            <div style={{ background: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: '12px', padding: '12px 14px' }}>
              <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                ⏱️ Latency
              </div>
              <div style={{ fontSize: '22px', fontWeight: 800, color: '#0F172A', marginTop: '4px' }}>
                {formatDuration(totalLatency)}
              </div>
              <div style={{ fontSize: '11px', color: '#64748B', marginTop: '2px' }}>
                Retrieval: <strong>{metrics.stages?.retrieval || metrics.retrieval_latency_ms ? formatDuration(metrics.retrieval_latency_ms || retrieval?.latency_ms) : 'Not invoked'}</strong>
              </div>
            </div>

            {/* Evidence Yield */}
            <div style={{ background: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: '12px', padding: '12px 14px' }}>
              <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                📚 Evidence Chunks
              </div>
              <div style={{ fontSize: '22px', fontWeight: 800, color: '#0F172A', marginTop: '4px' }}>
                {acceptedCount} <span style={{ fontSize: '12px', fontWeight: 500, color: '#64748B' }}>accepted</span>
              </div>
              <div style={{ fontSize: '11px', color: '#64748B', marginTop: '2px' }}>
                {metrics.executed_strategies && metrics.executed_strategies.length > 0
                  ? `from ${candidateCount || acceptedCount} total candidates`
                  : (metrics.stages?.retrieval ? `from ${candidateCount || acceptedCount} total candidates` : 'Retrieval not invoked')}
              </div>
            </div>
          </div>

          {/* Row 2: Step-by-Step Retrieval Journey Timeline */}
          <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '12px', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: '#0F172A', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span>🔍</span>
                <span>Retrieval Process Journey</span>
              </div>
              <span style={{ fontSize: '11px', color: '#64748B' }}>
                Pipeline Total: {formatDuration(metrics.retrieval_latency_ms || retrieval?.latency_ms)}
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {/* Step 1: Query Intent & Strategies */}
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', padding: '8px 10px', background: '#F8FAFC', borderRadius: '8px' }}>
                <span style={{ fontSize: '13px', marginTop: '2px' }}>🎯</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '12px', fontWeight: 600, color: '#1E293B' }}>1. Planned Retrieval Strategies</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '4px' }}>
                    {metrics.planned_strategies && metrics.planned_strategies.length > 0 ? (
                      metrics.planned_strategies.map((stg, i) => (
                        <span key={i} style={{ fontSize: '10px', fontWeight: 600, color: '#0369A1', background: '#E0F2FE', padding: '2px 6px', borderRadius: '4px' }}>
                          {stg.replace(/_/g, ' ').toUpperCase()}
                        </span>
                      ))
                    ) : (
                      <span style={{ fontSize: '11px', color: '#94A3B8', fontStyle: 'italic' }}>
                        Retrieval not invoked
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Step 2: Multi-Source Candidate Search */}
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', padding: '8px 10px', background: '#F8FAFC', borderRadius: '8px' }}>
                <span style={{ fontSize: '13px', marginTop: '2px' }}>🔎</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '12px', fontWeight: 600, color: '#1E293B' }}>2. Candidate Search Results</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', marginTop: '4px', fontSize: '11px', color: '#475569' }}>
                    {(metrics.executed_strategies && metrics.executed_strategies.length > 0) || metrics.stages?.retrieval ? (
                      <>
                        {metrics.vector_candidates !== undefined && (
                          <span>Vector Candidates: <strong>{metrics.vector_candidates}</strong></span>
                        )}
                        {metrics.keyword_candidates !== undefined && (
                          <span>BM25 Keyword: <strong>{metrics.keyword_candidates}</strong></span>
                        )}
                        {metrics.knowledge_candidates ? (
                          <span>Knowledge Graph: <strong>{metrics.knowledge_candidates}</strong></span>
                        ) : null}
                        {metrics.conversation_evidence_count ? (
                          <span>Conversation Threads: <strong>{metrics.conversation_evidence_count}</strong></span>
                        ) : null}
                      </>
                    ) : (
                      <span style={{ color: '#94A3B8', fontStyle: 'italic' }}>Retrieval not invoked</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Step 3: CrossEncoder Reranking */}
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', padding: '8px 10px', background: '#F8FAFC', borderRadius: '8px' }}>
                <span style={{ fontSize: '13px', marginTop: '2px' }}>📊</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '12px', fontWeight: 600, color: '#1E293B' }}>3. CrossEncoder Reranking & Filtering</div>
                  <div style={{ fontSize: '11px', color: '#475569', marginTop: '2px' }}>
                    {metrics.reranked_candidates !== undefined && metrics.reranked_candidates > 0 ? (
                      <span>Evaluated <strong>{metrics.reranked_candidates}</strong> candidates through CrossEncoder.</span>
                    ) : ((metrics.executed_strategies && metrics.executed_strategies.length > 0) || metrics.stages?.retrieval ? (
                      <span>RRF (Reciprocal Rank Fusion) hybrid scoring applied.</span>
                    ) : (
                      <span style={{ color: '#94A3B8', fontStyle: 'italic' }}>Reranking not executed</span>
                    ))}
                    {metrics.reranker_top_score !== undefined && metrics.reranker_top_score !== null && (
                      <span style={{ marginLeft: '6px' }}>Top Score: <strong>{metrics.reranker_top_score.toFixed(3)}</strong></span>
                    )}
                  </div>
                </div>
              </div>

              {/* Step 4: Evidence Sufficiency Evaluation */}
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', padding: '8px 10px', background: '#F8FAFC', borderRadius: '8px' }}>
                <span style={{ fontSize: '13px', marginTop: '2px' }}>⚖️</span>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: '#1E293B' }}>4. Evidence Sufficiency</span>
                    {metrics.sufficiency !== undefined && metrics.sufficiency !== null && (
                      <span
                        style={{
                          fontSize: '10px',
                          fontWeight: 700,
                          padding: '1px 6px',
                          borderRadius: '4px',
                          background: metrics.sufficiency ? '#DCFCE7' : '#FEF3C7',
                          color: metrics.sufficiency ? '#15803D' : '#B45309',
                        }}
                      >
                        {metrics.sufficiency ? '✓ SUFFICIENT' : '⚠ INSUFFICIENT'}
                      </span>
                    )}
                  </div>
                  {metrics.sufficiency_reason && (
                    <div style={{ fontSize: '11px', color: '#64748B', marginTop: '3px', fontStyle: 'italic' }}>
                      "{metrics.sufficiency_reason}"
                    </div>
                  )}
                </div>
              </div>

              {/* Step 5: Sub-stages Latency list if present */}
              {retrieval?.sub_stages && retrieval.sub_stages.length > 0 && (
                <div style={{ borderTop: '1px dashed #CBD5E1', paddingTop: '6px', marginTop: '4px' }}>
                  <div style={{ fontSize: '10px', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', marginBottom: '4px' }}>
                    Detailed Retrieval Sub-stages
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '6px' }}>
                    {retrieval.sub_stages.map((stg, idx) => (
                      <div key={idx} style={{ fontSize: '11px', color: '#475569', display: 'flex', justifyContent: 'space-between', background: '#F1F5F9', padding: '3px 8px', borderRadius: '4px' }}>
                        <span>{stg.name.replace(/_/g, ' ')}</span>
                        <strong>{formatDuration(stg.latency_ms)}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Row 3: Quality & Grounding Verification */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            {/* Grounding Verification */}
            <div style={{ background: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: '12px', padding: '14px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '12px', fontWeight: 600, color: '#1E293B' }}>🛡️ Grounding Verification</span>
                {metrics.grounding !== undefined && metrics.grounding !== null ? (
                  <span
                    style={{
                      fontSize: '10px',
                      fontWeight: 700,
                      padding: '2px 6px',
                      borderRadius: '4px',
                      background: metrics.grounding ? '#DCFCE7' : '#FEE2E2',
                      color: metrics.grounding ? '#15803D' : '#B91C1C',
                    }}
                  >
                    {metrics.grounding ? '✓ GROUNDED' : '✖ UNGROUNDED'}
                  </span>
                ) : (
                  <span
                    style={{
                      fontSize: '10px',
                      fontWeight: 600,
                      padding: '2px 6px',
                      borderRadius: '4px',
                      background: '#F1F5F9',
                      color: '#64748B',
                    }}
                  >
                    NOT EVALUATED
                  </span>
                )}
              </div>
              <div style={{ fontSize: '11px', color: '#64748B', marginTop: '6px' }}>
                Confidence: <strong>{metrics.grounding_confidence !== undefined && metrics.grounding_confidence !== null ? `${Math.round(metrics.grounding_confidence * 100)}%` : (metrics.grounding !== undefined && metrics.grounding !== null ? '0%' : 'Not evaluated')}</strong>
              </div>
              {metrics.repair_triggered && (
                <div style={{ fontSize: '10px', color: '#0284C7', marginTop: '4px', fontWeight: 600 }}>
                  ⚡ Auto-repair: Corrected unsupported phrasing
                </div>
              )}
            </div>

            {/* Generation & Token Reserve */}
            <div style={{ background: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: '12px', padding: '14px' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#1E293B' }}>💬 Generation Breakdown</div>
              <div style={{ fontSize: '11px', color: '#64748B', marginTop: '6px', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                <div>Generation Time: <strong>{metrics.stages?.generation || metrics.generation_latency_ms ? formatDuration(metrics.generation_latency_ms || generation?.latency_ms) : 'Not invoked'}</strong></div>
                <div>Tokens Output: <strong>{outputTokens > 0 ? `${outputTokens.toLocaleString()} tokens` : (metrics.stages?.generation ? '0 tokens' : 'N/A')}</strong></div>
              </div>
            </div>
          </div>

          {/* Toggle Raw JSON */}
          <div>
            <button
              onClick={() => setShowRaw(!showRaw)}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#0284C7',
                fontSize: '11px',
                fontWeight: 600,
                cursor: 'pointer',
                padding: '0',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}
            >
              <span>{showRaw ? '▲ Hide Raw Metrics JSON' : '▼ View Raw Metrics JSON'}</span>
            </button>
            {showRaw && (
              <pre
                style={{
                  marginTop: '8px',
                  background: '#0F172A',
                  color: '#F8FAFC',
                  borderRadius: '8px',
                  padding: '12px',
                  fontSize: '11px',
                  fontFamily: 'monospace',
                  overflowX: 'auto',
                  maxHeight: '200px',
                }}
              >
                {JSON.stringify(metrics, null, 2)}
              </pre>
            )}
          </div>
        </div>

        {/* Footer */}
        <div
          style={{
            padding: '12px 24px',
            borderTop: '1px solid #E2E8F0',
            background: '#F8FAFC',
            display: 'flex',
            justifyContent: 'flex-end',
          }}
        >
          <button
            onClick={onClose}
            style={{
              padding: '6px 16px',
              fontSize: '12px',
              fontWeight: 600,
              color: '#FFFFFF',
              background: '#0284C7',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              transition: 'background 0.15s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = '#0369A1')}
            onMouseLeave={(e) => (e.currentTarget.style.background = '#0284C7')}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  )
}
