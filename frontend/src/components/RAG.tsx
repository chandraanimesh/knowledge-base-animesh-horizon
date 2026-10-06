import { useState, useRef, useEffect, ChangeEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  askQuestion,
  createConversation,
  RetrievedChunk,
  ExecutionMetrics,
  getConversation,
} from '../api/chat'
import { uploadDocument } from '../api/documents'
import {
  listWorkspaces,
  WorkspaceItem,
  getProviderCapabilities,
  ProviderCapabilitiesSummary,
  listWorkspaceMembers,
  WorkspaceMemberItem,
} from '../api/workspaces'
import WorkspaceProviderSettingsModal from './WorkspaceProviderSettingsModal'
import { NavItem } from './Sidebar'
import { VisualizationRenderer, VisualizationSpec } from './visualizations'
import { MarkdownRenderer } from './MarkdownRenderer'

type Role = 'user' | 'assistant'
interface Source { title: string; excerpt: string; }
interface Message { id: number | string; role: Role; text: string; visualization?: VisualizationSpec | null; sources?: Source[]; metrics?: ExecutionMetrics | null; isTyping?: boolean; }

function SourceCard({ source }: { source: Source }) {
  return (
    <div style={{ background: 'var(--color-teal-light)', borderLeft: '2px solid var(--color-teal)', padding: '10px 14px', borderRadius: '4px', marginTop: '8px' }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.08em', color: 'var(--color-teal)', marginBottom: '4px', textTransform: 'uppercase' }}>{source.title}</div>
      <div style={{ fontSize: '13px', color: 'var(--color-ink-muted)', lineHeight: 1.5, fontStyle: 'italic' }}>"{source.excerpt}"</div>
    </div>
  )
}

function formatTokens(n: number | undefined | null): string {
  if (n === undefined || n === null) return '0 tokens'
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M tokens`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k tokens`
  return `${n} token${n !== 1 ? 's' : ''}`
}

function formatDurationCompact(ms: number | undefined | null): string {
  if (ms === undefined || ms === null) return '0.0s'
  const sec = ms / 1000
  return `${sec.toFixed(1)}s`
}

function formatDurationDetail(ms: number | undefined | null): string {
  if (ms === undefined || ms === null) return '0.00 s'
  const sec = ms / 1000
  return `${sec.toFixed(2)} s`
}

function ExecutionMetricsView({ metrics }: { metrics: ExecutionMetrics }) {
  const [expanded, setExpanded] = useState(false)

  const retrieval = metrics.stages?.retrieval
  const generation = metrics.stages?.generation
  const totalTokens = metrics.total_tokens || metrics.llm_usage?.total_tokens || 0
  const totalLatency = metrics.total_latency_ms || 0
  const acceptedCount = metrics.accepted_count || metrics.accepted_results || 0

  return (
    <div style={{ marginTop: '14px', fontFamily: 'var(--font-mono)' }}>
      {/* Compact summary bar */}
      <div
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '12px',
          background: 'var(--color-panel)',
          border: '1px solid var(--color-border)',
          borderRadius: '20px',
          padding: '4px 12px',
          fontSize: '11px',
          color: 'var(--color-ink-muted)',
          cursor: 'pointer',
          userSelect: 'none',
          transition: 'all 0.15s ease',
        }}
        onClick={() => setExpanded(!expanded)}
        title="Click to toggle execution details"
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--color-teal)' }}>
          <span style={{ fontSize: '10px' }}>⚡</span>
          <span>{formatTokens(totalTokens)}</span>
        </span>
        <span style={{ color: 'var(--color-border)' }}>•</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ fontSize: '10px' }}>⏱</span>
          <span>{formatDurationCompact(totalLatency)}</span>
        </span>
        <span style={{ color: 'var(--color-border)' }}>•</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ fontSize: '10px' }}>📚</span>
          <span>{acceptedCount} sources</span>
        </span>
        <span style={{ color: 'var(--color-border)' }}>•</span>
        <span style={{ fontSize: '10px', color: 'var(--color-ink-muted)', textDecoration: 'underline' }}>
          {expanded ? 'Hide details ▴' : 'View execution details ▾'}
        </span>
      </div>

      {/* Expanded details panel */}
      {expanded && (
        <div
          style={{
            marginTop: '10px',
            background: 'var(--color-surface)',
            border: '1px solid var(--color-border)',
            borderRadius: '8px',
            padding: '16px 18px',
            fontSize: '12px',
            lineHeight: 1.6,
            color: 'var(--color-ink)',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: '16px',
            animation: 'fadeIn 0.2s ease',
          }}
        >
          {/* Total & Request Info */}
          <div style={{ borderRight: '1px solid var(--color-border)', paddingRight: '12px' }}>
            <div style={{ fontSize: '10px', letterSpacing: '0.08em', color: 'var(--color-teal)', textTransform: 'uppercase', marginBottom: '6px', fontWeight: 600 }}>
              Execution Summary
            </div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-ink)' }}>
              Total: {formatDurationDetail(totalLatency)}
            </div>
            {metrics.provider && (
              <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
                Provider: <strong style={{ color: 'var(--color-ink)' }}>{metrics.provider}</strong> {metrics.model ? `(${metrics.model})` : ''}
              </div>
            )}
            <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', marginTop: '6px', wordBreak: 'break-all' }}>
              Request ID:
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--color-ink)', background: 'var(--color-ground)', padding: '2px 4px', borderRadius: '3px', marginTop: '2px' }}>
                {metrics.request_id}
              </div>
            </div>
            {metrics.trace_id && metrics.trace_id !== metrics.request_id && (
              <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', marginTop: '4px', wordBreak: 'break-all' }}>
                Trace ID:
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--color-ink)', background: 'var(--color-ground)', padding: '2px 4px', borderRadius: '3px', marginTop: '2px' }}>
                  {metrics.trace_id}
                </div>
              </div>
            )}
            {metrics.failure_categories && metrics.failure_categories.length > 0 && (
              <div style={{ marginTop: '8px' }}>
                <span style={{ fontSize: '10px', color: 'var(--color-coral)', fontWeight: 600 }}>TAXONOMY: </span>
                {metrics.failure_categories.map((fc, i) => (
                  <span key={i} style={{ fontSize: '9px', background: '#ffefef', color: '#d93838', border: '1px solid #ffd0d0', padding: '1px 5px', borderRadius: '4px', marginRight: '4px' }}>
                    {fc}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Retrieval Strategies & Candidates */}
          <div>
            <div style={{ fontSize: '10px', letterSpacing: '0.08em', color: 'var(--color-teal)', textTransform: 'uppercase', marginBottom: '6px', fontWeight: 600 }}>
              Retrieval & Evidence
            </div>
            <div style={{ fontWeight: 500 }}>
              {formatDurationDetail(metrics.retrieval_latency_ms || retrieval?.latency_ms || 0)}
            </div>
            {metrics.planned_strategies && metrics.planned_strategies.length > 0 && (
              <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
                Planned: {metrics.planned_strategies.join(' → ')}
              </div>
            )}
            {metrics.executed_strategies && metrics.executed_strategies.length > 0 && (
              <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>
                Executed: {metrics.executed_strategies.join(' → ')}
              </div>
            )}
            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
              Candidates: <strong>{metrics.candidate_count || 0}</strong> (Vector: {metrics.vector_candidates || 0}, KW: {metrics.keyword_candidates || 0}, Hier: {metrics.hierarchical_candidates || 0})
            </div>
            <div style={{ fontSize: '11px', color: 'var(--color-ink)' }}>
              Accepted Evidence: <strong>{acceptedCount} chunks</strong> {metrics.context_tokens ? `(${metrics.context_tokens} toks)` : ''}
            </div>
            {retrieval?.sub_stages && retrieval.sub_stages.length > 0 && (
              <div style={{ marginTop: '6px', borderTop: '1px dashed var(--color-border)', paddingTop: '4px' }}>
                {retrieval.sub_stages.map((stg, i) => (
                  <div key={i} style={{ fontSize: '10px', color: 'var(--color-ink-muted)', display: 'flex', justifyContent: 'space-between' }}>
                    <span>{stg.name.replace(/_/g, ' ')}:</span>
                    <span>{formatDurationDetail(stg.latency_ms)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Sufficiency & Grounding */}
          <div>
            <div style={{ fontSize: '10px', letterSpacing: '0.08em', color: 'var(--color-teal)', textTransform: 'uppercase', marginBottom: '6px', fontWeight: 600 }}>
              Verification & Quality
            </div>
            {metrics.sufficiency !== undefined && metrics.sufficiency !== null && (
              <div style={{ marginBottom: '6px' }}>
                <span style={{ fontSize: '11px' }}>Sufficiency: </span>
                <span style={{
                  fontSize: '10px',
                  fontWeight: 600,
                  padding: '2px 6px',
                  borderRadius: '4px',
                  background: metrics.sufficiency ? '#e6f7ec' : '#fff3e6',
                  color: metrics.sufficiency ? '#1c7430' : '#b25900',
                }}>
                  {metrics.sufficiency ? '✓ SUFFICIENT' : '⚠ INSUFFICIENT'}
                </span>
                {metrics.sufficiency_confidence !== undefined && metrics.sufficiency_confidence !== null && (
                  <span style={{ fontSize: '10px', color: 'var(--color-ink-muted)', marginLeft: '4px' }}>
                    ({(metrics.sufficiency_confidence * 100).toFixed(0)}% conf)
                  </span>
                )}
              </div>
            )}
            {metrics.grounding !== undefined && metrics.grounding !== null && (
              <div style={{ marginBottom: '6px' }}>
                <span style={{ fontSize: '11px' }}>Grounding: </span>
                <span style={{
                  fontSize: '10px',
                  fontWeight: 600,
                  padding: '2px 6px',
                  borderRadius: '4px',
                  background: metrics.grounding ? '#e6f7ec' : '#fce8e6',
                  color: metrics.grounding ? '#1c7430' : '#c5221f',
                }}>
                  {metrics.grounding ? '✓ GROUNDED' : '✖ UNGROUNDED'}
                </span>
                {metrics.grounding_confidence !== undefined && metrics.grounding_confidence !== null && (
                  <span style={{ fontSize: '10px', color: 'var(--color-ink-muted)', marginLeft: '4px' }}>
                    ({(metrics.grounding_confidence * 100).toFixed(0)}% conf)
                  </span>
                )}
              </div>
            )}
            {metrics.repair_triggered && (
              <div style={{ fontSize: '11px', color: 'var(--color-teal)' }}>
                ⚡ Auto-repair: Triggered
              </div>
            )}
            {metrics.regeneration_triggered && (
              <div style={{ fontSize: '11px', color: 'var(--color-teal)' }}>
                🔄 Constrained Regeneration: Triggered
              </div>
            )}
            {metrics.safe_limitation_triggered && (
              <div style={{ fontSize: '11px', color: '#b25900' }}>
                🛡 Safe Limitation Fallback: Active
              </div>
            )}
            {metrics.visualization_requested && (
              <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
                Visualization: {metrics.visualization_validated ? 'Validated ✓' : (metrics.visualization_dropped ? 'Dropped ✖' : 'Active')}
              </div>
            )}
          </div>

          {/* LLM Total Aggregated */}
          <div style={{ gridColumn: '1 / -1', borderTop: '1px solid var(--color-border)', paddingTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'center' }}>
            <span style={{ fontSize: '10px', letterSpacing: '0.08em', color: 'var(--color-teal)', textTransform: 'uppercase', fontWeight: 600 }}>
              Token Usage:
            </span>
            <span style={{ fontSize: '11px' }}>Input: <strong>{metrics.input_tokens || metrics.llm_usage?.input_tokens || 0}</strong></span>
            <span style={{ fontSize: '11px' }}>Output: <strong>{metrics.output_tokens || metrics.llm_usage?.output_tokens || 0}</strong></span>
            <span style={{ fontSize: '11px' }}>Total: <strong>{totalTokens}</strong></span>
            {generation && (
              <span style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginLeft: 'auto' }}>
                Generation latency: {formatDurationDetail(metrics.generation_latency_ms || generation.latency_ms)}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

const THINKING_STEPS = [
  { label: 'Doot is thinking…', icon: '◌' },
  { label: 'Using web search', icon: '⌕' },
  { label: 'Reading sources', icon: '⊡' },
  { label: 'Answer is about to come', icon: '◎' },
]

function TypingIndicator() {
  const [step, setStep] = useState(0)
  const [visible, setVisible] = useState(true)

  useEffect(() => {
    const interval = setInterval(() => {
      setVisible(false)
      setTimeout(() => { setStep((s) => (s + 1) % THINKING_STEPS.length); setVisible(true) }, 200)
    }, 1400)
    return () => clearInterval(interval)
  }, [])

  const current = THINKING_STEPS[step]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', opacity: visible ? 1 : 0, transition: 'opacity 0.2s ease' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '15px', color: 'var(--color-teal)', animation: 'spin 1.8s linear infinite', display: 'inline-block' }}>{current.icon}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-teal)', letterSpacing: '0.04em' }}>{current.label}</span>
      </div>
      <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
        {THINKING_STEPS.map((_, i) => (
          <div key={i} style={{ width: i === step ? '18px' : '5px', height: '5px', borderRadius: '3px', background: i === step ? 'var(--color-teal)' : 'var(--color-border)', transition: 'all 0.3s ease' }} />
        ))}
      </div>
      <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
    </div>
  )
}

function AssistantMessage({ msg, onVisualize }: { msg: Message; onVisualize?: (msgId: string | number) => void }) {
  return (
    <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
      <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'var(--color-teal)', flexShrink: 0, marginTop: '3px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: 'rgba(255,255,255,0.5)' }} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.1em', color: 'var(--color-teal)', textTransform: 'uppercase' }}>Doot</div>
          {onVisualize && !msg.isTyping && (
            <button
              onClick={() => onVisualize(msg.id)}
              title="Generate visual chart from this response"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
                padding: '3px 8px',
                fontSize: '11px',
                fontWeight: 600,
                color: 'var(--color-teal)',
                background: 'rgba(2, 132, 199, 0.08)',
                border: '1px solid rgba(2, 132, 199, 0.2)',
                borderRadius: '5px',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(2, 132, 199, 0.16)')}
              onMouseLeave={(e) => (e.currentTarget.style.background = 'rgba(2, 132, 199, 0.08)')}
            >
              <span>📊</span>
              <span>Visualize this</span>
            </button>
          )}
        </div>
        {msg.isTyping ? <TypingIndicator /> : (
          <>
            <MarkdownRenderer
              content={msg.text}
              style={{ margin: 0, fontFamily: "'Segoe UI', system-ui, sans-serif", fontSize: '16px', lineHeight: 1.7, color: 'var(--color-ink)' }}
            />
            {msg.visualization && <VisualizationRenderer spec={msg.visualization} />}
            {msg.sources && msg.sources.length > 0 && (
              <div style={{ marginTop: '12px' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.08em', color: 'var(--color-ink-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>{msg.sources.length} source{msg.sources.length !== 1 ? 's' : ''} retrieved</div>
                {msg.sources.map((s, i) => <SourceCard key={i} source={s} />)}
              </div>
            )}
            {msg.metrics && <ExecutionMetricsView metrics={msg.metrics} />}
          </>
        )}
      </div>
    </div>
  )
}

function UserMessage({ msg }: { msg: Message }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
      <div
        style={{
          maxWidth: '75%',
          background: 'var(--color-surface)',
          color: 'var(--color-ink)',
          border: '1px solid var(--color-border)',
          borderRadius: '14px 14px 3px 14px',
          padding: '12px 18px',
          fontSize: '15px',
          lineHeight: 1.6,
          fontFamily: 'var(--font-sans)',
          fontWeight: 400,
          whiteSpace: 'pre-wrap',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04)',
        }}
      >
        {msg.text}
      </div>
    </div>
  )
}

export interface DepartmentOption {
  code: string
  name: string
}

const DEFAULT_WELCOME_MESSAGE: Message = {
  id: 'default-welcome',
  role: 'assistant',
  text: "Hello. I'm Doot — a research assistant grounded in your document library. Ask me anything, and I'll retrieve relevant sources before answering.",
}

export interface RAGProps {
  token: string
  userName: string
  allowedUploadDepts: DepartmentOption[] | string[]
  activeConversationId?: string | null
  activeConversationTitle?: string
  onConversationCreatedOrUpdated?: (convId: string) => void
  onToggleSidebar?: () => void
  isSidebarCollapsed?: boolean
  workspaces?: WorkspaceItem[]
  activeWorkspaceId?: string | null
  onSelectWorkspace?: (wsId: string) => void
  onOpenWorkspaceSettings?: (ws: WorkspaceItem) => void
  onNavigateView?: (view: NavItem) => void
}

export default function RAG({
  token,
  userName,
  allowedUploadDepts,
  activeConversationId = null,
  activeConversationTitle,
  onConversationCreatedOrUpdated,
  onToggleSidebar,
  isSidebarCollapsed = false,
  workspaces = [],
  activeWorkspaceId = null,
  onSelectWorkspace,
  onOpenWorkspaceSettings,
  onNavigateView,
}: RAGProps) {
  const navigate = useNavigate()
  const normalizedUploadDepts: DepartmentOption[] = allowedUploadDepts.map((d) =>
    typeof d === 'string' ? { code: d, name: d } : d
  )

  const [messages, setMessages] = useState<Message[]>([DEFAULT_WELCOME_MESSAGE])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [topK, setTopK] = useState(5)

  // Sources panel state
  const [showSourcesPanel, setShowSourcesPanel] = useState(false)
  const [sourcesChunks, setSourcesChunks] = useState<RetrievedChunk[]>([])
  const [rerank, setRerank] = useState(false)

  const [department, setDepartment] = useState(
    normalizedUploadDepts.length > 0 ? normalizedUploadDepts[0].code : ''
  )
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)

  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const idRef = useRef(1)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const justCreatedConvIdRef = useRef<string | null>(null)

  // Workspaces state
  const [allWorkspaces, setAllWorkspaces] = useState<WorkspaceItem[]>(workspaces)
  const [chatWorkspaceId, setChatWorkspaceId] = useState<string>(activeWorkspaceId || '')

  // Provider Capabilities for RAG Run Experimentation Knobs
  const [capabilities, setCapabilities] = useState<ProviderCapabilitiesSummary>({
    llm: ['groq'],
    embedding: ['sentence_transformers'],
    document_reader: ['docling'],
    reranker: ['cross_encoder'],
    ocr: ['paddle_ocr'],
    web: ['firecrawl'],
  })

  // Per-Run Experiment Overrides State
  const [isUsingWorkspaceDefault, setIsUsingWorkspaceDefault] = useState<boolean>(true)
  const [showExperimentPanel, setShowExperimentPanel] = useState<boolean>(false)
  const [expLlm, setExpLlm] = useState<string>('')
  const [expEmbedding, setExpEmbedding] = useState<string>('')
  const [expReader, setExpReader] = useState<string>('')
  const [expReranker, setExpReranker] = useState<string>('')
  const [expOcr, setExpOcr] = useState<string>('')
  const [expWeb, setExpWeb] = useState<string>('')

  // Popovers state for Members & Knowledge
  const [showMembersModal, setShowMembersModal] = useState(false)
  const [workspaceMembers, setWorkspaceMembers] = useState<WorkspaceMemberItem[]>([])
  const [showKnowledgeModal, setShowKnowledgeModal] = useState(false)

  useEffect(() => {
    if (!token) return
    listWorkspaces(token)
      .then((ws) => setAllWorkspaces(ws))
      .catch(() => setAllWorkspaces([]))

    getProviderCapabilities(token)
      .then((caps) => {
        if (caps) setCapabilities(caps)
      })
      .catch(() => {})
  }, [token])

  useEffect(() => {
    if (workspaces && workspaces.length > 0) {
      setAllWorkspaces(workspaces)
    }
  }, [workspaces])

  useEffect(() => {
    if (activeWorkspaceId !== undefined) {
      setChatWorkspaceId(activeWorkspaceId || '')
    }
  }, [activeWorkspaceId])

  // Load conversation messages when activeConversationId changes
  useEffect(() => {
    setInput('')
    if (!token) return

    if (!activeConversationId) {
      setMessages([DEFAULT_WELCOME_MESSAGE])
      setIsLoading(false)
      setTimeout(() => {
        textareaRef.current?.focus()
      }, 50)
      return
    }

    if (activeConversationId === justCreatedConvIdRef.current) {
      justCreatedConvIdRef.current = null
      return
    }

    let isMounted = true
    setIsLoading(true)
    setMessages([DEFAULT_WELCOME_MESSAGE])
    getConversation(activeConversationId, token)
      .then((detail) => {
        if (!isMounted) return
        if (detail.workspace_id) {
          setChatWorkspaceId(detail.workspace_id)
        }
        if (detail.messages && detail.messages.length > 0) {
          const seenIds = new Set<string | number>()
          const loadedMsgs: Message[] = []
          for (const m of detail.messages) {
            if (m.id && !seenIds.has(m.id)) {
              seenIds.add(m.id)
              loadedMsgs.push({
                id: m.id,
                role: m.role as Role,
                text: m.content,
                visualization: m.visualization,
                sources: m.sources || undefined,
                metrics: m.execution_metrics || undefined,
              })
            }
          }
          setMessages(loadedMsgs)
        } else {
          setMessages([DEFAULT_WELCOME_MESSAGE])
        }
      })
      .catch((err) => {
        console.error('Failed to load conversation detail:', err)
        if (isMounted) setMessages([DEFAULT_WELCOME_MESSAGE])
      })
      .finally(() => {
        if (isMounted) {
          setIsLoading(false)
          setTimeout(() => {
            textareaRef.current?.focus()
          }, 50)
        }
      })

    return () => {
      isMounted = false
    }
  }, [activeConversationId, token])

  useEffect(() => {
    if (normalizedUploadDepts.length > 0) {
      if (!department || !normalizedUploadDepts.some((d) => d.code === department)) {
        setDepartment(normalizedUploadDepts[0].code)
      }
    } else {
      setDepartment('')
    }
  }, [allowedUploadDepts])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function autoResize() {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }

  function triggerUpload() {
    fileInputRef.current?.click()
  }

  async function handleUpload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file || !token || !department) return
    setUploading(true)
    setUploadStatus('Document is being processed… Queries will be enabled when ingestion completes.')
    try {
      const res = await uploadDocument(file, department, token, chatWorkspaceId || undefined)
      setUploadStatus(
        res.skipped
          ? `Document ready. ${res.file_name} was already indexed.`
          : `Document ready. ${res.file_name} indexed with ${res.chunks_created} chunks.`
      )
    } catch (err: any) {
      setUploadStatus(`Upload failed: ${err.message}`)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleResetToWorkspaceDefault = () => {
    setIsUsingWorkspaceDefault(true)
    setExpLlm('')
    setExpEmbedding('')
    setExpReader('')
    setExpReranker('')
    setExpOcr('')
    setExpWeb('')
  }

  const handleExperimentChange = (field: string, value: string) => {
    setIsUsingWorkspaceDefault(false)
    if (field === 'llm') setExpLlm(value)
    if (field === 'embedding') setExpEmbedding(value)
    if (field === 'document_reader') setExpReader(value)
    if (field === 'reranker') setExpReranker(value)
    if (field === 'ocr') setExpOcr(value)
    if (field === 'web') setExpWeb(value)
  }

  const currentWorkspace = allWorkspaces.find((w) => w.id === chatWorkspaceId)

  const handleFetchMembers = async () => {
    if (!chatWorkspaceId || !token) return
    try {
      const members = await listWorkspaceMembers(token, chatWorkspaceId)
      setWorkspaceMembers(members)
      setShowMembersModal(true)
    } catch (err) {
      console.error('Failed to load workspace members:', err)
    }
  }

  async function send(text: string) {
    if (!text.trim() || isLoading || uploading || !token) return
    const userMsg: Message = { id: `user-${idRef.current++}`, role: 'user', text: text.trim() }
    const typingMsg: Message = { id: `typing-${idRef.current++}`, role: 'assistant', text: '', isTyping: true }
    setMessages((m) => [...m, userMsg, typingMsg])
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    setIsLoading(true)

    // Build provider overrides payload if runtime experiment overrides are selected
    let overrides: Record<string, string> | undefined = undefined
    if (!isUsingWorkspaceDefault) {
      overrides = {}
      if (expLlm) overrides['llm'] = expLlm
      if (expEmbedding) overrides['embedding'] = expEmbedding
      if (expReader) overrides['document_reader'] = expReader
      if (expReranker) overrides['reranker'] = expReranker
      if (expOcr) overrides['ocr'] = expOcr
      if (expWeb) overrides['web'] = expWeb
      if (Object.keys(overrides).length === 0) overrides = undefined
    }

    try {
      let targetConvId = activeConversationId
      if (!targetConvId) {
        const newConv = await createConversation('New Chat', token, chatWorkspaceId || undefined)
        targetConvId = newConv.id
        justCreatedConvIdRef.current = newConv.id
        if (chatWorkspaceId) {
          navigate(`/app/workspace/${chatWorkspaceId}/thread/${targetConvId}`, { replace: true })
        }
      }

      const trimmed = text.trim()
      const isTypingVisualize = trimmed.toLowerCase() === 'visualize this' || trimmed.toLowerCase() === 'visulaize this'
      const lastAsst = [...messages].reverse().find((m) => m.role === 'assistant' && !m.isTyping)
      const targetMsgId = isTypingVisualize && lastAsst ? String(lastAsst.id) : undefined
      const actionVal = isTypingVisualize ? 'visualize' : undefined

      const res = await askQuestion(
        {
          query: isTypingVisualize ? 'Visualize this' : trimmed,
          top_k: topK,
          rerank,
          conversation_id: targetConvId,
          workspace_id: chatWorkspaceId || undefined,
          provider_overrides: overrides,
          action: actionVal,
          target_message_id: targetMsgId,
        },
        token
      )
      if (res.conversation_id && onConversationCreatedOrUpdated) {
        onConversationCreatedOrUpdated(res.conversation_id)
      }

      const mappedSources: Source[] = res.retrieved_chunks.map((c: RetrievedChunk) => ({
        title: `${c.file_name}${c.page_number ? ` (Page ${c.page_number})` : ''}`,
        excerpt: c.text,
      }))

      // Store raw retrieved chunks for Sources panel
      if (res.retrieved_chunks && res.retrieved_chunks.length > 0) {
        setSourcesChunks(res.retrieved_chunks)
      }

      setMessages((m) =>
        m.map((msg) =>
          msg.id === typingMsg.id
            ? { ...msg, id: res.message_id || msg.id, text: res.answer, visualization: res.visualization, sources: mappedSources, metrics: res.execution_metrics, isTyping: false }
            : msg
        )
      )
    } catch (err: any) {
      setMessages((m) =>
        m.map((msg) =>
          msg.id === typingMsg.id ? { ...msg, text: `Error: ${err.message}`, isTyping: false } : msg
        )
      )
    } finally {
      setIsLoading(false)
      setTimeout(() => textareaRef.current?.focus(), 50)
    }
  }

  async function handleVisualizeThis(targetMessageId: string | number) {
    if (isLoading) return
    const typingMsg: Message = { id: Date.now() + 1, role: 'assistant', text: '', isTyping: true }
    setMessages((m) => [...m, typingMsg])
    setIsLoading(true)

    try {
      const res = await askQuestion(
        {
          query: 'Visualize this',
          action: 'visualize',
          target_message_id: String(targetMessageId),
          conversation_id: activeConversationId || undefined,
          workspace_id: chatWorkspaceId || undefined,
        },
        token
      )

      const mappedSources: Source[] = (res.retrieved_chunks || []).map((c: RetrievedChunk) => ({
        title: `${c.file_name}${c.page_number ? ` (Page ${c.page_number})` : ''}`,
        excerpt: c.text,
      }))

      setMessages((m) =>
        m.map((msg) =>
          msg.id === typingMsg.id
            ? {
                ...msg,
                id: res.message_id || msg.id,
                text: res.answer,
                visualization: res.visualization,
                sources: mappedSources,
                metrics: res.execution_metrics,
                isTyping: false,
              }
            : msg
        )
      )
    } catch (err: any) {
      setMessages((m) =>
        m.map((msg) =>
          msg.id === typingMsg.id ? { ...msg, text: `Error: ${err.message}`, isTyping: false } : msg
        )
      )
    } finally {
      setIsLoading(false)
    }
  }

  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send(input)
    }
  }

  const isInputDisabled = isLoading || uploading

  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'row',
        minWidth: 0,
        height: '100vh',
        background: 'var(--color-ground)',
      }}
    >
      {/* Main Chat Column */}
      <div
        style={{
          flex: 1,
          display: 'grid',
          gridTemplateRows: 'auto auto 1fr auto',
          minWidth: 0,
          height: '100vh',
        }}
      >
      {/* 1. TOP HEADER: Workspace Name & Compact Workspace Menu */}
      <header
        style={{
          borderBottom: '1px solid var(--color-border)',
          minHeight: '56px',
          background: 'var(--color-surface)',
          zIndex: 10,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          flexShrink: 0,
        }}
      >
        <div
          style={{
            maxWidth: '1000px',
            width: '100%',
            margin: '0 auto',
            padding: '10px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            boxSizing: 'border-box',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', minWidth: 0 }}>
            {/* Sidebar toggle button */}
            {onToggleSidebar && (
              <button
                onClick={onToggleSidebar}
                title={isSidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '6px',
                  background: 'transparent',
                  border: '1px solid var(--color-border)',
                  color: 'var(--color-ink)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <line x1="9" y1="3" x2="9" y2="21"></line>
                </svg>
              </button>
            )}

            {/* Active Workspace / Title */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0 }}>
              <div
                style={{
                  width: '26px',
                  height: '26px',
                  borderRadius: '6px',
                  background: 'var(--color-teal-light)',
                  border: '1px solid var(--color-teal)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--color-teal)',
                  fontSize: '13px',
                  fontWeight: 'bold',
                  flexShrink: 0,
                }}
              >
                📁
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span
                    style={{
                      fontFamily: 'var(--font-sans)',
                      fontSize: '16px',
                      fontWeight: 600,
                      color: 'var(--color-ink)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {currentWorkspace ? currentWorkspace.name : 'Global Workspace'}
                  </span>
                  {currentWorkspace?.department_code && (
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '10px',
                        color: 'var(--color-teal)',
                        background: 'var(--color-teal-light)',
                        padding: '1px 6px',
                        borderRadius: '3px',
                      }}
                    >
                      {currentWorkspace.department_code}
                    </span>
                  )}
                </div>
                {activeConversationTitle && (
                  <span style={{ fontSize: '11px', color: 'var(--color-ink-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    Thread: {activeConversationTitle}
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Compact Workspace Menu & Settings Links */}
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <button
              onClick={() => onNavigateView && onNavigateView('sources')}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--color-ink)',
                fontSize: '12px',
                fontWeight: 500,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}
            >
              <span>📄</span> Documents
            </button>

            {currentWorkspace && (
              <>
                <button
                  onClick={handleFetchMembers}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--color-ink)',
                    fontSize: '12px',
                    fontWeight: 500,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                  }}
                >
                  <span>👥</span> Members
                </button>

                <button
                  onClick={() => setShowKnowledgeModal(true)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--color-ink)',
                    fontSize: '12px',
                    fontWeight: 500,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                  }}
                >
                  <span>🧠</span> Knowledge
                </button>

                <button
                  onClick={() => onOpenWorkspaceSettings && currentWorkspace && onOpenWorkspaceSettings(currentWorkspace)}
                  style={{
                    padding: '4px 10px',
                    borderRadius: '6px',
                    background: 'var(--color-panel)',
                    border: '1px solid var(--color-border)',
                    color: 'var(--color-teal)',
                    fontSize: '12px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                  }}
                >
                  <span>⚙</span> RAG / Providers
                </button>
              </>
            )}

            {/* Sources Panel Toggle */}
            <button
              onClick={() => setShowSourcesPanel(!showSourcesPanel)}
              title="Toggle Sources panel"
              style={{
                padding: '4px 10px',
                borderRadius: '6px',
                background: showSourcesPanel ? 'var(--color-teal-light)' : 'transparent',
                border: `1px solid ${showSourcesPanel ? 'var(--color-teal)' : 'var(--color-border)'}`,
                color: showSourcesPanel ? 'var(--color-teal)' : 'var(--color-ink-muted)',
                fontSize: '11px',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
              Sources{sourcesChunks.length > 0 ? ` (${sourcesChunks.length})` : ''}
            </button>

            {/* Experiment Controls Toggle */}
            <button
              onClick={() => setShowExperimentPanel(!showExperimentPanel)}
              title="Toggle per-run experiment override knobs"
              style={{
                padding: '4px 10px',
                borderRadius: '6px',
                background: isUsingWorkspaceDefault ? 'transparent' : 'var(--color-teal-light)',
                border: `1px solid ${isUsingWorkspaceDefault ? 'var(--color-border)' : 'var(--color-teal)'}`,
                color: isUsingWorkspaceDefault ? 'var(--color-ink-muted)' : 'var(--color-teal)',
                fontSize: '11px',
                fontFamily: 'var(--font-mono)',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              🧪 {isUsingWorkspaceDefault ? 'Overrides' : 'Override ACTIVE'}
            </button>
          </div>
        </div>
      </header>

      {/* 2. RAG EXPERIMENTATION / RUN OVERRIDES TOOLBAR */}
      {showExperimentPanel && (
        <div
          style={{
            background: 'var(--color-surface)',
            borderBottom: '1px solid var(--color-border)',
            padding: '10px 24px',
            fontSize: '12px',
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
            boxShadow: '0 2px 6px rgba(0,0,0,0.03)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontWeight: 600, color: 'var(--color-ink)', fontSize: '12px' }}>
                RAG Experimentation / Run Override Knobs:
              </span>
              <span
                style={{
                  fontSize: '10px',
                  fontFamily: 'var(--font-mono)',
                  padding: '2px 8px',
                  borderRadius: '10px',
                  background: isUsingWorkspaceDefault ? '#e2e8f0' : '#e0f2fe',
                  color: isUsingWorkspaceDefault ? '#475569' : '#0369a1',
                  fontWeight: 600,
                }}
              >
                {isUsingWorkspaceDefault ? 'WORKSPACE DEFAULT ACTIVE' : 'RUN OVERRIDE ACTIVE'}
              </span>
            </div>

            {/* Use Workspace Default Button */}
            <button
              onClick={handleResetToWorkspaceDefault}
              disabled={isUsingWorkspaceDefault}
              style={{
                padding: '3px 10px',
                borderRadius: '5px',
                background: isUsingWorkspaceDefault ? '#f1f5f9' : 'var(--color-teal)',
                border: '1px solid var(--color-border)',
                color: isUsingWorkspaceDefault ? '#94a3b8' : '#ffffff',
                fontSize: '11px',
                fontWeight: 600,
                cursor: isUsingWorkspaceDefault ? 'default' : 'pointer',
              }}
            >
              [Use Workspace Default]
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '8px' }}>
            {/* LLM */}
            <div>
              <label style={{ display: 'block', fontSize: '10px', color: 'var(--color-ink-muted)', marginBottom: '2px' }}>LLM Provider</label>
              <select
                value={expLlm}
                onChange={(e) => handleExperimentChange('llm', e.target.value)}
                style={{ width: '100%', padding: '4px 6px', borderRadius: '4px', border: '1px solid var(--color-border)', fontSize: '11px' }}
              >
                <option value="">(Workspace Default)</option>
                {capabilities.llm.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>

            {/* Embedding */}
            <div>
              <label style={{ display: 'block', fontSize: '10px', color: 'var(--color-ink-muted)', marginBottom: '2px' }}>Embedding</label>
              <select
                value={expEmbedding}
                onChange={(e) => handleExperimentChange('embedding', e.target.value)}
                style={{ width: '100%', padding: '4px 6px', borderRadius: '4px', border: '1px solid var(--color-border)', fontSize: '11px' }}
              >
                <option value="">(Workspace Default)</option>
                {capabilities.embedding.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>

            {/* Reader */}
            <div>
              <label style={{ display: 'block', fontSize: '10px', color: 'var(--color-ink-muted)', marginBottom: '2px' }}>Doc Reader</label>
              <select
                value={expReader}
                onChange={(e) => handleExperimentChange('document_reader', e.target.value)}
                style={{ width: '100%', padding: '4px 6px', borderRadius: '4px', border: '1px solid var(--color-border)', fontSize: '11px' }}
              >
                <option value="">(Workspace Default)</option>
                {capabilities.document_reader.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>

            {/* Reranker */}
            <div>
              <label style={{ display: 'block', fontSize: '10px', color: 'var(--color-ink-muted)', marginBottom: '2px' }}>Reranker</label>
              <select
                value={expReranker}
                onChange={(e) => handleExperimentChange('reranker', e.target.value)}
                style={{ width: '100%', padding: '4px 6px', borderRadius: '4px', border: '1px solid var(--color-border)', fontSize: '11px' }}
              >
                <option value="">(Workspace Default)</option>
                {capabilities.reranker.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>

            {/* OCR */}
            <div>
              <label style={{ display: 'block', fontSize: '10px', color: 'var(--color-ink-muted)', marginBottom: '2px' }}>OCR</label>
              <select
                value={expOcr}
                onChange={(e) => handleExperimentChange('ocr', e.target.value)}
                style={{ width: '100%', padding: '4px 6px', borderRadius: '4px', border: '1px solid var(--color-border)', fontSize: '11px' }}
              >
                <option value="">(Workspace Default)</option>
                {capabilities.ocr.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>

            {/* Web */}
            <div>
              <label style={{ display: 'block', fontSize: '10px', color: 'var(--color-ink-muted)', marginBottom: '2px' }}>Web Search</label>
              <select
                value={expWeb}
                onChange={(e) => handleExperimentChange('web', e.target.value)}
                style={{ width: '100%', padding: '4px 6px', borderRadius: '4px', border: '1px solid var(--color-border)', fontSize: '11px' }}
              >
                <option value="">(Workspace Default)</option>
                {capabilities.web.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>
          </div>
          <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', fontStyle: 'italic' }}>
            Changing experiment settings will NOT overwrite the Workspace default profile. Try different combinations on the fly!
          </div>
        </div>
      )}

      {/* 3. MAIN CHAT MESSAGES AREA */}
      <main style={{ overflowY: 'auto', padding: '32px 0', flex: 1, minHeight: 0 }}>
        <div
          style={{
            maxWidth: '760px',
            width: '100%',
            margin: '0 auto',
            padding: '0 24px',
            display: 'flex',
            flexDirection: 'column',
            gap: '32px',
            boxSizing: 'border-box',
          }}
        >
          {messages.map((msg) =>
            msg.role === 'assistant' ? (
              <AssistantMessage key={msg.id} msg={msg} onVisualize={handleVisualizeThis} />
            ) : (
              <UserMessage key={msg.id} msg={msg} />
            )
          )}
          <div ref={bottomRef} />
        </div>
      </main>

      {/* 4. BOTTOM COMPOSER AREA */}
      <footer
        style={{
          borderTop: '1px solid var(--color-border)',
          background: 'var(--color-surface)',
          padding: '16px 0 24px',
          flexShrink: 0,
        }}
      >
        <div style={{ maxWidth: '760px', width: '100%', margin: '0 auto', padding: '0 24px', boxSizing: 'border-box' }}>
          {uploadStatus && (
            <div
              style={{
                marginBottom: '12px',
                fontSize: '12px',
                color: uploading ? 'var(--color-teal)' : 'var(--color-ink-muted)',
                fontFamily: 'var(--font-mono)',
                textAlign: 'center',
              }}
            >
              {uploadStatus}
            </div>
          )}
          <div
            style={{
              display: 'flex',
              alignItems: 'flex-end',
              gap: '12px',
              background: isInputDisabled ? 'var(--color-panel)' : 'var(--color-ground)',
              border: '1px solid var(--color-border)',
              borderRadius: '12px',
              padding: '10px 12px 10px 16px',
              transition: 'border-color 0.15s, background 0.15s',
            }}
            onFocusCapture={(e) => {
              if (!isInputDisabled) e.currentTarget.style.borderColor = 'var(--color-teal)'
            }}
            onBlurCapture={(e) => (e.currentTarget.style.borderColor = 'var(--color-border)')}
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value)
                autoResize()
              }}
              onKeyDown={handleKey}
              placeholder={uploading ? 'Document is being processed...' : `Ask ${currentWorkspace ? currentWorkspace.name : 'Doot'}…`}
              disabled={isInputDisabled}
              rows={1}
              style={{
                flex: 1,
                fontFamily: 'var(--font-sans)',
                fontSize: '15px',
                color: 'var(--color-ink)',
                background: 'transparent',
                border: 'none',
                resize: 'none',
                lineHeight: 1.6,
                maxHeight: '160px',
                overflowY: 'auto',
                opacity: isInputDisabled ? 0.6 : 1,
              }}
            />
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              {normalizedUploadDepts.length > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span
                    style={{
                      fontSize: '11px',
                      color: 'var(--color-ink-muted)',
                      fontFamily: 'var(--font-mono)',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    Dept:
                  </span>
                  <select
                    value={department}
                    onChange={(e) => setDepartment(e.target.value)}
                    disabled={isInputDisabled}
                    style={{
                      background: 'var(--color-surface)',
                      border: '1px solid var(--color-border)',
                      borderRadius: '6px',
                      fontSize: '11px',
                      padding: '5px 8px',
                      color: 'var(--color-ink)',
                      cursor: isInputDisabled ? 'default' : 'pointer',
                    }}
                  >
                    {normalizedUploadDepts.map((d) => (
                      <option key={d.code} value={d.code}>
                        {d.name}
                      </option>
                    ))}
                  </select>

                  <button
                    onClick={triggerUpload}
                    disabled={isInputDisabled}
                    title="Upload Document into active Workspace"
                    style={{
                      width: '36px',
                      height: '36px',
                      borderRadius: '8px',
                      background: 'var(--color-panel)',
                      border: '1px solid var(--color-border)',
                      cursor: isInputDisabled ? 'default' : 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      opacity: isInputDisabled ? 0.6 : 1,
                    }}
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="17 8 12 3 7 8"></polyline>
                      <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                  </button>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleUpload}
                    style={{ display: 'none' }}
                    accept=".pdf,.txt,.md,.csv,.xlsx,.xls,.docx"
                  />
                </div>
              )}
              <button
                onClick={() => send(input)}
                disabled={!input.trim() || isInputDisabled}
                style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: '8px',
                  background:
                    input.trim() && !isInputDisabled ? 'var(--color-teal)' : 'var(--color-panel)',
                  border: 'none',
                  cursor: input.trim() && !isInputDisabled ? 'pointer' : 'default',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                  transition: 'background 0.15s',
                }}
              >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path
                    d="M3 8H13M13 8L9 4M13 8L9 12"
                    stroke={input.trim() && !isInputDisabled ? '#f5f5f0' : 'var(--color-ink-muted)'}
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </footer>

      {/* MEMBERS MODAL POPOVER */}
      {showMembersModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
          onClick={() => setShowMembersModal(false)}
        >
          <div
            style={{
              background: 'var(--color-surface)',
              borderRadius: '10px',
              padding: '24px',
              width: '90%',
              maxWidth: '480px',
              border: '1px solid var(--color-border)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
              <h3 style={{ margin: 0, fontSize: '16px' }}>Workspace Members ({currentWorkspace?.name})</h3>
              <button onClick={() => setShowMembersModal(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '16px' }}>✕</button>
            </div>
            {workspaceMembers.length === 0 ? (
              <div style={{ color: 'var(--color-ink-muted)', fontSize: '13px' }}>No explicit members assigned (Department authorization applies).</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {workspaceMembers.map((m) => (
                  <div key={m.user_id} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px', borderBottom: '1px solid var(--color-border)', fontSize: '13px' }}>
                    <span>{m.user_email || m.user_id}</span>
                    <span style={{ fontWeight: 600, color: 'var(--color-teal)' }}>{m.role}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* KNOWLEDGE MODAL POPOVER */}
      </div>{/* End Main Chat Column */}

      {/* ===== SOURCES PANEL (Right Side) ===== */}
      {showSourcesPanel && (
        <div
          style={{
            width: '300px',
            minWidth: '300px',
            height: '100vh',
            borderLeft: '1px solid var(--color-border)',
            background: 'var(--color-surface)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            flexShrink: 0,
          }}
        >
          {/* Sources Header */}
          <div
            style={{
              padding: '14px 16px',
              borderBottom: '1px solid var(--color-border)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexShrink: 0,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-teal)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
              <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--color-ink)' }}>Sources</span>
              <span style={{ fontSize: '11px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)' }}>
                {sourcesChunks.length} retrieved
              </span>
            </div>
            <button
              onClick={() => setShowSourcesPanel(false)}
              style={{ background: 'none', border: 'none', color: 'var(--color-ink-muted)', cursor: 'pointer', padding: '2px', borderRadius: '4px' }}
              title="Close Sources panel"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>

          {/* Sources List */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
            {sourcesChunks.length === 0 ? (
              <div style={{ padding: '24px 12px', textAlign: 'center', color: 'var(--color-ink-muted)', fontSize: '12px' }}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--color-border)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ margin: '0 auto 12px' }}>
                  <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>
                <div>No sources retrieved yet.</div>
                <div style={{ fontSize: '11px', marginTop: '4px' }}>Sources will appear here after asking a question.</div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {sourcesChunks.map((chunk, idx) => {
                  // Determine file type icon
                  const ext = chunk.file_name?.split('.').pop()?.toLowerCase() || ''
                  const getFileIcon = () => {
                    if (ext === 'pdf') return '📕'
                    if (ext === 'xlsx' || ext === 'xls' || ext === 'csv') return '📊'
                    if (ext === 'docx' || ext === 'doc') return '📘'
                    if (ext === 'md' || ext === 'txt') return '📝'
                    return '📄'
                  }
                  return (
                    <div
                      key={`${chunk.chunk_id || idx}`}
                      style={{
                        background: 'var(--color-ground)',
                        border: '1px solid var(--color-border)',
                        borderRadius: '10px',
                        padding: '12px 14px',
                        transition: 'all 0.12s ease',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = 'var(--color-teal)'
                        e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.04)'
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = 'var(--color-border)'
                        e.currentTarget.style.boxShadow = 'none'
                      }}
                    >
                      {/* File header */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                        <span style={{ fontSize: '16px' }}>{getFileIcon()}</span>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--color-ink)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {chunk.file_name}
                          </div>
                          <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)', display: 'flex', gap: '8px', marginTop: '1px' }}>
                            {chunk.page_number && <span>Page {chunk.page_number}</span>}
                            <span>Chunk #{chunk.chunk_index + 1}</span>
                            {chunk.distance !== undefined && <span>Score: {(1 - chunk.distance).toFixed(3)}</span>}
                          </div>
                        </div>
                      </div>
                      {/* Excerpt */}
                      <div
                        style={{
                          fontSize: '11px',
                          lineHeight: 1.5,
                          color: 'var(--color-ink-muted)',
                          fontStyle: 'italic',
                          maxHeight: '80px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          borderTop: '1px dashed var(--color-border)',
                          paddingTop: '6px',
                          marginTop: '2px',
                        }}
                      >
                        "{chunk.text.slice(0, 300)}{chunk.text.length > 300 ? '…' : ''}"
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {showKnowledgeModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
          onClick={() => setShowKnowledgeModal(false)}
        >
          <div
            style={{
              background: 'var(--color-surface)',
              borderRadius: '10px',
              padding: '24px',
              width: '90%',
              maxWidth: '520px',
              border: '1px solid var(--color-border)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
              <h3 style={{ margin: 0, fontSize: '16px' }}>Knowledge Graph & Vector Index ({currentWorkspace?.name})</h3>
              <button onClick={() => setShowKnowledgeModal(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '16px' }}>✕</button>
            </div>
            <div style={{ fontSize: '13px', lineHeight: 1.6, color: 'var(--color-ink-muted)' }}>
              <p>This Workspace accesses PostgreSQL vector embeddings (HNSW), BM25 keyword search, and Knowledge Graph entity relationships associated with <strong>{currentWorkspace?.department_code || 'all authorized departments'}</strong>.</p>
              <p style={{ fontSize: '12px', background: 'var(--color-panel)', padding: '10px', borderRadius: '6px' }}>
                🔒 Note: Existing document ACL / authorization rules strictly govern document chunk access for all queries.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
