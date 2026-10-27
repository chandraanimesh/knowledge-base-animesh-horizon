import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  WorkspaceItem,
  WorkspaceProviderProfile,
  ProviderCapabilitiesSummary,
  ProviderTestConnectionResponse,
  workspaceApi,
} from '../api/workspaces';
import CustomDropdown, { DropdownOption } from './CustomDropdown';
import WorkspaceMCPIntegrationsView from './WorkspaceMCPIntegrationsView';
import {
  SettingsIcon,
  UsersIcon,
  CpuIcon,
  BookIcon,
  GlobeIcon,
  PlugIcon,
  ShieldIcon,
  BarChartIcon,
  ZapIcon,
  LockIcon,
  SearchIcon,
  AlertCircleIcon,
  CheckIcon,
  ServerIcon,
} from './mcpIcons';

export function OpenAIIcon({ size = 16, color = 'currentColor', style }: { size?: number; color?: string; style?: React.CSSProperties }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ display: 'inline-block', verticalAlign: 'middle', flexShrink: 0, ...style }}
    >
      <path
        d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.5045 4.5045 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.8956zm16.0993 3.8558L12.5973 8.3829l2.0201-1.1685a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.3927-.6813zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.006 9.2298V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.5702 4.6069zm-9.6987 5.753l-2.8378-1.6372 2.8378-1.6372 2.8377 1.6372z"
        fill={color}
      />
    </svg>
  );
}

export function OpenAIBadge({ size = 18, iconSize = 12 }: { size?: number; iconSize?: number }) {
  return (
    <div
      style={{
        width: `${size}px`,
        height: `${size}px`,
        borderRadius: '50%',
        backgroundColor: '#f1f5f9',
        border: '1px solid #e2e8f0',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)',
      }}
    >
      <OpenAIIcon size={iconSize} color="#0f172a" />
    </div>
  );
}

interface WorkspaceSettingsViewProps {
  workspace: WorkspaceItem;
  token: string;
  isOwnerOrAdmin: boolean;
  onToast?: (type: 'success' | 'error', message: string) => void;
  onBackToChat?: () => void;
}

type SettingsSection =
  | 'general'
  | 'members'
  | 'ai_providers'
  | 'rag_retrieval'
  | 'web_search'
  | 'integrations'
  | 'security'
  | 'usage_limits'
  | 'advanced';

type ProviderSubTab =
  | 'all'
  | 'llm'
  | 'embedding'
  | 'reranker'
  | 'retrieval'
  | 'web'
  | 'other';

interface ActivityItem {
  id: string;
  title: string;
  subtitle?: string;
  timestamp: string;
  type: 'success' | 'info' | 'warning' | 'test';
}

const GROQ_MODELS = [
  'llama-3.3-70b-versatile',
  'llama-3.1-8b-instant',
  'mixtral-8x7b-32768',
  'gemma2-9b-it',
];

const OPENAI_MODELS = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'];
const MOCK_MODELS = ['mock-deterministic-v1'];
const RERANKER_MODELS = [
  'cross-encoder/ms-marco-MiniLM-L-6-v2',
  'BAAI/bge-reranker-large',
];

export default function WorkspaceSettingsView({
  workspace,
  token,
  isOwnerOrAdmin,
  onToast,
  onBackToChat,
}: WorkspaceSettingsViewProps) {
  // Navigation State
  const [activeSection, setActiveSection] = useState<SettingsSection>('ai_providers');
  const [activeSubTab, setActiveSubTab] = useState<ProviderSubTab>('all');

  // Loading & Saving States
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedBanner, setSavedBanner] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>('Just now');
  const [isSavedViewMode, setIsSavedViewMode] = useState<boolean>(false);

  // Original Profile for Strict Diff Tracking
  const [originalProfile, setOriginalProfile] = useState<WorkspaceProviderProfile | null>(null);

  // 1. LLM Provider State
  const [llmEnabled, setLlmEnabled] = useState(true);
  const [llmProvider, setLlmProvider] = useState<string>('openai');
  const [llmModel, setLlmModel] = useState<string>('gpt-4o');
  const [llmCredRef, setLlmCredRef] = useState<string>('');
  const [temperature, setTemperature] = useState<number>(0.2);
  const [isDefaultLlm, setIsDefaultLlm] = useState(true);

  // 2. Embedding Provider State
  const [embeddingEnabled, setEmbeddingEnabled] = useState(true);
  const [embeddingProvider, setEmbeddingProvider] = useState<string>('sentence_transformers');
  const [embeddingModel, setEmbeddingModel] = useState<string>('all-MiniLM-L6-v2 (384 dim)');
  const [isDefaultEmbedding, setIsDefaultEmbedding] = useState(true);

  // 3. Reranker State
  const [rerankerEnabled, setRerankerEnabled] = useState(true);
  const [rerankerProvider, setRerankerProvider] = useState<string>('cross_encoder');
  const [rerankerModel, setRerankerModel] = useState<string>('cross-encoder/ms-marco-MiniLM-L-6-v2');
  const [rerankerToken, setRerankerToken] = useState<string>('');
  const [showRerankerToken, setShowRerankerToken] = useState(false);
  const [enablePipelineRerank, setEnablePipelineRerank] = useState(true);

  // 4. Retrieval & Search Settings
  const [vectorSearchEnabled, setVectorSearchEnabled] = useState(true);
  const [ftsEnabled, setFtsEnabled] = useState(true);
  const [rrfEnabled, setRrfEnabled] = useState(true);
  const [topK, setTopK] = useState<number>(8);
  const [rerankerTopK, setRerankerTopK] = useState<number>(5);

  // 5. Web Search Provider
  const [webEnabled, setWebEnabled] = useState(true);
  const [webProvider, setWebProvider] = useState<string>('tavily');
  const [webCredRef, setWebCredRef] = useState<string>('');

  // 6. Vector Database
  const [vectorDbEnabled, setVectorDbEnabled] = useState(true);
  const [vectorDbProvider, setVectorDbProvider] = useState<string>('postgresql+pgvector');

  // Probe & Health States
  const [testingStatus, setTestingStatus] = useState<Record<string, boolean>>({});
  const [testResults, setTestResults] = useState<Record<string, ProviderTestConnectionResponse | null>>({});

  // Review & Confirmation Modal State
  const [showConfirmReview, setShowConfirmReview] = useState(false);

  // Dropdown Options
  const llmProviderOptions: DropdownOption[] = useMemo(
    () => [
      { value: 'groq', label: 'Groq', description: 'Ultra-fast cloud inference (Llama 3.3)', icon: <ZapIcon size={14} color="#18181b" /> },
      { value: 'openai', label: 'OpenAI', description: 'GPT-4o, GPT-4o-mini', icon: <OpenAIBadge size={18} iconSize={12} /> },
      { value: 'mock', label: 'Mock LLM', description: 'Deterministic test engine (Zero-network)', icon: <CpuIcon size={14} color="#71717a" /> },
      { value: 'anthropic', label: 'Anthropic', description: 'Claude 3.5 Sonnet', icon: <CpuIcon size={14} color="#a1a1aa" />, disabled: true, badge: 'Unavailable on server' },
      { value: 'gemini', label: 'Google Gemini', description: 'Gemini 1.5 Pro / Flash', icon: <CpuIcon size={14} color="#a1a1aa" />, disabled: true, badge: 'Unavailable on server' },
    ],
    []
  );

  const llmModelOptions: DropdownOption[] = useMemo(() => {
    if (llmProvider === 'groq') {
      return GROQ_MODELS.map((m) => ({ value: m, label: m }));
    }
    if (llmProvider === 'openai') {
      return OPENAI_MODELS.map((m) => ({ value: m, label: m }));
    }
    if (llmProvider === 'mock') {
      return MOCK_MODELS.map((m) => ({ value: m, label: m }));
    }
    return [{ value: llmModel, label: llmModel }];
  }, [llmProvider, llmModel]);

  const handleLlmProviderChange = (val: string) => {
    setLlmProvider(val);
    if (val === 'groq' && !GROQ_MODELS.includes(llmModel)) {
      setLlmModel('llama-3.3-70b-versatile');
    } else if (val === 'openai' && !OPENAI_MODELS.includes(llmModel)) {
      setLlmModel('gpt-4o');
    } else if (val === 'mock' && !MOCK_MODELS.includes(llmModel)) {
      setLlmModel('mock-deterministic-v1');
    }
  };

  const embeddingProviderOptions: DropdownOption[] = useMemo(
    () => [
      { value: 'sentence_transformers', label: 'Sentence Transformers', description: 'Local ONNX/PyTorch in-process', icon: <CpuIcon size={14} color="#18181b" /> },
    ],
    []
  );

  const embeddingModelOptions: DropdownOption[] = useMemo(
    () => [
      { value: 'all-MiniLM-L6-v2 (384 dim)', label: 'all-MiniLM-L6-v2 (384 dim)', badge: 'Compatible' },
      { value: 'text-embedding-3-small', label: 'text-embedding-3-small (1536 dim)', disabled: true, badge: 'Incompatible dimension' },
    ],
    []
  );

  const rerankerProviderOptions: DropdownOption[] = useMemo(
    () => [
      { value: 'cross_encoder', label: 'CrossEncoder', description: 'Local PyTorch cross-encoder', icon: <BarChartIcon size={14} color="#18181b" /> },
      { value: 'huggingface', label: 'HuggingFace', description: 'Transformer pipeline', icon: <BarChartIcon size={14} color="#71717a" /> },
    ],
    []
  );

  const rerankerModelOptions: DropdownOption[] = useMemo(
    () => RERANKER_MODELS.map((m) => ({ value: m, label: m })),
    []
  );

  const webProviderOptions: DropdownOption[] = useMemo(
    () => [
      { value: 'tinyfish', label: 'TinyFish Search API', description: 'Real-time agentic web search & fast retrieval', icon: <SearchIcon size={14} color="#18181b" /> },
      { value: 'tavily', label: 'Tavily Search API', description: 'Real-time AI search API', icon: <GlobeIcon size={14} color="#18181b" /> },
      { value: 'firecrawl', label: 'Firecrawl Search & Reader', description: 'Web search & document scraper', icon: <GlobeIcon size={14} color="#71717a" /> },
    ],
    []
  );

  // Recent Activity Log
  const [recentActivity, setRecentActivity] = useState<ActivityItem[]>([
    {
      id: 'act-1',
      title: 'Workspace configuration active',
      subtitle: `Bound to ${workspace.name}`,
      timestamp: 'Today',
      type: 'info',
    },
  ]);

  const addActivity = useCallback((title: string, subtitle?: string, type: 'success' | 'info' | 'warning' | 'test' = 'info') => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setRecentActivity((prev) => [
      {
        id: `act-${Date.now()}-${Math.random()}`,
        title,
        subtitle,
        timestamp: timeStr,
        type,
      },
      ...prev.slice(0, 8),
    ]);
  }, []);

  // Load Persisted Backend Profile
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const profile = await workspaceApi.getWorkspaceProviderProfile(token, workspace.id).catch(() => null);

      if (profile) {
        setOriginalProfile(profile);

        // LLM
        if (profile.llm_provider) setLlmProvider(profile.llm_provider);
        if (profile.llm_model) setLlmModel(profile.llm_model);
        if (profile.llm_credential_ref) setLlmCredRef(profile.llm_credential_ref);

        // Embedding
        if (profile.embedding_provider) setEmbeddingProvider(profile.embedding_provider);
        if (profile.embedding_model) setEmbeddingModel(profile.embedding_model);

        // Reranker
        if (profile.reranker_provider) setRerankerProvider(profile.reranker_provider);
        if (profile.reranker_model) setRerankerModel(profile.reranker_model);

        // Web
        if (profile.web_provider) setWebProvider(profile.web_provider);
        if (profile.web_credential_ref) setWebCredRef(profile.web_credential_ref);

        // JSONB configuration options
        const cfg = profile.configuration || {};
        if (typeof cfg.temperature === 'number') setTemperature(cfg.temperature);
        if (typeof cfg.is_default_llm === 'boolean') setIsDefaultLlm(cfg.is_default_llm);
        if (typeof cfg.is_default_embedding === 'boolean') setIsDefaultEmbedding(cfg.is_default_embedding);
        if (typeof cfg.vector_search_enabled === 'boolean') setVectorSearchEnabled(cfg.vector_search_enabled);
        if (typeof cfg.fts_enabled === 'boolean') setFtsEnabled(cfg.fts_enabled);
        if (typeof cfg.rrf_enabled === 'boolean') setRrfEnabled(cfg.rrf_enabled);
        if (typeof cfg.top_k === 'number') setTopK(cfg.top_k);
        if (typeof cfg.reranker_top_k === 'number') setRerankerTopK(cfg.reranker_top_k);
        if (typeof cfg.reranker_enabled === 'boolean') {
          setRerankerEnabled(cfg.reranker_enabled);
          setEnablePipelineRerank(cfg.reranker_enabled);
        }
        if (typeof cfg.llm_enabled === 'boolean') setLlmEnabled(cfg.llm_enabled);
        if (typeof cfg.embedding_enabled === 'boolean') setEmbeddingEnabled(cfg.embedding_enabled);
        if (typeof cfg.web_enabled === 'boolean') setWebEnabled(cfg.web_enabled);

        if (profile.updated_at) {
          try {
            setLastUpdated(new Date(profile.updated_at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }));
          } catch {
            setLastUpdated('Recently');
          }
        }
      } else {
        setOriginalProfile(null);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load workspace settings');
    } finally {
      setLoading(false);
    }
  }, [token, workspace.id]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Compute Material Changes Diff
  const changedFields = useMemo(() => {
    const changes: { field: string; label: string; from: string; to: string }[] = [];
    const orig = originalProfile || ({} as Partial<WorkspaceProviderProfile>);
    const origCfg = orig.configuration || {};

    const origLlmProv = orig.llm_provider || 'openai';
    if (origLlmProv !== llmProvider) {
      changes.push({ field: 'llm_provider', label: 'LLM Provider', from: origLlmProv, to: llmProvider });
    }

    const origLlmModel = orig.llm_model || 'gpt-4o';
    if (origLlmModel !== llmModel) {
      changes.push({ field: 'llm_model', label: 'Default Model', from: origLlmModel, to: llmModel });
    }

    const origRerankEnabled = origCfg.reranker_enabled !== undefined ? origCfg.reranker_enabled : true;
    if (origRerankEnabled !== rerankerEnabled) {
      changes.push({
        field: 'reranker_enabled',
        label: 'CrossEncoder reranking',
        from: origRerankEnabled ? 'ON' : 'OFF',
        to: rerankerEnabled ? 'ON' : 'OFF',
      });
    }

    const origRerankModel = orig.reranker_model || 'cross-encoder/ms-marco-MiniLM-L-6-v2';
    if (origRerankModel !== rerankerModel) {
      changes.push({ field: 'reranker_model', label: 'Reranker model', from: origRerankModel, to: rerankerModel });
    }

    const origVec = origCfg.vector_search_enabled !== undefined ? origCfg.vector_search_enabled : true;
    if (origVec !== vectorSearchEnabled) {
      changes.push({
        field: 'vector_search_enabled',
        label: 'Vector search',
        from: origVec ? 'ON' : 'OFF',
        to: vectorSearchEnabled ? 'ON' : 'OFF',
      });
    }

    const origFts = origCfg.fts_enabled !== undefined ? origCfg.fts_enabled : true;
    if (origFts !== ftsEnabled) {
      changes.push({
        field: 'fts_enabled',
        label: 'PostgreSQL FTS',
        from: origFts ? 'ON' : 'OFF',
        to: ftsEnabled ? 'ON' : 'OFF',
      });
    }

    const origRrf = origCfg.rrf_enabled !== undefined ? origCfg.rrf_enabled : true;
    if (origRrf !== rrfEnabled) {
      changes.push({
        field: 'rrf_enabled',
        label: 'RRF Fusion',
        from: origRrf ? 'ON' : 'OFF',
        to: rrfEnabled ? 'ON' : 'OFF',
      });
    }

    const origTopK = origCfg.top_k !== undefined ? origCfg.top_k : 8;
    if (origTopK !== topK) {
      changes.push({
        field: 'top_k',
        label: 'Top K (retrieval)',
        from: String(origTopK),
        to: String(topK),
      });
    }

    const origRerankTopK = origCfg.reranker_top_k !== undefined ? origCfg.reranker_top_k : 5;
    if (origRerankTopK !== rerankerTopK) {
      changes.push({
        field: 'reranker_top_k',
        label: 'Reranker Top K',
        from: String(origRerankTopK),
        to: String(rerankerTopK),
      });
    }

    const origTemp = origCfg.temperature !== undefined ? origCfg.temperature : 0.2;
    if (origTemp !== temperature) {
      changes.push({
        field: 'temperature',
        label: 'Temperature',
        from: String(origTemp),
        to: String(temperature),
      });
    }

    const origWebProv = orig.web_provider || 'tavily';
    if (origWebProv !== webProvider) {
      changes.push({
        field: 'web_provider',
        label: 'Web Search Provider',
        from: origWebProv === 'tinyfish' ? 'TinyFish Search' : origWebProv === 'firecrawl' ? 'Firecrawl' : 'Tavily',
        to: webProvider === 'tinyfish' ? 'TinyFish Search' : webProvider === 'firecrawl' ? 'Firecrawl' : 'Tavily',
      });
    }

    const origWebEnabled = origCfg.web_enabled !== undefined ? origCfg.web_enabled : true;
    if (origWebEnabled !== webEnabled) {
      changes.push({
        field: 'web_enabled',
        label: 'Web Search',
        from: origWebEnabled ? 'ON' : 'OFF',
        to: webEnabled ? 'ON' : 'OFF',
      });
    }

    const origWebCred = orig.web_credential_ref || '';
    if (origWebCred !== webCredRef.trim()) {
      changes.push({
        field: 'web_credential_ref',
        label: 'Web Search Credential Ref',
        from: origWebCred ? '••••••••••••••••' : 'None (Server ENV)',
        to: webCredRef.trim() ? '••••••••••••••••' : 'None (Server ENV)',
      });
    }

    return changes;
  }, [
    originalProfile,
    llmProvider,
    llmModel,
    rerankerEnabled,
    rerankerModel,
    vectorSearchEnabled,
    ftsEnabled,
    rrfEnabled,
    topK,
    rerankerTopK,
    temperature,
    webProvider,
    webEnabled,
    webCredRef,
  ]);

  const hasUnsavedChanges = changedFields.length > 0;

  // Run Backend Connection Test
  const handleTestConnection = async (capability: string) => {
    let prov = '';
    let model = '';
    let credRef = '';

    if (capability === 'llm') {
      prov = llmProvider;
      model = llmModel;
      credRef = llmCredRef;
    } else if (capability === 'embedding') {
      prov = embeddingProvider;
      model = embeddingModel;
    } else if (capability === 'reranker') {
      prov = rerankerProvider;
      model = rerankerModel;
      credRef = rerankerToken;
    } else if (capability === 'web') {
      prov = webProvider;
      credRef = webCredRef;
    } else if (capability === 'vector_store') {
      prov = 'pgvector';
    }

    setTestingStatus((prev) => ({ ...prev, [capability]: true }));
    setTestResults((prev) => ({ ...prev, [capability]: null }));

    try {
      const res = await workspaceApi.testProviderConnection(token, workspace.id, {
        capability,
        provider: prov,
        model: model.trim() || null,
        credential_ref: credRef.trim() || null,
      });
      setTestResults((prev) => ({ ...prev, [capability]: res }));
      if (res.success) {
        const msg = `${capability.toUpperCase()} connection verified (${res.latency_ms ? res.latency_ms + 'ms' : 'OK'}).`;
        onToast?.('success', msg);
        addActivity(`${prov.toUpperCase()} connection test passed`, `Model: ${model || 'default'} (${res.latency_ms ? res.latency_ms + 'ms' : 'OK'})`, 'test');
      } else {
        const msg = `${capability.toUpperCase()} test failed: ${res.message}`;
        onToast?.('error', msg);
        addActivity(`${prov.toUpperCase()} test probe failed`, res.message, 'warning');
      }
    } catch (err: any) {
      const failRes: ProviderTestConnectionResponse = {
        success: false,
        capability,
        provider: prov,
        model,
        message: err.message || 'Connection test request failed',
      };
      setTestResults((prev) => ({ ...prev, [capability]: failRes }));
      onToast?.('error', failRes.message);
      addActivity(`${capability.toUpperCase()} probe error`, err.message, 'warning');
    } finally {
      setTestingStatus((prev) => ({ ...prev, [capability]: false }));
    }
  };

  // Run All Health Checks for Right Panel
  const handleRefreshAllHealth = async () => {
    await Promise.all([
      handleTestConnection('llm'),
      handleTestConnection('embedding'),
      handleTestConnection('reranker'),
      handleTestConnection('web'),
      handleTestConnection('vector_store'),
    ]);
  };

  // Initiate Save (opens Review Changes modal if dirty)
  const handleInitiateSave = () => {
    if (!isOwnerOrAdmin) return;
    if (hasUnsavedChanges) {
      setShowConfirmReview(true);
    } else {
      setIsSavedViewMode(true);
    }
  };

  // Confirm Save & Apply to Backend
  const handleConfirmSave = async () => {
    setSaving(true);
    setError(null);

    const payload: Partial<WorkspaceProviderProfile> = {
      name: `${workspace.name} Profile`,
      is_active: true,
      llm_provider: llmProvider || null,
      llm_model: llmModel.trim() || null,
      llm_credential_ref: llmCredRef.trim() || null,
      embedding_provider: embeddingProvider || null,
      embedding_model: embeddingModel.trim() || null,
      reranker_provider: rerankerProvider || null,
      reranker_model: rerankerModel.trim() || null,
      web_provider: webProvider || null,
      web_credential_ref: webCredRef.trim() || null,
      configuration: {
        ...(originalProfile?.configuration || {}),
        temperature,
        is_default_llm: isDefaultLlm,
        is_default_embedding: isDefaultEmbedding,
        vector_search_enabled: vectorSearchEnabled,
        fts_enabled: ftsEnabled,
        rrf_enabled: rrfEnabled,
        top_k: topK,
        reranker_top_k: rerankerTopK,
        reranker_enabled: rerankerEnabled,
        llm_enabled: llmEnabled,
        embedding_enabled: embeddingEnabled,
        web_enabled: webEnabled,
      },
    };

    try {
      const saved = await workspaceApi.upsertWorkspaceProviderProfile(token, workspace.id, payload);
      setOriginalProfile(saved);
      setShowConfirmReview(false);
      setIsSavedViewMode(true);
      setSavedBanner('Configuration applied successfully');
      setLastUpdated(new Date().toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }));
      onToast?.('success', 'Workspace provider configuration saved and active.');

      // Add audit activities for material changes
      changedFields.forEach((c) => {
        addActivity(`${c.label} updated`, `${c.from} → ${c.to}`, 'success');
      });
      addActivity('Workspace configuration applied', `Applied to ${workspace.name}`, 'success');
    } catch (err: any) {
      const msg = err.message || 'Failed to update workspace provider profile';
      setError(msg);
      setShowConfirmReview(false);
      onToast?.('error', msg);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: '#f8fafc',
        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
        color: '#0f172a',
      }}
    >
      {/* Top Banner (Reference 3) */}
      {savedBanner && (
        <div
          style={{
            background: '#dcfce7',
            borderBottom: '1px solid #bbf7d0',
            padding: '10px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '13px',
            color: '#15803d',
            fontWeight: 500,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '15px' }}>✓</span>
            <span>{savedBanner}</span>
          </div>
          <button
            onClick={() => setSavedBanner(null)}
            style={{
              background: 'none',
              border: 'none',
              color: '#16a34a',
              cursor: 'pointer',
              fontSize: '16px',
              padding: '0 4px',
            }}
          >
            ×
          </button>
        </div>
      )}

      {/* Main Header / Breadcrumbs Bar */}
      <header
        style={{
          padding: '16px 28px',
          background: '#ffffff',
          borderBottom: '1px solid #e2e8f0',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>
            {onBackToChat && (
              <span
                onClick={onBackToChat}
                style={{ cursor: 'pointer', color: '#2563eb', fontWeight: 500 }}
              >
                ← Back to Chat
              </span>
            )}
            {onBackToChat && <span>/</span>}
            <span>Workspace Settings</span>
            <span>›</span>
            <span style={{ color: '#0f172a', fontWeight: 600 }}>
              {activeSection === 'integrations'
                ? 'MCP Integrations'
                : activeSection === 'rag_retrieval'
                ? 'RAG Retrieval'
                : activeSection === 'web_search'
                ? 'Web Search'
                : 'AI & Model Providers'}
            </span>
          </div>
          <h1 style={{ fontSize: '20px', fontWeight: 700, margin: 0, color: '#0f172a' }}>
            {activeSection === 'integrations'
              ? 'MCP Integrations'
              : activeSection === 'rag_retrieval'
              ? 'RAG Retrieval'
              : activeSection === 'web_search'
              ? 'Web Search'
              : 'AI & Model Providers'}
          </h1>
          <p style={{ margin: '2px 0 0 0', fontSize: '13px', color: '#64748b' }}>
            {activeSection === 'integrations'
              ? `Manage external MCP server connections and tools for ${workspace.name}.`
              : `Configure model providers, embeddings, reranking, and search services for ${workspace.name}.`}
          </p>
        </div>

        {/* Top Right Actions */}
        {activeSection !== 'integrations' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {isSavedViewMode && !hasUnsavedChanges ? (
            <>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: '#f1f5f9',
                  padding: '4px 10px',
                  borderRadius: '6px',
                  fontSize: '11px',
                  color: '#475569',
                }}
              >
                <span style={{ color: '#16a34a', fontWeight: 700 }}>✓</span>
                <span>Saved</span>
                <span style={{ color: '#94a3b8' }}>•</span>
                <span>Last updated: {lastUpdated}</span>
              </div>
              <button
                type="button"
                onClick={() => setIsSavedViewMode(false)}
                style={{
                  padding: '8px 16px',
                  borderRadius: '6px',
                  background: '#2563eb',
                  color: '#ffffff',
                  border: 'none',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >
                <span>⚙ Edit Configuration</span>
              </button>
            </>
          ) : (
            <>
              {hasUnsavedChanges && (
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    background: '#fef3c7',
                    color: '#92400e',
                    padding: '4px 10px',
                    borderRadius: '20px',
                    fontSize: '12px',
                    fontWeight: 600,
                  }}
                >
                  <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: '#d97706' }} />
                  <span>Unsaved changes</span>
                </div>
              )}
              <button
                type="button"
                onClick={handleInitiateSave}
                disabled={saving || !isOwnerOrAdmin}
                style={{
                  padding: '8px 18px',
                  borderRadius: '6px',
                  background: hasUnsavedChanges ? '#2563eb' : '#3b82f6',
                  color: '#ffffff',
                  border: 'none',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: !isOwnerOrAdmin ? 'not-allowed' : 'pointer',
                  boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
                }}
              >
                {saving ? 'Saving...' : 'Save & Apply'}
              </button>
            </>
          )}
        </div>
        )}
      </header>

      {/* Main Body Layout: Left Subnav + Center Content + Right Status Panels */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left Sub-Navigation Sidebar (Reference 4) */}
        <aside
          style={{
            width: '230px',
            background: '#ffffff',
            borderRight: '1px solid #e2e8f0',
            padding: '16px 10px',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
            flexShrink: 0,
          }}
        >
          <div style={{ fontSize: '11px', fontWeight: 600, color: '#94a3b8', padding: '4px 10px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Workspace Settings
          </div>

          {[
            { id: 'general', label: 'General', desc: 'Workspace details & preferences', icon: SettingsIcon },
            { id: 'members', label: 'Members', desc: 'Manage workspace members', icon: UsersIcon },
            { id: 'ai_providers', label: 'AI & Model Providers', desc: 'LLM, embeddings, reranker', icon: CpuIcon },
            { id: 'rag_retrieval', label: 'RAG & Retrieval', desc: 'Retrieval settings and sources', icon: BookIcon },
            { id: 'web_search', label: 'Web Search', desc: 'Web search providers', icon: SearchIcon },
            { id: 'integrations', label: 'Integrations', desc: 'Connect external tools (MCP/CRM)', icon: PlugIcon },
            { id: 'security', label: 'Security', desc: 'Access control and permissions', icon: ShieldIcon },
            { id: 'usage_limits', label: 'Usage & Limits', desc: 'Resource quotas and tokens', icon: BarChartIcon },
            { id: 'advanced', label: 'Advanced', desc: 'Configuration exports & API', icon: ZapIcon },
          ].map((item) => {
            const isActive = activeSection === item.id;
            const IconComponent = item.icon;
            return (
              <div
                key={item.id}
                onClick={() => setActiveSection(item.id as SettingsSection)}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '10px',
                  padding: '8px 10px',
                  borderRadius: '6px',
                  background: isActive ? '#f4f4f5' : 'transparent',
                  border: isActive ? '1px solid #e4e4e7' : '1px solid transparent',
                  cursor: 'pointer',
                  transition: 'background 0.12s ease',
                }}
              >
                <span style={{ marginTop: '2px', display: 'flex', alignItems: 'center' }}>
                  <IconComponent size={14} color={isActive ? '#18181b' : '#71717a'} />
                </span>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '13px', fontWeight: isActive ? 600 : 500, color: isActive ? '#18181b' : '#3f3f46' }}>
                    {item.label}
                  </span>
                  <span style={{ fontSize: '11px', color: '#71717a', lineHeight: 1.2, marginTop: '2px' }}>
                    {item.desc}
                  </span>
                </div>
              </div>
            );
          })}
        </aside>

        {/* Center Content Column */}
        <main style={{ flex: 1, overflowY: 'auto', padding: '24px 32px' }}>
          {error && (
            <div
              style={{
                padding: '12px 16px',
                borderRadius: '6px',
                background: '#fef2f2',
                border: '1px solid #fecaca',
                color: '#991b1b',
                fontSize: '13px',
                marginBottom: '16px',
              }}
            >
              {error}
            </div>
          )}

          {!isOwnerOrAdmin && (
            <div
              style={{
                padding: '10px 14px',
                borderRadius: '6px',
                background: '#fafafa',
                border: '1px solid #e4e4e7',
                color: '#52525b',
                fontSize: '12.5px',
                marginBottom: '16px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <LockIcon size={14} color="#71717a" style={{ flexShrink: 0 }} />
              <span>
                <strong style={{ color: '#18181b' }}>Read-Only Mode:</strong> Only Workspace Owners or System Administrators can modify provider infrastructure.
              </span>
            </div>
          )}

          {activeSection === 'integrations' ? (
            <WorkspaceMCPIntegrationsView
              workspace={workspace}
              token={token}
              isOwnerOrAdmin={isOwnerOrAdmin}
              onToast={onToast}
            />
          ) : activeSection !== 'ai_providers' && activeSection !== 'rag_retrieval' && activeSection !== 'web_search' ? (
            /* Unfinished Sections Explicitly Marked */
            <div
              style={{
                background: '#ffffff',
                border: '1px solid #e4e4e7',
                borderRadius: '8px',
                padding: '40px 24px',
                textAlign: 'center',
                maxWidth: '640px',
                margin: '20px auto',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '12px' }}>
                <LockIcon size={24} color="#a1a1aa" />
              </div>
              <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#18181b', margin: '0 0 6px 0' }}>
                {activeSection.replace('_', ' ').toUpperCase()} Configuration
              </h3>
              <p style={{ fontSize: '13px', color: '#71717a', maxWidth: '420px', margin: '0 auto 16px auto', lineHeight: 1.5 }}>
                This section is managed through platform policies and department permissions. Advanced controls for this capability are currently unavailable in this milestone.
              </p>
              <button
                type="button"
                onClick={() => setActiveSection('ai_providers')}
                style={{
                  padding: '7px 14px',
                  borderRadius: '6px',
                  background: '#18181b',
                  color: '#ffffff',
                  border: '1px solid #18181b',
                  fontSize: '12.5px',
                  fontWeight: 500,
                  cursor: 'pointer',
                }}
              >
                Go to AI & Model Providers
              </button>
            </div>
          ) : (
            /* AI & Model Providers Main Experience */
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', maxWidth: '880px' }}>
              {/* Horizontal Subtabs */}
              <div
                style={{
                  display: 'flex',
                  gap: '8px',
                  borderBottom: '1px solid #e2e8f0',
                  paddingBottom: '10px',
                }}
              >
                {[
                  { id: 'all', label: 'All Providers' },
                  { id: 'llm', label: 'LLM Providers' },
                  { id: 'embedding', label: 'Embedding Providers' },
                  { id: 'reranker', label: 'Reranker / Cross Encoder' },
                  { id: 'retrieval', label: 'Retrieval Settings' },
                  { id: 'web', label: 'Web Search' },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveSubTab(tab.id as ProviderSubTab)}
                    style={{
                      background: activeSubTab === tab.id ? '#eff6ff' : 'transparent',
                      color: activeSubTab === tab.id ? '#1d4ed8' : '#64748b',
                      border: activeSubTab === tab.id ? '1px solid #bfdbfe' : '1px solid transparent',
                      padding: '5px 12px',
                      borderRadius: '6px',
                      fontSize: '12px',
                      fontWeight: activeSubTab === tab.id ? 600 : 500,
                      cursor: 'pointer',
                    }}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* CARD 1: LLM PROVIDER */}
              {(activeSubTab === 'all' || activeSubTab === 'llm') && (
                <div
                  style={{
                    background: '#ffffff',
                    border: '1px solid #e4e4e7',
                    borderRadius: '8px',
                    padding: '20px 24px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <div
                        style={{
                          width: '32px',
                          height: '32px',
                          borderRadius: '6px',
                          background: '#f4f4f5',
                          border: '1px solid #e4e4e7',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                        }}
                      >
                        {llmProvider === 'openai' ? (
                          <OpenAIIcon size={18} color="#18181b" />
                        ) : llmProvider === 'groq' ? (
                          <ZapIcon size={16} color="#18181b" />
                        ) : (
                          <CpuIcon size={16} color="#71717a" />
                        )}
                      </div>
                      <div>
                        <div style={{ fontSize: '14px', fontWeight: 600, color: '#18181b' }}>OpenAI / Groq LLM</div>
                        <div style={{ fontSize: '12px', color: '#71717a' }}>Used for generating answers and planning (e.g. GPT-4, Llama 3.3)</div>
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: 500, color: '#3f3f46' }}>
                        <input
                          type="checkbox"
                          checked={llmEnabled}
                          onChange={(e) => setLlmEnabled(e.target.checked)}
                          disabled={!isOwnerOrAdmin}
                        />
                        <span>{llmEnabled ? 'Enabled' : 'Disabled'}</span>
                      </label>
                      <span
                        style={{
                          fontSize: '11px',
                          padding: '2px 8px',
                          borderRadius: '4px',
                          background: testResults.llm?.success ? '#f0fdf4' : '#f4f4f5',
                          color: testResults.llm?.success ? '#15803d' : '#71717a',
                          border: `1px solid ${testResults.llm?.success ? '#bbf7d0' : '#e4e4e7'}`,
                          fontWeight: 500,
                        }}
                      >
                        {testResults.llm?.success ? `Healthy ${testResults.llm.latency_ms ? testResults.llm.latency_ms + 'ms' : ''}` : 'Ready'}
                      </span>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                    <div>
                      <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '6px' }}>Provider</label>
                      <CustomDropdown
                        value={llmProvider}
                        onChange={handleLlmProviderChange}
                        options={llmProviderOptions}
                        disabled={!isOwnerOrAdmin}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '6px' }}>Default Model</label>
                      <CustomDropdown
                        value={llmModel}
                        onChange={(val) => setLlmModel(val)}
                        options={llmModelOptions}
                        disabled={!isOwnerOrAdmin}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '6px' }}>Temperature</label>
                      <input
                        type="number"
                        min="0"
                        max="2"
                        step="0.1"
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value) || 0)}
                        disabled={!isOwnerOrAdmin}
                        style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #e4e4e7', fontSize: '13px', background: '#ffffff', color: '#18181b' }}
                      />
                    </div>
                  </div>

                  {/* LLM Environment Variable Reference */}
                  <div style={{ marginTop: '16px' }}>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '6px' }}>
                      Environment variable name
                    </label>
                    <input
                      type="text"
                      value={llmCredRef}
                      onChange={(e) => setLlmCredRef(e.target.value)}
                      placeholder="e.g. OPENAI_API_KEY (Server Environment Variable)"
                      disabled={!isOwnerOrAdmin}
                      style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #e4e4e7', fontSize: '13px', background: '#ffffff', color: '#18181b' }}
                    />
                    <div style={{ fontSize: '11px', color: '#71717a', marginTop: '4px' }}>
                      Enter the name of a server-side environment variable (e.g. <code style={{ background: '#f4f4f5', padding: '1px 4px', borderRadius: '3px' }}>OPENAI_API_KEY</code> / <code style={{ background: '#f4f4f5', padding: '1px 4px', borderRadius: '3px' }}>GROQ_API_KEY</code>). Leave blank to use server default. Do not enter raw API keys.
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px', paddingTop: '12px', borderTop: '1px solid #f4f4f5' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#52525b', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={isDefaultLlm}
                        onChange={(e) => setIsDefaultLlm(e.target.checked)}
                        disabled={!isOwnerOrAdmin}
                      />
                      <span>Use as default LLM provider for this workspace</span>
                    </label>

                    <button
                      type="button"
                      onClick={() => handleTestConnection('llm')}
                      disabled={testingStatus.llm}
                      style={{
                        padding: '6px 14px',
                        borderRadius: '6px',
                        border: '1px solid #e4e4e7',
                        background: '#ffffff',
                        color: '#18181b',
                        fontSize: '12px',
                        fontWeight: 500,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '5px',
                      }}
                    >
                      <ZapIcon size={12} color="#71717a" />
                      <span>{testingStatus.llm ? 'Testing probe…' : 'Test Connection'}</span>
                    </button>
                  </div>
                </div>
              )}

              {/* CARD 2: EMBEDDING PROVIDER */}
              {(activeSubTab === 'all' || activeSubTab === 'embedding') && (
                <div
                  style={{
                    background: '#ffffff',
                    border: '1px solid #e4e4e7',
                    borderRadius: '8px',
                    padding: '20px 24px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <div style={{ width: '32px', height: '32px', borderRadius: '6px', background: '#f4f4f5', border: '1px solid #e4e4e7', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                        <CpuIcon size={16} color="#18181b" />
                      </div>
                      <div>
                        <div style={{ fontSize: '14px', fontWeight: 600, color: '#18181b' }}>Embedding Provider</div>
                        <div style={{ fontSize: '12px', color: '#71717a' }}>Used for document and query embeddings (vector representations)</div>
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: 500, color: '#3f3f46' }}>
                        <input
                          type="checkbox"
                          checked={embeddingEnabled}
                          onChange={(e) => setEmbeddingEnabled(e.target.checked)}
                          disabled={!isOwnerOrAdmin}
                        />
                        <span>{embeddingEnabled ? 'Enabled' : 'Disabled'}</span>
                      </label>
                      <span
                        style={{
                          fontSize: '11px',
                          padding: '2px 8px',
                          borderRadius: '4px',
                          background: testResults.embedding?.success ? '#f0fdf4' : '#f4f4f5',
                          color: testResults.embedding?.success ? '#15803d' : '#71717a',
                          border: `1px solid ${testResults.embedding?.success ? '#bbf7d0' : '#e4e4e7'}`,
                          fontWeight: 500,
                        }}
                      >
                        {testResults.embedding?.success ? `Healthy ${testResults.embedding.latency_ms ? testResults.embedding.latency_ms + 'ms' : ''}` : 'Connected'}
                      </span>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
                    <div>
                      <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '6px' }}>Provider</label>
                      <CustomDropdown
                        value={embeddingProvider}
                        onChange={(val) => setEmbeddingProvider(val)}
                        options={embeddingProviderOptions}
                        disabled={!isOwnerOrAdmin}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '6px' }}>Model</label>
                      <CustomDropdown
                        value={embeddingModel}
                        onChange={(val) => setEmbeddingModel(val)}
                        options={embeddingModelOptions}
                        disabled={!isOwnerOrAdmin}
                      />
                    </div>
                  </div>

                  <div
                    style={{
                      marginTop: '14px',
                      padding: '10px 14px',
                      background: '#fafafa',
                      border: '1px solid #e4e4e7',
                      borderRadius: '6px',
                      fontSize: '12px',
                      color: '#52525b',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                    }}
                  >
                    <ShieldIcon size={14} color="#71717a" style={{ flexShrink: 0 }} />
                    <span>This model runs locally in-process. 384-dimension vector requirement is enforced by the pgvector schema.</span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px', paddingTop: '12px', borderTop: '1px solid #f4f4f5' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#52525b', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={isDefaultEmbedding}
                        onChange={(e) => setIsDefaultEmbedding(e.target.checked)}
                        disabled={!isOwnerOrAdmin}
                      />
                      <span>Use for document and query embeddings</span>
                    </label>

                    <button
                      type="button"
                      onClick={() => handleTestConnection('embedding')}
                      disabled={testingStatus.embedding}
                      style={{
                        padding: '6px 14px',
                        borderRadius: '6px',
                        border: '1px solid #e4e4e7',
                        background: '#ffffff',
                        color: '#18181b',
                        fontSize: '12px',
                        fontWeight: 500,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '5px',
                      }}
                    >
                      <ZapIcon size={12} color="#71717a" />
                      <span>{testingStatus.embedding ? 'Validating…' : 'Test Connection'}</span>
                    </button>
                  </div>
                </div>
              )}

              {/* CARD 3: RERANKER / CROSS ENCODER (Reference 5) */}
              {(activeSubTab === 'all' || activeSubTab === 'reranker') && (
                <div
                  style={{
                    background: '#ffffff',
                    border: '1px solid #e4e4e7',
                    borderRadius: '8px',
                    padding: '20px 24px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <div style={{ width: '32px', height: '32px', borderRadius: '6px', background: '#f4f4f5', border: '1px solid #e4e4e7', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                        <BarChartIcon size={16} color="#18181b" />
                      </div>
                      <div>
                        <div style={{ fontSize: '14px', fontWeight: 600, color: '#18181b' }}>Reranker / Cross Encoder</div>
                        <div style={{ fontSize: '12px', color: '#71717a' }}>Used for reranking retrieved documents for precision & relevance</div>
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: 500, color: '#3f3f46' }}>
                        <input
                          type="checkbox"
                          checked={rerankerEnabled}
                          onChange={(e) => setRerankerEnabled(e.target.checked)}
                          disabled={!isOwnerOrAdmin}
                        />
                        <span>{rerankerEnabled ? 'Enabled' : 'Disabled'}</span>
                      </label>
                      <span
                        style={{
                          fontSize: '11px',
                          padding: '2px 8px',
                          borderRadius: '4px',
                          background: testResults.reranker?.success ? '#f0fdf4' : '#f4f4f5',
                          color: testResults.reranker?.success ? '#15803d' : '#71717a',
                          border: `1px solid ${testResults.reranker?.success ? '#bbf7d0' : '#e4e4e7'}`,
                          fontWeight: 500,
                        }}
                      >
                        {testResults.reranker?.success ? `Healthy ${testResults.reranker.latency_ms ? testResults.reranker.latency_ms + 'ms' : ''}` : 'Connected'}
                      </span>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
                    <div>
                      <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '6px' }}>Provider</label>
                      <CustomDropdown
                        value={rerankerProvider}
                        onChange={(val) => setRerankerProvider(val)}
                        options={rerankerProviderOptions}
                        disabled={!isOwnerOrAdmin}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '6px' }}>Model</label>
                      <CustomDropdown
                        value={rerankerModel}
                        onChange={(val) => setRerankerModel(val)}
                        options={rerankerModelOptions}
                        disabled={!isOwnerOrAdmin}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px', paddingTop: '12px', borderTop: '1px solid #f4f4f5' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#52525b', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={enablePipelineRerank}
                        onChange={(e) => {
                          setEnablePipelineRerank(e.target.checked);
                          setRerankerEnabled(e.target.checked);
                        }}
                        disabled={!isOwnerOrAdmin}
                      />
                      <span>Enable reranking in retrieval pipeline</span>
                    </label>

                    <button
                      type="button"
                      onClick={() => handleTestConnection('reranker')}
                      disabled={testingStatus.reranker}
                      style={{
                        padding: '6px 14px',
                        borderRadius: '6px',
                        border: '1px solid #e4e4e7',
                        background: '#ffffff',
                        color: '#18181b',
                        fontSize: '12px',
                        fontWeight: 500,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '5px',
                      }}
                    >
                      <ZapIcon size={12} color="#71717a" />
                      <span>{testingStatus.reranker ? 'Testing probe…' : 'Test Connection'}</span>
                    </button>
                  </div>
                </div>
              )}

              {/* CARD 4: RETRIEVAL & SEARCH SETTINGS (Reference 5) */}
              {(activeSubTab === 'all' || activeSubTab === 'retrieval') && (
                <div
                  style={{
                    background: '#ffffff',
                    border: '1px solid #e4e4e7',
                    borderRadius: '8px',
                    padding: '20px 24px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                    <div style={{ width: '32px', height: '32px', borderRadius: '6px', background: '#f4f4f5', border: '1px solid #e4e4e7', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                      <SearchIcon size={16} color="#18181b" />
                    </div>
                    <div>
                      <div style={{ fontSize: '14px', fontWeight: 600, color: '#18181b' }}>Retrieval & Search Settings</div>
                      <div style={{ fontSize: '12px', color: '#71717a' }}>Configure search components used in RAG retrieval pipeline</div>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '14px', marginBottom: '18px' }}>
                    {/* Toggle: Vector Search */}
                    <div
                      style={{
                        padding: '12px 14px',
                        border: '1px solid #e4e4e7',
                        borderRadius: '6px',
                        background: vectorSearchEnabled ? '#fafafa' : '#ffffff',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                      }}
                    >
                      <div>
                        <div style={{ fontSize: '13px', fontWeight: 600, color: '#18181b' }}>Vector Search (pgvector)</div>
                        <div style={{ fontSize: '11px', color: '#71717a' }}>Semantic search using embeddings</div>
                      </div>
                      <input
                        type="checkbox"
                        checked={vectorSearchEnabled}
                        onChange={(e) => setVectorSearchEnabled(e.target.checked)}
                        disabled={!isOwnerOrAdmin}
                        style={{ width: '16px', height: '16px', cursor: 'pointer' }}
                      />
                    </div>

                    {/* Toggle: PostgreSQL FTS */}
                    <div
                      style={{
                        padding: '12px 14px',
                        border: '1px solid #e4e4e7',
                        borderRadius: '6px',
                        background: ftsEnabled ? '#fafafa' : '#ffffff',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                      }}
                    >
                      <div>
                        <div style={{ fontSize: '13px', fontWeight: 600, color: '#18181b' }}>PostgreSQL FTS (BM25)</div>
                        <div style={{ fontSize: '11px', color: '#71717a' }}>Keyword search using full text search</div>
                      </div>
                      <input
                        type="checkbox"
                        checked={ftsEnabled}
                        onChange={(e) => setFtsEnabled(e.target.checked)}
                        disabled={!isOwnerOrAdmin}
                        style={{ width: '16px', height: '16px', cursor: 'pointer' }}
                      />
                    </div>

                    {/* Toggle: RRF Fusion */}
                    <div
                      style={{
                        padding: '12px 14px',
                        border: '1px solid #e4e4e7',
                        borderRadius: '6px',
                        background: rrfEnabled ? '#fafafa' : '#ffffff',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                      }}
                    >
                      <div>
                        <div style={{ fontSize: '13px', fontWeight: 600, color: '#18181b' }}>RRF Fusion</div>
                        <div style={{ fontSize: '11px', color: '#71717a' }}>Combine results using Reciprocal Rank Fusion</div>
                      </div>
                      <input
                        type="checkbox"
                        checked={rrfEnabled}
                        onChange={(e) => setRrfEnabled(e.target.checked)}
                        disabled={!isOwnerOrAdmin}
                        style={{ width: '16px', height: '16px', cursor: 'pointer' }}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                    <div>
                      <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '6px' }}>
                        Top K (Initial Retrieval)
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="50"
                        value={topK}
                        onChange={(e) => setTopK(parseInt(e.target.value, 10) || 1)}
                        disabled={!isOwnerOrAdmin}
                        style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #e4e4e7', fontSize: '13px', background: '#ffffff', color: '#18181b' }}
                      />
                      <span style={{ fontSize: '11px', color: '#71717a' }}>Number of candidates retrieved before reranking</span>
                    </div>

                    <div>
                      <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '6px' }}>
                        Reranker Top K
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="20"
                        value={rerankerTopK}
                        onChange={(e) => setRerankerTopK(parseInt(e.target.value, 10) || 1)}
                        disabled={!isOwnerOrAdmin}
                        style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #e4e4e7', fontSize: '13px', background: '#ffffff', color: '#18181b' }}
                      />
                      <span style={{ fontSize: '11px', color: '#71717a' }}>Final rescored chunks sent to context compiler</span>
                    </div>
                  </div>
                </div>
              )}

              {/* CARD 5: WEB SEARCH & VECTOR DATABASE ROW */}
              {(activeSubTab === 'all' || activeSubTab === 'web') && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '16px' }}>
                  {/* Web Search */}
                  <div
                    style={{
                      background: '#ffffff',
                      border: '1px solid #e4e4e7',
                      borderRadius: '8px',
                      padding: '18px 20px',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div style={{ width: '30px', height: '30px', borderRadius: '6px', background: '#f4f4f5', border: '1px solid #e4e4e7', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                          <GlobeIcon size={15} color="#18181b" />
                        </div>
                        <div>
                          <div style={{ fontSize: '14px', fontWeight: 600, color: '#18181b' }}>Web Search Provider</div>
                          <div style={{ fontSize: '11.5px', color: '#71717a' }}>Used for research and real-time information</div>
                        </div>
                      </div>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11.5px', color: '#52525b' }}>
                        <input
                          type="checkbox"
                          checked={webEnabled}
                          onChange={(e) => setWebEnabled(e.target.checked)}
                          disabled={!isOwnerOrAdmin}
                        />
                        <span>{webEnabled ? 'Enabled' : 'Disabled'}</span>
                      </label>
                    </div>

                    <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                      <div style={{ flex: 1 }}>
                        <CustomDropdown
                          value={webProvider}
                          onChange={(val) => setWebProvider(val)}
                          options={webProviderOptions}
                          disabled={!isOwnerOrAdmin}
                        />
                      </div>
                      <button
                        type="button"
                        onClick={() => handleTestConnection('web')}
                        disabled={testingStatus.web}
                        style={{ padding: '7px 12px', borderRadius: '6px', border: '1px solid #e4e4e7', background: '#ffffff', color: '#18181b', fontSize: '12px', cursor: 'pointer', fontWeight: 500 }}
                      >
                        {testingStatus.web ? 'Testing…' : 'Test'}
                      </button>
                    </div>

                    {/* Web Search Environment Variable Reference */}
                    <div style={{ marginTop: '12px' }}>
                      <label style={{ display: 'block', fontSize: '11.5px', fontWeight: 500, color: '#52525b', marginBottom: '5px' }}>
                        Environment variable name
                      </label>
                      <input
                        type="text"
                        value={webCredRef}
                        onChange={(e) => setWebCredRef(e.target.value)}
                        placeholder="e.g. TINYFISH_PROD_KEY (Server Environment Variable)"
                        disabled={!isOwnerOrAdmin}
                        style={{ width: '100%', padding: '7px 10px', borderRadius: '6px', border: '1px solid #e4e4e7', fontSize: '12.5px', background: '#ffffff', color: '#18181b' }}
                      />
                      <div style={{ fontSize: '10.5px', color: '#71717a', marginTop: '4px' }}>
                        Enter the name of a server-side environment variable (e.g. <code style={{ background: '#f4f4f5', padding: '1px 4px', borderRadius: '3px' }}>TINYFISH_PROD_KEY</code>). Leave blank to inherit server default (<code style={{ background: '#f4f4f5', padding: '1px 4px', borderRadius: '3px' }}>TINYFISH_API_KEY</code> / <code style={{ background: '#f4f4f5', padding: '1px 4px', borderRadius: '3px' }}>TAVILY_API_KEY</code>). Do not enter raw API keys.
                      </div>
                    </div>
                  </div>

                  {/* Vector Database */}
                  <div
                    style={{
                      background: '#ffffff',
                      border: '1px solid #e4e4e7',
                      borderRadius: '8px',
                      padding: '18px 20px',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div style={{ width: '30px', height: '30px', borderRadius: '6px', background: '#f4f4f5', border: '1px solid #e4e4e7', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                          <ServerIcon size={15} color="#18181b" />
                        </div>
                        <div>
                          <div style={{ fontSize: '14px', fontWeight: 600, color: '#18181b' }}>Vector Database</div>
                          <div style={{ fontSize: '11.5px', color: '#71717a' }}>Used for semantic search</div>
                        </div>
                      </div>
                      <span style={{ fontSize: '11px', color: '#15803d', fontWeight: 500, padding: '2px 7px', borderRadius: '4px', background: '#f0fdf4', border: '1px solid #bbf7d0' }}>Connected</span>
                    </div>

                    <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                      <input
                        type="text"
                        value="PostgreSQL + pgvector"
                        disabled
                        style={{ flex: 1, padding: '7px 10px', borderRadius: '6px', border: '1px solid #cbd5e1', background: '#f8fafc', fontSize: '12.5px' }}
                      />
                      <button
                        type="button"
                        onClick={() => handleTestConnection('vector_store')}
                        disabled={testingStatus.vector_store}
                        style={{ padding: '7px 12px', borderRadius: '6px', border: '1px solid #cbd5e1', background: '#f8fafc', fontSize: '12px', cursor: 'pointer' }}
                      >
                        {testingStatus.vector_store ? 'Checking…' : 'Probe'}
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </main>

        {/* Right Status Panel (Reference 3 & 4) */}
        <aside
          style={{
            width: '280px',
            background: '#ffffff',
            borderLeft: '1px solid #e2e8f0',
            padding: '20px 16px',
            display: 'flex',
            flexDirection: 'column',
            gap: '24px',
            flexShrink: 0,
            overflowY: 'auto',
          }}
        >
          {/* Panel 1: Provider Health Status */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <span style={{ fontSize: '13px', fontWeight: 700, color: '#0f172a' }}>Provider Health Status</span>
              <button
                type="button"
                onClick={handleRefreshAllHealth}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#2563eb',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                <span>↻</span>
                <span>Refresh</span>
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {[
                {
                  name: 'OpenAI / Groq',
                  status: testResults.llm?.success ? 'Healthy' : 'Ready',
                  latency: testResults.llm?.latency_ms ? `${testResults.llm.latency_ms}ms` : '120ms',
                  active: true,
                },
                {
                  name: 'Anthropic',
                  status: 'Disabled',
                  latency: '-',
                  active: false,
                },
                {
                  name: 'Google Gemini',
                  status: 'Disabled',
                  latency: '-',
                  active: false,
                },
                {
                  name: 'HuggingFace (Reranker)',
                  status: testResults.reranker?.success ? 'Healthy' : rerankerEnabled ? 'Healthy' : 'Disabled',
                  latency: testResults.reranker?.latency_ms ? `${testResults.reranker.latency_ms}ms` : '85ms',
                  active: rerankerEnabled,
                },
                {
                  name: 'Sentence Transformers',
                  status: 'Healthy',
                  latency: testResults.embedding?.latency_ms ? `${testResults.embedding.latency_ms}ms` : '72ms',
                  active: true,
                },
                {
                  name: webProvider === 'tinyfish' ? 'TinyFish Search' : webProvider === 'tavily' ? 'Tavily Search' : 'Firecrawl',
                  status: webEnabled ? (testResults.web ? (testResults.web.success ? 'Healthy' : 'Error') : 'Healthy') : 'Disabled',
                  latency: testResults.web?.latency_ms ? `${testResults.web.latency_ms}ms` : '180ms',
                  active: webEnabled,
                },
                {
                  name: 'PostgreSQL (pgvector)',
                  status: 'Healthy',
                  latency: testResults.vector_store?.latency_ms ? `${testResults.vector_store.latency_ms}ms` : '15ms',
                  active: true,
                },
              ].map((p) => (
                <div
                  key={p.name}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '6px 8px',
                    borderRadius: '6px',
                    background: '#f8fafc',
                    fontSize: '11.5px',
                  }}
                >
                  <span style={{ color: p.active ? '#1e293b' : '#94a3b8', fontWeight: 500 }}>{p.name}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        color: p.status === 'Healthy' ? '#16a34a' : '#64748b',
                        fontWeight: 600,
                      }}
                    >
                      <span style={{ fontSize: '8px' }}>●</span>
                      <span>{p.status}</span>
                    </span>
                    <span style={{ color: '#94a3b8', fontFamily: 'monospace', minWidth: '40px', textAlign: 'right' }}>
                      {p.latency}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Panel 2: Recent Activity (Reference 3) */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <span style={{ fontSize: '13px', fontWeight: 700, color: '#0f172a' }}>Recent Activity</span>
              <span style={{ fontSize: '11px', color: '#2563eb', fontWeight: 500 }}>Live Feed</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {recentActivity.map((act) => (
                <div
                  key={act.id}
                  style={{
                    padding: '8px 10px',
                    borderRadius: '6px',
                    background: '#f8fafc',
                    border: '1px solid #f1f5f9',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '2px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: '#1e293b' }}>{act.title}</span>
                    <span style={{ fontSize: '10px', color: '#94a3b8' }}>{act.timestamp}</span>
                  </div>
                  {act.subtitle && (
                    <div style={{ fontSize: '11px', color: '#64748b', lineHeight: 1.3 }}>
                      {act.subtitle}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>

      {/* MODAL: Review Changes Before Applying (Reference 1) */}
      {showConfirmReview && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(15, 23, 42, 0.6)',
            backdropFilter: 'blur(3px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}
          onClick={() => setShowConfirmReview(false)}
        >
          <div
            style={{
              background: '#ffffff',
              borderRadius: '14px',
              padding: '24px 28px',
              maxWidth: '520px',
              width: '92%',
              boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
              border: '1px solid #e2e8f0',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '18px' }}>⚙️</span>
                <span style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a' }}>
                  Apply configuration changes?
                </span>
              </div>
              <button
                type="button"
                onClick={() => setShowConfirmReview(false)}
                style={{ background: 'none', border: 'none', fontSize: '20px', color: '#94a3b8', cursor: 'pointer' }}
              >
                &times;
              </button>
            </div>

            <p style={{ fontSize: '12.5px', color: '#64748b', margin: '0 0 16px 0' }}>
              Please review the changes below before applying them to this workspace.
            </p>

            <div style={{ fontSize: '11.5px', fontWeight: 600, color: '#475569', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Configuration Changes
            </div>

            {/* Changes List */}
            <div
              style={{
                maxHeight: '220px',
                overflowY: 'auto',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                background: '#f8fafc',
                padding: '8px 12px',
                marginBottom: '16px',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
              }}
            >
              {changedFields.map((c) => (
                <div
                  key={c.field}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '4px 0',
                    borderBottom: '1px solid #f1f5f9',
                    fontSize: '12px',
                  }}
                >
                  <span style={{ fontWeight: 600, color: '#334155' }}>{c.label}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ color: '#94a3b8', textDecoration: 'line-through', fontSize: '11px' }}>{c.from}</span>
                    <span style={{ color: '#64748b' }}>→</span>
                    <span
                      style={{
                        color: '#0369a1',
                        background: '#e0f2fe',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontWeight: 600,
                      }}
                    >
                      {c.to}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Warning Callout (Reference 1) */}
            <div
              style={{
                padding: '10px 14px',
                borderRadius: '8px',
                background: '#fffbeb',
                border: '1px solid #fef3c7',
                color: '#b45309',
                fontSize: '12px',
                lineHeight: 1.4,
                marginBottom: '20px',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '8px',
              }}
            >
              <span style={{ fontSize: '15px' }}>⚠️</span>
              <div>
                <strong>Changes affect new retrieval requests in this workspace.</strong> Ongoing requests will not be affected.
              </div>
            </div>

            {/* Modal Actions */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button
                type="button"
                onClick={() => setShowConfirmReview(false)}
                disabled={saving}
                style={{
                  padding: '8px 16px',
                  borderRadius: '6px',
                  border: '1px solid #cbd5e1',
                  background: '#ffffff',
                  color: '#475569',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '13px',
                }}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleConfirmSave}
                disabled={saving}
                style={{
                  padding: '8px 20px',
                  borderRadius: '6px',
                  border: 'none',
                  background: '#2563eb',
                  color: '#ffffff',
                  cursor: saving ? 'not-allowed' : 'pointer',
                  fontWeight: 600,
                  fontSize: '13px',
                }}
              >
                {saving ? 'Applying…' : 'Confirm & Apply'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
