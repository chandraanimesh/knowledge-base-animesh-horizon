import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  WorkspaceItem,
  WorkspaceProviderProfile,
  ProviderCapabilitiesSummary,
  ProviderTestConnectionResponse,
  workspaceApi,
} from '../api/workspaces';

interface WorkspaceProviderSettingsModalProps {
  workspace: WorkspaceItem;
  token: string;
  isOwnerOrAdmin: boolean;
  onClose: () => void;
  onToast?: (type: 'success' | 'error', message: string) => void;
}

const GROQ_RECOMMENDED_MODELS = [
  'llama-3.3-70b-versatile',
  'llama-3.1-8b-instant',
  'mixtral-8x7b-32768',
  'gemma2-9b-it',
];

const MOCK_MODELS = ['mock-deterministic-v1'];

export default function WorkspaceProviderSettingsModal({
  workspace,
  token,
  isOwnerOrAdmin,
  onClose,
  onToast,
}: WorkspaceProviderSettingsModalProps) {
  const [capabilities, setCapabilities] = useState<ProviderCapabilitiesSummary>({
    llm: ['groq', 'mock'],
    embedding: ['sentence_transformers'],
    document_reader: ['standard', 'docling'],
    reranker: ['cross_encoder'],
    ocr: ['paddle_ocr'],
    web: ['tinyfish', 'tavily', 'firecrawl'],
  });

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Original Profile for Diff Tracking
  const [originalProfile, setOriginalProfile] = useState<WorkspaceProviderProfile | null>(null);

  // Profile Form State
  const [llmProvider, setLlmProvider] = useState<string>('groq');
  const [llmModel, setLlmModel] = useState<string>('llama-3.3-70b-versatile');
  const [llmCredRef, setLlmCredRef] = useState<string>('');

  const [embeddingProvider, setEmbeddingProvider] = useState<string>('sentence_transformers');
  const [embeddingModel, setEmbeddingModel] = useState<string>('sentence-transformers/all-MiniLM-L6-v2');

  const [docReaderProvider, setDocReaderProvider] = useState<string>('standard');

  const [rerankerProvider, setRerankerProvider] = useState<string>('cross_encoder');
  const [rerankerModel, setRerankerModel] = useState<string>('cross-encoder/ms-marco-MiniLM-L-6-v2');

  const [ocrProvider, setOcrProvider] = useState<string>('paddle_ocr');

  const [webProvider, setWebProvider] = useState<string>('firecrawl');
  const [webCredRef, setWebCredRef] = useState<string>('');

  // Connection Testing State
  const [testingStatus, setTestingStatus] = useState<Record<string, boolean>>({});
  const [testResults, setTestResults] = useState<Record<string, ProviderTestConnectionResponse | null>>({});

  // Review & Confirmation Modal State
  const [showConfirmReview, setShowConfirmReview] = useState(false);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [caps, profile] = await Promise.all([
        workspaceApi.getProviderCapabilities(token).catch(() => null),
        workspaceApi.getWorkspaceProviderProfile(token, workspace.id).catch(() => null),
      ]);

      if (caps) {
        setCapabilities(caps);
      }

      if (profile) {
        setOriginalProfile(profile);
        if (profile.llm_provider) setLlmProvider(profile.llm_provider);
        if (profile.llm_model) setLlmModel(profile.llm_model);
        if (profile.llm_credential_ref) setLlmCredRef(profile.llm_credential_ref);

        if (profile.embedding_provider) setEmbeddingProvider(profile.embedding_provider);
        if (profile.embedding_model) setEmbeddingModel(profile.embedding_model);

        if (profile.document_reader_provider) setDocReaderProvider(profile.document_reader_provider);

        if (profile.reranker_provider) setRerankerProvider(profile.reranker_provider);
        if (profile.reranker_model) setRerankerModel(profile.reranker_model);

        if (profile.ocr_provider) setOcrProvider(profile.ocr_provider);

        if (profile.web_provider) setWebProvider(profile.web_provider);
        if (profile.web_credential_ref) setWebCredRef(profile.web_credential_ref);
      } else {
        setOriginalProfile(null);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load workspace provider profile');
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
    const orig = originalProfile || {} as Partial<WorkspaceProviderProfile>;

    if ((orig.llm_provider || 'groq') !== llmProvider) {
      changes.push({ field: 'llm_provider', label: 'LLM Provider', from: orig.llm_provider || 'Default (groq)', to: llmProvider });
    }
    if ((orig.llm_model || '') !== llmModel.trim()) {
      changes.push({ field: 'llm_model', label: 'LLM Model', from: orig.llm_model || 'Platform Default', to: llmModel.trim() || 'Platform Default' });
    }
    if ((orig.llm_credential_ref || '') !== llmCredRef.trim()) {
      changes.push({ field: 'llm_cred_ref', label: 'LLM Credential Ref', from: orig.llm_credential_ref || 'None', to: llmCredRef.trim() || 'None' });
    }
    if ((orig.embedding_provider || 'sentence_transformers') !== embeddingProvider) {
      changes.push({ field: 'embedding_provider', label: 'Embedding Provider', from: orig.embedding_provider || 'sentence_transformers', to: embeddingProvider });
    }
    if ((orig.embedding_model || '') !== embeddingModel.trim()) {
      changes.push({ field: 'embedding_model', label: 'Embedding Model', from: orig.embedding_model || 'Default', to: embeddingModel.trim() || 'Default' });
    }
    if ((orig.reranker_provider || 'cross_encoder') !== rerankerProvider) {
      changes.push({ field: 'reranker_provider', label: 'Reranker Provider', from: orig.reranker_provider || 'cross_encoder', to: rerankerProvider });
    }
    if ((orig.reranker_model || '') !== rerankerModel.trim()) {
      changes.push({ field: 'reranker_model', label: 'Reranker Model', from: orig.reranker_model || 'Default', to: rerankerModel.trim() || 'Default' });
    }
    if ((orig.web_provider || 'firecrawl') !== webProvider) {
      changes.push({ field: 'web_provider', label: 'Web Search Provider', from: orig.web_provider || 'firecrawl', to: webProvider });
    }
    if ((orig.web_credential_ref || '') !== webCredRef.trim()) {
      changes.push({ field: 'web_cred_ref', label: 'Web Credential Ref', from: orig.web_credential_ref || 'None', to: webCredRef.trim() || 'None' });
    }

    return changes;
  }, [
    originalProfile,
    llmProvider,
    llmModel,
    llmCredRef,
    embeddingProvider,
    embeddingModel,
    rerankerProvider,
    rerankerModel,
    webProvider,
    webCredRef,
  ]);

  const isDirty = changedFields.length > 0;

  // Execute Backend Connection Test
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
    } else if (capability === 'web') {
      prov = webProvider;
      credRef = webCredRef;
    } else if (capability === 'document_reader') {
      prov = docReaderProvider;
    } else if (capability === 'ocr') {
      prov = ocrProvider;
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
      if (res.success && onToast) {
        onToast('success', `${capability.toUpperCase()} connection verified (${res.latency_ms ? res.latency_ms + 'ms' : 'OK'}).`);
      } else if (!res.success && onToast) {
        onToast('error', `${capability.toUpperCase()} probe failed: ${res.message}`);
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
      if (onToast) onToast('error', failRes.message);
    } finally {
      setTestingStatus((prev) => ({ ...prev, [capability]: false }));
    }
  };

  const handleInitiateSave = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isOwnerOrAdmin) return;
    if (changedFields.length > 0) {
      setShowConfirmReview(true);
    } else {
      onClose();
    }
  };

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
      document_reader_provider: docReaderProvider || null,
      reranker_provider: rerankerProvider || null,
      reranker_model: rerankerModel.trim() || null,
      ocr_provider: ocrProvider || null,
      web_provider: webProvider || null,
      web_credential_ref: webCredRef.trim() || null,
    };

    try {
      await workspaceApi.upsertWorkspaceProviderProfile(token, workspace.id, payload);
      if (onToast) onToast('success', 'Workspace provider configuration saved and active.');
      setShowConfirmReview(false);
      onClose();
    } catch (err: any) {
      const msg = err.message || 'Failed to update workspace provider profile';
      setError(msg);
      setShowConfirmReview(false);
      if (onToast) onToast('error', msg);
    } finally {
      setSaving(false);
    }
  };

  const handleCancelOrClose = () => {
    if (isDirty) {
      const confirmed = window.confirm('You have unsaved changes in provider configuration. Are you sure you want to discard them?');
      if (!confirmed) return;
    }
    onClose();
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.65)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
      }}
      onClick={handleCancelOrClose}
    >
      <div
        style={{
          background: '#ffffff',
          borderRadius: '16px',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          width: '94%',
          maxWidth: '760px',
          maxHeight: '92vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          border: '1px solid #e2e8f0',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: '20px 28px',
            borderBottom: '1px solid #e2e8f0',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            background: 'linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)',
          }}
        >
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0f172a' }}>
                Workspace AI & Provider Settings
              </span>
              <span
                style={{
                  fontSize: '0.75rem',
                  padding: '3px 10px',
                  borderRadius: '20px',
                  background: '#e0f2fe',
                  color: '#0369a1',
                  fontWeight: 600,
                }}
              >
                Adaptive Engine
              </span>
              {isDirty && (
                <span
                  style={{
                    fontSize: '0.75rem',
                    padding: '3px 8px',
                    borderRadius: '20px',
                    background: '#fef3c7',
                    color: '#92400e',
                    fontWeight: 600,
                  }}
                >
                  Unsaved Changes
                </span>
              )}
            </div>
            <p style={{ margin: '4px 0 0 0', fontSize: '0.85rem', color: '#64748b' }}>
              Configure infrastructure providers for <strong>{workspace.name}</strong> ({workspace.code})
            </p>
          </div>
          <button
            onClick={handleCancelOrClose}
            aria-label="Close modal"
            style={{
              background: 'none',
              border: 'none',
              fontSize: '1.5rem',
              color: '#94a3b8',
              cursor: 'pointer',
              lineHeight: 1,
            }}
          >
            &times;
          </button>
        </div>

        {/* Form Body */}
        <div style={{ padding: '24px 28px', overflowY: 'auto', flex: 1 }}>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '60px 20px', color: '#64748b' }}>
              <div style={{ fontSize: '1rem', fontWeight: 500 }}>Loading workspace provider profile...</div>
            </div>
          ) : (
            <form onSubmit={handleInitiateSave} style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
              {error && (
                <div
                  style={{
                    padding: '12px 16px',
                    borderRadius: '8px',
                    background: '#fef2f2',
                    border: '1px solid #fecaca',
                    color: '#991b1b',
                    fontSize: '0.875rem',
                  }}
                >
                  {error}
                </div>
              )}

              {!isOwnerOrAdmin && (
                <div
                  style={{
                    padding: '12px 16px',
                    borderRadius: '8px',
                    background: '#fffbe8',
                    border: '1px solid #ffe58f',
                    color: '#8c6b00',
                    fontSize: '0.85rem',
                  }}
                >
                  🔒 <strong>Read-Only Mode:</strong> Only Workspace Owners or System Administrators can modify provider infrastructure.
                </div>
              )}

              {/* Secret Policy Notice */}
              <div
                style={{
                  padding: '10px 14px',
                  borderRadius: '8px',
                  background: '#f0fdf4',
                  border: '1px solid #bbf7d0',
                  color: '#166534',
                  fontSize: '0.8rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                }}
              >
                <span>🛡️</span>
                <span>
                  <strong>Zero-Secret Storage Policy:</strong> API keys are read server-side through secure configuration.
                  Enter secret reference names (e.g. <code>GROQ_DEFAULT</code>, <code>ENV_KEY</code>) instead of actual tokens.
                </span>
              </div>

              {/* 1. LLM Provider */}
              <div
                style={{
                  padding: '18px',
                  border: '1px solid #e2e8f0',
                  borderRadius: '12px',
                  background: '#ffffff',
                  boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.05)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '1rem', fontWeight: 600, color: '#0f172a' }}>
                      1. LLM Provider (Generation & Writing)
                    </span>
                    <span style={{ fontSize: '0.75rem', padding: '2px 8px', borderRadius: '4px', background: '#f1f5f9', color: '#475569' }}>
                      Pluggable
                    </span>
                  </div>
                  <button
                    type="button"
                    disabled={testingStatus['llm']}
                    onClick={() => handleTestConnection('llm')}
                    style={{
                      padding: '5px 12px',
                      borderRadius: '6px',
                      border: '1px solid #cbd5e1',
                      background: '#f8fafc',
                      color: '#0f172a',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                      cursor: testingStatus['llm'] ? 'not-allowed' : 'pointer',
                    }}
                  >
                    {testingStatus['llm'] ? 'Testing Probe...' : '⚡ Test Connection'}
                  </button>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '14px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: '6px' }}>
                      Provider Selection
                    </label>
                    <select
                      value={llmProvider}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => {
                        const nextProv = e.target.value;
                        setLlmProvider(nextProv);
                        if (nextProv === 'mock') setLlmModel('mock-deterministic-v1');
                        else if (nextProv === 'groq') setLlmModel('llama-3.3-70b-versatile');
                      }}
                      style={{ width: '100%', padding: '9px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                    >
                      {capabilities.llm.map((p) => (
                        <option key={p} value={p}>{p === 'mock' ? 'Mock (Deterministic Test Engine)' : p.toUpperCase()}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: '6px' }}>
                      Model Identifier
                    </label>
                    <input
                      type="text"
                      list="llm-model-options"
                      placeholder="e.g. llama-3.3-70b-versatile"
                      value={llmModel}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setLlmModel(e.target.value)}
                      style={{ width: '100%', padding: '9px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                    />
                    <datalist id="llm-model-options">
                      {(llmProvider === 'mock' ? MOCK_MODELS : GROQ_RECOMMENDED_MODELS).map((m) => (
                        <option key={m} value={m} />
                      ))}
                    </datalist>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: '6px' }}>
                      Credential Reference
                    </label>
                    <input
                      type="text"
                      placeholder="e.g. GROQ_DEFAULT"
                      value={llmCredRef}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setLlmCredRef(e.target.value)}
                      style={{ width: '100%', padding: '9px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                    />
                  </div>
                </div>

                {/* Connection probe outcome */}
                {testResults['llm'] && (
                  <div
                    style={{
                      marginTop: '12px',
                      padding: '8px 12px',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      background: testResults['llm']?.success ? '#f0fdf4' : '#fef2f2',
                      border: `1px solid ${testResults['llm']?.success ? '#bbf7d0' : '#fecaca'}`,
                      color: testResults['llm']?.success ? '#166534' : '#991b1b',
                    }}
                  >
                    {testResults['llm']?.success ? '✓' : '✕'} {testResults['llm']?.message}
                    {testResults['llm']?.latency_ms && ` (${testResults['llm']?.latency_ms}ms)`}
                  </div>
                )}
              </div>

              {/* 2. Embedding Provider */}
              <div
                style={{
                  padding: '18px',
                  border: '1px solid #e2e8f0',
                  borderRadius: '12px',
                  background: '#ffffff',
                  boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.05)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '1rem', fontWeight: 600, color: '#0f172a' }}>
                      2. Embedding Provider (Vector Index)
                    </span>
                    <span style={{ fontSize: '0.75rem', padding: '2px 8px', borderRadius: '4px', background: '#dcfce7', color: '#15803d', fontWeight: 600 }}>
                      384-Dimensions Locked
                    </span>
                  </div>
                  <button
                    type="button"
                    disabled={testingStatus['embedding']}
                    onClick={() => handleTestConnection('embedding')}
                    style={{
                      padding: '5px 12px',
                      borderRadius: '6px',
                      border: '1px solid #cbd5e1',
                      background: '#f8fafc',
                      color: '#0f172a',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                      cursor: testingStatus['embedding'] ? 'not-allowed' : 'pointer',
                    }}
                  >
                    {testingStatus['embedding'] ? 'Verifying...' : '⚡ Validate 384-Dim'}
                  </button>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: '6px' }}>
                      Provider
                    </label>
                    <select
                      value={embeddingProvider}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setEmbeddingProvider(e.target.value)}
                      style={{ width: '100%', padding: '9px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                    >
                      {capabilities.embedding.map((p) => (
                        <option key={p} value={p}>{p}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: '6px' }}>
                      Model (Must generate 384 dimensions)
                    </label>
                    <input
                      type="text"
                      placeholder="sentence-transformers/all-MiniLM-L6-v2"
                      value={embeddingModel}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setEmbeddingModel(e.target.value)}
                      style={{ width: '100%', padding: '9px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                    />
                  </div>
                </div>

                {testResults['embedding'] && (
                  <div
                    style={{
                      marginTop: '12px',
                      padding: '8px 12px',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      background: testResults['embedding']?.success ? '#f0fdf4' : '#fef2f2',
                      border: `1px solid ${testResults['embedding']?.success ? '#bbf7d0' : '#fecaca'}`,
                      color: testResults['embedding']?.success ? '#166534' : '#991b1b',
                    }}
                  >
                    {testResults['embedding']?.success ? '✓' : '✕'} {testResults['embedding']?.message}
                  </div>
                )}
              </div>

              {/* 3. Reranker Provider */}
              <div
                style={{
                  padding: '18px',
                  border: '1px solid #e2e8f0',
                  borderRadius: '12px',
                  background: '#ffffff',
                  boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.05)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '1rem', fontWeight: 600, color: '#0f172a' }}>
                      3. Reranker Provider (Cross-Encoder / RRF)
                    </span>
                  </div>
                  <button
                    type="button"
                    disabled={testingStatus['reranker']}
                    onClick={() => handleTestConnection('reranker')}
                    style={{
                      padding: '5px 12px',
                      borderRadius: '6px',
                      border: '1px solid #cbd5e1',
                      background: '#f8fafc',
                      color: '#0f172a',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                      cursor: testingStatus['reranker'] ? 'not-allowed' : 'pointer',
                    }}
                  >
                    {testingStatus['reranker'] ? 'Testing...' : '⚡ Test Reranker'}
                  </button>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: '6px' }}>
                      Provider
                    </label>
                    <select
                      value={rerankerProvider}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setRerankerProvider(e.target.value)}
                      style={{ width: '100%', padding: '9px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                    >
                      {capabilities.reranker.map((p) => (
                        <option key={p} value={p}>{p}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: '6px' }}>
                      Reranker Model
                    </label>
                    <input
                      type="text"
                      placeholder="cross-encoder/ms-marco-MiniLM-L-6-v2"
                      value={rerankerModel}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setRerankerModel(e.target.value)}
                      style={{ width: '100%', padding: '9px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                    />
                  </div>
                </div>

                {testResults['reranker'] && (
                  <div
                    style={{
                      marginTop: '12px',
                      padding: '8px 12px',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      background: testResults['reranker']?.success ? '#f0fdf4' : '#fef2f2',
                      border: `1px solid ${testResults['reranker']?.success ? '#bbf7d0' : '#fecaca'}`,
                      color: testResults['reranker']?.success ? '#166534' : '#991b1b',
                    }}
                  >
                    {testResults['reranker']?.success ? '✓' : '✕'} {testResults['reranker']?.message}
                  </div>
                )}
              </div>

              {/* 4. Document Reader & Web Providers */}
              <div
                style={{
                  padding: '18px',
                  border: '1px solid #e2e8f0',
                  borderRadius: '12px',
                  background: '#ffffff',
                  boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.05)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
                  <span style={{ fontSize: '1rem', fontWeight: 600, color: '#0f172a' }}>
                    4. Ingestion & Web Research Providers
                  </span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: '6px' }}>
                      Document Reader
                    </label>
                    <select
                      value={docReaderProvider}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setDocReaderProvider(e.target.value)}
                      style={{ width: '100%', padding: '9px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                    >
                      {capabilities.document_reader.map((p) => (
                        <option key={p} value={p}>{p}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: '6px' }}>
                      Web Search Provider
                    </label>
                    <select
                      value={webProvider}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setWebProvider(e.target.value)}
                      style={{ width: '100%', padding: '9px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                    >
                      {capabilities.web.map((p) => (
                        <option key={p} value={p}>{p}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', marginTop: '8px' }}>
                <button
                  type="button"
                  onClick={handleCancelOrClose}
                  style={{
                    padding: '9px 20px',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    background: '#ffffff',
                    color: '#475569',
                    cursor: 'pointer',
                    fontWeight: 600,
                    fontSize: '0.875rem',
                  }}
                >
                  Cancel
                </button>
                {isOwnerOrAdmin && (
                  <button
                    type="submit"
                    disabled={saving || !isDirty}
                    style={{
                      padding: '9px 24px',
                      borderRadius: '8px',
                      border: 'none',
                      background: isDirty ? '#0284c7' : '#94a3b8',
                      color: '#ffffff',
                      cursor: isDirty && !saving ? 'pointer' : 'not-allowed',
                      fontWeight: 600,
                      fontSize: '0.875rem',
                      boxShadow: isDirty ? '0 4px 6px -1px rgba(2, 132, 199, 0.3)' : 'none',
                    }}
                  >
                    {saving ? 'Saving...' : 'Review & Apply Changes'}
                  </button>
                )}
              </div>
            </form>
          )}
        </div>
      </div>

      {/* Material Changes Confirmation Modal */}
      {showConfirmReview && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.55)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1100,
          }}
          onClick={() => setShowConfirmReview(false)}
        >
          <div
            style={{
              background: '#ffffff',
              borderRadius: '14px',
              padding: '24px 28px',
              maxWidth: '520px',
              width: '90%',
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.2)',
              border: '1px solid #e2e8f0',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
              Confirm Workspace Provider Configuration
            </div>
            <p style={{ fontSize: '0.85rem', color: '#64748b', marginBottom: '16px' }}>
              Review the infrastructure changes that will be applied to <strong>{workspace.name}</strong>:
            </p>

            <div
              style={{
                maxHeight: '220px',
                overflowY: 'auto',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                padding: '12px',
                background: '#f8fafc',
                marginBottom: '16px',
              }}
            >
              {changedFields.map((c) => (
                <div
                  key={c.field}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '6px 0',
                    borderBottom: '1px solid #edf2f7',
                    fontSize: '0.8rem',
                  }}
                >
                  <span style={{ fontWeight: 600, color: '#334155' }}>{c.label}:</span>
                  <span style={{ color: '#0284c7' }}>
                    <s style={{ color: '#94a3b8', marginRight: '6px' }}>{c.from}</s> → {c.to}
                  </span>
                </div>
              ))}
            </div>

            <div
              style={{
                padding: '8px 12px',
                borderRadius: '6px',
                background: '#eff6ff',
                color: '#1d4ed8',
                fontSize: '0.78rem',
                marginBottom: '20px',
              }}
            >
              ℹ️ Ongoing requests in flight will continue with their bound instances. Subsequent queries in this workspace will immediately adopt these settings.
            </div>

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
                  fontSize: '0.85rem',
                }}
              >
                Back to Edit
              </button>
              <button
                type="button"
                onClick={handleConfirmSave}
                disabled={saving}
                style={{
                  padding: '8px 20px',
                  borderRadius: '6px',
                  border: 'none',
                  background: '#0284c7',
                  color: '#ffffff',
                  cursor: saving ? 'not-allowed' : 'pointer',
                  fontWeight: 600,
                  fontSize: '0.85rem',
                }}
              >
                {saving ? 'Applying...' : 'Confirm & Apply'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
