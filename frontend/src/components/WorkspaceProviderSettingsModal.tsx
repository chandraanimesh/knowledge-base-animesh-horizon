import React, { useState, useEffect, useCallback } from 'react';
import {
  WorkspaceItem,
  WorkspaceProviderProfile,
  ProviderCapabilitiesSummary,
  workspaceApi,
} from '../api/workspaces';

interface WorkspaceProviderSettingsModalProps {
  workspace: WorkspaceItem;
  token: string;
  isOwnerOrAdmin: boolean;
  onClose: () => void;
  onToast?: (type: 'success' | 'error', message: string) => void;
}

export default function WorkspaceProviderSettingsModal({
  workspace,
  token,
  isOwnerOrAdmin,
  onClose,
  onToast,
}: WorkspaceProviderSettingsModalProps) {
  const [capabilities, setCapabilities] = useState<ProviderCapabilitiesSummary>({
    llm: ['groq'],
    embedding: ['sentence_transformers'],
    document_reader: ['docling'],
    reranker: ['cross_encoder'],
    ocr: ['paddle_ocr'],
    web: ['firecrawl'],
  });

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Profile Form State
  const [llmProvider, setLlmProvider] = useState<string>('groq');
  const [llmModel, setLlmModel] = useState<string>('');
  const [llmCredRef, setLlmCredRef] = useState<string>('');

  const [embeddingProvider, setEmbeddingProvider] = useState<string>('sentence_transformers');
  const [embeddingModel, setEmbeddingModel] = useState<string>('');

  const [docReaderProvider, setDocReaderProvider] = useState<string>('docling');

  const [rerankerProvider, setRerankerProvider] = useState<string>('cross_encoder');
  const [rerankerModel, setRerankerModel] = useState<string>('');

  const [ocrProvider, setOcrProvider] = useState<string>('paddle_ocr');

  const [webProvider, setWebProvider] = useState<string>('firecrawl');
  const [webCredRef, setWebCredRef] = useState<string>('');

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

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isOwnerOrAdmin) return;
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
      if (onToast) onToast('success', 'Workspace provider configuration saved successfully.');
      onClose();
    } catch (err: any) {
      const msg = err.message || 'Failed to update workspace provider profile';
      setError(msg);
      if (onToast) onToast('error', msg);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        fontFamily: 'var(--font-sans)',
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'var(--color-surface, #ffffff)',
          borderRadius: '12px',
          boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04)',
          width: '90%',
          maxWidth: '680px',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          border: '1px solid var(--color-border, #e5e7eb)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: '20px 24px',
            borderBottom: '1px solid var(--color-border, #e5e7eb)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            background: '#f8fafc',
          }}
        >
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--color-text, #111827)' }}>
                Workspace Settings
              </span>
              <span
                style={{
                  fontSize: '0.75rem',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  background: '#e0f2fe',
                  color: '#0369a1',
                  fontWeight: 500,
                }}
              >
                RAG / Providers
              </span>
            </div>
            <p style={{ margin: '4px 0 0 0', fontSize: '0.875rem', color: '#6b7280' }}>
              Configure infrastructure providers for {workspace.name} ({workspace.code})
            </p>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '1.5rem',
              color: '#9ca3af',
              cursor: 'pointer',
            }}
          >
            &times;
          </button>
        </div>

        {/* Form Body */}
        <div style={{ padding: '24px', overflowY: 'auto', flex: 1 }}>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '40px', color: '#6b7280' }}>
              Loading provider settings...
            </div>
          ) : (
            <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
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
                    padding: '10px 14px',
                    borderRadius: '6px',
                    background: '#fffbe8',
                    border: '1px solid #ffe58f',
                    color: '#8c6b00',
                    fontSize: '0.8125rem',
                  }}
                >
                  Read-only: Only Workspace OWNER or ADMIN can modify provider configuration.
                </div>
              )}

              {/* 1. LLM */}
              <div style={{ padding: '16px', border: '1px solid #e2e8f0', borderRadius: '8px', background: '#ffffff' }}>
                <h4 style={{ margin: '0 0 12px 0', fontSize: '0.95rem', fontWeight: 600, color: '#1e293b' }}>
                  1. LLM Provider (Generation)
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 500, color: '#475569', marginBottom: '4px' }}>
                      Provider
                    </label>
                    <select
                      value={llmProvider}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setLlmProvider(e.target.value)}
                      style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                    >
                      {capabilities.llm.map((p) => (
                        <option key={p} value={p}>{p}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 500, color: '#475569', marginBottom: '4px' }}>
                      Model (Optional)
                    </label>
                    <input
                      type="text"
                      placeholder="e.g. llama-3.3-70b-versatile"
                      value={llmModel}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setLlmModel(e.target.value)}
                      style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 500, color: '#475569', marginBottom: '4px' }}>
                      Credential Ref (Secret Name)
                    </label>
                    <input
                      type="text"
                      placeholder="e.g. GROQ_PROD"
                      value={llmCredRef}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setLlmCredRef(e.target.value)}
                      style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                    />
                  </div>
                </div>
              </div>

              {/* 2. Embedding */}
              <div style={{ padding: '16px', border: '1px solid #e2e8f0', borderRadius: '8px', background: '#ffffff' }}>
                <h4 style={{ margin: '0 0 12px 0', fontSize: '0.95rem', fontWeight: 600, color: '#1e293b' }}>
                  2. Embedding Provider
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 500, color: '#475569', marginBottom: '4px' }}>
                      Provider
                    </label>
                    <select
                      value={embeddingProvider}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setEmbeddingProvider(e.target.value)}
                      style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                    >
                      {capabilities.embedding.map((p) => (
                        <option key={p} value={p}>{p}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 500, color: '#475569', marginBottom: '4px' }}>
                      Model (Optional)
                    </label>
                    <input
                      type="text"
                      placeholder="e.g. sentence-transformers/all-MiniLM-L6-v2"
                      value={embeddingModel}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setEmbeddingModel(e.target.value)}
                      style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                    />
                  </div>
                </div>
              </div>

              {/* 3. Document Reader */}
              <div style={{ padding: '16px', border: '1px solid #e2e8f0', borderRadius: '8px', background: '#ffffff' }}>
                <h4 style={{ margin: '0 0 12px 0', fontSize: '0.95rem', fontWeight: 600, color: '#1e293b' }}>
                  3. Document Reader Provider
                </h4>
                <div>
                  <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 500, color: '#475569', marginBottom: '4px' }}>
                    Provider
                  </label>
                  <select
                    value={docReaderProvider}
                    disabled={!isOwnerOrAdmin}
                    onChange={(e) => setDocReaderProvider(e.target.value)}
                    style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                  >
                    {capabilities.document_reader.map((p) => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* 4. Reranker */}
              <div style={{ padding: '16px', border: '1px solid #e2e8f0', borderRadius: '8px', background: '#ffffff' }}>
                <h4 style={{ margin: '0 0 12px 0', fontSize: '0.95rem', fontWeight: 600, color: '#1e293b' }}>
                  4. Reranker Provider
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 500, color: '#475569', marginBottom: '4px' }}>
                      Provider
                    </label>
                    <select
                      value={rerankerProvider}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setRerankerProvider(e.target.value)}
                      style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                    >
                      {capabilities.reranker.map((p) => (
                        <option key={p} value={p}>{p}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 500, color: '#475569', marginBottom: '4px' }}>
                      Model (Optional)
                    </label>
                    <input
                      type="text"
                      placeholder="e.g. ms-marco-MiniLM-L-6-v2"
                      value={rerankerModel}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setRerankerModel(e.target.value)}
                      style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                    />
                  </div>
                </div>
              </div>

              {/* 5. OCR */}
              <div style={{ padding: '16px', border: '1px solid #e2e8f0', borderRadius: '8px', background: '#ffffff' }}>
                <h4 style={{ margin: '0 0 12px 0', fontSize: '0.95rem', fontWeight: 600, color: '#1e293b' }}>
                  5. OCR Provider
                </h4>
                <div>
                  <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 500, color: '#475569', marginBottom: '4px' }}>
                    Provider
                  </label>
                  <select
                    value={ocrProvider}
                    disabled={!isOwnerOrAdmin}
                    onChange={(e) => setOcrProvider(e.target.value)}
                    style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                  >
                    {capabilities.ocr.map((p) => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* 6. Web */}
              <div style={{ padding: '16px', border: '1px solid #e2e8f0', borderRadius: '8px', background: '#ffffff' }}>
                <h4 style={{ margin: '0 0 12px 0', fontSize: '0.95rem', fontWeight: 600, color: '#1e293b' }}>
                  6. Web Search Provider
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 500, color: '#475569', marginBottom: '4px' }}>
                      Provider
                    </label>
                    <select
                      value={webProvider}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setWebProvider(e.target.value)}
                      style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                    >
                      {capabilities.web.map((p) => (
                        <option key={p} value={p}>{p}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 500, color: '#475569', marginBottom: '4px' }}>
                      Credential Ref (Secret Name)
                    </label>
                    <input
                      type="text"
                      placeholder="e.g. FIRECRAWL_PROD"
                      value={webCredRef}
                      disabled={!isOwnerOrAdmin}
                      onChange={(e) => setWebCredRef(e.target.value)}
                      style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                    />
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', marginTop: '12px' }}>
                <button
                  type="button"
                  onClick={onClose}
                  style={{
                    padding: '8px 16px',
                    borderRadius: '6px',
                    border: '1px solid #cbd5e1',
                    background: '#ffffff',
                    color: '#475569',
                    cursor: 'pointer',
                    fontWeight: 500,
                  }}
                >
                  Cancel
                </button>
                {isOwnerOrAdmin && (
                  <button
                    type="submit"
                    disabled={saving}
                    style={{
                      padding: '8px 16px',
                      borderRadius: '6px',
                      border: 'none',
                      background: '#0284c7',
                      color: '#ffffff',
                      cursor: saving ? 'not-allowed' : 'pointer',
                      fontWeight: 500,
                    }}
                  >
                    {saving ? 'Saving...' : 'Save Configuration'}
                  </button>
                )}
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
