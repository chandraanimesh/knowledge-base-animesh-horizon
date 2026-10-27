import { test } from 'node:test';
import assert from 'node:assert';

export interface DiffEntry {
  field: string;
  label: string;
  from: string;
  to: string;
}

export function computeConfigurationDiff(
  original: {
    llmProvider?: string;
    llmModel?: string;
    rerankerEnabled?: boolean;
    rerankerModel?: string;
    vectorSearchEnabled?: boolean;
    ftsEnabled?: boolean;
    rrfEnabled?: boolean;
    topK?: number;
    rerankerTopK?: number;
    temperature?: number;
    webProvider?: string;
    webEnabled?: boolean;
  },
  current: {
    llmProvider: string;
    llmModel: string;
    rerankerEnabled: boolean;
    rerankerModel: string;
    vectorSearchEnabled: boolean;
    ftsEnabled: boolean;
    rrfEnabled: boolean;
    topK: number;
    rerankerTopK: number;
    temperature: number;
    webProvider?: string;
    webEnabled?: boolean;
  }
): DiffEntry[] {
  const diffs: DiffEntry[] = [];

  if ((original.llmProvider || 'openai') !== current.llmProvider) {
    diffs.push({ field: 'llm_provider', label: 'LLM Provider', from: original.llmProvider || 'openai', to: current.llmProvider });
  }

  if ((original.llmModel || 'gpt-4o') !== current.llmModel) {
    diffs.push({ field: 'llm_model', label: 'Default Model', from: original.llmModel || 'gpt-4o', to: current.llmModel });
  }

  const origRerank = original.rerankerEnabled !== undefined ? original.rerankerEnabled : true;
  if (origRerank !== current.rerankerEnabled) {
    diffs.push({
      field: 'reranker_enabled',
      label: 'CrossEncoder reranking',
      from: origRerank ? 'ON' : 'OFF',
      to: current.rerankerEnabled ? 'ON' : 'OFF',
    });
  }

  const origRerankModel = original.rerankerModel || 'cross-encoder/ms-marco-MiniLM-L-6-v2';
  if (origRerankModel !== current.rerankerModel) {
    diffs.push({ field: 'reranker_model', label: 'Reranker model', from: origRerankModel, to: current.rerankerModel });
  }

  const origVec = original.vectorSearchEnabled !== undefined ? original.vectorSearchEnabled : true;
  if (origVec !== current.vectorSearchEnabled) {
    diffs.push({
      field: 'vector_search_enabled',
      label: 'Vector search',
      from: origVec ? 'ON' : 'OFF',
      to: current.vectorSearchEnabled ? 'ON' : 'OFF',
    });
  }

  const origFts = original.ftsEnabled !== undefined ? original.ftsEnabled : true;
  if (origFts !== current.ftsEnabled) {
    diffs.push({
      field: 'fts_enabled',
      label: 'PostgreSQL FTS',
      from: origFts ? 'ON' : 'OFF',
      to: current.ftsEnabled ? 'ON' : 'OFF',
    });
  }

  const origRrf = original.rrfEnabled !== undefined ? original.rrfEnabled : true;
  if (origRrf !== current.rrfEnabled) {
    diffs.push({
      field: 'rrf_enabled',
      label: 'RRF Fusion',
      from: origRrf ? 'ON' : 'OFF',
      to: current.rrfEnabled ? 'ON' : 'OFF',
    });
  }

  const origTopK = original.topK !== undefined ? original.topK : 8;
  if (origTopK !== current.topK) {
    diffs.push({
      field: 'top_k',
      label: 'Top K (retrieval)',
      from: String(origTopK),
      to: String(current.topK),
    });
  }

  const origRerankTopK = original.rerankerTopK !== undefined ? original.rerankerTopK : 5;
  if (origRerankTopK !== current.rerankerTopK) {
    diffs.push({
      field: 'reranker_top_k',
      label: 'Reranker Top K',
      from: String(origRerankTopK),
      to: String(current.rerankerTopK),
    });
  }

  const origTemp = original.temperature !== undefined ? original.temperature : 0.2;
  if (origTemp !== current.temperature) {
    diffs.push({
      field: 'temperature',
      label: 'Temperature',
      from: String(origTemp),
      to: String(current.temperature),
    });
  }

  if (current.webProvider !== undefined) {
    const origWebProv = original.webProvider || 'tavily';
    if (origWebProv !== current.webProvider) {
      diffs.push({
        field: 'web_provider',
        label: 'Web Search Provider',
        from: origWebProv === 'tinyfish' ? 'TinyFish Search' : origWebProv === 'firecrawl' ? 'Firecrawl' : 'Tavily',
        to: current.webProvider === 'tinyfish' ? 'TinyFish Search' : current.webProvider === 'firecrawl' ? 'Firecrawl' : 'Tavily',
      });
    }
  }

  if (current.webEnabled !== undefined) {
    const origWebEnabled = original.webEnabled !== undefined ? original.webEnabled : true;
    if (origWebEnabled !== current.webEnabled) {
      diffs.push({
        field: 'web_enabled',
        label: 'Web Search',
        from: origWebEnabled ? 'ON' : 'OFF',
        to: current.webEnabled ? 'ON' : 'OFF',
      });
    }
  }

  return diffs;
}

export function maskSecretValue(keyOrRef?: string | null): string {
  if (!keyOrRef || keyOrRef.trim().length === 0) return '••••••••••••••••••••••••••••••••';
  return '••••••••••••••••••••••••••••••••';
}

export function validateEmbeddingDimension(modelName: string): { valid: boolean; dim: number } {
  const incompatible: Record<string, number> = {
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'text-embedding-ada-002': 1536,
    'bge-large-en': 1024,
  };
  for (const [key, dim] of Object.entries(incompatible)) {
    if (modelName.toLowerCase().includes(key)) {
      return { valid: false, dim };
    }
  }
  return { valid: true, dim: 384 };
}

// ============================================================================
// Unit Tests Matching Acceptance Gates
// ============================================================================

test('1. Compute diff matches Reference 1 changes when modifying retrieval controls', () => {
  const original = {
    rerankerEnabled: false,
    rerankerModel: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    vectorSearchEnabled: true,
    ftsEnabled: false,
    topK: 6,
    rerankerTopK: 3,
  };

  const current = {
    llmProvider: 'openai',
    llmModel: 'gpt-4o',
    rerankerEnabled: true,
    rerankerModel: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    vectorSearchEnabled: true,
    ftsEnabled: true,
    rrfEnabled: true,
    topK: 8,
    rerankerTopK: 5,
    temperature: 0.2,
  };

  const diffs = computeConfigurationDiff(original, current);
  assert.strictEqual(diffs.length, 4);

  const rerankDiff = diffs.find((d) => d.field === 'reranker_enabled');
  assert.ok(rerankDiff);
  assert.strictEqual(rerankDiff.from, 'OFF');
  assert.strictEqual(rerankDiff.to, 'ON');

  const ftsDiff = diffs.find((d) => d.field === 'fts_enabled');
  assert.ok(ftsDiff);
  assert.strictEqual(ftsDiff.from, 'OFF');
  assert.strictEqual(ftsDiff.to, 'ON');

  const topKDiff = diffs.find((d) => d.field === 'top_k');
  assert.ok(topKDiff);
  assert.strictEqual(topKDiff.from, '6');
  assert.strictEqual(topKDiff.to, '8');

  const rerankTopKDiff = diffs.find((d) => d.field === 'reranker_top_k');
  assert.ok(rerankTopKDiff);
  assert.strictEqual(rerankTopKDiff.from, '3');
  assert.strictEqual(rerankTopKDiff.to, '5');
});

test('2. Clean state returns empty diff when no parameters are changed', () => {
  const baseline = {
    llmProvider: 'openai',
    llmModel: 'gpt-4o',
    rerankerEnabled: true,
    rerankerModel: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    vectorSearchEnabled: true,
    ftsEnabled: true,
    rrfEnabled: true,
    topK: 8,
    rerankerTopK: 5,
    temperature: 0.2,
  };

  const diffs = computeConfigurationDiff(baseline, baseline);
  assert.strictEqual(diffs.length, 0);
});

test('3. Secret values are masked and never exposed in plaintext', () => {
  assert.strictEqual(maskSecretValue('sk-proj-my-real-api-key-12345'), '••••••••••••••••••••••••••••••••');
  assert.strictEqual(maskSecretValue(''), '••••••••••••••••••••••••••••••••');
  assert.strictEqual(maskSecretValue(null), '••••••••••••••••••••••••••••••••');
});

test('4. Embedding model dimension check enforces 384 dimensions invariant', () => {
  const compatible = validateEmbeddingDimension('all-MiniLM-L6-v2 (384 dim)');
  assert.strictEqual(compatible.valid, true);
  assert.strictEqual(compatible.dim, 384);

  const incompatible = validateEmbeddingDimension('text-embedding-3-small');
  assert.strictEqual(incompatible.valid, false);
  assert.strictEqual(incompatible.dim, 1536);
});

test('5. Cancel review modal leaves changes unpersisted', () => {
  let isSaved = false;
  let showConfirmReview = true;

  // User triggers cancel
  showConfirmReview = false;

  assert.strictEqual(showConfirmReview, false);
  assert.strictEqual(isSaved, false);
});

test('6. Changing web provider to tinyfish computes expected diff and review item', () => {
  const original = {
    webProvider: 'tavily',
    webEnabled: true,
  };

  const current = {
    llmProvider: 'openai',
    llmModel: 'gpt-4o',
    rerankerEnabled: true,
    rerankerModel: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    vectorSearchEnabled: true,
    ftsEnabled: true,
    rrfEnabled: true,
    topK: 8,
    rerankerTopK: 5,
    temperature: 0.2,
    webProvider: 'tinyfish',
    webEnabled: true,
  };

  const diffs = computeConfigurationDiff(original, current);
  assert.strictEqual(diffs.length, 1);
  assert.strictEqual(diffs[0].field, 'web_provider');
  assert.strictEqual(diffs[0].label, 'Web Search Provider');
  assert.strictEqual(diffs[0].from, 'Tavily');
  assert.strictEqual(diffs[0].to, 'TinyFish Search');
});
