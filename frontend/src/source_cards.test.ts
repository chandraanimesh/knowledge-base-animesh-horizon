import { test } from 'node:test';
import assert from 'node:assert';

import {
  isValidWebUrl,
  extractHostname,
  deduplicateSources,
  normalizeSources,
} from './components/sourceCardDefinitions.ts';
import type { SourceItem, RetrievedChunk } from './api/chat.ts';

// ─── 1. External Source Cards Using Actual Citation Fields ─────────────────
test('1. External source cards: parses real API web research metadata safely', () => {
  const rawSources: SourceItem[] = [
    {
      type: 'web_research',
      chunk_id: 'web_abc123',
      document_id: 'https://finance.yahoo.com/news/q3-results',
      file_name: 'finance.yahoo.com',
      title: 'Web: Q3 Financial Summary',
      url: 'https://finance.yahoo.com/news/q3-results',
      excerpt: 'Total Q3 revenue grew 14% year over year to $4.2B.',
    },
  ];

  const normalized = normalizeSources(rawSources);
  assert.strictEqual(normalized.length, 1);
  const card = normalized[0];
  assert.strictEqual(card.type, 'web_research');
  assert.strictEqual(card.url, 'https://finance.yahoo.com/news/q3-results');
  assert.strictEqual(extractHostname(card.url), 'finance.yahoo.com');
  assert.strictEqual(card.excerpt, 'Total Q3 revenue grew 14% year over year to $4.2B.');
  assert.ok(isValidWebUrl(card.url));
});

// ─── 2. Internal Document Cards ───────────────────────────────────────────
test('2. Internal document cards: correctly displays filename, page, and excerpt without external links', () => {
  const rawSources: SourceItem[] = [
    {
      type: 'document',
      chunk_id: 'chunk_doc_99',
      document_id: 'doc-uuid-555',
      department_id: 'dept-fin',
      file_name: '01_finance_demo_document.docx',
      page_number: 3,
      title: '01_finance_demo_document.docx (Page 3)',
      excerpt: 'The department maintained an emergency operating reserve of $1.5M.',
      url: null,
    },
  ];

  const normalized = normalizeSources(rawSources);
  assert.strictEqual(normalized.length, 1);
  const card = normalized[0];
  assert.strictEqual(card.type, 'document');
  assert.strictEqual(card.file_name, '01_finance_demo_document.docx');
  assert.strictEqual(card.page_number, 3);
  assert.strictEqual(card.url, undefined);
  assert.strictEqual(card.excerpt, 'The department maintained an emergency operating reserve of $1.5M.');
  assert.strictEqual(isValidWebUrl(card.url), false);
});

// ─── 3. Missing or Malformed URLs ──────────────────────────────────────────
test('3. Malformed and dangerous URL validation rejects non-HTTP protocols', () => {
  assert.strictEqual(isValidWebUrl('javascript:alert(document.cookie)'), false);
  assert.strictEqual(isValidWebUrl('data:text/html,<script>alert(1)</script>'), false);
  assert.strictEqual(isValidWebUrl('file:///etc/passwd'), false);
  assert.strictEqual(isValidWebUrl('ftp://files.example.com'), false);
  assert.strictEqual(isValidWebUrl('not a url'), false);
  assert.strictEqual(isValidWebUrl(''), false);
  assert.strictEqual(isValidWebUrl(null), false);
  assert.strictEqual(isValidWebUrl(undefined), false);

  assert.strictEqual(isValidWebUrl('https://example.com/report.pdf'), true);
  assert.strictEqual(isValidWebUrl('http://insecure-internal.corp/data'), true);

  // Hostname extraction safety
  assert.strictEqual(extractHostname('javascript:alert(1)'), null);
  assert.strictEqual(extractHostname('https://www.bloomberg.com/news/articles'), 'bloomberg.com');
  assert.strictEqual(extractHostname('https://sub.domain.org:8080/path?q=1'), 'sub.domain.org');
});

// ─── 4. Missing Optional Metadata & Graceful Fallbacks ─────────────────────
test('4. Missing optional metadata: graceful fallbacks for title, page, and excerpt', () => {
  const minimalSources: SourceItem[] = [
    {
      // No title, no page_number, no excerpt, no url
      file_name: 'Sales_Discount_Policy.pdf',
    },
    {
      // Only url, no title, no file_name
      url: 'https://sec.gov/edgar/search',
    },
  ];

  const normalized = normalizeSources(minimalSources);
  assert.strictEqual(normalized.length, 2);

  // First source: internal doc fallback
  assert.strictEqual(normalized[0].title, 'Sales_Discount_Policy.pdf');
  assert.strictEqual(normalized[0].page_number, null);
  assert.strictEqual(normalized[0].excerpt, undefined);
  assert.strictEqual(normalized[0].url, undefined);

  // Second source: web fallback derived from hostname
  assert.strictEqual(normalized[1].type, 'web_research');
  assert.strictEqual(normalized[1].title, 'sec.gov');
  assert.strictEqual(normalized[1].url, 'https://sec.gov/edgar/search');
});

// ─── 5. Empty Citation and Source Lists ─────────────────────────────────────
test('5. Empty states: returns empty array when no sources or citations are provided', () => {
  assert.deepStrictEqual(normalizeSources([], [], []), []);
  assert.deepStrictEqual(normalizeSources(null, null, null), []);
  assert.deepStrictEqual(normalizeSources(undefined, undefined, undefined), []);
  assert.deepStrictEqual(normalizeSources([], null, undefined), []);
});

// ─── 6. Deduplication and Citation Identifier Mapping ───────────────────────
test('6. Deduplication: repeated citations or chunks do not create duplicate cards', () => {
  const duplicateSources: SourceItem[] = [
    {
      chunk_id: 'chk-1',
      file_name: 'DocA.pdf',
      page_number: 1,
      excerpt: 'First passage',
    },
    {
      chunk_id: 'chk-1', // Exact same chunk referenced twice in LLM answer
      file_name: 'DocA.pdf',
      page_number: 1,
      excerpt: 'First passage',
    },
    {
      url: 'https://reuters.com/market-news',
      title: 'Reuters News',
    },
    {
      url: 'https://reuters.com/market-news', // Duplicate web URL
      title: 'Reuters News',
    },
    {
      chunk_id: 'chk-2',
      file_name: 'DocB.pdf',
      page_number: 2,
    },
  ];

  const deduplicated = deduplicateSources(duplicateSources);
  assert.strictEqual(deduplicated.length, 3);
  assert.strictEqual(deduplicated[0].chunk_id, 'chk-1');
  assert.strictEqual(deduplicated[1].url, 'https://reuters.com/market-news');
  assert.strictEqual(deduplicated[2].chunk_id, 'chk-2');
});

// ─── 7. Safe Handling of External URLs and Retrieved Excerpts ──────────────
test('7. Security: malicious scripts or javascript links in sources are neutralized', () => {
  const unsafeSources: SourceItem[] = [
    {
      type: 'web_research',
      title: '<script>alert("xss")</script> Malicious Page',
      url: 'javascript:alert(1)', // Malicious protocol
      excerpt: '<img src=x onerror=alert(1)> Some plain evidence',
    },
  ];

  const normalized = normalizeSources(unsafeSources);
  assert.strictEqual(normalized.length, 1);
  // Malicious javascript: URL is stripped to undefined
  assert.strictEqual(normalized[0].url, undefined);
  // Plain text excerpt is preserved safely without script execution
  assert.ok(normalized[0].excerpt?.includes('Some plain evidence'));
});

// ─── 8. Backward Compatibility: Mapping Raw Retrieved Chunks ───────────────
test('8. Backward compatibility: converts raw retrieved_chunks into structured cards', () => {
  const rawChunks: RetrievedChunk[] = [
    {
      chunk_id: 'chunk-normal-1',
      document_id: 'doc-uuid-1',
      department_id: 'dept-sales',
      file_name: 'Sales_Strategy_2026.pdf',
      chunk_index: 0,
      text: 'Enterprise discount thresholds require director sign-off.',
      page_number: 5,
    },
    {
      chunk_id: 'web_transient_2',
      document_id: 'https://partner.com/specs',
      department_id: 'transient_web',
      file_name: 'partner.com',
      chunk_index: 0,
      text: 'Partner API SLA guarantees 99.9% availability.',
    },
  ];

  const normalized = normalizeSources(null, null, rawChunks);
  assert.strictEqual(normalized.length, 2);

  // Internal document chunk
  assert.strictEqual(normalized[0].type, 'document');
  assert.strictEqual(normalized[0].file_name, 'Sales_Strategy_2026.pdf');
  assert.strictEqual(normalized[0].page_number, 5);
  assert.strictEqual(normalized[0].excerpt, 'Enterprise discount thresholds require director sign-off.');
  assert.strictEqual(normalized[0].url, undefined);

  // External web chunk
  assert.strictEqual(normalized[1].type, 'web_research');
  assert.strictEqual(normalized[1].url, 'https://partner.com/specs');
  assert.strictEqual(extractHostname(normalized[1].url), 'partner.com');
  assert.strictEqual(normalized[1].excerpt, 'Partner API SLA guarantees 99.9% availability.');
});

// ─── 9. Backward Compatibility: Mapping Legacy Citation Strings ─────────────
test('9. Backward compatibility: converts citation string arrays into cards', () => {
  const rawCitations = [
    'Annual_Financial_Report.pdf, page 12',
    'https://techcrunch.com/2026/01/ai-trends',
    'General_Company_Handbook.docx',
  ];

  const normalized = normalizeSources(null, rawCitations, null);
  assert.strictEqual(normalized.length, 3);

  // Document with page parsed from citation string
  assert.strictEqual(normalized[0].type, 'document');
  assert.strictEqual(normalized[0].file_name, 'Annual_Financial_Report.pdf');
  assert.strictEqual(normalized[0].page_number, 12);

  // Web citation string
  assert.strictEqual(normalized[1].type, 'web_research');
  assert.strictEqual(normalized[1].url, 'https://techcrunch.com/2026/01/ai-trends');
  assert.strictEqual(normalized[1].title, 'techcrunch.com');

  // Simple document filename
  assert.strictEqual(normalized[2].type, 'document');
  assert.strictEqual(normalized[2].file_name, 'General_Company_Handbook.docx');
  assert.strictEqual(normalized[2].page_number, null);
});

// ─── 10. Shared Knowledge and Personal Memory Types ────────────────────────
test('10. Specialized source types: preserves shared knowledge and personal memory sources', () => {
  const specializedSources: SourceItem[] = [
    {
      type: 'shared_knowledge',
      chunk_id: 'shared_kn_44',
      file_name: 'Company_Global_Glossary.pdf',
      page_number: 1,
      excerpt: 'Definition of ACV: Annual Contract Value.',
    },
    {
      type: 'personal_memory',
      title: '🧠 Personal Preference: Prefers concise bullet points',
      excerpt: 'User explicitly requested summary points over long narratives.',
    },
    {
      type: 'thread',
      title: 'Thread: "Q2 Pricing Discussion"',
      excerpt: 'Agreed on 10% discount cap during Friday team sync.',
    },
  ];

  const normalized = normalizeSources(specializedSources);
  assert.strictEqual(normalized.length, 3);
  assert.strictEqual(normalized[0].type, 'shared_knowledge');
  assert.strictEqual(normalized[1].type, 'personal_memory');
  assert.strictEqual(normalized[2].type, 'thread');
});

// ─── 11. Chunk Indices Tracking across Deduplicated Web Sources ───────────
test('11. Preserves and aggregates chunk indices when multiple chunks come from the same URL', () => {
  const multiChunkSources: SourceItem[] = [
    { chunk_id: 'web_chunk_1', url: 'https://glassdoor.co.in/salaries', title: 'Glassdoor' },
    { chunk_id: 'web_chunk_2', url: 'https://glassdoor.co.in/salaries', title: 'Glassdoor' },
    { chunk_id: 'web_chunk_3', url: 'https://payscale.com/salaries', title: 'Payscale' },
  ];

  const deduplicated = deduplicateSources(multiChunkSources);
  assert.strictEqual(deduplicated.length, 2);
  assert.strictEqual(deduplicated[0].url, 'https://glassdoor.co.in/salaries');
  assert.deepStrictEqual(deduplicated[0].indices, [1, 2]);
  assert.strictEqual(deduplicated[1].url, 'https://payscale.com/salaries');
  assert.deepStrictEqual(deduplicated[1].indices, [3]);
});
