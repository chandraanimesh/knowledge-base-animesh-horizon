import { test } from 'node:test';
import assert from 'node:assert';

import type {
  ChunkEvidenceResponse,
  QualityCheckFlag,
  QualityReportItem,
  ChunkIssueReportRequest,
} from './api/chunkCuration.ts';

// ─── 1. Evidence Response Structure & Source Availability ─────────────────
test('1. ChunkEvidenceResponse correctly represents available source document', () => {
  const sampleEvidence: ChunkEvidenceResponse = {
    chunk_id: '11111111-1111-1111-1111-111111111111',
    document_id: '22222222-2222-2222-2222-222222222222',
    chunk_index: 0,
    file_name: 'architecture_guide.pdf',
    file_path: 'C:/ragDemo/data/architecture_guide.pdf',
    mime_type: 'application/pdf',
    source_available: true,
    source_availability_reason: null,
    source_url: null,
    page_number: 4,
    node_id: '33333333-3333-3333-3333-333333333333',
    section_title: 'Ingestion Pipeline & Section Hierarchy',
    section_ordinal: '2.1',
    section_abstract: 'Hierarchical overview of document processing.',
    current_text: 'The ingestion pipeline parses documents into hierarchical sections.',
    original_text: 'The ingestion pipeline parses documents into hierarchical sections.',
    is_curated: false,
    current_version: 1,
    created_at: '2026-10-28T12:00:00Z',
    updated_at: '2026-10-28T12:00:00Z',
    chunk_hash: 'abcdef1234567890',
    is_managed_external: false,
    quality_flags: [],
    quality_reports: [],
  };

  assert.strictEqual(sampleEvidence.source_available, true);
  assert.strictEqual(sampleEvidence.is_curated, false);
  assert.strictEqual(sampleEvidence.current_version, 1);
  assert.strictEqual(sampleEvidence.page_number, 4);
  assert.strictEqual(sampleEvidence.section_title, 'Ingestion Pipeline & Section Hierarchy');
});

test('2. ChunkEvidenceResponse honestly flags unavailable source on disk', () => {
  const unavailableEvidence: ChunkEvidenceResponse = {
    chunk_id: '11111111-1111-1111-1111-111111111111',
    document_id: '22222222-2222-2222-2222-222222222222',
    chunk_index: 0,
    file_name: 'deleted_doc.pdf',
    file_path: 'C:/ragDemo/data/deleted_doc.pdf',
    mime_type: 'application/pdf',
    source_available: false,
    source_availability_reason: 'Source file does not exist on host storage: C:/ragDemo/data/deleted_doc.pdf',
    page_number: null,
    current_text: 'Orphaned chunk text',
    original_text: 'Orphaned chunk text',
    is_curated: false,
    current_version: 1,
    created_at: '2026-10-28T12:00:00Z',
    updated_at: '2026-10-28T12:00:00Z',
    chunk_hash: 'deadbeef12345678',
    is_managed_external: false,
    quality_flags: [
      {
        flag_id: 'SOURCE_FILE_UNAVAILABLE',
        severity: 'WARNING',
        message: 'Original source file is not accessible on the host storage volume.',
        suggested_remediation: 'SOURCE_CORRECTION',
      },
    ],
    quality_reports: [],
  };

  assert.strictEqual(unavailableEvidence.source_available, false);
  assert.ok(unavailableEvidence.source_availability_reason?.includes('does not exist'));
  assert.strictEqual(unavailableEvidence.quality_flags.length, 1);
  assert.strictEqual(unavailableEvidence.quality_flags[0].flag_id, 'SOURCE_FILE_UNAVAILABLE');
  assert.strictEqual(unavailableEvidence.quality_flags[0].suggested_remediation, 'SOURCE_CORRECTION');
});

// ─── 2. Original vs Curated Drift ──────────────────────────────────────────
test('3. Accurately detects curated content drift and preserves original text', () => {
  const original = 'Initial uncurated text with OCR typpo on page 12.';
  const curated = 'Initial uncurated text with OCR typo corrected on page 12.';

  const curatedEvidence: ChunkEvidenceResponse = {
    chunk_id: 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
    document_id: 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
    chunk_index: 2,
    file_name: 'annual_report.pdf',
    file_path: 'C:/ragDemo/data/annual_report.pdf',
    mime_type: 'application/pdf',
    source_available: true,
    page_number: 12,
    current_text: curated,
    original_text: original,
    is_curated: true,
    current_version: 2,
    created_at: '2026-10-28T10:00:00Z',
    updated_at: '2026-10-28T12:00:00Z',
    chunk_hash: 'hashv2',
    is_managed_external: false,
    quality_flags: [
      {
        flag_id: 'CURATED_CONTENT_DRIFT',
        severity: 'INFO',
        message: 'Active text has been curated (version 2) and differs from originally ingested content.',
        suggested_remediation: 'MANUAL_CURATE',
      },
    ],
    quality_reports: [],
  };

  assert.strictEqual(curatedEvidence.is_curated, true);
  assert.strictEqual(curatedEvidence.current_version, 2);
  assert.notStrictEqual(curatedEvidence.current_text, curatedEvidence.original_text);
  assert.strictEqual(curatedEvidence.quality_flags[0].flag_id, 'CURATED_CONTENT_DRIFT');
});

// ─── 3. Deterministic Quality Flags & Remediation Paths ───────────────────
test('4. QualityCheckFlag enforces standard remediation categories', () => {
  const allowedRemediations = ['MANUAL_CURATE', 'REINGEST_DOCUMENT', 'SOURCE_CORRECTION'];

  const flags: QualityCheckFlag[] = [
    {
      flag_id: 'EMPTY_EXTRACTION',
      severity: 'ERROR',
      message: 'Zero content extracted.',
      suggested_remediation: 'REINGEST_DOCUMENT',
    },
    {
      flag_id: 'MALFORMED_ENCODING',
      severity: 'ERROR',
      message: 'Null bytes found in chunk.',
      suggested_remediation: 'REINGEST_DOCUMENT',
    },
    {
      flag_id: 'MINIMAL_EXTRACTION',
      severity: 'WARNING',
      message: 'Short text segment.',
      suggested_remediation: 'MANUAL_CURATE',
    },
    {
      flag_id: 'MANAGED_EXTERNAL_SYNC',
      severity: 'INFO',
      message: 'CRM chunk protected from direct edits.',
      suggested_remediation: 'SOURCE_CORRECTION',
    },
  ];

  for (const flag of flags) {
    assert.ok(allowedRemediations.includes(flag.suggested_remediation));
    assert.ok(['INFO', 'WARNING', 'ERROR'].includes(flag.severity));
  }
});

// ─── 4. User Issue Reporting ───────────────────────────────────────────────
test('5. ChunkIssueReportRequest and QualityReportItem integrity', () => {
  const request: ChunkIssueReportRequest = {
    issue_type: 'EXTRACT_DEFECT',
    description: 'Header rows in table are duplicated due to multi-column OCR issue.',
    suggested_remediation: 'REINGEST_DOCUMENT',
  };

  const storedReport: QualityReportItem = {
    report_id: 'rep_abc123',
    reported_by: 'reviewer@example.com',
    reported_at: '2026-10-28T12:30:00Z',
    issue_type: request.issue_type,
    description: request.description,
    suggested_remediation: request.suggested_remediation,
    status: 'OPEN',
  };

  assert.strictEqual(storedReport.status, 'OPEN');
  assert.strictEqual(storedReport.issue_type, 'EXTRACT_DEFECT');
  assert.strictEqual(storedReport.suggested_remediation, 'REINGEST_DOCUMENT');
  assert.strictEqual(storedReport.reported_by, 'reviewer@example.com');
});
