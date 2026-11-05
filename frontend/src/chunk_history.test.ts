import { test } from 'node:test';
import assert from 'node:assert';

import {
  parseUnifiedDiff,
  checkCurationPermissions,
  formatChunkHash,
  formatAuditTimestamp,
  isOptimisticConflictError,
  canRollbackToVersion,
} from './components/chunkDiffDefinitions.ts';

// ─── 1. Unified Diff Parsing ───────────────────────────────────────────────
test('1. parseUnifiedDiff: parses hunks, line numbers, additions, and deletions', () => {
  const sampleDiff = `--- chunk:v1
+++ chunk:v2
@@ -1,4 +1,5 @@
 Constant header line
-Old line that was removed
+New updated line that replaced it
+Additional added line
 Trailing unchanged line`;

  const parsed = parseUnifiedDiff(sampleDiff);
  assert.strictEqual(parsed.length, 8);

  // Line 0: Meta
  assert.strictEqual(parsed[0].type, 'meta');
  assert.strictEqual(parsed[0].content, '--- chunk:v1');

  // Line 1: Meta
  assert.strictEqual(parsed[1].type, 'meta');
  assert.strictEqual(parsed[1].content, '+++ chunk:v2');

  // Line 2: Hunk header
  assert.strictEqual(parsed[2].type, 'header');
  assert.strictEqual(parsed[2].content, '@@ -1,4 +1,5 @@');

  // Line 3: Unchanged context
  assert.strictEqual(parsed[3].type, 'context');
  assert.strictEqual(parsed[3].content, 'Constant header line');
  assert.strictEqual(parsed[3].oldLineNumber, 1);
  assert.strictEqual(parsed[3].newLineNumber, 1);

  // Line 4: Deletion
  assert.strictEqual(parsed[4].type, 'deletion');
  assert.strictEqual(parsed[4].content, 'Old line that was removed');
  assert.strictEqual(parsed[4].oldLineNumber, 2);
  assert.strictEqual(parsed[4].newLineNumber, undefined);

  // Line 5: Addition 1
  assert.strictEqual(parsed[5].type, 'addition');
  assert.strictEqual(parsed[5].content, 'New updated line that replaced it');
  assert.strictEqual(parsed[5].oldLineNumber, undefined);
  assert.strictEqual(parsed[5].newLineNumber, 2);

  // Line 6: Addition 2
  assert.strictEqual(parsed[6].type, 'addition');
  assert.strictEqual(parsed[6].content, 'Additional added line');
  assert.strictEqual(parsed[6].oldLineNumber, undefined);
  assert.strictEqual(parsed[6].newLineNumber, 3);

  // Line 7: Trailing context
  assert.strictEqual(parsed[7].type, 'context');
  assert.strictEqual(parsed[7].content, 'Trailing unchanged line');
  assert.strictEqual(parsed[7].oldLineNumber, 3);
  assert.strictEqual(parsed[7].newLineNumber, 4);
});

test('2. parseUnifiedDiff: handles empty string or whitespace safely', () => {
  assert.deepStrictEqual(parseUnifiedDiff(''), []);
  assert.deepStrictEqual(parseUnifiedDiff(null as any), []);
  assert.deepStrictEqual(parseUnifiedDiff(undefined as any), []);
});

// ─── 2. Permission Evaluation & Managed CRM Invariants ──────────────────────
test('3. checkCurationPermissions: grants full curate and rollback to ADMIN, OWNER, and EDITOR', () => {
  const adminCheck = checkCurationPermissions('ADMIN', false);
  assert.strictEqual(adminCheck.canEdit, true);
  assert.strictEqual(adminCheck.canRollback, true);
  assert.strictEqual(adminCheck.isManaged, false);

  const ownerCheck = checkCurationPermissions('OWNER', false);
  assert.strictEqual(ownerCheck.canEdit, true);
  assert.strictEqual(ownerCheck.canRollback, true);

  const editorCheck = checkCurationPermissions('EDITOR', false);
  assert.strictEqual(editorCheck.canEdit, true);
  assert.strictEqual(editorCheck.canRollback, true);
});

test('4. checkCurationPermissions: blocks VIEWER role from editing and rolling back', () => {
  const viewerCheck = checkCurationPermissions('VIEWER', false);
  assert.strictEqual(viewerCheck.canEdit, false);
  assert.strictEqual(viewerCheck.canRollback, false);
  assert.ok(viewerCheck.reason?.toLowerCase().includes('read-only'));
});

test('5. checkCurationPermissions: blocks CRM managed chunks even for ADMIN users', () => {
  const managedAdminCheck = checkCurationPermissions('ADMIN', true);
  assert.strictEqual(managedAdminCheck.canEdit, false);
  assert.strictEqual(managedAdminCheck.canRollback, false);
  assert.strictEqual(managedAdminCheck.isManaged, true);
  assert.ok(managedAdminCheck.reason?.includes('external CRM') || managedAdminCheck.reason?.includes('Twenty CRM'));
});

// ─── 3. Optimistic Concurrency & 409 Conflict Handling ──────────────────────
test('6. isOptimisticConflictError: correctly identifies 409 conflict errors', () => {
  assert.strictEqual(isOptimisticConflictError({ status: 409, message: 'Conflict' }), true);
  assert.strictEqual(isOptimisticConflictError(new Error('409 Conflict: base_version mismatch')), true);
  assert.strictEqual(isOptimisticConflictError('HTTP 409 base_version out of date'), true);
  assert.strictEqual(isOptimisticConflictError(new Error('500 Internal Server Error')), false);
  assert.strictEqual(isOptimisticConflictError(null), false);
});

// ─── 4. Formatting Utilities ───────────────────────────────────────────────
test('7. formatChunkHash: safely formats and truncates SHA-256 hashes', () => {
  const fullHash = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
  const truncated = formatChunkHash(fullHash, 8);
  assert.strictEqual(truncated, 'e3b0c442…');

  assert.strictEqual(formatChunkHash('abc', 8), 'abc');
  assert.strictEqual(formatChunkHash(null), '—');
  assert.strictEqual(formatChunkHash(undefined), '—');
});

test('8. formatAuditTimestamp: formats ISO dates and handles invalid inputs', () => {
  const formatted = formatAuditTimestamp('2026-10-28T10:30:00Z');
  assert.ok(formatted.length > 5);
  assert.ok(formatted !== '—');

  assert.strictEqual(formatAuditTimestamp('invalid-date-string'), 'invalid-date-string');
  assert.strictEqual(formatAuditTimestamp(null), '—');
});

// ─── 5. Rollback Target Validation ─────────────────────────────────────────
test('9. canRollbackToVersion: prevents rollback to active version or without permission', () => {
  // Cannot rollback to version that is already the current version
  assert.strictEqual(canRollbackToVersion(3, 3, true), false);

  // Can rollback to an older version (e.g. v1 when current is v3)
  assert.strictEqual(canRollbackToVersion(1, 3, true), true);

  // Cannot rollback if user lacks permission
  assert.strictEqual(canRollbackToVersion(1, 3, false), false);

  // Invalid versions
  assert.strictEqual(canRollbackToVersion(0, 3, true), false);
  assert.strictEqual(canRollbackToVersion(-1, 3, true), false);
});
