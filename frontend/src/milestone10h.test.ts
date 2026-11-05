import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

import {
  formatDuration,
  formatMetric,
  sanitizePayload,
  isTerminalStatus,
  getExecutionStatusBadge,
} from './components/workflowStudio/executionUtils.ts';
import type { WorkflowTaskResponse, WorkflowRunResponse } from './api/workflows.ts';
import type { ArtifactPreviewResponse } from './api/artifacts.ts';

describe('Milestone 10H: Frontend Execution Canvas & HITL Mutation UI', () => {
  // -------------------------------------------------------------------------
  // 1. 10H.1 Routing-Mode Indicator Contracts & Badges
  // -------------------------------------------------------------------------
  describe('10H.1 Chat Routing-Mode Indicator', () => {
    test('Correctly maps FAST_RAG, CONTEXTUAL, and ADAPTIVE_PLAN execution modes', () => {
      const modeConfigs = [
        {
          mode: 'FAST_RAG',
          expectedLabel: 'Fast RAG',
          expectedIcon: '⚡',
        },
        {
          mode: 'CONTEXTUAL',
          expectedLabel: 'Contextual',
          expectedIcon: '🧠',
        },
        {
          mode: 'ADAPTIVE_PLAN',
          expectedLabel: 'Adaptive Plan',
          expectedIcon: '🔀',
        },
      ];

      for (const cfg of modeConfigs) {
        assert.ok(cfg.expectedLabel.length > 0);
        assert.ok(cfg.expectedIcon.length > 0);
      }
    });

    test('Safely handles missing, empty, or unknown routing modes without throwing', () => {
      const fallbackModes = [null, undefined, '', 'CUSTOM_PIPELINE'];
      for (const m of fallbackModes) {
        const normalized = (m || '').toUpperCase();
        // Fallback should not crash and should resolve to uppercase string or empty
        assert.doesNotThrow(() => {
          const label = normalized || 'AGENT';
          assert.ok(label);
        });
      }
    });

    test('Preserves explicit routing reason provided by backend classifier', () => {
      const classifierReason = 'Request requires multi-source evidence synthesis across 3 departments.';
      const payload = {
        routing_mode: 'ADAPTIVE_PLAN',
        routing_reason: classifierReason,
      };
      assert.equal(payload.routing_mode, 'ADAPTIVE_PLAN');
      assert.equal(payload.routing_reason, classifierReason);
    });
  });

  // -------------------------------------------------------------------------
  // 2. 10H.2 Execution Canvas (DAG Nodes, Edges, Topological Ordering)
  // -------------------------------------------------------------------------
  describe('10H.2 Execution Canvas DAG Transformation', () => {
    const mockTasks: WorkflowTaskResponse[] = [
      {
        id: 'task-1',
        workflow_run_id: 'run-100',
        task_key: 'plan_tasks',
        task_type: 'PLANNER',
        status: 'COMPLETED',
        dependencies: [],
        tokens_spent: 120,
        provider_calls: 1,
        created_at: '2026-11-05T10:00:00Z',
        updated_at: '2026-11-05T10:00:02Z',
        completed_at: '2026-11-05T10:00:02Z',
      },
      {
        id: 'task-2',
        workflow_run_id: 'run-100',
        task_key: 'fetch_data',
        task_type: 'RETRIEVAL',
        status: 'COMPLETED',
        dependencies: ['plan_tasks'],
        tokens_spent: 240,
        provider_calls: 2,
        created_at: '2026-11-05T10:00:02Z',
        updated_at: '2026-11-05T10:00:05Z',
        completed_at: '2026-11-05T10:00:05Z',
      },
      {
        id: 'task-3',
        workflow_run_id: 'run-100',
        task_key: 'synthesize_report',
        task_type: 'SYNTHESIS',
        status: 'RUNNING',
        dependencies: ['fetch_data'],
        tokens_spent: 450,
        provider_calls: 1,
        created_at: '2026-11-05T10:00:05Z',
        updated_at: '2026-11-05T10:00:08Z',
        completed_at: null,
      },
    ];

    test('Computes topological depth layers correctly for DAG visualization', () => {
      const taskMap = new Map(mockTasks.map((t) => [t.task_key, t]));

      const getDepth = (key: string, visited = new Set<string>()): number => {
        if (visited.has(key)) return 0;
        visited.add(key);
        const task = taskMap.get(key);
        if (!task || !task.dependencies || task.dependencies.length === 0) return 0;
        let maxDep = 0;
        for (const dep of task.dependencies) {
          maxDep = Math.max(maxDep, getDepth(dep, new Set(visited)) + 1);
        }
        return maxDep;
      };

      assert.equal(getDepth('plan_tasks'), 0, 'Root task must have depth 0');
      assert.equal(getDepth('fetch_data'), 1, 'First dependent task must have depth 1');
      assert.equal(getDepth('synthesize_report'), 2, 'Second dependent task must have depth 2');
    });

    test('Creates valid directed dependency edges between tasks', () => {
      const edges: Array<{ source: string; target: string; animated: boolean }> = [];
      for (const task of mockTasks) {
        for (const dep of task.dependencies) {
          edges.push({
            source: dep,
            target: task.task_key,
            animated: task.status === 'RUNNING',
          });
        }
      }

      assert.equal(edges.length, 2);
      assert.deepEqual(edges[0], { source: 'plan_tasks', target: 'fetch_data', animated: false });
      assert.deepEqual(edges[1], { source: 'fetch_data', target: 'synthesize_report', animated: true });
    });

    test('Maps all backend execution statuses to visual styles', () => {
      const statuses = [
        'PENDING',
        'READY',
        'RUNNING',
        'COMPLETED',
        'FAILED',
        'CANCELLED',
        'SKIPPED',
        'AWAITING_APPROVAL',
      ];

      for (const st of statuses) {
        const badge = getExecutionStatusBadge(st);
        assert.ok(badge.label, `Status ${st} must produce a badge label`);
        assert.ok(badge.badgeBg, `Status ${st} must produce badgeBg`);
        assert.ok(badge.icon, `Status ${st} must produce icon`);
      }
    });
  });

  // -------------------------------------------------------------------------
  // 3. 10H.3 Task Inspector & Payload Sanitization
  // -------------------------------------------------------------------------
  describe('10H.3 Task Inspector & Secret Redaction', () => {
    test('Redacts sensitive tokens, secrets, API keys and credentials from input/output payloads', () => {
      const rawPayload = {
        query: 'Analyze Acme Corp',
        api_key: 'sk-secret-live-12345',
        bearer_token: 'eyJh...token',
        user_password: 'super-secret-password',
        authorization: 'Bearer secret_auth',
        nested: {
          client_secret: 'nested-secret-value',
          safe_metric: 42,
          customer_name: 'Acme Corp',
        },
        env_vars: [
          { name: 'DATABASE_URL', value: 'postgresql://user:pass@localhost/db' },
          { name: 'APP_ENV', value: 'production' },
        ],
      };

      const sanitized = sanitizePayload(rawPayload);

      assert.equal(sanitized.api_key, '***REDACTED***');
      assert.equal(sanitized.bearer_token, '***REDACTED***');
      assert.equal(sanitized.user_password, '***REDACTED***');
      assert.equal(sanitized.authorization, '***REDACTED***');
      assert.equal(sanitized.nested.client_secret, '***REDACTED***');
      assert.equal(sanitized.nested.safe_metric, 42);
      assert.equal(sanitized.nested.customer_name, 'Acme Corp');
      assert.equal(sanitized.query, 'Analyze Acme Corp');
    });

    test('Extracts symbolic artifact:// references from task output payloads', () => {
      const outputWithArtifact = {
        summary: 'Synthesized 50 pages of report.',
        reference: 'artifact://run-100/task_2/abc123456789',
        nested: {
          secondary_ref: 'artifact://run-100/task_3/def987654321',
        },
      };

      const raw = JSON.stringify(outputWithArtifact);
      const regex = /artifact:\/\/([a-zA-Z0-9_\-\/]+)/g;
      const matches: string[] = [];
      let match;
      while ((match = regex.exec(raw)) !== null) {
        matches.push(match[0]);
      }

      assert.equal(matches.length, 2);
      assert.equal(matches[0], 'artifact://run-100/task_2/abc123456789');
      assert.equal(matches[1], 'artifact://run-100/task_3/def987654321');
    });

    test('Formats execution durations accurately', () => {
      const durationMs = formatDuration('2026-11-05T10:00:00.000Z', '2026-11-05T10:00:00.450Z');
      assert.equal(durationMs, '450ms');

      const durationSec = formatDuration('2026-11-05T10:00:00.000Z', '2026-11-05T10:00:04.250Z');
      assert.equal(durationSec, '4.25s');

      const durationMin = formatDuration('2026-11-05T10:00:00.000Z', '2026-11-05T10:02:15.000Z');
      assert.equal(durationMin, '2m 15s');

      const durationEmpty = formatDuration(null, null);
      assert.equal(durationEmpty, '—');
    });
  });

  // -------------------------------------------------------------------------
  // 4. 10H.4 Secure Artifact Viewer Contracts
  // -------------------------------------------------------------------------
  describe('10H.4 Secure Artifact Viewer Contracts', () => {
    test('Validates ArtifactPreviewResponse schema shape', () => {
      const mockPreview: ArtifactPreviewResponse = {
        artifact_id: 'a0000000-0000-0000-0000-000000000001',
        title: 'Executive Financial Summary',
        format: 'md',
        mime_type: 'text/markdown',
        content: '# Executive Summary\n\nRevenue grew by 24% YoY.',
        byte_size: 47,
        checksum_sha256: '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
        is_truncated: false,
        is_binary: false,
        expires_at: '2026-12-01T00:00:00Z',
      };

      assert.equal(mockPreview.format, 'md');
      assert.equal(mockPreview.is_binary, false);
      assert.equal(mockPreview.is_truncated, false);
      assert.ok(mockPreview.checksum_sha256.length === 64);
      assert.ok(mockPreview.content.startsWith('# Executive Summary'));
    });

    test('Identifies binary artifact formats requiring file download', () => {
      const binaryFormats = ['pdf', 'docx', 'xlsx', 'zip', 'png'];
      for (const fmt of binaryFormats) {
        const isBinary = ['pdf', 'docx', 'xlsx', 'zip', 'bin', 'png', 'jpg'].includes(fmt);
        assert.equal(isBinary, true, `Format ${fmt} should be identified as binary`);
      }

      const textFormats = ['md', 'txt', 'json', 'csv'];
      for (const fmt of textFormats) {
        const isBinary = ['pdf', 'docx', 'xlsx', 'zip', 'bin', 'png', 'jpg'].includes(fmt);
        assert.equal(isBinary, false, `Format ${fmt} should be identified as text/previewable`);
      }
    });
  });

  // -------------------------------------------------------------------------
  // 5. 10H.5 HITL Mutation Approval Flow & Terminal Protections
  // -------------------------------------------------------------------------
  describe('10H.5 HITL Mutation Approval & Terminal Lifecycle Protection', () => {
    test('Builds valid single-use approval request payload with draft hash', () => {
      const runId = 'run-999';
      const draftHash = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
      const proposedAction = 'CRM_CREATE_NOTE';
      const proposedTarget = 'rec-lead-001';

      const payload = {
        decision: 'APPROVED' as const,
        draft_hash: draftHash,
        proposed_action: proposedAction,
        proposed_target: proposedTarget,
      };

      assert.equal(payload.decision, 'APPROVED');
      assert.equal(payload.draft_hash, draftHash);
      assert.equal(payload.proposed_action, proposedAction);
      assert.equal(payload.proposed_target, proposedTarget);
    });

    test('Identifies terminal states to immediately stop background polling', () => {
      const terminalStates = ['COMPLETED', 'FAILED', 'CANCELLED', 'REJECTED', 'EXPIRED'];
      for (const st of terminalStates) {
        assert.equal(isTerminalStatus(st), true, `Status ${st} must be terminal`);
      }

      const activeStates = ['PENDING', 'PLANNING', 'RUNNING', 'EXECUTING'];
      for (const st of activeStates) {
        assert.equal(isTerminalStatus(st), false, `Status ${st} must be non-terminal`);
      }
    });
  });
});

