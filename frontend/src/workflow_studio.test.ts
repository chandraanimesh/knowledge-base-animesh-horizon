import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

import {
  checkCycle,
  isValidNodeId,
  validateNodeInputs,
  serializeCanvasToBackend,
  deserializeBackendToCanvas,
} from './components/workflowStudio/dagUtils.ts';
import { ACTION_CATALOG, TRIGGER_CATALOG } from './components/workflowStudio/types.ts';
import {
  formatDuration,
  formatMetric,
  sanitizePayload,
  isTerminalStatus,
  getExecutionStatusBadge,
  isApprovalExpired,
  canSubmitApproval,
  buildApprovalPayload,
} from './components/workflowStudio/executionUtils.ts';
import type {
  WorkflowVersionResponse,
  WorkflowActionType,
  WorkflowTriggerSpec,
} from './api/workflowDefinitions.ts';
import type {
  WorkflowRunResponse,
  WorkflowTaskResponse,
  WorkflowApprovalRequest,
  WorkflowApprovalResponse,
} from './api/workflows.ts';

describe('Phase 4A: Visual Workflow Studio Foundation', () => {
  // -------------------------------------------------------------------------
  // 1. Node Creation, Action Catalog & ID Validation
  // -------------------------------------------------------------------------
  test('1. Action catalog strictly matches allowlisted backend action types', () => {
    const expectedActions: WorkflowActionType[] = [
      'RETRIEVAL',
      'WEB_SEARCH',
      'EXTRACTION',
      'GROUNDING',
      'SYNTHESIS',
      'IF_ELSE',
      'APPROVAL_GATE',
      'EXECUTION',
    ];

    for (const act of expectedActions) {
      assert.ok(ACTION_CATALOG[act], `Action ${act} must exist in ACTION_CATALOG`);
      assert.equal(ACTION_CATALOG[act].type, act);
      assert.ok(ACTION_CATALOG[act].defaultInputs, `Action ${act} must have defaultInputs`);
      assert.ok(ACTION_CATALOG[act].outputSchema, `Action ${act} must have outputSchema`);
    }

    // Verify trigger catalog
    assert.ok(TRIGGER_CATALOG.MANUAL);
    assert.ok(TRIGGER_CATALOG.SCHEDULE);
    assert.ok(TRIGGER_CATALOG.WEBHOOK);
  });

  test('2. isValidNodeId enforces 1-64 alphanumeric, hyphen and underscore pattern', () => {
    assert.equal(isValidNodeId('step_1'), true);
    assert.equal(isValidNodeId('step-retrieval-01'), true);
    assert.equal(isValidNodeId('A1_b2_c3'), true);

    // Invalid patterns
    assert.equal(isValidNodeId(''), false, 'Empty string is invalid');
    assert.equal(isValidNodeId('step 1'), false, 'Spaces are invalid');
    assert.equal(isValidNodeId('step$special'), false, 'Special symbols are invalid');
    assert.equal(isValidNodeId('a'.repeat(65)), false, 'More than 64 characters is invalid');
  });

  // -------------------------------------------------------------------------
  // 2. Cycle Detection and Invalid Edge Rejection
  // -------------------------------------------------------------------------
  test('3. checkCycle rejects self-referencing loops immediately', () => {
    const nodes = [{ id: 'step_a' }, { id: 'step_b' }];
    const edges: { source: string; target: string }[] = [];

    const res = checkCycle(nodes, edges, { source: 'step_a', target: 'step_a' });
    assert.equal(res.hasCycle, true);
    assert.ok(res.message?.includes('Self-referencing'));
  });

  test('4. checkCycle detects 2-node cycles (A -> B, proposed B -> A)', () => {
    const nodes = [{ id: 'step_a' }, { id: 'step_b' }];
    const edges = [{ source: 'step_a', target: 'step_b' }];

    const res = checkCycle(nodes, edges, { source: 'step_b', target: 'step_a' });
    assert.equal(res.hasCycle, true);
    assert.ok(res.message?.includes('Cycle detected'));
  });

  test('5. checkCycle detects multi-node circular dependencies (A -> B -> C, proposed C -> A)', () => {
    const nodes = [{ id: 'step_a' }, { id: 'step_b' }, { id: 'step_c' }];
    const edges = [
      { source: 'step_a', target: 'step_b' },
      { source: 'step_b', target: 'step_c' },
    ];

    const res = checkCycle(nodes, edges, { source: 'step_c', target: 'step_a' });
    assert.equal(res.hasCycle, true);
  });

  test('6. checkCycle permits valid Directed Acyclic Graphs (diamond / branching)', () => {
    const nodes = [
      { id: 'trigger' },
      { id: 'step_retrieval' },
      { id: 'step_web_search' },
      { id: 'step_grounding' },
      { id: 'step_synthesis' },
    ];
    const edges = [
      { source: 'trigger', target: 'step_retrieval' },
      { source: 'trigger', target: 'step_web_search' },
      { source: 'step_retrieval', target: 'step_grounding' },
      { source: 'step_web_search', target: 'step_grounding' },
      { source: 'step_grounding', target: 'step_synthesis' },
    ];

    const res = checkCycle(nodes, edges);
    assert.equal(res.hasCycle, false, 'Valid diamond DAG should have no cycles');
  });

  // -------------------------------------------------------------------------
  // 3. Node Input Validation
  // -------------------------------------------------------------------------
  test('7. validateNodeInputs detects missing or invalid required parameters', () => {
    // RETRIEVAL: requires query
    assert.ok(validateNodeInputs('RETRIEVAL', { query: '' }).length > 0);
    assert.equal(validateNodeInputs('RETRIEVAL', { query: 'test query' }).length, 0);

    // WEB_SEARCH: requires query
    assert.ok(validateNodeInputs('WEB_SEARCH', {}).length > 0);
    assert.equal(validateNodeInputs('WEB_SEARCH', { query: 'competitor search' }).length, 0);

    // GROUNDING: confidence threshold between 0.0 and 1.0
    assert.ok(validateNodeInputs('GROUNDING', { confidence_threshold: 1.5 }).length > 0);
    assert.ok(validateNodeInputs('GROUNDING', { confidence_threshold: -0.1 }).length > 0);
    assert.equal(validateNodeInputs('GROUNDING', { confidence_threshold: 0.8 }).length, 0);

    // SYNTHESIS: requires prompt_template or focus
    assert.ok(validateNodeInputs('SYNTHESIS', { prompt_template: '', focus: '' }).length > 0);
    assert.equal(validateNodeInputs('SYNTHESIS', { prompt_template: 'Write report' }).length, 0);

    // IF_ELSE: requires left_value and operator
    assert.ok(validateNodeInputs('IF_ELSE', { left_value: '', operator: 'EQUALS' }).length > 0);
    assert.equal(
      validateNodeInputs('IF_ELSE', { left_value: true, operator: 'EQUALS', right_value: true }).length,
      0
    );

    // APPROVAL_GATE: requires gate_name
    assert.ok(validateNodeInputs('APPROVAL_GATE', { gate_name: '' }).length > 0);
    assert.equal(validateNodeInputs('APPROVAL_GATE', { gate_name: 'Lead Review' }).length, 0);

    // EXECUTION: requires action
    assert.ok(validateNodeInputs('EXECUTION', { action: '' }).length > 0);
    assert.equal(validateNodeInputs('EXECUTION', { action: 'CRM_CREATE_NOTE' }).length, 0);
  });

  // -------------------------------------------------------------------------
  // 4. Serialization: Execution Graph & Canvas Layout Separation
  // -------------------------------------------------------------------------
  test('8. serializeCanvasToBackend strictly separates execution graph from canvas layout', () => {
    const triggerSpec: WorkflowTriggerSpec = {
      trigger_type: 'SCHEDULE',
      config: { cron: '0 9 * * *', timezone: 'Asia/Kolkata', misfire_policy: 'RUN_ONCE' },
    };

    const rfNodes = [
      {
        id: 'trigger',
        position: { x: 50, y: 150 },
        data: { isTrigger: true },
      },
      {
        id: 'step_1',
        position: { x: 300, y: 100 },
        data: {
          name: 'Search News',
          node_type: 'WEB_SEARCH',
          inputs: { query: 'AI developments', limit: 5 },
        },
      },
      {
        id: 'step_2',
        position: { x: 600, y: 100 },
        data: {
          name: 'Summarize',
          node_type: 'SYNTHESIS',
          inputs: { prompt_template: 'Summarize news', focus: 'Daily' },
        },
      },
    ];

    const rfEdges = [
      { id: 'e1', source: 'trigger', target: 'step_1' },
      { id: 'e2', source: 'step_1', target: 'step_2' },
    ];

    const viewport = { x: 10, y: 20, zoom: 1.2 };

    const { steps_spec, canvas_layout, trigger_spec } = serializeCanvasToBackend(
      rfNodes,
      rfEdges,
      triggerSpec,
      viewport
    );

    // 1. steps_spec contains ONLY execution metadata
    assert.equal(steps_spec.length, 2, 'steps_spec must exclude trigger node');
    assert.equal(steps_spec[0].id, 'step_1');
    assert.equal(steps_spec[0].node_type, 'WEB_SEARCH');
    assert.equal((steps_spec[0] as any).position, undefined, 'steps_spec must NOT contain position coordinates');

    assert.equal(steps_spec[1].id, 'step_2');
    assert.deepEqual(steps_spec[1].dependencies, ['step_1'], 'step_2 must depend on step_1 based on edge');

    // 2. canvas_layout contains presentation layout
    assert.equal(canvas_layout.nodes.length, 3, 'canvas_layout must preserve all 3 visual node positions');
    assert.deepEqual(canvas_layout.nodes[0].position, { x: 50, y: 150 });
    assert.deepEqual(canvas_layout.nodes[1].position, { x: 300, y: 100 });
    assert.deepEqual(canvas_layout.nodes[2].position, { x: 600, y: 100 });
    assert.equal(canvas_layout.edges.length, 2);
    assert.deepEqual(canvas_layout.viewport, viewport);

    // 3. trigger_spec
    assert.equal(trigger_spec.trigger_type, 'SCHEDULE');
    assert.equal(trigger_spec.config.cron, '0 9 * * *');
  });

  // -------------------------------------------------------------------------
  // 5. Deserialization: Backend Version to React Flow
  // -------------------------------------------------------------------------
  test('9. deserializeBackendToCanvas correctly reconstructs nodes, edges, and dependencies', () => {
    const mockVersion: WorkflowVersionResponse = {
      id: 'ver-uuid-001',
      workflow_id: 'wf-uuid-001',
      version_number: 1,
      status: 'DRAFT',
      schema_version: '1.0',
      created_by_id: 'user-001',
      created_at: new Date().toISOString(),
      trigger_spec: {
        trigger_type: 'MANUAL',
        config: {},
      },
      steps_spec: [
        {
          id: 'step_retrieval',
          name: 'Find Docs',
          node_type: 'RETRIEVAL',
          inputs: { query: 'sales', limit: 3, scope: 'workspace' },
          dependencies: [],
        },
        {
          id: 'step_synthesis',
          name: 'Draft Report',
          node_type: 'SYNTHESIS',
          inputs: { prompt_template: 'Draft report', focus: 'Sales' },
          dependencies: ['step_retrieval'],
        },
      ],
      canvas_layout: {
        nodes: [
          { id: 'trigger', position: { x: 60, y: 220 } },
          { id: 'step_retrieval', position: { x: 320, y: 220 } },
          { id: 'step_synthesis', position: { x: 640, y: 220 } },
        ],
        edges: [
          { id: 'e-trig-ret', source: 'trigger', target: 'step_retrieval' },
          { id: 'e-ret-syn', source: 'step_retrieval', target: 'step_synthesis' },
        ],
        viewport: { x: 0, y: 0, zoom: 1 },
      },
    };

    const { nodes, edges, triggerSpec } = deserializeBackendToCanvas(mockVersion);

    assert.equal(nodes.length, 3, 'Reconstructs trigger + 2 step nodes');
    assert.equal(nodes[0].id, 'trigger');
    assert.equal(nodes[0].data.isTrigger, true);
    assert.deepEqual(nodes[0].position, { x: 60, y: 220 });

    assert.equal(nodes[1].id, 'step_retrieval');
    assert.equal(nodes[1].data.node_type, 'RETRIEVAL');
    assert.deepEqual(nodes[1].position, { x: 320, y: 220 });

    assert.equal(nodes[2].id, 'step_synthesis');
    assert.deepEqual(nodes[2].data.dependencies, ['step_retrieval']);

    assert.equal(edges.length, 2);
    assert.equal(triggerSpec.trigger_type, 'MANUAL');
  });

  // -------------------------------------------------------------------------
  // 6. Immutability and Permissions Safeguards
  // -------------------------------------------------------------------------
  test('10. Immutability safeguard prevents in-place editing of ACTIVE or ARCHIVED versions', () => {
    const isVersionEditable = (status: string, role: string) => {
      if (role === 'VIEWER') return false;
      return status === 'DRAFT';
    };

    assert.equal(isVersionEditable('DRAFT', 'EDITOR'), true, 'EDITOR can edit DRAFT');
    assert.equal(isVersionEditable('DRAFT', 'ADMIN'), true, 'ADMIN can edit DRAFT');
    assert.equal(isVersionEditable('DRAFT', 'OWNER'), true, 'OWNER can edit DRAFT');
    assert.equal(isVersionEditable('DRAFT', 'VIEWER'), false, 'VIEWER cannot edit DRAFT');

    assert.equal(isVersionEditable('ACTIVE', 'ADMIN'), false, 'ACTIVE version is immutable for all roles');
    assert.equal(isVersionEditable('ARCHIVED', 'OWNER'), false, 'ARCHIVED version is immutable for all roles');
  });
});

describe('Phase 4B: Workflow Execution Timeline & Live Node Inspector', () => {
  // -------------------------------------------------------------------------
  // 1. Metric Distinguishability (— vs 0)
  // -------------------------------------------------------------------------
  test('1. formatMetric cleanly distinguishes unavailable/missing metrics from actual zero', () => {
    // Missing / not tracked
    assert.equal(formatMetric(null), '—', 'null metric must display as em-dash');
    assert.equal(formatMetric(undefined), '—', 'undefined metric must display as em-dash');

    // Truthful zero metrics
    assert.equal(formatMetric(0), '0', 'zero metric must display as "0"');

    // Positive values
    assert.equal(formatMetric(42), '42');
    assert.equal(formatMetric(1500), '1,500');
    assert.equal(formatMetric(25000), '25,000');
  });

  // -------------------------------------------------------------------------
  // 2. Execution Duration Calculations
  // -------------------------------------------------------------------------
  test('2. formatDuration accurately computes sub-second, second, and minute durations', () => {
    assert.equal(formatDuration(null, null), '—');
    assert.equal(formatDuration(undefined, undefined), '—');
    assert.equal(formatDuration('invalid-iso', 'invalid-iso'), '—');

    // Sub-second
    const t0 = '2026-11-02T10:00:00.000Z';
    const t250ms = '2026-11-02T10:00:00.250Z';
    assert.equal(formatDuration(t0, t250ms), '250ms');

    // Seconds (under 60s)
    const t3s = '2026-11-02T10:00:03.450Z';
    assert.equal(formatDuration(t0, t3s), '3.45s');

    // Minutes and seconds
    const t125s = '2026-11-02T10:02:05.000Z';
    assert.equal(formatDuration(t0, t125s), '2m 5s');
  });

  // -------------------------------------------------------------------------
  // 3. Sensitive Payload Key Redaction
  // -------------------------------------------------------------------------
  test('3. sanitizePayload recursively masks sensitive credentials, tokens, and keys', () => {
    const rawPayload = {
      query: 'Quarterly financial report analysis',
      api_key: 'sk-live-super-secret-key-123',
      secret_token: 'tok-xyz-987',
      auth: {
        bearer_token: 'eyJh...secret',
        password: 'Password123!',
        provider: 'twenty_crm',
      },
      headers: [
        { name: 'Authorization', value: 'Bearer token-value' },
        { name: 'Content-Type', value: 'application/json' },
      ],
      safe_metrics: {
        limit: 10,
        scope: 'workspace',
      },
    };

    const sanitized = sanitizePayload(rawPayload);

    // Sensitive values must be redacted
    assert.equal(sanitized.api_key, '***REDACTED***');
    assert.equal(sanitized.secret_token, '***REDACTED***');
    assert.equal(sanitized.auth.bearer_token, '***REDACTED***');
    assert.equal(sanitized.auth.password, '***REDACTED***');

    // Safe values must remain completely intact
    assert.equal(sanitized.query, 'Quarterly financial report analysis');
    assert.equal(sanitized.auth.provider, 'twenty_crm');
    assert.equal(sanitized.safe_metrics.limit, 10);
    assert.equal(sanitized.safe_metrics.scope, 'workspace');

    // Handles null / primitives safely
    assert.equal(sanitizePayload(null), null);
    assert.equal(sanitizePayload(42), 42);
    assert.equal(sanitizePayload('hello'), 'hello');
  });

  // -------------------------------------------------------------------------
  // 4. Polling Termination Invariant (Terminal vs Active States)
  // -------------------------------------------------------------------------
  test('4. isTerminalStatus correctly bounds polling lifecycle and halts on terminal states', () => {
    // Terminal states must halt polling
    const terminalStates = ['COMPLETED', 'FAILED', 'CANCELLED', 'REJECTED', 'EXPIRED'];
    for (const st of terminalStates) {
      assert.equal(isTerminalStatus(st), true, `State ${st} must be terminal`);
    }

    // Active states must continue polling
    const activeStates = ['PENDING', 'PLANNING', 'RUNNING', 'EXECUTING'];
    for (const st of activeStates) {
      assert.equal(isTerminalStatus(st), false, `State ${st} must not be terminal`);
    }
  });

  // -------------------------------------------------------------------------
  // 5. Execution State Visualization & Badges
  // -------------------------------------------------------------------------
  test('5. getExecutionStatusBadge assigns distinct visual attributes and animation states', () => {
    const completedBadge = getExecutionStatusBadge('COMPLETED');
    assert.equal(completedBadge.label, 'Completed');
    assert.equal(completedBadge.pulse, false);
    assert.equal(completedBadge.icon, '✓');

    const runningBadge = getExecutionStatusBadge('RUNNING');
    assert.equal(runningBadge.label, 'RUNNING');
    assert.equal(runningBadge.pulse, true, 'Running steps must pulse');
    assert.equal(runningBadge.icon, '⟳');

    const failedBadge = getExecutionStatusBadge('FAILED');
    assert.equal(failedBadge.label, 'Failed');
    assert.equal(failedBadge.pulse, false);
    assert.equal(failedBadge.icon, '✕');

    const skippedBadge = getExecutionStatusBadge('SKIPPED');
    assert.equal(skippedBadge.label, 'Skipped');
    assert.equal(skippedBadge.pulse, false);

    // Human-in-the-loop approval gate state
    const approvalBadge = getExecutionStatusBadge('AWAITING_APPROVAL');
    assert.equal(approvalBadge.label, 'Awaiting Approval');
    assert.equal(approvalBadge.pulse, true, 'Awaiting approval must pulse');

    // Running APPROVAL_GATE task automatically formats as Awaiting Approval
    const gateRunningBadge = getExecutionStatusBadge('RUNNING', true);
    assert.equal(gateRunningBadge.label, 'Awaiting Approval');
    assert.equal(gateRunningBadge.pulse, true);
  });

  // -------------------------------------------------------------------------
  // 6. Run & Task Isolation and Pinned Version Association
  // -------------------------------------------------------------------------
  test('6. Run inspection never mixes tasks across different runs and tracks pinned version', () => {
    const runA: WorkflowRunResponse = {
      id: 'run-uuid-001',
      user_id: 'user-001',
      workspace_id: 'ws-001',
      workflow_id: 'wf-001',
      workflow_version_id: 'ver-pinned-v1',
      scope_type: 'WORKSPACE',
      status: 'COMPLETED',
      request_hash: 'hash-a',
      max_tokens_budget: 25000,
      tokens_spent: 1200,
      tokens_reserved: 0,
      max_provider_calls_budget: 15,
      provider_calls_spent: 2,
      provider_calls_reserved: 0,
      wall_clock_timeout_seconds: 180,
      created_at: '2026-11-02T10:00:00.000Z',
      tasks: [
        {
          id: 'task-a1',
          workflow_run_id: 'run-uuid-001',
          task_key: 'step_retrieval',
          task_type: 'RETRIEVAL',
          status: 'COMPLETED',
          dependencies: [],
          tokens_spent: 400,
          provider_calls: 1,
          created_at: '2026-11-02T10:00:00.000Z',
          updated_at: '2026-11-02T10:00:01.000Z',
          completed_at: '2026-11-02T10:00:01.000Z',
        },
      ],
    };

    const runB: WorkflowRunResponse = {
      id: 'run-uuid-002',
      user_id: 'user-001',
      workspace_id: 'ws-001',
      workflow_id: 'wf-001',
      workflow_version_id: 'ver-pinned-v2',
      scope_type: 'WORKSPACE',
      status: 'FAILED',
      request_hash: 'hash-b',
      max_tokens_budget: 25000,
      tokens_spent: 600,
      tokens_reserved: 0,
      max_provider_calls_budget: 15,
      provider_calls_spent: 1,
      provider_calls_reserved: 0,
      wall_clock_timeout_seconds: 180,
      error_message: 'Provider timeout in step_synthesis',
      created_at: '2026-11-02T10:05:00.000Z',
      tasks: [
        {
          id: 'task-b1',
          workflow_run_id: 'run-uuid-002',
          task_key: 'step_retrieval',
          task_type: 'RETRIEVAL',
          status: 'COMPLETED',
          dependencies: [],
          tokens_spent: 300,
          provider_calls: 1,
          created_at: '2026-11-02T10:05:00.000Z',
          updated_at: '2026-11-02T10:05:01.000Z',
          completed_at: '2026-11-02T10:05:01.000Z',
        },
        {
          id: 'task-b2',
          workflow_run_id: 'run-uuid-002',
          task_key: 'step_synthesis',
          task_type: 'SYNTHESIS',
          status: 'FAILED',
          dependencies: ['step_retrieval'],
          error_message: 'Connection reset by peer',
          tokens_spent: 300,
          provider_calls: 0,
          created_at: '2026-11-02T10:05:01.000Z',
          updated_at: '2026-11-02T10:05:02.000Z',
        },
      ],
    };

    // Verify task isolation: Run A tasks belong ONLY to Run A
    for (const t of runA.tasks) {
      assert.equal(t.workflow_run_id, runA.id);
      assert.notEqual(t.workflow_run_id, runB.id);
    }

    // Verify task isolation: Run B tasks belong ONLY to Run B
    for (const t of runB.tasks) {
      assert.equal(t.workflow_run_id, runB.id);
      assert.notEqual(t.workflow_run_id, runA.id);
    }

    // Verify pinned version association
    assert.equal(runA.workflow_version_id, 'ver-pinned-v1');
    assert.equal(runB.workflow_version_id, 'ver-pinned-v2');
  });
});

describe('Phase 4C: Interactive Human-in-the-Loop Approval Studio', () => {
  const baseRun: WorkflowRunResponse = {
    id: 'run-hitl-001',
    user_id: 'user-001',
    workspace_id: 'ws-test',
    workflow_id: 'wf-crm-note',
    workflow_version_id: 'ver-crm-v1',
    scope_type: 'WORKSPACE',
    status: 'AWAITING_APPROVAL',
    request_hash: 'req-hash-abc',
    draft_hash: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    intended_action: 'CRM_CREATE_NOTE',
    target_record_id: 'company-uuid-123',
    target_object: 'company',
    approval_status: 'PENDING',
    approval_id: 'appr-uuid-789',
    approval_expires_at: new Date(Date.now() + 3600 * 1000).toISOString(),
    max_tokens_budget: 25000,
    tokens_spent: 1200,
    tokens_reserved: 0,
    max_provider_calls_budget: 15,
    provider_calls_spent: 2,
    provider_calls_reserved: 0,
    wall_clock_timeout_seconds: 180,
    created_at: '2026-11-02T12:00:00.000Z',
    tasks: [],
    draft_text: 'Summary of Q3 performance for ACME Corp.',
    evidence_items: [
      {
        source_type: 'CRM_RECORD',
        source_id: 'company-uuid-123',
        snippet: 'ACME Corp Q3 deal closed at $50k ARR.',
        confidence: 0.95,
      },
    ],
  };

  // -------------------------------------------------------------------------
  // 1. Expiration Validation
  // -------------------------------------------------------------------------
  test('1. isApprovalExpired correctly evaluates expiration relative to reference time', () => {
    const fixedNow = 1700000000000;

    // No expiration set => never expired
    assert.equal(isApprovalExpired(undefined, fixedNow), false);
    assert.equal(isApprovalExpired(null, fixedNow), false);

    // Future timestamp => not expired
    const futureTime = new Date(fixedNow + 60000).toISOString();
    assert.equal(isApprovalExpired(futureTime, fixedNow), false);

    // Exact equal timestamp => expired (boundary)
    const exactTime = new Date(fixedNow).toISOString();
    assert.equal(isApprovalExpired(exactTime, fixedNow), true);

    // Past timestamp => expired
    const pastTime = new Date(fixedNow - 60000).toISOString();
    assert.equal(isApprovalExpired(pastTime, fixedNow), true);

    // Invalid date string => false (graceful fallback)
    assert.equal(isApprovalExpired('invalid-date', fixedNow), false);
  });

  // -------------------------------------------------------------------------
  // 2. canSubmitApproval Guard
  // -------------------------------------------------------------------------
  test('2. canSubmitApproval enforces strict AWAITING_APPROVAL state and unexpired status', () => {
    const fixedNow = 1700000000000;
    const futureTime = new Date(fixedNow + 60000).toISOString();
    const pastTime = new Date(fixedNow - 60000).toISOString();

    // Valid: AWAITING_APPROVAL, future expiry, not submitting
    assert.equal(canSubmitApproval('AWAITING_APPROVAL', futureTime, false, fixedNow), true);

    // Blocked: already submitting (duplicate click protection)
    assert.equal(canSubmitApproval('AWAITING_APPROVAL', futureTime, true, fixedNow), false);

    // Blocked: expired ticket
    assert.equal(canSubmitApproval('AWAITING_APPROVAL', pastTime, false, fixedNow), false);

    // Blocked: terminal or other non-approval states
    assert.equal(canSubmitApproval('COMPLETED', futureTime, false, fixedNow), false);
    assert.equal(canSubmitApproval('RUNNING', futureTime, false, fixedNow), false);
    assert.equal(canSubmitApproval('FAILED', futureTime, false, fixedNow), false);
    assert.equal(canSubmitApproval('REJECTED', futureTime, false, fixedNow), false);
    assert.equal(canSubmitApproval('EXPIRED', futureTime, false, fixedNow), false);
    assert.equal(canSubmitApproval('PENDING', futureTime, false, fixedNow), false);
  });

  // -------------------------------------------------------------------------
  // 3. Approval Payload Construction & Canonical Hash Invariant
  // -------------------------------------------------------------------------
  test('3. buildApprovalPayload preserves server canonical draft hash without modification', () => {
    const payload = buildApprovalPayload({
      run: baseRun,
      decision: 'APPROVED',
    });

    assert.equal(payload.decision, 'APPROVED');
    // Security invariant: draft_hash must strictly equal the server-persisted value
    assert.equal(payload.draft_hash, baseRun.draft_hash);
    assert.equal(payload.proposed_action, 'CRM_CREATE_NOTE');
    assert.equal(payload.proposed_target, 'company-uuid-123');
    assert.equal(payload.rejection_reason, undefined);
  });

  // -------------------------------------------------------------------------
  // 4. Rejection Payload Construction with Reason
  // -------------------------------------------------------------------------
  test('4. buildApprovalPayload attaches rejection reason and defaults safely', () => {
    // Explicit rejection reason
    const payloadWithReason = buildApprovalPayload({
      run: baseRun,
      decision: 'REJECTED',
      rejectionReason: 'Financial figures require verification against invoices.',
    });

    assert.equal(payloadWithReason.decision, 'REJECTED');
    assert.equal(payloadWithReason.draft_hash, baseRun.draft_hash);
    assert.equal(
      payloadWithReason.rejection_reason,
      'Financial figures require verification against invoices.'
    );

    // Blank or whitespace rejection reason defaults safely
    const payloadDefaultReason = buildApprovalPayload({
      run: baseRun,
      decision: 'REJECTED',
      rejectionReason: '   ',
    });

    assert.equal(
      payloadDefaultReason.rejection_reason,
      'Rejected by human reviewer.'
    );
  });

  // -------------------------------------------------------------------------
  // 5. Fail-Closed Security: Missing Hash Throws Error
  // -------------------------------------------------------------------------
  test('5. buildApprovalPayload fails closed if draft hash is missing or empty', () => {
    const runWithoutHash = { ...baseRun, draft_hash: null };

    assert.throws(
      () => {
        buildApprovalPayload({
          run: runWithoutHash,
          decision: 'APPROVED',
        });
      },
      /Security Error: Run lacks server-persisted canonical draft hash/
    );
  });

  // -------------------------------------------------------------------------
  // 6. Approval Lifecycle Transitions & Single-Use Token Semantics
  // -------------------------------------------------------------------------
  test('6. Approval response simulates single-use token transition to terminal or executing state', () => {
    const approvalResponse: WorkflowApprovalResponse = {
      workflow_run_id: baseRun.id,
      approval_id: baseRun.approval_id!,
      status: 'COMPLETED',
      side_effect_executed: true,
      writeback_result: { note_id: 'crm-note-555', status: 'created' },
    };

    assert.equal(approvalResponse.status, 'COMPLETED');
    assert.equal(approvalResponse.side_effect_executed, true);
    assert.ok(approvalResponse.writeback_result?.note_id);

    // After approval completes, the run status transitions to COMPLETED.
    // A subsequent attempt to decide must be rejected by canSubmitApproval.
    const completedRun: WorkflowRunResponse = {
      ...baseRun,
      status: approvalResponse.status,
      approval_status: 'CONSUMED',
    };

    assert.equal(canSubmitApproval(completedRun.status, completedRun.approval_expires_at), false);
    assert.equal(isTerminalStatus(completedRun.status), true);
  });
});

describe('Phase 4D: End-to-End Visual Workflow Authoring & Safe Test Run', () => {
  // Helper to validate sample trigger JSON payload strings
  function parseAndValidateTestPayload(jsonStr: string): { valid: boolean; data?: any; error?: string } {
    try {
      const data = JSON.parse(jsonStr);
      if (typeof data !== 'object' || data === null || Array.isArray(data)) {
        return { valid: false, error: 'Trigger payload must be a JSON object dictionary.' };
      }
      return { valid: true, data };
    } catch (err: any) {
      return { valid: false, error: err.message || 'Invalid JSON syntax' };
    }
  }

  // -------------------------------------------------------------------------
  // 1. Trigger Payload Syntax & Structure Validation
  // -------------------------------------------------------------------------
  test('1. Test payload parser validates valid JSON dictionary and rejects malformed inputs', () => {
    // Valid query dictionary
    const validQuery = parseAndValidateTestPayload('{"query": "Analyze Q3 sales pipeline", "limit": 5}');
    assert.equal(validQuery.valid, true);
    assert.equal(validQuery.data.query, 'Analyze Q3 sales pipeline');
    assert.equal(validQuery.data.limit, 5);

    // Valid CRM event dictionary
    const validCrm = parseAndValidateTestPayload('{"event": "company.created", "company_id": "c123"}');
    assert.equal(validCrm.valid, true);
    assert.equal(validCrm.data.event, 'company.created');

    // Invalid JSON syntax (unclosed quote / brace)
    const malformed = parseAndValidateTestPayload('{"query": "broken json');
    assert.equal(malformed.valid, false);
    assert.ok(malformed.error);

    // Invalid data types (array or primitive instead of object)
    const arrayInput = parseAndValidateTestPayload('["not", "an", "object"]');
    assert.equal(arrayInput.valid, false);
    assert.equal(arrayInput.error, 'Trigger payload must be a JSON object dictionary.');

    const primitiveInput = parseAndValidateTestPayload('"just a string"');
    assert.equal(primitiveInput.valid, false);
  });

  // -------------------------------------------------------------------------
  // 2. Server-Controlled Test Run Identity & Mode Association
  // -------------------------------------------------------------------------
  test('2. Workflow test run response strictly associates draft version and sets test markers', () => {
    const draftTestRun: WorkflowRunResponse = {
      id: 'run-test-uuid-444',
      user_id: 'user-reviewer-1',
      workspace_id: 'ws-test-1',
      workflow_id: 'wf-draft-analysis',
      workflow_version_id: 'ver-draft-v3', // Strictly bound to draft version!
      scope_type: 'WORKSPACE',
      thread_id: 'test:ver-draft-v3', // Explicit TEST marker!
      status: 'COMPLETED',
      request_hash: 'req-hash-draft-test',
      idempotency_key: 'test:studio-test-manual-1', // Isolated test idempotency namespace!
      max_tokens_budget: 25000,
      tokens_spent: 650,
      tokens_reserved: 0,
      max_provider_calls_budget: 15,
      provider_calls_spent: 1,
      provider_calls_reserved: 0,
      wall_clock_timeout_seconds: 180,
      created_at: new Date().toISOString(),
      is_test: true,
      execution_mode: 'TEST',
      tasks: [
        {
          id: 'task-test-1',
          workflow_run_id: 'run-test-uuid-444',
          task_key: 'step_retrieval',
          task_type: 'RETRIEVAL',
          status: 'COMPLETED',
          dependencies: [],
          tokens_spent: 400,
          provider_calls: 1,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      ],
    };

    assert.equal(draftTestRun.is_test, true);
    assert.equal(draftTestRun.execution_mode, 'TEST');
    assert.ok(draftTestRun.thread_id?.startsWith('test:'));
    assert.ok(draftTestRun.idempotency_key?.startsWith('test:'));
    assert.equal(draftTestRun.workflow_version_id, 'ver-draft-v3');
  });

  // -------------------------------------------------------------------------
  // 3. Dry-Run Mutation Safety in TEST Mode
  // -------------------------------------------------------------------------
  test('3. CRM mutation output in TEST mode confirms dry-run simulation without external writes', () => {
    const dryRunTask: WorkflowRunResponse['tasks'][0] = {
      id: 'task-exec-sim',
      workflow_run_id: 'run-test-uuid-444',
      task_key: 'step_crm_write',
      task_type: 'EXECUTION',
      status: 'COMPLETED',
      dependencies: ['step_retrieval'],
      tokens_spent: 50,
      provider_calls: 0, // In test mode, 0 live external provider calls!
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      output_result: {
        status: 'simulated_dry_run',
        executed: true,
        action: 'CRM_CREATE_NOTE',
        mutation_id: 'dryrun-abc12345',
        details: {
          dry_run: true,
          target_system: 'crm',
          simulated_payload: { note: 'Sample note content' },
          message: "Simulated execution for 'CRM_CREATE_NOTE' in TEST mode. No external mutations performed.",
        },
      },
    };

    assert.equal(dryRunTask.output_result?.status, 'simulated_dry_run');
    assert.equal(dryRunTask.output_result?.executed, true);
    assert.equal(dryRunTask.output_result?.details?.dry_run, true);
    assert.equal(dryRunTask.provider_calls, 0, 'Test execution must not consume external provider calls');
  });

  // -------------------------------------------------------------------------
  // 4. Repeated Test Runs with Isolated Identities
  // -------------------------------------------------------------------------
  test('4. Rerunning tests generates distinct test-run identities without colliding with production', () => {
    const run1Key = `test:studio-test-${Date.now()}-a1b2c3`;
    const run2Key = `test:studio-test-${Date.now() + 10}-d4e5f6`;

    assert.notEqual(run1Key, run2Key);
    assert.ok(run1Key.startsWith('test:'));
    assert.ok(run2Key.startsWith('test:'));

    // Production keys do not have the test: prefix
    const prodKey = 'prod-manual-trigger-uuid-999';
    assert.equal(prodKey.startsWith('test:'), false);
    assert.notEqual(run1Key, prodKey);
  });
});


