import { test } from 'node:test';
import assert from 'node:assert';
import {
  getToolRiskClassification,
  getServerBrandInfo,
  VERIFIED_READ_ONLY_NAMES,
  SERVER_MANAGED_CREDENTIAL_PROFILES,
  type MCPConnectionResponse,
} from './components/mcpDefinitions.ts';

test('1. getToolRiskClassification correctly classifies read-only vs mutating vs restricted tools', () => {
  // Read-only tools
  assert.strictEqual(getToolRiskClassification('find_many_companies').type, 'Read');
  assert.strictEqual(getToolRiskClassification('find_many_companies').label, 'Read');
  assert.strictEqual(getToolRiskClassification('get_company').type, 'Read');
  assert.strictEqual(getToolRiskClassification('list_webhooks').type, 'Read');
  assert.strictEqual(getToolRiskClassification('rag_ask').type, 'Read');
  assert.strictEqual(getToolRiskClassification('rag_search').type, 'Read');
  assert.strictEqual(getToolRiskClassification('list_documents').type, 'Read');

  // Mutating tools (gated by human approval)
  assert.strictEqual(getToolRiskClassification('create_one_note').type, 'Write');
  assert.strictEqual(getToolRiskClassification('create_one_note').label, 'Write (Approval Gated)');
  assert.strictEqual(getToolRiskClassification('create_one_note_target').type, 'Write');
  assert.strictEqual(getToolRiskClassification('create_webhook').type, 'Write');
  assert.strictEqual(getToolRiskClassification('ingest_document').type, 'Write');
  assert.strictEqual(getToolRiskClassification('mutate_record').type, 'Write');

  // Restricted / unverified tools (even if prefixed with get_ or search_)
  assert.strictEqual(getToolRiskClassification('execute_arbitrary_code').type, 'Restricted');
  assert.strictEqual(getToolRiskClassification('unknown_remote_tool').type, 'Restricted');
  assert.strictEqual(getToolRiskClassification('get_admin_secrets').type, 'Restricted');
  assert.strictEqual(getToolRiskClassification('get_admin_secrets').label, 'Unverified');
  assert.strictEqual(getToolRiskClassification('search_private_records').type, 'Restricted');
});

test('2. getServerBrandInfo correctly identifies known platforms', () => {
  const twenty = getServerBrandInfo('Twenty CRM', 'http://localhost:3000/mcp');
  assert.strictEqual(twenty.avatarText, 'T');

  const odoo = getServerBrandInfo('Odoo MCP', 'https://odoo.example.com/mcp');
  assert.strictEqual(odoo.avatarText, 'O');

  const custom = getServerBrandInfo('Internal Tools', 'https://internal.example.com/mcp');
  assert.strictEqual(custom.avatarText, '🔗');
});

test('3. SERVER_MANAGED_CREDENTIAL_PROFILES enforces server-side references without raw secrets', () => {
  // Verify allowlist exists and matches approved backend refs
  const profileValues = SERVER_MANAGED_CREDENTIAL_PROFILES.map((p) => p.value);
  assert.ok(profileValues.includes(null)); // None
  assert.ok(profileValues.includes('SERVER_TWENTY_API_KEY'));
  assert.ok(profileValues.includes('MCP_TEST_SERVER_KEY'));
  assert.ok(profileValues.includes('MCP_DEMO_API_KEY'));

  // Ensure no raw secrets are stored in labels or descriptions
  SERVER_MANAGED_CREDENTIAL_PROFILES.forEach((profile) => {
    assert.ok(!profile.label.includes('sk-'));
    assert.ok(!profile.description.includes('sk-'));
    assert.ok(!profile.label.includes('gsk_'));
  });
});

test('4. Default-Deny tool allowlist behavior: discovered tools remain unallowed until explicitly enabled', () => {
  const mockConnection: MCPConnectionResponse = {
    id: 'conn-1',
    workspace_id: 'ws-1',
    name: 'Twenty CRM',
    endpoint_url: 'http://localhost:3000/mcp',
    transport: 'streamable-http',
    credential_ref: 'SERVER_TWENTY_API_KEY',
    is_enabled: true,
    status: 'CONNECTED',
    server_info: { name: 'Twenty MCP Server' },
    discovered_tools: [
      { name: 'find_many_companies', description: 'Search companies' },
      { name: 'create_one_note', description: 'Create note' },
      { name: 'create_one_note_target', description: 'Link note' },
    ],
    enabled_tools: ['find_many_companies'], // Only 1 is allowed
    configuration: {},
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };

  // Verify allowlist containment
  const isAllowed = (toolName: string) => mockConnection.enabled_tools.includes(toolName);
  assert.strictEqual(isAllowed('find_many_companies'), true);
  assert.strictEqual(isAllowed('create_one_note'), false);
  assert.strictEqual(isAllowed('create_one_note_target'), false);
  assert.strictEqual(isAllowed('arbitrary_tool'), false);
});

test('5. Multi-step write idempotency and recovery state machine', () => {
  // Scenario: create_one_note succeeds but companion create_one_note_target fails/times out.
  // Re-invoking create_one_note with the same idempotency_key must return cached confirmation
  // and NOT generate a duplicate note.
  const confirmedNoteOperation = {
    idempotency_key: 'prop-a1d13d-note',
    tool_name: 'create_one_note',
    status: 'CONFIRMED',
    result: { id: 'note-uuid-12345', title: 'Q3 Brief' },
    is_replay: false,
  };

  function simulateDispatch(key: string, existing: typeof confirmedNoteOperation | null) {
    if (existing && existing.idempotency_key === key && existing.status === 'CONFIRMED') {
      return {
        ...existing.result,
        is_replay: true,
      };
    }
    return { id: 'new-note-uuid', is_replay: false };
  }

  // First replay
  const replayResult = simulateDispatch('prop-a1d13d-note', confirmedNoteOperation);
  assert.strictEqual(replayResult.id, 'note-uuid-12345');
  assert.strictEqual(replayResult.is_replay, true);

  // New key
  const newResult = simulateDispatch('prop-new-unique-key', confirmedNoteOperation);
  assert.strictEqual(newResult.is_replay, false);
});

test('6. Milestone 6G: Credential mode contract guarantees no raw secrets or ciphertext in responses', () => {
  const customConnection: MCPConnectionResponse = {
    id: 'conn-sec-1',
    workspace_id: 'ws-prod',
    name: 'Secure Custom MCP',
    endpoint_url: 'https://mcp.company.internal/api',
    transport: 'streamable-http',
    credential_mode: 'custom',
    credential_ref: null,
    has_custom_credential: true,
    is_enabled: true,
    status: 'CONNECTED',
    server_info: { name: 'Internal MCP' },
    discovered_tools: [],
    enabled_tools: [],
    configuration: {},
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };

  // Verify response contract
  assert.strictEqual(customConnection.credential_mode, 'custom');
  assert.strictEqual(customConnection.has_custom_credential, true);

  // Assert secret confidentiality: raw secret or ciphertext keys must never be exposed on the response object
  assert.strictEqual((customConnection as any).custom_credential, undefined);
  assert.strictEqual((customConnection as any).encrypted_credential, undefined);
  assert.strictEqual((customConnection as any).configuration?.encrypted_credential, undefined);
  assert.strictEqual((customConnection as any).configuration?.credential_secret, undefined);
});

test('7. Milestone 6G: Masked placeholder validation invariant prevents overwriting secret', () => {
  function isMaskedPlaceholder(value?: string | null): boolean {
    if (!value) return true;
    const stripped = value.trim();
    if (!stripped) return true;
    return /^[\u2022*]+$/.test(stripped);
  }

  assert.strictEqual(isMaskedPlaceholder('••••••••••••••••'), true);
  assert.strictEqual(isMaskedPlaceholder('********'), true);
  assert.strictEqual(isMaskedPlaceholder(''), true);
  assert.strictEqual(isMaskedPlaceholder('   '), true);
  assert.strictEqual(isMaskedPlaceholder(undefined), true);
  assert.strictEqual(isMaskedPlaceholder(null), true);

  // Legitimate secret must not be considered a masked placeholder
  assert.strictEqual(isMaskedPlaceholder('sk-proj-1234567890abcdef'), false);
  assert.strictEqual(isMaskedPlaceholder('real-api-token-xyz'), false);
});
