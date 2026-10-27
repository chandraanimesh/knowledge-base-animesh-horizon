export type MCPTransportType = 'streamable-http' | 'sse';

export type MCPConnectionStatus = 'CONNECTED' | 'DISCONNECTED' | 'ERROR' | 'SYNCING';

export interface DiscoveredToolItem {
  name: string;
  description?: string | null;
  input_schema?: Record<string, any> | null;
  output_schema?: Record<string, any> | null;
}

export type MCPCredentialMode = 'default' | 'custom' | 'env_ref';

export interface MCPConnectionResponse {
  id: string;
  workspace_id: string;
  name: string;
  endpoint_url: string;
  transport: MCPTransportType;
  credential_mode: MCPCredentialMode;
  has_custom_credential: boolean;
  credential_ref?: string | null;
  is_enabled: boolean;
  status: MCPConnectionStatus;
  error_message?: string | null;
  last_connected_at?: string | null;
  last_synced_at?: string | null;
  server_info: Record<string, any>;
  discovered_tools: DiscoveredToolItem[];
  enabled_tools: string[];
  configuration: Record<string, any>;
  created_by?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ServerManagedCredentialOption {
  value: string | null;
  label: string;
  description: string;
  allowedHosts: string[];
}

export const SERVER_MANAGED_CREDENTIAL_PROFILES: ServerManagedCredentialOption[] = [
  {
    value: null,
    label: 'None / Public (No authentication required)',
    description: 'Endpoint is publicly accessible or runs on a trusted internal network without credentials.',
    allowedHosts: ['*'],
  },
  {
    value: 'SERVER_TWENTY_API_KEY',
    label: 'Twenty CRM Server Profile (TWENTY_API_KEY)',
    description: 'Server-managed Twenty CRM API token. Authorized for twenty.com and localhost.',
    allowedHosts: ['twenty.com', 'localhost', '127.0.0.1'],
  },
  {
    value: 'MCP_TEST_SERVER_KEY',
    label: 'Integration Test Server Profile (MCP_TEST_SERVER_KEY)',
    description: 'Server-managed key for automated test harnesses and mock servers.',
    allowedHosts: ['localhost', '127.0.0.1', 'mockserver', 'example.com'],
  },
  {
    value: 'MCP_DEMO_API_KEY',
    label: 'Demo MCP Server Profile (MCP_DEMO_API_KEY)',
    description: 'Server-managed key for staging/demo environments.',
    allowedHosts: ['localhost', '127.0.0.1', 'mockserver', 'example.com'],
  },
];

export const VERIFIED_READ_ONLY_NAMES = new Set([
  'find_many_companies',
  'find_one_company',
  'find_many_people',
  'find_many_opportunities',
  'get_company',
  'get_contact',
  'list_companies',
  'list_contacts',
  'read_query',
  'get_search_status',
  'echo_read',
  'read_report',
  'list_webhooks',
  'rag_ask',
  'rag_search',
  'list_documents',
]);

export const APPROVED_WRITE_BACK_TOOLS = new Set([
  'create_one_note',
  'create_one_note_target',
  'create_webhook',
  'ingest_document',
  'run_diagnostics',
]);

export function getToolRiskClassification(toolName: string): {
  type: 'Read' | 'Write' | 'Restricted';
  label: string;
  badgeBg: string;
  badgeColor: string;
  border: string;
  notice?: string;
} {
  const lower = (toolName || '').toLowerCase();

  // STRICT AUTHORITATIVE BACKEND ALIGNMENT:
  // Only explicitly verified tools in the server catalog are classified as Read.
  // Client-side prefix matching (e.g. startsWith 'get_') must NEVER classify unknown tools as Read.
  if (VERIFIED_READ_ONLY_NAMES.has(lower)) {
    return {
      type: 'Read',
      label: 'Read',
      badgeBg: '#eff6ff',
      badgeColor: '#1d4ed8',
      border: '#bfdbfe',
    };
  }

  if (
    APPROVED_WRITE_BACK_TOOLS.has(lower) ||
    lower.startsWith('create_') ||
    lower.startsWith('update_') ||
    lower.startsWith('mutate_') ||
    lower.startsWith('delete_') ||
    lower.startsWith('ingest_') ||
    lower.includes('write') ||
    lower.includes('target')
  ) {
    return {
      type: 'Write',
      label: 'Write (Approval Gated)',
      badgeBg: '#fff7ed',
      badgeColor: '#c2410c',
      border: '#ffedd5',
      notice: 'Allowlisting enables tool proposal; external mutation strictly requires cryptographic human approval.',
    };
  }

  // Unknown tools default to Restricted / Unverified
  return {
    type: 'Restricted',
    label: 'Unverified',
    badgeBg: '#f8fafc',
    badgeColor: '#475569',
    border: '#e2e8f0',
    notice: 'Unverified tool: Not in server read-only catalog. Gated by Owner/Admin permissions.',
  };
}

export function getServerBrandInfo(name: string, endpoint: string): {
  avatarText: string;
  avatarBg: string;
  avatarColor: string;
} {
  const combined = ((name || '') + ' ' + (endpoint || '')).toLowerCase();
  if (combined.includes('twenty')) {
    return { avatarText: 'T', avatarBg: '#f43f5e', avatarColor: '#ffffff' };
  }
  if (combined.includes('odoo')) {
    return { avatarText: 'O', avatarBg: '#8b5cf6', avatarColor: '#ffffff' };
  }
  if (combined.includes('hubspot')) {
    return { avatarText: 'H', avatarBg: '#f97316', avatarColor: '#ffffff' };
  }
  if (combined.includes('salesforce')) {
    return { avatarText: 'S', avatarBg: '#0ea5e9', avatarColor: '#ffffff' };
  }
  return { avatarText: '🔗', avatarBg: '#64748b', avatarColor: '#ffffff' };
}
