import { apiClient } from './client';
export * from '../components/mcpDefinitions';
import type {
  MCPTransportType,
  MCPConnectionResponse,
  MCPCredentialMode,
} from '../components/mcpDefinitions';

export interface MCPConnectionCreate {
  name: string;
  endpoint_url: string;
  transport?: MCPTransportType;
  credential_mode?: MCPCredentialMode;
  credential_ref?: string | null;
  custom_credential?: string | null;
  configuration?: Record<string, any>;
}

export interface MCPConnectionUpdate {
  name?: string;
  endpoint_url?: string;
  transport?: MCPTransportType;
  credential_mode?: MCPCredentialMode;
  credential_ref?: string | null;
  custom_credential?: string | null;
  clear_custom_credential?: boolean;
  is_enabled?: boolean;
  configuration?: Record<string, any>;
}

export interface MCPConnectionTestResult {
  status: string;
  latency_ms: number;
  server_info: Record<string, any>;
  discovered_tool_count: number;
  credential_mode: string;
  message: string;
}

export interface MCPToolToggleRequest {
  tool_name: string;
  enabled: boolean;
}

export interface MCPToolAllowlistUpdate {
  enabled_tools: string[];
}

export interface RAGServerMCPStatus {
  status: string;
  name: string;
  transport: string;
  sse_endpoint: string;
  messages_endpoint: string;
  local_base_url: string;
  tools_count: number;
  instructions: string;
}

// ── API Methods ──────────────────────────────────────────────────────────────

export async function listMCPConnections(
  token: string,
  workspaceId: string
): Promise<MCPConnectionResponse[]> {
  return apiClient<MCPConnectionResponse[]>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp-connections`,
    { method: 'GET' },
    token
  );
}

export async function getMCPConnection(
  token: string,
  workspaceId: string,
  connectionId: string
): Promise<MCPConnectionResponse> {
  return apiClient<MCPConnectionResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp-connections/${encodeURIComponent(connectionId)}`,
    { method: 'GET' },
    token
  );
}

export async function createMCPConnection(
  token: string,
  workspaceId: string,
  data: MCPConnectionCreate
): Promise<MCPConnectionResponse> {
  return apiClient<MCPConnectionResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp-connections`,
    {
      method: 'POST',
      body: JSON.stringify(data),
    },
    token
  );
}

export async function updateMCPConnection(
  token: string,
  workspaceId: string,
  connectionId: string,
  data: MCPConnectionUpdate
): Promise<MCPConnectionResponse> {
  return apiClient<MCPConnectionResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp-connections/${encodeURIComponent(connectionId)}`,
    {
      method: 'PATCH',
      body: JSON.stringify(data),
    },
    token
  );
}

export async function deleteMCPConnection(
  token: string,
  workspaceId: string,
  connectionId: string
): Promise<void> {
  return apiClient<void>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp-connections/${encodeURIComponent(connectionId)}`,
    { method: 'DELETE' },
    token
  );
}

export async function syncMCPTools(
  token: string,
  workspaceId: string,
  connectionId: string
): Promise<MCPConnectionResponse> {
  return apiClient<MCPConnectionResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp-connections/${encodeURIComponent(connectionId)}/sync`,
    { method: 'POST' },
    token
  );
}

export async function testMCPConnection(
  token: string,
  workspaceId: string,
  connectionId: string
): Promise<MCPConnectionTestResult> {
  return apiClient<MCPConnectionTestResult>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp-connections/${encodeURIComponent(connectionId)}/test`,
    { method: 'POST' },
    token
  );
}

export async function toggleMCPTool(
  token: string,
  workspaceId: string,
  connectionId: string,
  toolName: string,
  enabled: boolean
): Promise<MCPConnectionResponse> {
  const payload: MCPToolToggleRequest = { tool_name: toolName, enabled };
  return apiClient<MCPConnectionResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp-connections/${encodeURIComponent(connectionId)}/tools/toggle`,
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    token
  );
}

export async function updateMCPToolAllowlist(
  token: string,
  workspaceId: string,
  connectionId: string,
  enabledTools: string[]
): Promise<MCPConnectionResponse> {
  const payload: MCPToolAllowlistUpdate = { enabled_tools: enabledTools };
  return apiClient<MCPConnectionResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp-connections/${encodeURIComponent(connectionId)}/tools/allowlist`,
    {
      method: 'PUT',
      body: JSON.stringify(payload),
    },
    token
  );
}

export async function getRAGServerMCPStatus(token?: string): Promise<RAGServerMCPStatus> {
  return apiClient<RAGServerMCPStatus>('/mcp', { method: 'GET' }, token);
}
