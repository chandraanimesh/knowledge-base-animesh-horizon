import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  MCPConnectionResponse,
  MCPConnectionCreate,
  MCPConnectionUpdate,
  MCPCredentialMode,
  DiscoveredToolItem,
  SERVER_MANAGED_CREDENTIAL_PROFILES,
  RAGServerMCPStatus,
  listMCPConnections,
  createMCPConnection,
  updateMCPConnection,
  deleteMCPConnection,
  syncMCPTools,
  testMCPConnection,
  toggleMCPTool,
  getRAGServerMCPStatus,
} from '../api/mcpConnections';
import { WorkspaceItem } from '../api/workspaces';

interface WorkspaceMCPIntegrationsViewProps {
  workspace: WorkspaceItem;
  token: string;
  isOwnerOrAdmin: boolean;
  onToast?: (type: 'success' | 'error', message: string) => void;
}

type MainTab = 'clients' | 'server';

interface ActivityLogItem {
  id: string;
  title: string;
  detail: string;
  timestamp: string;
  status: 'success' | 'info' | 'warning' | 'error';
}

import {
  getToolRiskClassification,
  getServerBrandInfo,
} from './mcpDefinitions';
import {
  ShieldIcon,
  KeyIcon,
  GlobeIcon,
  PlugIcon,
  ServerIcon,
  LockIcon,
  EyeIcon,
  EyeOffIcon,
  AlertCircleIcon,
  CheckIcon,
  CopyIcon,
  EditIcon,
  TrashIcon,
  SettingsIcon,
  ZapIcon,
  PlayIcon,
  RefreshCwIcon,
  XIcon,
} from './mcpIcons';

export default function WorkspaceMCPIntegrationsView({
  workspace,
  token,
  isOwnerOrAdmin,
  onToast,
}: WorkspaceMCPIntegrationsViewProps) {
  const [activeTab, setActiveTab] = useState<MainTab>('clients');

  // Connections state
  const [connections, setConnections] = useState<MCPConnectionResponse[]>([]);
  const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Syncing & mutating status tracking
  const [syncingMap, setSyncingMap] = useState<Record<string, boolean>>({});
  const [togglingTools, setTogglingTools] = useState<Record<string, boolean>>({});
  const [updatingConnectionId, setUpdatingConnectionId] = useState<string | null>(null);

  // Modals state
  const [showAddEditModal, setShowAddEditModal] = useState<boolean>(false);
  const [editingConnection, setEditingConnection] = useState<MCPConnectionResponse | null>(null);
  const [showDeleteModal, setShowDeleteModal] = useState<boolean>(false);
  const [deletingConnection, setDeletingConnection] = useState<MCPConnectionResponse | null>(null);
  const [modalLoading, setModalLoading] = useState<boolean>(false);
  const [modalError, setModalError] = useState<string | null>(null);

  // Form fields for Add/Edit
  const [formName, setFormName] = useState<string>('');
  const [formEndpoint, setFormEndpoint] = useState<string>('');
  const [formTransport, setFormTransport] = useState<'streamable-http' | 'sse'>('streamable-http');
  const [formCredMode, setFormCredMode] = useState<MCPCredentialMode>('default');
  const [formCredRef, setFormCredRef] = useState<string>('');
  const [formCustomCred, setFormCustomCred] = useState<string>('');
  const [formClearCustomCred, setFormClearCustomCred] = useState<boolean>(false);
  const [showCustomCredPlaintext, setShowCustomCredPlaintext] = useState<boolean>(false);
  const [formAllowPrivate, setFormAllowPrivate] = useState<boolean>(true);
  const [formTimeout, setFormTimeout] = useState<number>(10);
  const [testingConnectionId, setTestingConnectionId] = useState<string | null>(null);

  // Copy indicator state
  const [copiedEndpoint, setCopiedEndpoint] = useState<string | null>(null);

  // RAG Server tab state
  const [ragStatus, setRagStatus] = useState<RAGServerMCPStatus | null>(null);
  const [ragLoading, setRagLoading] = useState<boolean>(false);
  const [ragError, setRagError] = useState<string | null>(null);

  // Activity log state
  const [activityLogs, setActivityLogs] = useState<ActivityLogItem[]>([
    {
      id: 'log-init',
      title: 'Workspace MCP Manager Ready',
      detail: `Scoped to ${workspace.name}`,
      timestamp: 'Just now',
      status: 'info',
    },
  ]);

  const addLog = useCallback(
    (title: string, detail: string, status: 'success' | 'info' | 'warning' | 'error' = 'info') => {
      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      setActivityLogs((prev) => [
        {
          id: `log-${Date.now()}-${Math.random()}`,
          title,
          detail,
          timestamp: timeStr,
          status,
        },
        ...prev.slice(0, 7),
      ]);
    },
    []
  );

  // Fetch connections for current workspace
  const fetchConnections = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await listMCPConnections(token, workspace.id);
      setConnections(list);
      // Auto-select first connection if none selected
      if (list.length > 0) {
        setSelectedConnectionId((prev) => {
          if (prev && list.some((c) => c.id === prev)) return prev;
          return list[0].id;
        });
      } else {
        setSelectedConnectionId(null);
      }
    } catch (err: any) {
      const msg = err.message || 'Failed to load workspace MCP connections.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [token, workspace.id]);

  useEffect(() => {
    fetchConnections();
  }, [fetchConnections]);

  // Selected Connection
  const selectedConnection = useMemo(() => {
    return connections.find((c) => c.id === selectedConnectionId) || null;
  }, [connections, selectedConnectionId]);

  // Copy to clipboard helper
  const handleCopyEndpoint = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedEndpoint(text);
    setTimeout(() => setCopiedEndpoint(null), 2000);
    onToast?.('success', 'Endpoint URL copied to clipboard');
  };

  // Sync / Test Connection
  const handleSyncConnection = async (connId: string) => {
    setSyncingMap((prev) => ({ ...prev, [connId]: true }));
    try {
      const updated = await syncMCPTools(token, workspace.id, connId);
      setConnections((prev) => prev.map((c) => (c.id === connId ? updated : c)));
      const count = updated.discovered_tools?.length || 0;
      addLog(
        `${updated.name} synced successfully`,
        `${count} tools discovered`,
        'success'
      );
      onToast?.('success', `Connection synced! ${count} tools discovered.`);
    } catch (err: any) {
      const msg = err.message || 'Synchronization failed.';
      addLog('Connection sync failed', msg, 'error');
      onToast?.('error', msg);
      // Refresh list to capture error status
      fetchConnections();
    } finally {
      setSyncingMap((prev) => ({ ...prev, [connId]: false }));
    }
  };

  // Toggle Connection Enabled / Disabled
  const handleToggleConnectionEnabled = async (conn: MCPConnectionResponse) => {
    if (!isOwnerOrAdmin) return;
    const newStatus = !conn.is_enabled;
    setUpdatingConnectionId(conn.id);
    try {
      const updated = await updateMCPConnection(token, workspace.id, conn.id, {
        is_enabled: newStatus,
      });
      setConnections((prev) => prev.map((c) => (c.id === conn.id ? updated : c)));
      addLog(
        `${conn.name} ${newStatus ? 'enabled' : 'disabled'}`,
        newStatus ? 'Available to workspace agent' : 'Unavailable to agent',
        'info'
      );
      onToast?.('success', `Connection ${newStatus ? 'enabled' : 'disabled'}.`);
    } catch (err: any) {
      onToast?.('error', err.message || 'Failed to update connection status.');
    } finally {
      setUpdatingConnectionId(null);
    }
  };

  // Toggle Individual Tool in Allowlist
  const handleToggleTool = async (connId: string, toolName: string, currentEnabled: boolean) => {
    if (!isOwnerOrAdmin) return;
    const newEnabled = !currentEnabled;
    const toggleKey = `${connId}-${toolName}`;
    setTogglingTools((prev) => ({ ...prev, [toggleKey]: true }));
    try {
      const updated = await toggleMCPTool(token, workspace.id, connId, toolName, newEnabled);
      setConnections((prev) => prev.map((c) => (c.id === connId ? updated : c)));
      addLog(
        `Tool '${toolName}' ${newEnabled ? 'allowed' : 'disabled'}`,
        `Allowlist updated for connection`,
        newEnabled ? 'success' : 'info'
      );
      onToast?.(
        'success',
        `Tool '${toolName}' ${newEnabled ? 'added to allowlist' : 'disabled'}.`
      );
    } catch (err: any) {
      onToast?.('error', err.message || `Failed to toggle tool '${toolName}'.`);
    } finally {
      setTogglingTools((prev) => ({ ...prev, [toggleKey]: false }));
    }
  };

  // Test Connection
  const handleTestConnection = async (connId: string) => {
    setTestingConnectionId(connId);
    try {
      const res = await testMCPConnection(token, workspace.id, connId);
      addLog(
        `Connection test passed`,
        `${res.message} (${res.latency_ms}ms)`,
        'success'
      );
      onToast?.('success', res.message);
    } catch (err: any) {
      const msg = err.message || 'Connection test failed.';
      addLog('Connection test failed', msg, 'error');
      onToast?.('error', msg);
    } finally {
      setTestingConnectionId(null);
    }
  };

  // Open Add Modal
  const handleOpenAddModal = () => {
    setEditingConnection(null);
    setFormName('');
    setFormEndpoint('http://localhost:3000/mcp');
    setFormTransport('streamable-http');
    setFormCredMode('default');
    setFormCredRef('SERVER_TWENTY_API_KEY');
    setFormCustomCred('');
    setFormClearCustomCred(false);
    setShowCustomCredPlaintext(false);
    setFormAllowPrivate(true);
    setFormTimeout(15);
    setModalError(null);
    setShowAddEditModal(true);
  };

  // Open Edit Modal
  const handleOpenEditModal = (conn: MCPConnectionResponse) => {
    setEditingConnection(conn);
    setFormName(conn.name);
    setFormEndpoint(conn.endpoint_url);
    setFormTransport(conn.transport || 'streamable-http');
    setFormCredMode(conn.credential_mode || (conn.credential_ref ? 'env_ref' : 'default'));
    setFormCredRef(conn.credential_ref || 'SERVER_TWENTY_API_KEY');
    setFormCustomCred('');
    setFormClearCustomCred(false);
    setShowCustomCredPlaintext(false);
    setFormAllowPrivate(bool(conn.configuration?.allow_private_hosts ?? true));
    setFormTimeout(Number(conn.configuration?.timeout_seconds || 15));
    setModalError(null);
    setShowAddEditModal(true);
  };

  // Helper for boolean conversion
  function bool(v: any): boolean {
    return Boolean(v);
  }

  // Submit Add / Edit
  const handleSubmitModal = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formName.trim() || !formEndpoint.trim()) {
      setModalError('Connection name and endpoint URL are required.');
      return;
    }

    setModalLoading(true);
    setModalError(null);

    const config: Record<string, any> = {
      allow_private_hosts: formAllowPrivate,
      timeout_seconds: formTimeout,
    };

    try {
      if (editingConnection) {
        const updatePayload: MCPConnectionUpdate = {
          name: formName.trim(),
          endpoint_url: formEndpoint.trim(),
          transport: formTransport,
          credential_mode: formCredMode,
          credential_ref: formCredMode === 'env_ref' && formCredRef.trim() ? formCredRef.trim() : null,
          custom_credential: formCredMode === 'custom' && formCustomCred.trim() ? formCustomCred.trim() : undefined,
          clear_custom_credential: formClearCustomCred || (formCredMode !== 'custom' && editingConnection.has_custom_credential),
          configuration: {
            ...(editingConnection.configuration || {}),
            ...config,
          },
        };
        const updated = await updateMCPConnection(
          token,
          workspace.id,
          editingConnection.id,
          updatePayload
        );
        setConnections((prev) =>
          prev.map((c) => (c.id === editingConnection.id ? updated : c))
        );
        addLog(`Updated connection '${updated.name}'`, 'Settings saved', 'success');
        onToast?.('success', 'MCP connection settings updated.');
      } else {
        const createPayload: MCPConnectionCreate = {
          name: formName.trim(),
          endpoint_url: formEndpoint.trim(),
          transport: formTransport,
          credential_mode: formCredMode,
          credential_ref: formCredMode === 'env_ref' && formCredRef.trim() ? formCredRef.trim() : null,
          custom_credential: formCredMode === 'custom' && formCustomCred.trim() ? formCustomCred.trim() : null,
          configuration: config,
        };
        const created = await createMCPConnection(token, workspace.id, createPayload);
        setConnections((prev) => [created, ...prev]);
        setSelectedConnectionId(created.id);
        addLog(`Registered connection '${created.name}'`, 'Attempting initial sync', 'info');
        onToast?.('success', 'MCP connection registered successfully.');
        // Trigger initial sync automatically
        handleSyncConnection(created.id);
      }
      setShowAddEditModal(false);
    } catch (err: any) {
      setModalError(err.message || 'Failed to save MCP connection.');
    } finally {
      setModalLoading(false);
    }
  };

  // Confirm Delete
  const handleConfirmDelete = async () => {
    if (!deletingConnection) return;
    setModalLoading(true);
    setModalError(null);
    try {
      await deleteMCPConnection(token, workspace.id, deletingConnection.id);
      setConnections((prev) => prev.filter((c) => c.id !== deletingConnection.id));
      if (selectedConnectionId === deletingConnection.id) {
        setSelectedConnectionId(null);
      }
      addLog(
        `Disconnected ${deletingConnection.name}`,
        'Server removed from workspace',
        'warning'
      );
      onToast?.('success', `Connection '${deletingConnection.name}' removed.`);
      setShowDeleteModal(false);
      setDeletingConnection(null);
    } catch (err: any) {
      setModalError(err.message || 'Failed to delete MCP connection.');
    } finally {
      setModalLoading(false);
    }
  };

  // Test RAG Server MCP
  const handleTestRAGServer = async () => {
    setRagLoading(true);
    setRagError(null);
    try {
      const res = await getRAGServerMCPStatus(token);
      setRagStatus(res);
      addLog('Tested RAG MCP Endpoint', `${res.tools_count} tools available`, 'success');
      onToast?.('success', 'Local RAG MCP endpoint verified and ready.');
    } catch (err: any) {
      const msg = err.message || 'Failed to reach RAG MCP server.';
      setRagError(msg);
      addLog('RAG MCP Test failed', msg, 'error');
      onToast?.('error', msg);
    } finally {
      setRagLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'server' && !ragStatus && !ragLoading) {
      handleTestRAGServer();
    }
  }, [activeTab]);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '20px',
        maxWidth: '1240px',
        margin: '0 auto',
        width: '100%',
        color: '#0f172a',
      }}
    >
      {/* Top Header / Breadcrumb Card */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        <div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '12px',
              color: '#64748b',
              marginBottom: '4px',
            }}
          >
            <span>Settings</span>
            <span>›</span>
            <span>Integrations</span>
            <span>›</span>
            <span style={{ color: '#0f172a', fontWeight: 600 }}>MCP Servers</span>
          </div>
          <h1 style={{ fontSize: '22px', fontWeight: 700, margin: '0 0 4px 0', color: '#0f172a' }}>
            MCP Integrations
          </h1>
          <p style={{ margin: 0, fontSize: '13px', color: '#64748b' }}>
            Connect external tools to your workspace and expose your RAG tools to other MCP clients.
          </p>
        </div>

        {/* Workspace Indicator Badge */}
        <div
          style={{
            background: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: '10px',
            padding: '8px 14px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            boxShadow: '0 1px 2px rgba(0,0,0,0.03)',
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '10px', textTransform: 'uppercase', color: '#94a3b8', fontWeight: 700 }}>
              Active Workspace
            </span>
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#1e293b' }}>
              {workspace.name}
            </span>
          </div>
          <span
            style={{
              background: '#eff6ff',
              color: '#1d4ed8',
              fontSize: '10.5px',
              fontWeight: 600,
              padding: '2px 8px',
              borderRadius: '12px',
            }}
          >
            Scoped
          </span>
        </div>
      </div>

      {/* Read-Only Warning if Viewer */}
      {!isOwnerOrAdmin && (
        <div
          style={{
            padding: '9px 14px',
            borderRadius: '6px',
            background: '#fafafa',
            border: '1px solid #e4e4e7',
            color: '#52525b',
            fontSize: '12.5px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <LockIcon size={14} color="#71717a" />
          <span>
            <strong style={{ color: '#27272a' }}>Read-Only Mode:</strong> Only Workspace Owners or Editors can add, edit, or toggle tool permissions.
          </span>
        </div>
      )}

      {/* Main Tab Navigation Buttons */}
      <div
        style={{
          display: 'flex',
          gap: '8px',
          borderBottom: '1px solid #e4e4e7',
          paddingBottom: '10px',
        }}
      >
        <button
          type="button"
          onClick={() => setActiveTab('clients')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '7px 14px',
            borderRadius: '6px',
            border: activeTab === 'clients' ? '1px solid #18181b' : '1px solid #e4e4e7',
            background: activeTab === 'clients' ? '#f4f4f5' : '#ffffff',
            color: activeTab === 'clients' ? '#18181b' : '#71717a',
            fontWeight: activeTab === 'clients' ? 600 : 500,
            fontSize: '13px',
            cursor: 'pointer',
            transition: 'all 0.15s ease',
          }}
        >
          <PlugIcon size={14} color={activeTab === 'clients' ? '#18181b' : '#71717a'} />
          <span>Connected MCP Servers</span>
          {connections.length > 0 && (
            <span
              style={{
                background: activeTab === 'clients' ? '#e4e4e7' : '#f4f4f5',
                color: activeTab === 'clients' ? '#18181b' : '#71717a',
                padding: '1px 6px',
                borderRadius: '8px',
                fontSize: '11px',
                fontWeight: 600,
              }}
            >
              {connections.length}
            </span>
          )}
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('server')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '7px 14px',
            borderRadius: '6px',
            border: activeTab === 'server' ? '1px solid #18181b' : '1px solid #e4e4e7',
            background: activeTab === 'server' ? '#f4f4f5' : '#ffffff',
            color: activeTab === 'server' ? '#18181b' : '#71717a',
            fontWeight: activeTab === 'server' ? 600 : 500,
            fontSize: '13px',
            cursor: 'pointer',
            transition: 'all 0.15s ease',
          }}
        >
          <ServerIcon size={14} color={activeTab === 'server' ? '#18181b' : '#71717a'} />
          <span>Our MCP Server</span>
          <span
            style={{
              background: '#f0fdf4',
              color: '#15803d',
              border: '1px solid #dcfce7',
              padding: '1px 6px',
              borderRadius: '8px',
              fontSize: '10.5px',
              fontWeight: 600,
            }}
          >
            Live
          </span>
        </button>
      </div>

      {/* TAB 1: Connected MCP Servers */}
      {activeTab === 'clients' && (
        <div style={{ display: 'flex', gap: '24px', alignItems: 'flex-start', flexWrap: 'wrap' }}>
          {/* Left Column: Server Cards & Activity (width: ~58%) */}
          <div style={{ flex: '1 1 540px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Header + Add Server Button */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <div>
                <h2 style={{ fontSize: '15px', fontWeight: 700, margin: 0, color: '#0f172a' }}>
                  Connected MCP Servers
                </h2>
                <p style={{ margin: '2px 0 0 0', fontSize: '12px', color: '#64748b' }}>
                  Enabled servers are available to the agent simultaneously.
                </p>
              </div>

              {isOwnerOrAdmin && (
                <button
                  type="button"
                  onClick={handleOpenAddModal}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '7px 14px',
                    borderRadius: '6px',
                    background: '#18181b',
                    color: '#ffffff',
                    border: '1px solid #18181b',
                    fontSize: '13px',
                    fontWeight: 500,
                    cursor: 'pointer',
                  }}
                >
                  <span>+</span>
                  <span>Add MCP Server</span>
                </button>
              )}
            </div>

            {/* Error Banner */}
            {error && (
              <div
                style={{
                  padding: '10px 14px',
                  borderRadius: '6px',
                  background: '#fef2f2',
                  border: '1px solid #fecaca',
                  color: '#991b1b',
                  fontSize: '12.5px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <span>{error}</span>
                <button
                  onClick={fetchConnections}
                  style={{ background: 'none', border: 'none', color: '#dc2626', fontWeight: 600, cursor: 'pointer' }}
                >
                  Retry
                </button>
              </div>
            )}

            {/* Loading Skeleton */}
            {loading && (
              <div
                style={{
                  padding: '36px',
                  textAlign: 'center',
                  background: '#ffffff',
                  borderRadius: '8px',
                  border: '1px solid #e4e4e7',
                  color: '#71717a',
                }}
              >
                <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'center' }}>
                  <RefreshCwIcon size={20} color="#71717a" />
                </div>
                <div style={{ fontSize: '13px', fontWeight: 500 }}>Loading workspace MCP connections...</div>
              </div>
            )}

            {/* Empty State */}
            {!loading && connections.length === 0 && (
              <div
                style={{
                  background: '#ffffff',
                  border: '1px dashed #d4d4d8',
                  borderRadius: '8px',
                  padding: '36px 20px',
                  textAlign: 'center',
                }}
              >
                <div
                  style={{
                    width: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    background: '#f4f4f5',
                    border: '1px solid #e4e4e7',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto 12px auto',
                  }}
                >
                  <PlugIcon size={18} color="#71717a" />
                </div>
                <h3 style={{ fontSize: '14px', fontWeight: 600, color: '#18181b', margin: '0 0 4px 0' }}>
                  No MCP Servers Connected
                </h3>
                <p style={{ fontSize: '12px', color: '#71717a', maxWidth: '380px', margin: '0 auto 14px auto', lineHeight: 1.4 }}>
                  Connect external tools (such as Twenty CRM, Odoo, or internal tools) to empower your workspace agent with real-time external capabilities.
                </p>
                {isOwnerOrAdmin && (
                  <button
                    type="button"
                    onClick={handleOpenAddModal}
                    style={{
                      padding: '7px 14px',
                      borderRadius: '6px',
                      background: '#18181b',
                      color: '#ffffff',
                      border: '1px solid #18181b',
                      fontSize: '13px',
                      fontWeight: 500,
                      cursor: 'pointer',
                    }}
                  >
                    + Connect Your First MCP Server
                  </button>
                )}
              </div>
            )}

            {/* Connection Cards List */}
            {!loading &&
              connections.map((conn) => {
                const isSelected = conn.id === selectedConnectionId;
                const brand = getServerBrandInfo(conn.name, conn.endpoint_url);
                const isSyncing = Boolean(syncingMap[conn.id]);
                const isUpdating = updatingConnectionId === conn.id;

                const isConnected = conn.status === 'CONNECTED';
                const isError = conn.status === 'ERROR';

                return (
                  <div
                    key={conn.id}
                    onClick={() => setSelectedConnectionId(conn.id)}
                    style={{
                      background: '#ffffff',
                      border: isSelected ? '1px solid #18181b' : '1px solid #e4e4e7',
                      borderRadius: '8px',
                      padding: '16px 18px',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                      boxShadow: isSelected
                        ? '0 1px 3px rgba(0, 0, 0, 0.05)'
                        : 'none',
                    }}
                  >
                    {/* Top Row: Avatar, Title, Status Badges, Toggle */}
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        marginBottom: '10px',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        {/* Avatar */}
                        <div
                          style={{
                            width: '32px',
                            height: '32px',
                            borderRadius: '6px',
                            background: '#18181b',
                            color: '#ffffff',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontWeight: 600,
                            fontSize: '13px',
                            flexShrink: 0,
                          }}
                        >
                          {brand.avatarText}
                        </div>

                        {/* Name + Status + Transport */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                          <span style={{ fontSize: '14px', fontWeight: 600, color: '#18181b' }}>
                            {conn.name}
                          </span>

                          {/* Status Badge */}
                          <div
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '5px',
                              padding: '2px 7px',
                              borderRadius: '4px',
                              fontSize: '11px',
                              fontWeight: 500,
                              background: isConnected ? '#f0fdf4' : isError ? '#fef2f2' : '#f4f4f5',
                              color: isConnected ? '#15803d' : isError ? '#b91c1c' : '#71717a',
                              border: `1px solid ${isConnected ? '#bbf7d0' : isError ? '#fecaca' : '#e4e4e7'}`,
                            }}
                          >
                            <span
                              style={{
                                width: '5px',
                                height: '5px',
                                borderRadius: '50%',
                                background: isConnected ? '#16a34a' : isError ? '#dc2626' : '#a1a1aa',
                              }}
                            />
                            <span>{conn.status}</span>
                          </div>

                          {/* Transport Badge */}
                          <span
                            style={{
                              padding: '2px 7px',
                              borderRadius: '4px',
                              background: '#f4f4f5',
                              border: '1px solid #e4e4e7',
                              fontSize: '11px',
                              color: '#71717a',
                              fontWeight: 500,
                            }}
                          >
                            {conn.transport === 'streamable-http' ? 'Streamable HTTP' : 'SSE'}
                          </span>

                          {/* Credential Mode Badge */}
                          <span
                            style={{
                              padding: '2px 7px',
                              borderRadius: '4px',
                              background: '#f4f4f5',
                              border: '1px solid #e4e4e7',
                              fontSize: '11px',
                              color: '#52525b',
                              fontWeight: 500,
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '4px',
                            }}
                            title={
                              conn.credential_mode === 'custom' || conn.has_custom_credential
                                ? 'Encrypted Custom Secret (Fernet)'
                                : conn.credential_mode === 'env_ref'
                                ? `Environment Ref: ${conn.credential_ref}`
                                : 'Platform Default Credentials'
                            }
                          >
                            {conn.credential_mode === 'custom' || conn.has_custom_credential ? (
                              <>
                                <KeyIcon size={11} color="#52525b" />
                                <span>Custom Secret</span>
                              </>
                            ) : conn.credential_mode === 'env_ref' ? (
                              <>
                                <GlobeIcon size={11} color="#52525b" />
                                <span>{conn.credential_ref || 'Env Ref'}</span>
                              </>
                            ) : (
                              <>
                                <ShieldIcon size={11} color="#52525b" />
                                <span>Default</span>
                              </>
                            )}
                          </span>
                        </div>
                      </div>

                      {/* Enable/Disable Toggle */}
                      <div
                        onClick={(e) => {
                          e.stopPropagation();
                          handleToggleConnectionEnabled(conn);
                        }}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          cursor: isOwnerOrAdmin ? 'pointer' : 'not-allowed',
                          opacity: isUpdating ? 0.6 : 1,
                        }}
                      >
                        <div
                          style={{
                            width: '32px',
                            height: '18px',
                            borderRadius: '9px',
                            background: conn.is_enabled ? '#18181b' : '#e4e4e7',
                            position: 'relative',
                            transition: 'background 0.2s',
                          }}
                        >
                          <div
                            style={{
                              width: '14px',
                              height: '14px',
                              borderRadius: '50%',
                              background: '#ffffff',
                              position: 'absolute',
                              top: '2px',
                              left: conn.is_enabled ? '16px' : '2px',
                              transition: 'left 0.2s',
                              boxShadow: '0 1px 2px rgba(0,0,0,0.15)',
                            }}
                          />
                        </div>
                        <span style={{ fontSize: '11.5px', fontWeight: 500, color: conn.is_enabled ? '#18181b' : '#71717a' }}>
                          {conn.is_enabled ? 'Active' : 'Disabled'}
                        </span>
                      </div>
                    </div>

                    {/* Endpoint Box with Copy Button */}
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        background: '#fafafa',
                        border: '1px solid #e4e4e7',
                        borderRadius: '6px',
                        padding: '6px 10px',
                        marginBottom: '10px',
                        fontSize: '12px',
                        fontFamily: 'monospace',
                        color: '#3f3f46',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', overflow: 'hidden' }}>
                        <span style={{ color: '#a1a1aa', fontSize: '11px' }}>Endpoint:</span>
                        <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                          {conn.endpoint_url}
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCopyEndpoint(conn.endpoint_url);
                        }}
                        title="Copy endpoint"
                        style={{
                          background: 'none',
                          border: 'none',
                          color: copiedEndpoint === conn.endpoint_url ? '#16a34a' : '#71717a',
                          cursor: 'pointer',
                          fontSize: '12px',
                          padding: '2px 6px',
                        }}
                      >
                        {copiedEndpoint === conn.endpoint_url ? (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '3px', color: '#16a34a' }}>
                            <CheckIcon size={12} color="#16a34a" />
                            <span>Copied</span>
                          </span>
                        ) : (
                          <CopyIcon size={12} color="#71717a" />
                        )}
                      </button>
                    </div>

                    {/* Metadata line: Last connected & Tools count */}
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        fontSize: '11.5px',
                        color: '#71717a',
                        flexWrap: 'wrap',
                        gap: '8px',
                      }}
                    >
                      <div>
                        <span>Last connected: </span>
                        <strong style={{ color: '#27272a' }}>
                          {conn.last_synced_at
                            ? new Date(conn.last_synced_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                            : 'Never'}
                        </strong>
                        <span> • </span>
                        <span>
                          <strong style={{ color: '#27272a' }}>{conn.discovered_tools?.length || 0}</strong> tools discovered (
                          <strong style={{ color: '#18181b' }}>{conn.enabled_tools?.length || 0}</strong> allowed)
                        </span>
                      </div>

                      {/* Action buttons on card */}
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                        }}
                        onClick={(e) => e.stopPropagation()}
                      >
                        <button
                          type="button"
                          onClick={() => setSelectedConnectionId(conn.id)}
                          style={{
                            padding: '4px 9px',
                            borderRadius: '5px',
                            background: isSelected ? '#18181b' : '#ffffff',
                            color: isSelected ? '#ffffff' : '#3f3f46',
                            border: isSelected ? '1px solid #18181b' : '1px solid #e4e4e7',
                            fontSize: '12px',
                            fontWeight: 500,
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                          }}
                        >
                          <SettingsIcon size={12} color={isSelected ? '#ffffff' : '#71717a'} />
                          <span>Tools</span>
                        </button>

                        {isOwnerOrAdmin && (
                          <button
                            type="button"
                            onClick={() => handleTestConnection(conn.id)}
                            disabled={testingConnectionId === conn.id || !conn.is_enabled}
                            style={{
                              padding: '4px 9px',
                              borderRadius: '5px',
                              background: '#ffffff',
                              color: '#3f3f46',
                              border: '1px solid #e4e4e7',
                              fontSize: '12px',
                              fontWeight: 500,
                              cursor: conn.is_enabled && testingConnectionId !== conn.id ? 'pointer' : 'not-allowed',
                              opacity: !conn.is_enabled ? 0.5 : 1,
                              display: 'flex',
                              alignItems: 'center',
                              gap: '4px',
                            }}
                            title="Test connection using configured credentials"
                          >
                            {testingConnectionId === conn.id ? (
                              <RefreshCwIcon size={12} color="#71717a" />
                            ) : (
                              <ZapIcon size={12} color="#71717a" />
                            )}
                            <span>{testingConnectionId === conn.id ? 'Testing...' : 'Test'}</span>
                          </button>
                        )}

                        {isOwnerOrAdmin && (
                          <button
                            type="button"
                            onClick={() => handleOpenEditModal(conn)}
                            style={{
                              padding: '4px 8px',
                              borderRadius: '5px',
                              background: '#ffffff',
                              color: '#52525b',
                              border: '1px solid #e4e4e7',
                              fontSize: '12px',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '3px',
                            }}
                            title="Edit Connection Settings"
                          >
                            <EditIcon size={12} color="#71717a" />
                            <span>Edit</span>
                          </button>
                        )}

                        {isOwnerOrAdmin && (
                          <button
                            type="button"
                            onClick={() => {
                              setDeletingConnection(conn);
                              setShowDeleteModal(true);
                            }}
                            style={{
                              padding: '4px 8px',
                              borderRadius: '5px',
                              background: '#ffffff',
                              color: '#71717a',
                              border: '1px solid #e4e4e7',
                              fontSize: '12px',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                            }}
                            title="Disconnect Connection"
                          >
                            <TrashIcon size={12} color="#71717a" />
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Error callout if connection is in ERROR state */}
                    {conn.error_message && (
                      <div
                        style={{
                          marginTop: '10px',
                          padding: '8px 12px',
                          borderRadius: '6px',
                          background: '#fff1f2',
                          border: '1px solid #fecdd3',
                          color: '#9f1239',
                          fontSize: '11.5px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          gap: '8px',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <AlertCircleIcon size={13} color="#9f1239" />
                          <span>{conn.error_message}</span>
                        </div>
                        {isOwnerOrAdmin && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSyncConnection(conn.id);
                            }}
                            style={{
                              background: 'none',
                              border: 'none',
                              color: '#be123c',
                              fontWeight: 600,
                              fontSize: '11px',
                              cursor: 'pointer',
                            }}
                          >
                            Retry Sync
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}

            {/* Connection Activity Card */}
            <div
              style={{
                background: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '12px',
                padding: '16px 20px',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginBottom: '12px',
                }}
              >
                <h3 style={{ fontSize: '14px', fontWeight: 700, margin: 0, color: '#0f172a' }}>
                  Connection Activity
                </h3>
                <span style={{ fontSize: '11px', color: '#64748b' }}>Live log</span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {activityLogs.map((log) => {
                  const dotColor =
                    log.status === 'success'
                      ? '#16a34a'
                      : log.status === 'error'
                      ? '#dc2626'
                      : log.status === 'warning'
                      ? '#d97706'
                      : '#3b82f6';
                  return (
                    <div
                      key={log.id}
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '10px',
                        fontSize: '12px',
                      }}
                    >
                      <span
                        style={{
                          width: '7px',
                          height: '7px',
                          borderRadius: '50%',
                          background: dotColor,
                          marginTop: '5px',
                          flexShrink: 0,
                        }}
                      />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
                          <span style={{ fontWeight: 600, color: '#1e293b' }}>{log.title}</span>
                          <span style={{ fontSize: '11px', color: '#94a3b8', flexShrink: 0 }}>
                            {log.timestamp}
                          </span>
                        </div>
                        <div style={{ fontSize: '11.5px', color: '#64748b', marginTop: '1px' }}>
                          {log.detail}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Right Column: Tools Table & Discovered Allowlist (width: ~40%) */}
          <div style={{ flex: '1 1 420px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {selectedConnection ? (
              <div
                style={{
                  background: '#ffffff',
                  border: '1px solid #e2e8f0',
                  borderRadius: '12px',
                  padding: '20px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.02)',
                }}
              >
                {/* Header: Name + Status + Test Button */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: '6px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <h3 style={{ fontSize: '16px', fontWeight: 700, margin: 0, color: '#0f172a' }}>
                      {selectedConnection.name} — Tools
                    </h3>
                    <span
                      style={{
                        fontSize: '11px',
                        fontWeight: 600,
                        padding: '2px 7px',
                        borderRadius: '10px',
                        background:
                          selectedConnection.status === 'CONNECTED' ? '#dcfce7' : '#f1f5f9',
                        color:
                          selectedConnection.status === 'CONNECTED' ? '#15803d' : '#475569',
                      }}
                    >
                      {selectedConnection.status}
                    </span>
                  </div>

                  {isOwnerOrAdmin && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <button
                        type="button"
                        onClick={() => handleTestConnection(selectedConnection.id)}
                        disabled={testingConnectionId === selectedConnection.id || !selectedConnection.is_enabled}
                        style={{
                          padding: '5px 11px',
                          borderRadius: '6px',
                          background: '#ffffff',
                          border: '1px solid #e4e4e7',
                          fontSize: '12px',
                          fontWeight: 500,
                          color: '#3f3f46',
                          cursor:
                            selectedConnection.is_enabled && testingConnectionId !== selectedConnection.id
                              ? 'pointer'
                              : 'not-allowed',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '5px',
                        }}
                        title="Probe endpoint with configured credentials and measure latency"
                      >
                        {testingConnectionId === selectedConnection.id ? (
                          <RefreshCwIcon size={12} color="#71717a" />
                        ) : (
                          <ZapIcon size={12} color="#71717a" />
                        )}
                        <span>{testingConnectionId === selectedConnection.id ? 'Testing...' : 'Test Connection'}</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => handleSyncConnection(selectedConnection.id)}
                        disabled={Boolean(syncingMap[selectedConnection.id]) || !selectedConnection.is_enabled}
                        style={{
                          padding: '5px 11px',
                          borderRadius: '6px',
                          background: '#ffffff',
                          border: '1px solid #e4e4e7',
                          fontSize: '12px',
                          fontWeight: 500,
                          color: '#3f3f46',
                          cursor: selectedConnection.is_enabled ? 'pointer' : 'not-allowed',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '5px',
                        }}
                      >
                        {syncingMap[selectedConnection.id] ? (
                          <RefreshCwIcon size={12} color="#71717a" />
                        ) : (
                          <PlayIcon size={10} color="#71717a" />
                        )}
                        <span>{syncingMap[selectedConnection.id] ? 'Syncing...' : 'Sync Tools'}</span>
                      </button>
                    </div>
                  )}
                </div>

                <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '14px' }}>
                  Select which tools are available to the agent in this workspace.
                </div>

                {/* Discovered Tools List / Table */}
                {selectedConnection.discovered_tools && selectedConnection.discovered_tools.length > 0 ? (
                  <div
                    style={{
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      overflow: 'hidden',
                      marginBottom: '16px',
                    }}
                  >
                    {/* Table Header */}
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '1fr 65px 50px',
                        padding: '8px 12px',
                        background: '#f8fafc',
                        borderBottom: '1px solid #e2e8f0',
                        fontSize: '11px',
                        fontWeight: 700,
                        color: '#64748b',
                        textTransform: 'uppercase',
                        letterSpacing: '0.04em',
                      }}
                    >
                      <span>Tool & Description</span>
                      <span style={{ textAlign: 'center' }}>Type</span>
                      <span style={{ textAlign: 'right' }}>Allow</span>
                    </div>

                    {/* Rows */}
                    {selectedConnection.discovered_tools.map((t) => {
                      const isAllowed = selectedConnection.enabled_tools?.includes(t.name);
                      const risk = getToolRiskClassification(t.name);
                      const isToggling = Boolean(togglingTools[`${selectedConnection.id}-${t.name}`]);

                      return (
                        <div
                          key={t.name}
                          style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 65px 50px',
                            padding: '10px 12px',
                            alignItems: 'center',
                            borderBottom: '1px solid #f1f5f9',
                            background: isAllowed ? '#ffffff' : '#fafafa',
                            transition: 'background 0.15s',
                          }}
                        >
                          {/* Name + Description */}
                          <div style={{ paddingRight: '8px', minWidth: 0 }}>
                            <div
                              style={{
                                fontSize: '13px',
                                fontWeight: 600,
                                color: isAllowed ? '#4f6492ff' : '#64748b',
                                fontFamily: 'monospace',
                              }}
                            >
                              {t.name}
                            </div>
                            <div
                              style={{
                                fontSize: '11.5px',
                                color: '#64748b',
                                lineHeight: 1.3,
                                marginTop: '2px',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                              }}
                              title={t.description || ''}
                            >
                              {t.description || 'No description provided.'}
                            </div>
                          </div>

                          {/* Risk Classification Badge */}
                          <div style={{ textAlign: 'center' }}>
                            <span
                              style={{
                                fontSize: '10px',
                                fontWeight: 700,
                                padding: '2px 7px',
                                borderRadius: '4px',
                                background: risk.badgeBg,
                                color: risk.badgeColor,
                                border: `1px solid ${risk.border}`,
                                textTransform: 'uppercase',
                                cursor: 'help',
                              }}
                              title={risk.notice || `${risk.label} capability`}
                            >
                              {risk.label}
                            </span>
                          </div>

                          {/* Toggle Switch */}
                          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                            <div
                              onClick={() => handleToggleTool(selectedConnection.id, t.name, isAllowed)}
                              style={{
                                width: '32px',
                                height: '18px',
                                borderRadius: '9px',
                                background: isAllowed ? '#73947fff' : '#cbd5e1',
                                position: 'relative',
                                cursor: isOwnerOrAdmin && !isToggling ? 'pointer' : 'not-allowed',
                                opacity: isToggling ? 0.5 : 1,
                                transition: 'background 0.2s',
                              }}
                              title={
                                isOwnerOrAdmin
                                  ? isAllowed
                                    ? `Click to revoke '${t.name}'`
                                    : `Click to permit '${t.name}'`
                                  : 'Only Workspace Owners or Editors can toggle tools.'
                              }
                            >
                              <div
                                style={{
                                  width: '14px',
                                  height: '14px',
                                  borderRadius: '50%',
                                  background: '#ffffff',
                                  position: 'absolute',
                                  top: '2px',
                                  left: isAllowed ? '16px' : '2px',
                                  transition: 'left 0.2s',
                                  boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div
                    style={{
                      padding: '30px 16px',
                      background: '#fafafa',
                      borderRadius: '8px',
                      border: '1px dashed #e4e4e7',
                      textAlign: 'center',
                      marginBottom: '16px',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '8px' }}>
                      <PlugIcon size={20} color="#71717a" />
                    </div>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: '#18181b' }}>
                      No Tools Discovered Yet
                    </div>
                    <p style={{ fontSize: '12px', color: '#71717a', maxWidth: '300px', margin: '4px auto 12px auto' }}>
                      Click Sync Tools above to connect to this server and discover its available capabilities.
                    </p>
                    {isOwnerOrAdmin && (
                      <button
                        type="button"
                        onClick={() => handleSyncConnection(selectedConnection.id)}
                        disabled={Boolean(syncingMap[selectedConnection.id])}
                        style={{
                          padding: '6px 14px',
                          borderRadius: '6px',
                          background: '#18181b',
                          color: '#ffffff',
                          border: '1px solid #18181b',
                          fontSize: '12px',
                          fontWeight: 500,
                          cursor: 'pointer',
                        }}
                      >
                        Run Tool Discovery
                      </button>
                    )}
                  </div>
                )}

                {/* Security Note at bottom */}
                <div
                  style={{
                    padding: '10px 14px',
                    borderRadius: '6px',
                    background: '#fafafa',
                    border: '1px solid #e4e4e7',
                    fontSize: '12px',
                    color: '#52525b',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '8px',
                    lineHeight: 1.4,
                  }}
                >
                  <ShieldIcon size={14} color="#71717a" style={{ marginTop: '2px', flexShrink: 0 }} />
                  <span>
                    <strong style={{ color: '#18181b' }}>Default-Deny Policy:</strong> Only enabled tools are accessible to the agent in this workspace. Write tools are gated by cryptographic human approval.
                  </span>
                </div>
              </div>
            ) : (
              <div
                style={{
                  background: '#ffffff',
                  border: '1px dashed #e4e4e7',
                  borderRadius: '8px',
                  padding: '40px 24px',
                  textAlign: 'center',
                  color: '#71717a',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '8px' }}>
                  <ServerIcon size={24} color="#a1a1aa" />
                </div>
                <h4 style={{ fontSize: '14px', fontWeight: 600, margin: '0 0 4px 0', color: '#18181b' }}>
                  Select a Server
                </h4>
                <p style={{ fontSize: '12.5px', margin: 0 }}>
                  Select a connected MCP server on the left to review its discovered tools and configure its allowlist.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: Our MCP Server (Expose Tools) */}
      {activeTab === 'server' && (
        <div
          style={{
            background: '#ffffff',
            border: '1px solid #e4e4e7',
            borderRadius: '8px',
            padding: '24px',
            maxWidth: '820px',
          }}
        >
          {/* Header */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '8px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h2 style={{ fontSize: '16px', fontWeight: 600, margin: 0, color: '#18181b' }}>
                Expose our RAG via MCP
              </h2>
              <span
                style={{
                  fontSize: '11px',
                  fontWeight: 500,
                  padding: '2px 8px',
                  borderRadius: '4px',
                  background: '#f0fdf4',
                  color: '#15803d',
                  border: '1px solid #bbf7d0',
                }}
              >
                Configured
              </span>
            </div>

            <button
              type="button"
              onClick={handleTestRAGServer}
              disabled={ragLoading}
              style={{
                padding: '6px 14px',
                borderRadius: '6px',
                background: '#ffffff',
                border: '1px solid #e4e4e7',
                fontSize: '12.5px',
                fontWeight: 500,
                color: '#18181b',
                cursor: ragLoading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              {ragLoading ? <RefreshCwIcon size={12} color="#71717a" /> : <PlayIcon size={10} color="#71717a" />}
              <span>{ragLoading ? 'Testing...' : 'Test Endpoint'}</span>
            </button>
          </div>

          <p style={{ fontSize: '13px', color: '#71717a', margin: '0 0 16px 0', lineHeight: 1.4 }}>
            Expose selected RAG tools so that external MCP clients (e.g. Claude Desktop, IDEs, Cursor, or other AI agents) can use them.
          </p>

          {/* Endpoint Box with Copy */}
          <div
            style={{
              background: '#fafafa',
              border: '1px solid #e4e4e7',
              borderRadius: '6px',
              padding: '12px 16px',
              marginBottom: '16px',
            }}
          >
            <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#71717a', fontWeight: 600, marginBottom: '4px' }}>
              Server SSE Endpoint URL
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
              <code style={{ fontSize: '13px', color: '#18181b', fontWeight: 500 }}>
                {ragStatus?.local_base_url || 'http://localhost:8000/mcp/sse'}
              </code>
              <button
                type="button"
                onClick={() =>
                  handleCopyEndpoint(ragStatus?.local_base_url || 'http://localhost:8000/mcp/sse')
                }
                style={{
                  background: '#ffffff',
                  border: '1px solid #e4e4e7',
                  borderRadius: '5px',
                  padding: '4px 10px',
                  fontSize: '12px',
                  fontWeight: 500,
                  color: copiedEndpoint ? '#16a34a' : '#18181b',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                {copiedEndpoint ? (
                  <>
                    <CheckIcon size={12} color="#16a34a" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <CopyIcon size={12} color="#71717a" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Status Alert */}
          <div
            style={{
              padding: '10px 14px',
              borderRadius: '6px',
              background: ragError ? '#fef2f2' : '#f0fdf4',
              border: ragError ? '1px solid #fecaca' : '1px solid #bbf7d0',
              color: ragError ? '#991b1b' : '#166534',
              fontSize: '12.5px',
              marginBottom: '20px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            {ragError ? <AlertCircleIcon size={14} color="#991b1b" /> : <CheckIcon size={14} color="#166534" />}
            <span>
              {ragError
                ? `Endpoint Test Failed: ${ragError}`
                : `Server endpoint verified. ${ragStatus?.tools_count || 5} tools available for MCP clients.`}
            </span>
          </div>

          {/* Built-in RAG Tools */}
          <h4 style={{ fontSize: '13.5px', fontWeight: 600, margin: '0 0 10px 0', color: '#18181b' }}>
            Built-In Tools Exposed to MCP
          </h4>

          <div style={{ border: '1px solid #e4e4e7', borderRadius: '6px', overflow: 'hidden' }}>
            {[
              {
                name: 'rag_ask',
                desc: 'Ask an authoritative question with verified citations from your RAG corpus.',
                type: 'Read',
                enabled: true,
              },
              {
                name: 'rag_search',
                desc: 'Perform hybrid dense vector and keyword full-text search with RRF scoring.',
                type: 'Read',
                enabled: true,
              },
              {
                name: 'list_documents',
                desc: 'Inspect accessible documents, chunks, and metadata within caller workspace.',
                type: 'Read',
                enabled: true,
              },
              {
                name: 'ingest_document',
                desc: 'Upload and parse a new document into workspace knowledge index.',
                type: 'Write',
                enabled: false,
              },
              {
                name: 'run_diagnostics',
                desc: 'Inspect vector database health, tokenizer metrics, and index integrity.',
                type: 'Write',
                enabled: false,
              },
            ].map((tool, idx) => (
              <div
                key={tool.name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 14px',
                  borderBottom: idx < 4 ? '1px solid #f4f4f5' : 'none',
                  background: tool.enabled ? '#ffffff' : '#fafafa',
                }}
              >
                <div>
                  <div style={{ fontSize: '13px', fontWeight: 600, color: '#18181b', fontFamily: 'monospace' }}>
                    {tool.name}
                  </div>
                  <div style={{ fontSize: '12px', color: '#71717a', marginTop: '2px' }}>
                    {tool.desc}
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span
                    style={{
                      fontSize: '11px',
                      fontWeight: 500,
                      padding: '2px 7px',
                      borderRadius: '4px',
                      background: '#f4f4f5',
                      color: '#52525b',
                      border: '1px solid #e4e4e7',
                    }}
                  >
                    {tool.type}
                  </span>

                  <span
                    style={{
                      fontSize: '12px',
                      fontWeight: 500,
                      color: tool.enabled ? '#15803d' : '#a1a1aa',
                    }}
                  >
                    {tool.enabled ? 'Enabled' : 'Restricted'}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div
            style={{
              marginTop: '16px',
              fontSize: '12px',
              color: '#71717a',
              lineHeight: 1.4,
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <LockIcon size={12} color="#71717a" />
            <span>Write/side-effect tools are gated by admin approval and workspace department permissions.</span>
          </div>
        </div>
      )}

      {/* MODAL 1: Add / Edit Connection Modal */}
      {showAddEditModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15, 23, 42, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 999,
            padding: '20px',
          }}
        >
          <div
            style={{
              background: '#ffffff',
              borderRadius: '8px',
              maxWidth: '540px',
              width: '100%',
              boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.05)',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              border: '1px solid #e4e4e7',
            }}
          >
            {/* Modal Header */}
            <div
              style={{
                padding: '14px 18px',
                borderBottom: '1px solid #e4e4e7',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <h3 style={{ fontSize: '15px', fontWeight: 600, margin: 0, color: '#18181b' }}>
                {editingConnection ? 'Edit MCP Server' : 'Add External MCP Server'}
              </h3>
              <button
                type="button"
                onClick={() => setShowAddEditModal(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  padding: '4px',
                  color: '#71717a',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: '4px',
                }}
              >
                <XIcon size={16} color="#71717a" />
              </button>
            </div>

            {/* Modal Body / Form */}
            <form onSubmit={handleSubmitModal} style={{ padding: '18px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
              {modalError && (
                <div
                  style={{
                    padding: '8px 12px',
                    borderRadius: '6px',
                    background: '#fef2f2',
                    border: '1px solid #fecaca',
                    color: '#991b1b',
                    fontSize: '12px',
                  }}
                >
                  {modalError}
                </div>
              )}

              {/* Server Name */}
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '5px' }}>
                  Server Display Name
                </label>
                <input
                  type="text"
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                  placeholder="e.g. Twenty CRM, Odoo MCP, Custom Tools"
                  required
                  style={{
                    width: '100%',
                    padding: '8px 11px',
                    borderRadius: '6px',
                    border: '1px solid #e4e4e7',
                    fontSize: '13px',
                    color: '#18181b',
                    boxSizing: 'border-box',
                    background: '#ffffff',
                  }}
                />
              </div>

              {/* Endpoint URL */}
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '5px' }}>
                  Endpoint URL
                </label>
                <input
                  type="url"
                  value={formEndpoint}
                  onChange={(e) => setFormEndpoint(e.target.value)}
                  placeholder="http://localhost:3000/mcp or https://mcp.example.com/mcp"
                  required
                  style={{
                    width: '100%',
                    padding: '8px 11px',
                    borderRadius: '6px',
                    border: '1px solid #e4e4e7',
                    fontSize: '13px',
                    fontFamily: 'monospace',
                    color: '#18181b',
                    boxSizing: 'border-box',
                    background: '#ffffff',
                  }}
                />
              </div>

              {/* Transport Type */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '5px' }}>
                    Transport Protocol
                  </label>
                  <select
                    value={formTransport}
                    onChange={(e) => setFormTransport(e.target.value as 'streamable-http' | 'sse')}
                    style={{
                      width: '100%',
                      padding: '8px 10px',
                      borderRadius: '6px',
                      border: '1px solid #e4e4e7',
                      fontSize: '13px',
                      background: '#ffffff',
                      color: '#18181b',
                    }}
                  >
                    <option value="streamable-http">Streamable HTTP (Recommended)</option>
                    <option value="sse">Server-Sent Events (SSE)</option>
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#52525b', marginBottom: '5px' }}>
                    Request Timeout (seconds)
                  </label>
                  <input
                    type="number"
                    min={5}
                    max={60}
                    value={formTimeout}
                    onChange={(e) => setFormTimeout(Number(e.target.value))}
                    style={{
                      width: '100%',
                      padding: '8px 10px',
                      borderRadius: '6px',
                      border: '1px solid #e4e4e7',
                      fontSize: '13px',
                      color: '#18181b',
                      boxSizing: 'border-box',
                      background: '#ffffff',
                    }}
                  />
                </div>
              </div>

              {/* Credential Mode & Configuration */}
              <div
                style={{
                  border: '1px solid #e4e4e7',
                  borderRadius: '6px',
                  padding: '12px',
                  background: '#fafafa',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '10px',
                }}
              >
                <div>
                  <label
                    style={{
                      display: 'block',
                      fontSize: '12px',
                      fontWeight: 500,
                      color: '#27272a',
                      marginBottom: '6px',
                    }}
                  >
                    Authentication / Credential Source
                  </label>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    {[
                      { id: 'default', label: 'Platform Default', desc: 'Pre-configured credentials', icon: ShieldIcon },
                      { id: 'custom', label: 'Workspace Custom Key', desc: 'Fernet encrypted at rest', icon: KeyIcon },
                      { id: 'env_ref', label: 'Server Env Reference', desc: 'Server-managed variable', icon: GlobeIcon },
                    ].map((modeOption) => {
                      const isSelected = formCredMode === modeOption.id;
                      const IconComponent = modeOption.icon;
                      return (
                        <button
                          key={modeOption.id}
                          type="button"
                          onClick={() => {
                            setFormCredMode(modeOption.id as MCPCredentialMode);
                            if (modeOption.id !== 'custom') {
                              setFormCustomCred('');
                            }
                          }}
                          style={{
                            flex: '1 1 140px',
                            padding: '9px 11px',
                            borderRadius: '6px',
                            border: isSelected ? '1.5px solid #18181b' : '1px solid #e4e4e7',
                            background: isSelected ? '#f4f4f5' : '#ffffff',
                            color: isSelected ? '#18181b' : '#3f3f46',
                            textAlign: 'left',
                            cursor: 'pointer',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '3px',
                            transition: 'all 0.15s ease',
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <IconComponent size={12} color={isSelected ? '#18181b' : '#71717a'} />
                            <span style={{ fontSize: '12px', fontWeight: 600 }}>{modeOption.label}</span>
                          </div>
                          <span style={{ fontSize: '10.5px', color: '#71717a', paddingLeft: '18px' }}>
                            {modeOption.desc}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Mode: Default */}
                {formCredMode === 'default' && (
                  <div
                    style={{
                      fontSize: '12px',
                      color: '#52525b',
                      background: '#ffffff',
                      padding: '9px 11px',
                      borderRadius: '6px',
                      border: '1px solid #e4e4e7',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '8px',
                    }}
                  >
                    <ShieldIcon size={14} color="#71717a" style={{ marginTop: '1px', flexShrink: 0 }} />
                    <span>
                      <strong style={{ color: '#18181b' }}>Platform-Managed Credentials:</strong> Uses server-level platform credentials (such as platform Twenty CRM environment variables). No custom keys required.
                    </span>
                  </div>
                )}

                {/* Mode: Custom */}
                {formCredMode === 'custom' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <label style={{ fontSize: '12px', fontWeight: 500, color: '#52525b' }}>
                        Workspace Custom API Key
                      </label>
                      {editingConnection?.has_custom_credential && !formClearCustomCred && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span
                            style={{
                              fontSize: '11px',
                              fontWeight: 500,
                              color: '#15803d',
                              background: '#f0fdf4',
                              border: '1px solid #dcfce7',
                              padding: '2px 6px',
                              borderRadius: '4px',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '4px',
                            }}
                          >
                            <span style={{ width: '5px', height: '5px', borderRadius: '50%', background: '#16a34a' }} />
                            <span>Configured (Encrypted)</span>
                          </span>
                          <button
                            type="button"
                            onClick={() => {
                              setFormClearCustomCred(true);
                              setFormCustomCred('');
                            }}
                            style={{
                              fontSize: '11px',
                              color: '#dc2626',
                              background: 'none',
                              border: 'none',
                              textDecoration: 'underline',
                              cursor: 'pointer',
                              padding: '0',
                            }}
                          >
                            Clear Stored Secret
                          </button>
                        </div>
                      )}
                      {formClearCustomCred && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span
                            style={{
                              fontSize: '11px',
                              fontWeight: 500,
                              color: '#b45309',
                              background: '#fef3c7',
                              padding: '2px 6px',
                              borderRadius: '4px',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '4px',
                            }}
                          >
                            <AlertCircleIcon size={11} color="#b45309" />
                            <span>Secret will be cleared on save</span>
                          </span>
                          <button
                            type="button"
                            onClick={() => setFormClearCustomCred(false)}
                            style={{
                              fontSize: '11px',
                              color: '#18181b',
                              background: 'none',
                              border: 'none',
                              textDecoration: 'underline',
                              cursor: 'pointer',
                              padding: '0',
                            }}
                          >
                            Undo
                          </button>
                        </div>
                      )}
                    </div>

                    <div style={{ position: 'relative' }}>
                      <input
                        type={showCustomCredPlaintext ? 'text' : 'password'}
                        value={formCustomCred}
                        onChange={(e) => {
                          setFormCustomCred(e.target.value);
                          if (formClearCustomCred) setFormClearCustomCred(false);
                        }}
                        placeholder={
                          editingConnection?.has_custom_credential && !formClearCustomCred
                            ? '•••••••••••••••• (Leave blank to keep existing encrypted secret)'
                            : 'Enter API key or bearer token...'
                        }
                        autoComplete="new-password"
                        style={{
                          width: '100%',
                          padding: '8px 36px 8px 10px',
                          borderRadius: '6px',
                          border: '1px solid #e4e4e7',
                          fontSize: '13px',
                          boxSizing: 'border-box',
                          background: '#ffffff',
                          color: '#18181b',
                          fontFamily: showCustomCredPlaintext ? 'monospace' : 'inherit',
                        }}
                      />
                      <button
                        type="button"
                        onClick={() => setShowCustomCredPlaintext((prev) => !prev)}
                        style={{
                          position: 'absolute',
                          right: '8px',
                          top: '50%',
                          transform: 'translateY(-50%)',
                          background: 'none',
                          border: 'none',
                          cursor: 'pointer',
                          color: '#71717a',
                          padding: '3px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                        title={showCustomCredPlaintext ? 'Hide key' : 'Show key'}
                      >
                        {showCustomCredPlaintext ? <EyeOffIcon size={14} color="#71717a" /> : <EyeIcon size={14} color="#71717a" />}
                      </button>
                    </div>

                    <div style={{ fontSize: '11px', color: '#71717a', lineHeight: 1.4, display: 'flex', alignItems: 'flex-start', gap: '6px' }}>
                      <LockIcon size={12} color="#71717a" style={{ marginTop: '2px', flexShrink: 0 }} />
                      <span>
                        <strong style={{ color: '#27272a' }}>Multi-Tenant Encryption:</strong> Encrypted at rest using Fernet (AES-128-CBC + HMAC-SHA256) with zero-downtime key rotation. Secrets are never stored in browser storage, logged, or returned in public API payloads.
                      </span>
                    </div>
                  </div>
                )}

                {/* Mode: Server Environment Variable Reference */}
                {formCredMode === 'env_ref' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    <label style={{ fontSize: '12px', fontWeight: 500, color: '#52525b' }}>
                      Allowlisted Server Environment Reference
                    </label>
                    <select
                      value={formCredRef}
                      onChange={(e) => setFormCredRef(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px 10px',
                        borderRadius: '6px',
                        border: '1px solid #e4e4e7',
                        fontSize: '13px',
                        background: '#ffffff',
                        color: '#18181b',
                      }}
                    >
                      {SERVER_MANAGED_CREDENTIAL_PROFILES.map((p) => (
                        <option key={p.value || 'none'} value={p.value || ''}>
                          {p.label}
                        </option>
                      ))}
                    </select>
                    <div style={{ fontSize: '11px', color: '#92400e', background: '#fefce8', padding: '7px 10px', borderRadius: '6px', border: '1px solid #fef08a', display: 'flex', alignItems: 'flex-start', gap: '6px' }}>
                      <AlertCircleIcon size={13} color="#b45309" style={{ marginTop: '1px', flexShrink: 0 }} />
                      <span>
                        <strong style={{ color: '#78350f' }}>Server Reference:</strong> References an environment variable present on the backend server that must be in the server allowlist.
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Allow Private Hosts Toggle */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  background: '#fafafa',
                  border: '1px solid #e4e4e7',
                  borderRadius: '6px',
                  padding: '8px 12px',
                }}
              >
                <input
                  type="checkbox"
                  id="allowPrivateHosts"
                  checked={formAllowPrivate}
                  onChange={(e) => setFormAllowPrivate(e.target.checked)}
                />
                <label htmlFor="allowPrivateHosts" style={{ fontSize: '12px', color: '#52525b', cursor: 'pointer' }}>
                  Allow local loopback hosts (<code>localhost</code> / <code>127.0.0.1</code>) for <strong>development & testing</strong> (strictly rejected in production by server policy)
                </label>
              </div>

              {/* Modal Actions */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'flex-end',
                  gap: '8px',
                  marginTop: '8px',
                  borderTop: '1px solid #e4e4e7',
                  paddingTop: '14px',
                }}
              >
                <button
                  type="button"
                  onClick={() => setShowAddEditModal(false)}
                  style={{
                    padding: '7px 14px',
                    borderRadius: '6px',
                    border: '1px solid #e4e4e7',
                    background: '#ffffff',
                    color: '#3f3f46',
                    fontSize: '13px',
                    fontWeight: 500,
                    cursor: 'pointer',
                  }}
                >
                  Cancel
                </button>

                <button
                  type="submit"
                  disabled={modalLoading}
                  style={{
                    padding: '7px 16px',
                    borderRadius: '6px',
                    background: '#18181b',
                    color: '#ffffff',
                    border: '1px solid #18181b',
                    fontSize: '13px',
                    fontWeight: 500,
                    cursor: modalLoading ? 'not-allowed' : 'pointer',
                    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
                  }}
                >
                  {modalLoading ? 'Saving...' : editingConnection ? 'Update Server' : 'Connect Server'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* MODAL 2: Disconnect Confirmation Modal */}
      {showDeleteModal && deletingConnection && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15, 23, 42, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 999,
            padding: '20px',
          }}
        >
          <div
            style={{
              background: '#ffffff',
              borderRadius: '8px',
              maxWidth: '440px',
              width: '100%',
              boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)',
              overflow: 'hidden',
              padding: '22px',
              border: '1px solid #e4e4e7',
            }}
          >
            <div style={{ marginBottom: '10px' }}>
              <AlertCircleIcon size={22} color="#dc2626" />
            </div>
            <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#18181b', margin: '0 0 6px 0' }}>
              Disconnect {deletingConnection.name}?
            </h3>
            <p style={{ fontSize: '12.5px', color: '#71717a', margin: '0 0 16px 0', lineHeight: 1.4 }}>
              Are you sure you want to disconnect this MCP server from <strong>{workspace.name}</strong>? All discovered tools from this server will immediately become unavailable to the workspace agent.
            </p>

            {modalError && (
              <div
                style={{
                  padding: '8px 12px',
                  borderRadius: '6px',
                  background: '#fef2f2',
                  border: '1px solid #fecaca',
                  color: '#991b1b',
                  fontSize: '12px',
                  marginBottom: '14px',
                }}
              >
                {modalError}
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
              <button
                type="button"
                onClick={() => {
                  setShowDeleteModal(false);
                  setDeletingConnection(null);
                }}
                style={{
                  padding: '7px 14px',
                  borderRadius: '6px',
                  border: '1px solid #e4e4e7',
                  background: '#ffffff',
                  fontSize: '13px',
                  fontWeight: 500,
                  color: '#3f3f46',
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>

              <button
                type="button"
                onClick={handleConfirmDelete}
                disabled={modalLoading}
                style={{
                  padding: '7px 16px',
                  borderRadius: '6px',
                  background: '#dc2626',
                  color: '#ffffff',
                  border: '1px solid #dc2626',
                  fontSize: '13px',
                  fontWeight: 500,
                  cursor: modalLoading ? 'not-allowed' : 'pointer',
                }}
              >
                {modalLoading ? 'Disconnecting...' : 'Disconnect Server'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
