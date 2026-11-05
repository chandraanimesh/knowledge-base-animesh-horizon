import React, { useState } from 'react';
import { ACTION_CATALOG, TRIGGER_CATALOG, StudioNodeData } from './types.ts';
import {
  WorkflowTriggerType,
  WorkflowTriggerSpec,
} from '../../api/workflowDefinitions.ts';
import type { WorkflowRunResponse, WorkflowTaskResponse } from '../../api/workflows.ts';
import { isValidNodeId } from './dagUtils.ts';
import {
  formatDuration,
  formatMetric,
  sanitizePayload,
  getExecutionStatusBadge,
} from './executionUtils.ts';

interface NodeConfigPanelProps {
  selectedNode: any | null;
  triggerSpec: WorkflowTriggerSpec;
  allNodes: any[];
  onUpdateNodeData: (nodeId: string, updatedData: Partial<StudioNodeData>) => void;
  onUpdateTriggerSpec: (updatedSpec: WorkflowTriggerSpec) => void;
  onDeleteNode: (nodeId: string) => void;
  onClose: () => void;
  isReadOnly?: boolean;
  isInspectionMode?: boolean;
  activeRun?: WorkflowRunResponse | null;
  onOpenApprovalModal?: () => void;
}

export const NodeConfigPanel: React.FC<NodeConfigPanelProps> = ({
  selectedNode,
  triggerSpec,
  allNodes,
  onUpdateNodeData,
  onUpdateTriggerSpec,
  onDeleteNode,
  onClose,
  isReadOnly = false,
  isInspectionMode = false,
  activeRun = null,
  onOpenApprovalModal,
}) => {
  const [activeTab, setActiveTab] = useState<'execution' | 'config'>('execution');

  if (!selectedNode) {
    return (
      <aside className="w-80 bg-[#fafaf7] border-l border-[#d4d4cc] p-4 flex flex-col justify-center items-center text-center text-slate-400 select-none h-full shrink-0 min-h-0">
        <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-xl mb-2 text-slate-300">
          ⚙
        </div>
        <p className="text-xs font-semibold text-slate-600">No Component Selected</p>
        <p className="text-[11px] text-slate-400 mt-1 max-w-[200px]">
          {isInspectionMode
            ? 'Click on any step or trigger on the canvas to inspect its live execution metrics, inputs, outputs, and errors.'
            : 'Click on any trigger or action node on the canvas to inspect its configuration and variable bindings.'}
        </p>
      </aside>
    );
  }

  const isTrigger = Boolean(selectedNode.data?.isTrigger);

  // -------------------------------------------------------------------------
  // TRIGGER NODE INSPECTION & CONFIGURATION
  // -------------------------------------------------------------------------
  if (isTrigger) {
    const triggerType = (selectedNode.data?.triggerType || triggerSpec.trigger_type) as WorkflowTriggerType;
    const triggerMeta = TRIGGER_CATALOG[triggerType] || TRIGGER_CATALOG.MANUAL;
    const config = selectedNode.data?.triggerConfig || triggerSpec.config || {};

    const handleConfigChange = (key: string, value: any) => {
      if (isReadOnly || isInspectionMode) return;
      onUpdateTriggerSpec({
        ...triggerSpec,
        config: {
          ...config,
          [key]: value,
        },
      });
    };

    const handleTypeChange = (newType: WorkflowTriggerType) => {
      if (isReadOnly || isInspectionMode) return;
      onUpdateTriggerSpec({
        trigger_type: newType,
        config: TRIGGER_CATALOG[newType].defaultConfig,
      });
    };

    return (
      <aside className="w-80 bg-[#fafaf7] border-l border-[#d4d4cc] flex flex-col h-full overflow-y-auto shrink-0 min-h-0 overscroll-contain">
        {/* Header */}
        <div className="p-4 border-b border-[#d4d4cc] bg-[#f0f0eb] flex items-center justify-between">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-blue-700 bg-blue-100 px-2 py-0.5 rounded">
              {isInspectionMode ? 'Execution Trigger' : 'Trigger Configuration'}
            </span>
            <h3 className="text-sm font-bold text-slate-800 mt-1">{triggerMeta.name}</h3>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 text-sm p-1 rounded"
          >
            ✕
          </button>
        </div>

        <div className="p-4 space-y-4 flex-1">
          {/* Inspection Mode Trigger Banner */}
          {isInspectionMode && activeRun && (
            <div className="space-y-3">
              <div className="p-3 rounded-lg border border-emerald-300 bg-emerald-50">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-emerald-800 flex items-center gap-1.5">
                    <span>⚡</span>
                    <span>Trigger Fired</span>
                  </span>
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-200 text-emerald-800">
                    {activeRun.status}
                  </span>
                </div>
              </div>

              <div className="bg-white p-3 rounded-lg border border-slate-200 space-y-2">
                <span className="text-[11px] font-bold text-slate-700 uppercase block">Run Summary</span>
                <div className="space-y-1 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Run ID:</span>
                    <span className="font-mono text-slate-700 font-semibold truncate max-w-[150px]" title={activeRun.id}>
                      {activeRun.id.slice(0, 13)}...
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Triggered At:</span>
                    <span className="font-mono text-slate-700">{new Date(activeRun.created_at).toLocaleTimeString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Idempotency Key:</span>
                    <span className="font-mono text-slate-700 truncate max-w-[150px]" title={activeRun.idempotency_key || ''}>
                      {activeRun.idempotency_key || '—'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Tokens Spent:</span>
                    <span className="font-mono text-slate-700">
                      {formatMetric(activeRun.tokens_spent)} / {formatMetric(activeRun.max_tokens_budget)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Provider Calls:</span>
                    <span className="font-mono text-slate-700">
                      {formatMetric(activeRun.provider_calls_spent)} / {formatMetric(activeRun.max_provider_calls_budget)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Trigger Type Selection */}
          <div>
            <label className="block text-[11px] font-bold text-slate-700 uppercase mb-1">
              Trigger Type
            </label>
            <select
              disabled={isReadOnly || isInspectionMode}
              value={triggerType}
              onChange={(e) => handleTypeChange(e.target.value as WorkflowTriggerType)}
              className="w-full text-xs px-2.5 py-2 rounded border border-[#d4d4cc] bg-white text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-600 disabled:opacity-60"
            >
              <option value="MANUAL">Manual / Authenticated API</option>
              <option value="SCHEDULE">Cron / Periodic Schedule</option>
              <option value="WEBHOOK">Twenty CRM Webhook (HMAC)</option>
            </select>
            <p className="text-[10px] text-slate-500 mt-1">{triggerMeta.description}</p>
          </div>

          {/* Conditional Trigger Settings */}
          {triggerType === 'SCHEDULE' && (
            <div className="space-y-3 bg-white p-3 rounded-lg border border-slate-200">
              <span className="text-[11px] font-bold text-slate-800 block">Schedule Details</span>
              <div>
                <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                  Cron Expression (5 fields)
                </label>
                <input
                  type="text"
                  disabled={isReadOnly || isInspectionMode}
                  value={config.cron || '0 9 * * *'}
                  onChange={(e) => handleConfigChange('cron', e.target.value)}
                  className="w-full text-xs font-mono px-2 py-1.5 rounded border border-slate-200 focus:ring-1 focus:ring-blue-500"
                  placeholder="0 9 * * *"
                />
                <span className="text-[9px] text-slate-400">e.g. 0 9 * * * (daily 9am), */15 * * * *</span>
              </div>

              <div>
                <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                  Timezone
                </label>
                <select
                  disabled={isReadOnly || isInspectionMode}
                  value={config.timezone || 'UTC'}
                  onChange={(e) => handleConfigChange('timezone', e.target.value)}
                  className="w-full text-xs px-2 py-1.5 rounded border border-slate-200 bg-white"
                >
                  <option value="UTC">UTC (+00:00)</option>
                  <option value="Asia/Kolkata">Asia/Kolkata (IST +05:30)</option>
                  <option value="America/New_York">America/New_York (EST/EDT)</option>
                  <option value="Europe/London">Europe/London (GMT/BST)</option>
                </select>
              </div>

              <div>
                <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                  Misfire Policy
                </label>
                <select
                  disabled={isReadOnly || isInspectionMode}
                  value={config.misfire_policy || 'RUN_ONCE'}
                  onChange={(e) => handleConfigChange('misfire_policy', e.target.value)}
                  className="w-full text-xs px-2 py-1.5 rounded border border-slate-200 bg-white"
                >
                  <option value="RUN_ONCE">RUN_ONCE (Coalesce missed into 1 run)</option>
                  <option value="SKIP">SKIP (Skip all missed occurrences)</option>
                </select>
              </div>
            </div>
          )}

          {triggerType === 'WEBHOOK' && (
            <div className="space-y-2 bg-white p-3 rounded-lg border border-slate-200">
              <span className="text-[11px] font-bold text-slate-800 block">Webhook Ingress</span>
              <p className="text-[11px] text-slate-600">
                Endpoint: <code className="bg-slate-100 px-1 rounded text-[10px]">/api/v1/workflows/triggers/webhook</code>
              </p>
              <div className="text-[10px] text-slate-500 space-y-1">
                <p>• Requires HMAC-SHA256 signature in <code className="text-slate-700">X-Twenty-Webhook-Signature</code>.</p>
                <p>• Pinned active workflow version is recorded in Transactional Outbox before HTTP 200 acknowledgment.</p>
              </div>
            </div>
          )}

          {/* Available Trigger Variables */}
          <div className="bg-blue-50/50 p-3 rounded-lg border border-blue-100">
            <span className="text-[11px] font-bold text-blue-900 block mb-1">
              Available Trigger Variables
            </span>
            <p className="text-[10px] text-slate-600 mb-2">
              Downstream action nodes can reference these trigger variables:
            </p>
            <div className="space-y-1 text-[10px] font-mono">
              <div className="bg-white px-2 py-1 rounded border border-blue-100 text-blue-800">
                {'{{trigger.query}}'}
              </div>
              <div className="bg-white px-2 py-1 rounded border border-blue-100 text-blue-800">
                {'{{trigger.record_id}}'}
              </div>
              <div className="bg-white px-2 py-1 rounded border border-blue-100 text-blue-800">
                {'{{trigger.event_name}}'}
              </div>
            </div>
          </div>
        </div>
      </aside>
    );
  }

  // -------------------------------------------------------------------------
  // ACTION STEP NODE INSPECTION & CONFIGURATION
  // -------------------------------------------------------------------------
  const nodeData = selectedNode.data as StudioNodeData;
  const actionMeta = ACTION_CATALOG[nodeData.node_type] || ACTION_CATALOG.RETRIEVAL;
  const inputs = nodeData.inputs || {};
  const taskData = nodeData.taskData as WorkflowTaskResponse | null;

  const handleInputChange = (field: string, value: any) => {
    if (isReadOnly || isInspectionMode) return;
    onUpdateNodeData(selectedNode.id, {
      inputs: {
        ...inputs,
        [field]: value,
      },
    });
  };

  const handleIdChange = (newId: string) => {
    if (isReadOnly || isInspectionMode) return;
    onUpdateNodeData(selectedNode.id, { id: newId });
  };

  const handleNameChange = (newName: string) => {
    if (isReadOnly || isInspectionMode) return;
    onUpdateNodeData(selectedNode.id, { name: newName });
  };

  const insertVariable = (field: string, varExpression: string) => {
    if (isReadOnly || isInspectionMode) return;
    const current = inputs[field] || '';
    handleInputChange(field, `${current} ${varExpression}`.trim());
  };

  const upstreamNodes = allNodes.filter(
    (n) => n.id !== selectedNode.id && !n.data?.isTrigger
  );

  // Status badge styling in inspection mode
  const execStatus = nodeData.executionStatus || taskData?.status || 'PENDING';
  const execBadge = getExecutionStatusBadge(
    execStatus,
    nodeData.node_type === 'APPROVAL_GATE'
  );

  // Evidence items associated with task or run
  const evidenceItems: any[] = [];
  if (taskData?.output_result && Array.isArray((taskData.output_result as any).evidence)) {
    evidenceItems.push(...(taskData.output_result as any).evidence);
  } else if (activeRun?.evidence_items && activeRun.evidence_items.length > 0) {
    // If run has evidence items, show them
    evidenceItems.push(...activeRun.evidence_items);
  }

  return (
    <aside className="w-80 bg-[#fafaf7] border-l border-[#d4d4cc] flex flex-col h-full overflow-y-auto shrink-0 min-h-0 overscroll-contain">
      {/* Header */}
      <div className="p-4 border-b border-[#d4d4cc] bg-[#f0f0eb] flex items-center justify-between">
        <div>
          <span
            className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded"
            style={{
              backgroundColor: actionMeta.color.badgeBg,
              color: actionMeta.color.text,
            }}
          >
            {actionMeta.type}
          </span>
          <h3 className="text-sm font-bold text-slate-800 mt-1">{nodeData.name || selectedNode.id}</h3>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-slate-600 text-sm p-1 rounded"
        >
          ✕
        </button>
      </div>

      {/* Mode Sub-Tabs (if in inspection mode) */}
      {isInspectionMode && (
        <div className="flex border-b border-[#d4d4cc] bg-[#fafaf7] px-4 pt-2 gap-2">
          <button
            type="button"
            onClick={() => setActiveTab('execution')}
            className={`text-xs font-semibold pb-2 border-b-2 transition-colors ${
              activeTab === 'execution'
                ? 'border-teal-700 text-teal-800'
                : 'border-transparent text-slate-500 hover:text-slate-700'
            }`}
          >
            Live Execution
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('config')}
            className={`text-xs font-semibold pb-2 border-b-2 transition-colors ${
              activeTab === 'config'
                ? 'border-teal-700 text-teal-800'
                : 'border-transparent text-slate-500 hover:text-slate-700'
            }`}
          >
            Step Definition
          </button>
        </div>
      )}

      <div className="p-4 space-y-4 flex-1">
        {/* ================================================================= */}
        {/* TAB 1: LIVE TASK EXECUTION INSPECTOR */}
        {/* ================================================================= */}
        {isInspectionMode && activeTab === 'execution' ? (
          <div className="space-y-4">
            {/* Status Card */}
            <div className={`p-3 rounded-lg border ${execBadge.borderColor} ${execBadge.badgeBg}`}>
              <div className="flex items-center justify-between">
                <span className={`text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 ${execBadge.badgeText}`}>
                  <span>{execBadge.icon}</span>
                  <span>{execBadge.label}</span>
                </span>
                <span className="text-[11px] font-mono text-slate-700 font-semibold">
                  ⏱ {formatDuration(taskData?.created_at, taskData?.completed_at)}
                </span>
              </div>
              {taskData?.error_message && (
                <div className="mt-2 text-[11px] text-red-800 bg-red-100/90 p-2 rounded border border-red-300">
                  <div className="font-bold mb-0.5">Execution Failure:</div>
                  <div className="font-mono text-[10px] break-words">{taskData.error_message}</div>
                </div>
              )}
            </div>

            {/* Approval Gate Decision Action */}
            {onOpenApprovalModal && (execStatus === 'AWAITING_APPROVAL' || activeRun?.status === 'AWAITING_APPROVAL') && (
              <div className="p-3 bg-amber-50 border border-amber-300 rounded-xl space-y-2">
                <div className="flex items-center gap-1.5 text-xs font-bold text-amber-900">
                  <span>⏸</span>
                  <span>Approval Gate Interrupted</span>
                </div>
                <p className="text-[10px] text-amber-800">
                  This execution requires human reviewer sign-off before proceeding with CRM mutation.
                </p>
                <button
                  type="button"
                  onClick={onOpenApprovalModal}
                  className="w-full py-2 px-3 text-xs font-bold bg-amber-600 hover:bg-amber-700 text-white rounded-lg shadow-xs transition-colors flex items-center justify-center gap-1.5"
                >
                  <span>🛡</span>
                  <span>Review & Decide Ticket</span>
                </button>
              </div>
            )}

            {/* Lifecycle Timings */}
            <div className="bg-white p-3 rounded-lg border border-slate-200 space-y-2">
              <span className="text-[11px] font-bold text-slate-700 uppercase block">Lifecycle Timings</span>
              <div className="grid grid-cols-2 gap-2 text-[10px]">
                <div>
                  <span className="text-slate-400 block">Created At</span>
                  <span className="font-mono text-slate-700">
                    {taskData?.created_at ? new Date(taskData.created_at).toLocaleTimeString() : '—'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block">Completed At</span>
                  <span className="font-mono text-slate-700">
                    {taskData?.completed_at
                      ? new Date(taskData.completed_at).toLocaleTimeString()
                      : taskData?.status === 'RUNNING'
                      ? 'In Progress...'
                      : '—'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block">Duration</span>
                  <span className="font-mono text-slate-700 font-semibold">
                    {formatDuration(taskData?.created_at, taskData?.completed_at)}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block">Worker Lock</span>
                  <span className="font-mono text-slate-700 truncate block" title={taskData?.locked_by || ''}>
                    {taskData?.locked_by || '—'}
                  </span>
                </div>
              </div>
            </div>

            {/* Resource Accounting */}
            <div className="bg-white p-3 rounded-lg border border-slate-200">
              <span className="text-[11px] font-bold text-slate-700 uppercase block mb-2">Resource Accounting</span>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="bg-slate-50 p-2 rounded border border-slate-100">
                  <span className="text-[9px] uppercase font-bold text-slate-400 block">Tokens</span>
                  <span className="text-xs font-mono font-bold text-slate-800">
                    {formatMetric(taskData?.tokens_spent)}
                  </span>
                </div>
                <div className="bg-slate-50 p-2 rounded border border-slate-100">
                  <span className="text-[9px] uppercase font-bold text-slate-400 block">Calls</span>
                  <span className="text-xs font-mono font-bold text-slate-800">
                    {formatMetric(taskData?.provider_calls)}
                  </span>
                </div>
                <div className="bg-slate-50 p-2 rounded border border-slate-100">
                  <span className="text-[9px] uppercase font-bold text-slate-400 block">Retries</span>
                  <span className="text-xs font-mono font-bold text-slate-800">
                    {taskData ? `${taskData.retry_count ?? 0}/${taskData.max_retries ?? 3}` : '—'}
                  </span>
                </div>
              </div>
            </div>

            {/* Input Payload (Sanitized) */}
            <div className="bg-white p-3 rounded-lg border border-slate-200 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold text-slate-700 uppercase">Input Payload</span>
                <span className="text-[9px] text-slate-400">🔒 Redacted</span>
              </div>
              <pre className="text-[10px] font-mono text-slate-700 bg-slate-50 p-2 rounded max-h-48 overflow-y-auto overflow-x-auto border border-slate-100">
                {JSON.stringify(
                  sanitizePayload(taskData?.input_payload || nodeData.inputs || {}),
                  null,
                  2
                )}
              </pre>
            </div>

            {/* Output Result (Sanitized) */}
            <div className="bg-white p-3 rounded-lg border border-slate-200 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold text-slate-700 uppercase">Output Result</span>
                {taskData?.output_result && <span className="text-[9px] text-slate-400">🔒 Redacted</span>}
              </div>
              {taskData?.output_result ? (
                <pre className="text-[10px] font-mono text-slate-700 bg-slate-50 p-2 rounded max-h-56 overflow-y-auto overflow-x-auto border border-slate-100">
                  {JSON.stringify(sanitizePayload(taskData.output_result), null, 2)}
                </pre>
              ) : (
                <div className="text-[10px] text-slate-400 italic bg-slate-50 p-3 rounded border border-slate-100 text-center">
                  {taskData?.status === 'RUNNING'
                    ? 'Execution in progress... awaiting task completion.'
                    : taskData?.status === 'FAILED'
                    ? 'Task failed before producing output.'
                    : taskData?.status === 'SKIPPED'
                    ? 'Task skipped by conditional branch.'
                    : 'No output recorded yet.'}
                </div>
              )}
            </div>

            {/* Evidence References */}
            {evidenceItems.length > 0 && (
              <div className="bg-white p-3 rounded-lg border border-slate-200 space-y-2">
                <span className="text-[11px] font-bold text-slate-700 uppercase block">
                  Evidence References ({evidenceItems.length})
                </span>
                <div className="space-y-1.5 max-h-40 overflow-y-auto">
                  {evidenceItems.map((ev: any, idx: number) => (
                    <div key={idx} className="bg-slate-50 p-2 rounded border border-slate-200 text-[10px]">
                      <div className="flex items-center justify-between text-slate-600 font-semibold mb-0.5">
                        <span>{ev.source_type || 'Evidence'}</span>
                        {ev.confidence !== undefined && (
                          <span className="text-teal-700">{(ev.confidence * 100).toFixed(0)}% conf</span>
                        )}
                      </div>
                      <p className="text-slate-500 line-clamp-2">{ev.snippet || JSON.stringify(ev)}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          /* ================================================================= */
          /* TAB 2: STEP CONFIGURATION & PARAMETERS */
          /* ================================================================= */
          <>
            {/* Validation Errors Alert */}
            {!isInspectionMode && nodeData.validationErrors && nodeData.validationErrors.length > 0 && (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-2.5">
                <span className="text-[11px] font-bold text-amber-900 block mb-1">
                  ⚠️ Incomplete Configuration
                </span>
                <ul className="list-disc list-inside text-[10px] text-amber-700 space-y-0.5">
                  {nodeData.validationErrors.map((err, i) => (
                    <li key={i}>{err}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Identity & Basic Info */}
            <div className="space-y-2.5">
              <div>
                <label className="block text-[11px] font-bold text-slate-700 uppercase mb-0.5">
                  Node Label
                </label>
                <input
                  type="text"
                  disabled={isReadOnly || isInspectionMode}
                  value={nodeData.name || ''}
                  onChange={(e) => handleNameChange(e.target.value)}
                  className="w-full text-xs px-2.5 py-1.5 rounded border border-[#d4d4cc] bg-white text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-600"
                />
              </div>

              <div>
                <label className="block text-[11px] font-bold text-slate-700 uppercase mb-0.5">
                  Node ID (Slug)
                </label>
                <input
                  type="text"
                  disabled={isReadOnly || isInspectionMode}
                  value={selectedNode.id}
                  onChange={(e) => handleIdChange(e.target.value)}
                  className={`w-full text-xs font-mono px-2.5 py-1.5 rounded border bg-white ${
                    isValidNodeId(selectedNode.id)
                      ? 'border-[#d4d4cc] focus:ring-1 focus:ring-teal-600'
                      : 'border-red-400 focus:ring-1 focus:ring-red-500'
                  }`}
                />
                {!isValidNodeId(selectedNode.id) && (
                  <span className="text-[9px] text-red-500">
                    1-64 alphanumeric chars, underscores, or hyphens.
                  </span>
                )}
              </div>
            </div>

            {/* Typed Inputs Section */}
            <div className="space-y-3 pt-2 border-t border-slate-200">
              <span className="text-[11px] font-bold text-slate-700 uppercase block">
                Step Parameters
              </span>

              {/* 1. RETRIEVAL */}
              {nodeData.node_type === 'RETRIEVAL' && (
                <>
                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Search Query *
                    </label>
                    <input
                      type="text"
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.query || ''}
                      onChange={(e) => handleInputChange('query', e.target.value)}
                      placeholder="{{trigger.query}}"
                      className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-200 bg-white"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                        Limit (1-20)
                      </label>
                      <input
                        type="number"
                        min={1}
                        max={20}
                        disabled={isReadOnly || isInspectionMode}
                        value={inputs.limit ?? 3}
                        onChange={(e) => handleInputChange('limit', parseInt(e.target.value, 10))}
                        className="w-full text-xs px-2 py-1.5 rounded border border-slate-200 bg-white"
                      />
                    </div>
                    <div>
                      <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                        Scope
                      </label>
                      <select
                        disabled={isReadOnly || isInspectionMode}
                        value={inputs.scope || 'workspace'}
                        onChange={(e) => handleInputChange('scope', e.target.value)}
                        className="w-full text-xs px-2 py-1.5 rounded border border-slate-200 bg-white"
                      >
                        <option value="workspace">Workspace</option>
                        <option value="workspace_mcp">Workspace + MCP</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      CrossEncoder Reranker
                    </label>
                    <select
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.rerank === undefined || inputs.rerank === null ? '' : inputs.rerank ? 'true' : 'false'}
                      onChange={(e) => {
                        const val = e.target.value;
                        handleInputChange('rerank', val === '' ? undefined : val === 'true');
                      }}
                      className="w-full text-xs px-2 py-1.5 rounded border border-slate-200 bg-white"
                    >
                      <option value="">Inherit Workspace Default</option>
                      <option value="true">Enable Reranker (High Accuracy)</option>
                      <option value="false">Disable Reranker (Fast Retrieval)</option>
                    </select>
                  </div>
                </>
              )}

              {/* 2. WEB_SEARCH */}
              {nodeData.node_type === 'WEB_SEARCH' && (
                <>
                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Search Query *
                    </label>
                    <input
                      type="text"
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.query || ''}
                      onChange={(e) => handleInputChange('query', e.target.value)}
                      placeholder="e.g. competitor news"
                      className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-200 bg-white"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                        Result Limit (1-10)
                      </label>
                      <input
                        type="number"
                        min={1}
                        max={10}
                        disabled={isReadOnly || isInspectionMode}
                        value={inputs.limit ?? 5}
                        onChange={(e) => handleInputChange('limit', parseInt(e.target.value, 10))}
                        className="w-full text-xs px-2 py-1.5 rounded border border-slate-200 bg-white"
                      />
                    </div>
                    <div>
                      <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                        Provider Override
                      </label>
                      <select
                        disabled={isReadOnly || isInspectionMode}
                        value={inputs.search_provider || ''}
                        onChange={(e) => handleInputChange('search_provider', e.target.value || undefined)}
                        className="w-full text-xs px-2 py-1.5 rounded border border-slate-200 bg-white"
                      >
                        <option value="">Inherit Default</option>
                        <option value="tinyfish">TinyFish</option>
                        <option value="firecrawl">Firecrawl</option>
                      </select>
                    </div>
                  </div>
                </>
              )}

              {/* 3. EXTRACTION */}
              {nodeData.node_type === 'EXTRACTION' && (
                <>
                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Focus Query
                    </label>
                    <input
                      type="text"
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.query || ''}
                      onChange={(e) => handleInputChange('query', e.target.value)}
                      placeholder="e.g. Extract revenue and dates"
                      className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-200 bg-white"
                    />
                  </div>

                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Target Fields (comma-separated)
                    </label>
                    <input
                      type="text"
                      disabled={isReadOnly || isInspectionMode}
                      value={(inputs.fields || []).join(', ')}
                      onChange={(e) =>
                        handleInputChange(
                          'fields',
                          e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
                        )
                      }
                      placeholder="e.g. revenue, ceo, competitors"
                      className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-200 bg-white"
                    />
                  </div>
                </>
              )}

              {/* 4. GROUNDING */}
              {nodeData.node_type === 'GROUNDING' && (
                <>
                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Claim to Verify
                    </label>
                    <textarea
                      rows={2}
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.claim || ''}
                      onChange={(e) => handleInputChange('claim', e.target.value)}
                      placeholder="{{steps.synthesis.output.synthesis_text}}"
                      className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-200 bg-white"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-[10px] font-semibold text-slate-600 mb-0.5">
                      <span>Confidence Threshold</span>
                      <span>{inputs.confidence_threshold ?? 0.7}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.confidence_threshold ?? 0.7}
                      onChange={(e) => handleInputChange('confidence_threshold', parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </>
              )}

              {/* 5. SYNTHESIS */}
              {nodeData.node_type === 'SYNTHESIS' && (
                <>
                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Prompt Template *
                    </label>
                    <textarea
                      rows={4}
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.prompt_template || ''}
                      onChange={(e) => handleInputChange('prompt_template', e.target.value)}
                      placeholder="Synthesize brief based on evidence: {{trigger.query}}"
                      className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-200 bg-white font-mono"
                    />
                  </div>

                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Focus Label
                    </label>
                    <input
                      type="text"
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.focus || ''}
                      onChange={(e) => handleInputChange('focus', e.target.value)}
                      placeholder="e.g. Executive Summary"
                      className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-200 bg-white"
                    />
                  </div>

                  <div className="pt-2 border-t border-slate-100 flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-bold text-slate-700 uppercase">
                        AI Model &amp; Generation Overrides
                      </span>
                      {inputs.model_tier || inputs.temperature !== undefined || inputs.max_tokens ? (
                        <span className="text-[9px] font-semibold text-amber-700 bg-amber-50 border border-amber-200 px-1.5 py-0.5 rounded">
                          Node Override
                        </span>
                      ) : (
                        <span className="text-[9px] font-semibold text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded">
                          Inherited Default
                        </span>
                      )}
                    </div>

                    <div>
                      <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                        Model Tier
                      </label>
                      <select
                        disabled={isReadOnly || isInspectionMode}
                        value={inputs.model_tier || ''}
                        onChange={(e) => handleInputChange('model_tier', e.target.value || undefined)}
                        className="w-full text-xs px-2 py-1.5 rounded border border-slate-200 bg-white"
                      >
                        <option value="">Inherit Workspace Default</option>
                        <option value="FAST">FAST (Llama 3.1 8B Instant)</option>
                        <option value="BALANCED">BALANCED (Llama 3.3 70B Versatile)</option>
                        <option value="REASONING">REASONING (High Capacity / Reasoning)</option>
                      </select>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <div className="flex justify-between text-[10px] font-semibold text-slate-600 mb-0.5">
                          <span>Temperature</span>
                          <span>{inputs.temperature ?? 'Default'}</span>
                        </div>
                        <input
                          type="number"
                          step={0.1}
                          min={0.0}
                          max={1.0}
                          disabled={isReadOnly || isInspectionMode}
                          placeholder="e.g. 0.2"
                          value={inputs.temperature ?? ''}
                          onChange={(e) => {
                            const val = e.target.value === '' ? undefined : parseFloat(e.target.value);
                            handleInputChange('temperature', val);
                          }}
                          className="w-full text-xs px-2 py-1.5 rounded border border-slate-200 bg-white"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                          Max Tokens (50-4096)
                        </label>
                        <input
                          type="number"
                          min={50}
                          max={4096}
                          disabled={isReadOnly || isInspectionMode}
                          placeholder="e.g. 2000"
                          value={inputs.max_tokens ?? ''}
                          onChange={(e) => {
                            const val = e.target.value === '' ? undefined : parseInt(e.target.value, 10);
                            handleInputChange('max_tokens', val);
                          }}
                          className="w-full text-xs px-2 py-1.5 rounded border border-slate-200 bg-white"
                        />
                      </div>
                    </div>
                  </div>
                </>
              )}

              {/* 6. IF_ELSE */}
              {nodeData.node_type === 'IF_ELSE' && (
                <>
                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Left-Hand Operand *
                    </label>
                    <input
                      type="text"
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.left_value ?? ''}
                      onChange={(e) => handleInputChange('left_value', e.target.value)}
                      placeholder="{{steps.grounding.output.is_grounded}}"
                      className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-200 bg-white font-mono"
                    />
                  </div>

                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Operator *
                    </label>
                    <select
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.operator || 'EQUALS'}
                      onChange={(e) => handleInputChange('operator', e.target.value)}
                      className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-200 bg-white"
                    >
                      <option value="EQUALS">EQUALS (==)</option>
                      <option value="NOT_EQUALS">NOT_EQUALS (!=)</option>
                      <option value="GREATER_THAN">GREATER_THAN (&gt;)</option>
                      <option value="LESS_THAN">LESS_THAN (&lt;)</option>
                      <option value="CONTAINS">CONTAINS</option>
                      <option value="NOT_CONTAINS">NOT_CONTAINS</option>
                      <option value="IS_TRUE">IS_TRUE</option>
                      <option value="IS_FALSE">IS_FALSE</option>
                      <option value="IS_NOT_EMPTY">IS_NOT_EMPTY</option>
                      <option value="IS_EMPTY">IS_EMPTY</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Right-Hand Operand
                    </label>
                    <input
                      type="text"
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.right_value ?? ''}
                      onChange={(e) => handleInputChange('right_value', e.target.value)}
                      placeholder="true"
                      className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-200 bg-white font-mono"
                    />
                  </div>
                </>
              )}

              {/* 7. APPROVAL_GATE */}
              {nodeData.node_type === 'APPROVAL_GATE' && (
                <>
                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Gate Name *
                    </label>
                    <input
                      type="text"
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.gate_name || ''}
                      onChange={(e) => handleInputChange('gate_name', e.target.value)}
                      placeholder="Manager Review"
                      className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-200 bg-white"
                    />
                  </div>

                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Required Role
                    </label>
                    <select
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.required_role || 'ADMIN'}
                      onChange={(e) => handleInputChange('required_role', e.target.value)}
                      className="w-full text-xs px-2 py-1.5 rounded border border-slate-200 bg-white"
                    >
                      <option value="ADMIN">ADMIN</option>
                      <option value="OWNER">OWNER</option>
                      <option value="EDITOR">EDITOR</option>
                    </select>
                  </div>
                </>
              )}

              {/* 8. EXECUTION */}
              {nodeData.node_type === 'EXECUTION' && (
                <>
                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Target System
                    </label>
                    <input
                      type="text"
                      disabled
                      value={inputs.target_system || 'crm'}
                      className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-200 bg-slate-50 text-slate-500"
                    />
                  </div>

                  <div>
                    <label className="block text-[10px] font-semibold text-slate-600 mb-0.5">
                      Action Name *
                    </label>
                    <input
                      type="text"
                      disabled={isReadOnly || isInspectionMode}
                      value={inputs.action || ''}
                      onChange={(e) => handleInputChange('action', e.target.value)}
                      placeholder="CRM_CREATE_NOTE"
                      className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-200 bg-white font-mono"
                    />
                  </div>
                </>
              )}
            </div>

            {/* Upstream Variable Helper */}
            {!isInspectionMode && (
              <div className="bg-slate-100 p-2.5 rounded-lg border border-slate-200 space-y-1.5">
                <span className="text-[10px] font-bold text-slate-700 uppercase block">
                  Insert Variable
                </span>
                <p className="text-[9px] text-slate-500">
                  Click to append into active input:
                </p>
                <div className="flex flex-wrap gap-1">
                  <button
                    type="button"
                    onClick={() => insertVariable('query', '{{trigger.query}}')}
                    className="text-[9px] font-mono bg-white px-1.5 py-0.5 rounded border border-slate-200 text-teal-700 hover:bg-teal-50"
                  >
                    {'{{trigger.query}}'}
                  </button>
                  {upstreamNodes.map((un) => (
                    <button
                      key={un.id}
                      type="button"
                      onClick={() => insertVariable('query', `{{steps.${un.id}.output}}`)}
                      className="text-[9px] font-mono bg-white px-1.5 py-0.5 rounded border border-slate-200 text-teal-700 hover:bg-teal-50"
                    >
                      {`{{steps.${un.id}.output}}`}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Output Schema Reference */}
            <div className="border border-slate-200 rounded-lg p-2.5 bg-white">
              <span className="text-[10px] font-bold text-slate-700 uppercase block mb-1">
                Output Contract
              </span>
              <pre className="text-[9px] font-mono text-slate-600 bg-slate-50 p-2 rounded overflow-x-auto">
                {JSON.stringify(actionMeta.outputSchema, null, 2)}
              </pre>
            </div>

            {/* Delete Node Action */}
            {!isReadOnly && !isInspectionMode && (
              <div className="pt-2">
                <button
                  type="button"
                  onClick={() => onDeleteNode(selectedNode.id)}
                  className="w-full py-1.5 px-3 text-xs font-semibold text-red-600 bg-red-50 hover:bg-red-100 border border-red-200 rounded-lg transition-colors"
                >
                  Delete Step
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </aside>
  );
};
