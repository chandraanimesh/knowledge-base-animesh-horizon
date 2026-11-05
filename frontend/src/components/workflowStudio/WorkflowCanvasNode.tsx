import { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { ACTION_CATALOG, TRIGGER_CATALOG, StudioNodeData } from './types.ts';
import { getExecutionStatusBadge, formatDuration } from './executionUtils.ts';

export const WorkflowCanvasNode = memo(({ data, selected }: NodeProps) => {
  const nodeData = data as StudioNodeData;
  const isTrigger = Boolean(nodeData.isTrigger);
  const isInspectionMode = Boolean(nodeData.isInspectionMode);

  if (isTrigger) {
    const triggerType = nodeData.triggerType || 'MANUAL';
    const triggerMeta = TRIGGER_CATALOG[triggerType];
    const config = nodeData.triggerConfig || {};

    let summary = 'Manual / On-Demand';
    if (triggerType === 'SCHEDULE') {
      summary = `Cron: ${config.cron || '0 9 * * *'} (${config.timezone || 'UTC'})`;
    } else if (triggerType === 'WEBHOOK') {
      summary = 'Inbound HMAC-SHA256 Webhook';
    }

    return (
      <div
        className={`min-w-[200px] max-w-[260px] rounded-xl border-2 bg-white shadow-sm transition-all ${
          selected
            ? 'border-blue-600 ring-2 ring-blue-200'
            : isInspectionMode
            ? 'border-emerald-400 shadow-xs'
            : 'border-blue-200 hover:border-blue-300'
        }`}
      >
        <div className="flex items-center gap-2 rounded-t-[10px] bg-blue-50 px-3 py-2 border-b border-blue-100">
          <div className="flex h-6 w-6 items-center justify-center rounded-md bg-blue-600 text-white font-bold text-xs">
            ⚡
          </div>
          <div className="flex-1 min-w-0">
            <span className="text-[10px] font-semibold tracking-wider text-blue-600 uppercase block">
              Workflow Trigger
            </span>
            <span className="text-xs font-semibold text-slate-800 truncate block">
              {triggerMeta?.name || 'Trigger'}
            </span>
          </div>
          {isInspectionMode && (
            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-300">
              ✓ FIRED
            </span>
          )}
        </div>

        <div className="p-3">
          <p className="text-[11px] text-slate-500 font-mono truncate">{summary}</p>
        </div>

        {/* Source Handle (Outbound only) */}
        <Handle
          type="source"
          position={Position.Right}
          className="!h-3 !w-3 !bg-blue-600 !border-2 !border-white"
        />
      </div>
    );
  }

  const actionMeta = ACTION_CATALOG[nodeData.node_type] || ACTION_CATALOG.RETRIEVAL;
  const hasErrors = !isInspectionMode && nodeData.validationErrors && nodeData.validationErrors.length > 0;

  // Phase 4B: Execution inspection badge
  const execStatus = nodeData.executionStatus || 'PENDING';
  const execBadge = getExecutionStatusBadge(
    execStatus,
    nodeData.node_type === 'APPROVAL_GATE'
  );
  const durationStr = formatDuration(
    nodeData.taskData?.created_at,
    nodeData.taskData?.completed_at
  );

  let borderClasses = 'border-slate-200 hover:border-slate-300';
  if (selected) {
    borderClasses = 'border-teal-700 ring-2 ring-teal-200 shadow-md';
  } else if (isInspectionMode) {
    borderClasses = `${execBadge.borderColor} ${execBadge.pulse ? 'animate-pulse' : ''}`;
  } else if (hasErrors) {
    borderClasses = 'border-amber-400 hover:border-amber-500';
  }

  return (
    <div
      className={`min-w-[220px] max-w-[280px] rounded-xl border-2 bg-white shadow-sm transition-all ${borderClasses}`}
    >
      {/* Target Handle (Inbound) */}
      <Handle
        type="target"
        position={Position.Left}
        className="!h-3 !w-3 !bg-slate-400 !border-2 !border-white"
      />

      {/* Node Header */}
      <div
        className="flex items-center justify-between rounded-t-[10px] px-3 py-2 border-b"
        style={{
          backgroundColor: actionMeta.color.bg,
          borderColor: actionMeta.color.border,
        }}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="text-[10px] font-semibold px-1.5 py-0.5 rounded tracking-wide uppercase"
            style={{
              backgroundColor: actionMeta.color.badgeBg,
              color: actionMeta.color.text,
            }}
          >
            {actionMeta.type}
          </span>
          <span className="text-xs font-medium text-slate-800 truncate">
            {nodeData.name || nodeData.id}
          </span>
        </div>

        {isInspectionMode ? (
          <span
            className={`text-[9px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider flex items-center gap-1 ${
              execBadge.badgeBg
            } ${execBadge.badgeText} ${execBadge.pulse ? 'animate-pulse' : ''}`}
          >
            <span>{execBadge.icon}</span>
            <span>{execBadge.label}</span>
          </span>
        ) : hasErrors ? (
          <span
            className="flex h-5 w-5 items-center justify-center rounded-full bg-amber-500 text-white text-[10px] font-bold"
            title={`${nodeData.validationErrors?.length} validation warning(s)`}
          >
            !
          </span>
        ) : (
          <span className="text-emerald-500 text-xs">✓</span>
        )}
      </div>

      {/* Node Body */}
      <div className="p-3">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] font-mono text-slate-400 truncate max-w-[130px]">
            id: {nodeData.id}
          </span>

          {isInspectionMode ? (
            <span className="text-[10px] font-mono text-slate-500 font-medium">
              ⏱ {durationStr}
            </span>
          ) : (
            nodeData.dependencies && nodeData.dependencies.length > 0 && (
              <span className="text-[10px] text-slate-400">
                {nodeData.dependencies.length} dep{nodeData.dependencies.length > 1 ? 's' : ''}
              </span>
            )
          )}
        </div>

        {/* Inspection Mode Details or Input Preview */}
        {isInspectionMode && nodeData.taskData ? (
          <div className="space-y-1 mt-1">
            {nodeData.taskData.error_message && (
              <div className="text-[10px] text-red-700 bg-red-50 p-1 rounded border border-red-200 line-clamp-2">
                ⚠ {nodeData.taskData.error_message}
              </div>
            )}
            <div className="flex items-center justify-between text-[10px] text-slate-500 bg-slate-50 p-1.5 rounded border border-slate-100">
              <span>Tokens: {nodeData.taskData.tokens_spent ?? 0}</span>
              <span>Calls: {nodeData.taskData.provider_calls ?? 0}</span>
              {nodeData.taskData.retry_count ? (
                <span className="text-amber-600 font-semibold">
                  Retries: {nodeData.taskData.retry_count}
                </span>
              ) : null}
            </div>
          </div>
        ) : (
          nodeData.inputs && (
            <div className="text-[11px] text-slate-600 line-clamp-2 font-sans bg-slate-50 p-1.5 rounded border border-slate-100 mt-1">
              {nodeData.node_type === 'RETRIEVAL' && `Query: ${nodeData.inputs.query || '—'}`}
              {nodeData.node_type === 'WEB_SEARCH' && `Search: ${nodeData.inputs.query || '—'}`}
              {nodeData.node_type === 'SYNTHESIS' && `Focus: ${nodeData.inputs.focus || 'Summary'}`}
              {nodeData.node_type === 'IF_ELSE' &&
                `Condition: ${nodeData.inputs.operator || 'EQUALS'}`}
              {nodeData.node_type === 'GROUNDING' &&
                `Threshold: ${nodeData.inputs.confidence_threshold ?? 0.7}`}
              {nodeData.node_type === 'EXTRACTION' &&
                `Fields: ${(nodeData.inputs.fields || []).join(', ') || 'all'}`}
              {nodeData.node_type === 'APPROVAL_GATE' &&
                `Role: ${nodeData.inputs.required_role || 'ADMIN'}`}
              {nodeData.node_type === 'EXECUTION' && `Action: ${nodeData.inputs.action || 'CRM write'}`}
            </div>
          )
        )}
      </div>

      {/* Source Handle (Outbound) */}
      <Handle
        type="source"
        position={Position.Right}
        className="!h-3 !w-3 !bg-teal-700 !border-2 !border-white"
      />
    </div>
  );
});
