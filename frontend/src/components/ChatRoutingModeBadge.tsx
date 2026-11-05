import React, { useState } from 'react';

export interface ChatRoutingModeBadgeProps {
  mode?: string | null;
  reason?: string | null;
  workflowRunId?: string | null;
  workflowStatus?: string | null;
  onOpenCanvas?: (runId: string) => void;
}

export function ChatRoutingModeBadge({
  mode,
  reason,
  workflowRunId,
  workflowStatus,
  onOpenCanvas,
}: ChatRoutingModeBadgeProps) {
  const [showTooltip, setShowTooltip] = useState(false);

  if (!mode && !workflowRunId) {
    return null;
  }

  const normalizedMode = (mode || '').toUpperCase();

  let modeLabel = normalizedMode || 'AGENT';
  let modeIcon = '✦';
  let bgStyle = '#F1F5F9';
  let textStyle = '#475569';
  let borderStyle = '#CBD5E1';
  let defaultDescription = 'Standard query execution mode.';

  if (normalizedMode === 'FAST_RAG' || normalizedMode === 'FAST_PATH') {
    modeLabel = 'Fast RAG';
    modeIcon = '⚡';
    bgStyle = '#F0F9FF';
    textStyle = '#0369A1';
    borderStyle = '#BAE6FD';
    defaultDescription = 'Single-shot direct knowledge retrieval and grounded answer synthesis.';
  } else if (normalizedMode === 'CONTEXTUAL') {
    modeLabel = 'Contextual';
    modeIcon = '🧠';
    bgStyle = '#FAF5FF';
    textStyle = '#7E22CE';
    borderStyle = '#E9D5FF';
    defaultDescription = 'Multi-source retrieval fusing internal documents, personal memory, and living profile context.';
  } else if (normalizedMode === 'ADAPTIVE_PLAN' || normalizedMode === 'ADAPTIVE_WORKFLOW' || normalizedMode === 'WORKFLOW') {
    modeLabel = 'Adaptive Plan';
    modeIcon = '🔀';
    bgStyle = '#ECFDF5';
    textStyle = '#047857';
    borderStyle = '#A7F3D0';
    defaultDescription = 'Multi-step autonomous task planning, capability routing, and dependency DAG execution.';
  }

  const explanation = reason || defaultDescription;

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', position: 'relative' }}>
      {mode && (
        <div
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            padding: '2px 7px',
            borderRadius: '4px',
            fontSize: '10.5px',
            fontWeight: 600,
            background: bgStyle,
            color: textStyle,
            border: `1px solid ${borderStyle}`,
            cursor: 'help',
            userSelect: 'none',
            transition: 'all 0.15s ease',
          }}
          title={explanation}
        >
          <span>{modeIcon}</span>
          <span>{modeLabel}</span>
        </div>
      )}

      {showTooltip && (
        <div
          style={{
            position: 'absolute',
            bottom: 'calc(100% + 6px)',
            left: 0,
            zIndex: 100,
            width: '260px',
            padding: '8px 10px',
            borderRadius: '6px',
            background: '#1E293B',
            color: '#F8FAFC',
            fontSize: '11px',
            lineHeight: 1.4,
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            pointerEvents: 'none',
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: '2px', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span>{modeIcon}</span>
            <span>{modeLabel} Execution</span>
          </div>
          <div style={{ color: '#CBD5E1', fontSize: '10.5px' }}>{explanation}</div>
        </div>
      )}

      {workflowRunId && onOpenCanvas && (
        <button
          onClick={() => onOpenCanvas(workflowRunId)}
          title={`Open interactive execution canvas for run ${workflowRunId}`}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            padding: '2px 8px',
            fontSize: '10.5px',
            fontWeight: 600,
            color: '#0D9488',
            background: '#F0FDFA',
            border: '1px solid #99F6E4',
            borderRadius: '4px',
            cursor: 'pointer',
            transition: 'all 0.15s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = '#CCFBF1';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = '#F0FDFA';
          }}
        >
          <span>📊</span>
          <span>Execution Canvas</span>
          {workflowStatus && (
            <span
              style={{
                fontSize: '9.5px',
                padding: '1px 4px',
                borderRadius: '3px',
                background: workflowStatus === 'COMPLETED' ? '#D1FAE5' : workflowStatus === 'AWAITING_APPROVAL' ? '#FEF3C7' : '#E2E8F0',
                color: workflowStatus === 'COMPLETED' ? '#065F46' : workflowStatus === 'AWAITING_APPROVAL' ? '#92400E' : '#475569',
              }}
            >
              {workflowStatus}
            </span>
          )}
        </button>
      )}
    </div>
  );
}

