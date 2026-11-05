import { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { getExecutionStatusBadge, formatDuration } from './workflowStudio/executionUtils.ts';

export interface ExecutionTaskNodeData {
  taskKey: string;
  taskType: string;
  status: string;
  dependencies: string[];
  tokensSpent: number;
  providerCalls: number;
  retryCount?: number;
  errorMessage?: string | null;
  startedAt?: string | null;
  completedAt?: string | null;
  isSelected?: boolean;
}

export const ExecutionTaskNode = memo(({ data, selected }: NodeProps) => {
  const nodeData = data as unknown as ExecutionTaskNodeData;
  const statusBadge = getExecutionStatusBadge(nodeData.status, false);
  const duration = formatDuration(nodeData.startedAt, nodeData.completedAt);

  return (
    <div
      style={{
        minWidth: '220px',
        maxWidth: '260px',
        borderRadius: '10px',
        border: selected ? '2px solid #2563EB' : '1.5px solid #CBD5E1',
        background: '#FFFFFF',
        boxShadow: selected
          ? '0 0 0 3px rgba(37, 99, 235, 0.2), 0 4px 6px -1px rgba(0, 0, 0, 0.1)'
          : '0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        transition: 'all 0.15s ease',
        overflow: 'hidden',
        cursor: 'pointer',
      }}
    >
      {/* Incoming Dependency Handle */}
      <Handle
        type="target"
        position={Position.Left}
        style={{
          width: '10px',
          height: '10px',
          background: '#64748B',
          border: '2px solid #FFFFFF',
        }}
      />

      {/* Header Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '6px 10px',
          background: '#F8FAFC',
          borderBottom: '1px solid #E2E8F0',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span
            style={{
              fontSize: '10px',
              fontFamily: 'monospace',
              padding: '1px 5px',
              borderRadius: '3px',
              background: '#E2E8F0',
              color: '#334155',
              fontWeight: 700,
            }}
          >
            {nodeData.taskType || 'TASK'}
          </span>
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '3px',
            fontSize: '10px',
            fontWeight: 700,
            padding: '2px 6px',
            borderRadius: '4px',
            border: `1px solid ${statusBadge.borderColor}`,
          }}
          className={`${statusBadge.badgeBg} ${statusBadge.badgeText}`}
        >
          <span>{statusBadge.icon}</span>
          <span>{statusBadge.label}</span>
        </div>
      </div>

      {/* Body */}
      <div style={{ padding: '8px 10px' }}>
        <div
          style={{
            fontSize: '12px',
            fontWeight: 600,
            color: '#0F172A',
            fontFamily: 'monospace',
            marginBottom: '4px',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
          title={nodeData.taskKey}
        >
          {nodeData.taskKey}
        </div>

        {/* Quick metrics row */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '10.5px',
            color: '#64748B',
            marginTop: '6px',
            paddingTop: '6px',
            borderTop: '1px dashed #E2E8F0',
          }}
        >
          <span>⏱ {duration}</span>
          <span>🪙 {nodeData.tokensSpent > 0 ? `${nodeData.tokensSpent} toks` : '—'}</span>
        </div>

        {nodeData.errorMessage && (
          <div
            style={{
              marginTop: '6px',
              padding: '3px 6px',
              borderRadius: '4px',
              background: '#FEF2F2',
              color: '#DC2626',
              fontSize: '10px',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
            title={nodeData.errorMessage}
          >
            ⚠️ {nodeData.errorMessage}
          </div>
        )}
      </div>

      {/* Outgoing Dependency Handle */}
      <Handle
        type="source"
        position={Position.Right}
        style={{
          width: '10px',
          height: '10px',
          background: '#2563EB',
          border: '2px solid #FFFFFF',
        }}
      />
    </div>
  );
});

