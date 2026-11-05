import React, { useState } from 'react';
import type { WorkflowTaskResponse, WorkflowRunResponse } from '../api/workflows.ts';
import {
  sanitizePayload,
  formatDuration,
  getExecutionStatusBadge,
} from './workflowStudio/executionUtils.ts';

export interface TaskInspectorPanelProps {
  task: WorkflowTaskResponse | null;
  run: WorkflowRunResponse;
  token: string;
  onClose?: () => void;
  onOpenArtifact?: (artifactId: string) => void;
}

export function TaskInspectorPanel({
  task,
  run,
  token,
  onClose,
  onOpenArtifact,
}: TaskInspectorPanelProps) {
  const [activeTab, setActiveTab] = useState<'details' | 'input' | 'output'>('details');
  const [copiedPayload, setCopiedPayload] = useState<string | null>(null);

  if (!task) {
    return (
      <div
        style={{
          padding: '24px',
          textAlign: 'center',
          color: '#64748B',
          fontSize: '13px',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#F8FAFC',
        }}
      >
        <div style={{ fontSize: '32px', marginBottom: '8px' }}>🔍</div>
        <div style={{ fontWeight: 600, color: '#1E293B' }}>No Task Selected</div>
        <p style={{ fontSize: '12px', marginTop: '4px', maxWidth: '240px' }}>
          Click on any task node in the canvas to inspect its execution metrics, parameters, outputs, and evidence.
        </p>
      </div>
    );
  }

  const badge = getExecutionStatusBadge(task.status, false);
  const duration = formatDuration(task.created_at, task.completed_at);
  const sanitizedInput = sanitizePayload(task.input_payload || {});
  const sanitizedOutput = sanitizePayload(task.output_result || {});

  // Extract any artifact:// symbolic references or UUIDs
  const extractArtifactReferences = (data: any): string[] => {
    const raw = JSON.stringify(data || '');
    const regex = /artifact:\/\/([a-zA-Z0-9_\-\/]+)/g;
    const matches: string[] = [];
    let match;
    while ((match = regex.exec(raw)) !== null) {
      matches.push(match[0]);
    }
    return Array.from(new Set(matches));
  };

  const artifactRefs = [
    ...extractArtifactReferences(task.input_payload),
    ...extractArtifactReferences(task.output_result),
  ];

  const handleCopy = (content: any, type: string) => {
    navigator.clipboard.writeText(JSON.stringify(content, null, 2));
    setCopiedPayload(type);
    setTimeout(() => setCopiedPayload(null), 2000);
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: '#FFFFFF',
        borderLeft: '1px solid #E2E8F0',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '14px 16px',
          borderBottom: '1px solid #E2E8F0',
          background: '#F8FAFC',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
            <span
              style={{
                fontSize: '10.5px',
                fontFamily: 'monospace',
                padding: '1px 6px',
                borderRadius: '3px',
                background: '#E2E8F0',
                color: '#334155',
                fontWeight: 700,
              }}
            >
              {task.task_type}
            </span>
            <div
              style={{
                fontSize: '10px',
                fontWeight: 700,
                padding: '2px 6px',
                borderRadius: '4px',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '3px',
                border: `1px solid ${badge.borderColor}`,
              }}
              className={`${badge.badgeBg} ${badge.badgeText}`}
            >
              <span>{badge.icon}</span>
              <span>{badge.label}</span>
            </div>
          </div>
          <h4
            style={{
              margin: 0,
              fontSize: '14px',
              fontWeight: 700,
              color: '#0F172A',
              fontFamily: 'monospace',
            }}
          >
            {task.task_key}
          </h4>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              fontSize: '16px',
              color: '#64748B',
              cursor: 'pointer',
              padding: '2px 6px',
            }}
            title="Close Inspector"
          >
            ✕
          </button>
        )}
      </div>

      {/* Tabs */}
      <div
        style={{
          display: 'flex',
          borderBottom: '1px solid #E2E8F0',
          background: '#FFFFFF',
        }}
      >
        <button
          onClick={() => setActiveTab('details')}
          style={{
            flex: 1,
            padding: '8px 12px',
            fontSize: '12px',
            fontWeight: activeTab === 'details' ? 600 : 500,
            color: activeTab === 'details' ? '#2563EB' : '#64748B',
            borderBottom: activeTab === 'details' ? '2px solid #2563EB' : '2px solid transparent',
            background: 'none',
            borderTop: 'none',
            borderLeft: 'none',
            borderRight: 'none',
            cursor: 'pointer',
          }}
        >
          Details
        </button>
        <button
          onClick={() => setActiveTab('input')}
          style={{
            flex: 1,
            padding: '8px 12px',
            fontSize: '12px',
            fontWeight: activeTab === 'input' ? 600 : 500,
            color: activeTab === 'input' ? '#2563EB' : '#64748B',
            borderBottom: activeTab === 'input' ? '2px solid #2563EB' : '2px solid transparent',
            background: 'none',
            borderTop: 'none',
            borderLeft: 'none',
            borderRight: 'none',
            cursor: 'pointer',
          }}
        >
          Input
        </button>
        <button
          onClick={() => setActiveTab('output')}
          style={{
            flex: 1,
            padding: '8px 12px',
            fontSize: '12px',
            fontWeight: activeTab === 'output' ? 600 : 500,
            color: activeTab === 'output' ? '#2563EB' : '#64748B',
            borderBottom: activeTab === 'output' ? '2px solid #2563EB' : '2px solid transparent',
            background: 'none',
            borderTop: 'none',
            borderLeft: 'none',
            borderRight: 'none',
            cursor: 'pointer',
          }}
        >
          Output
        </button>
      </div>

      {/* Tab Body */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
        {activeTab === 'details' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {/* Error Banner */}
            {task.error_message && (
              <div
                style={{
                  padding: '10px 12px',
                  borderRadius: '6px',
                  background: '#FEF2F2',
                  border: '1px solid #FCA5A5',
                  color: '#991B1B',
                  fontSize: '12px',
                }}
              >
                <div style={{ fontWeight: 600, marginBottom: '2px' }}>⚠️ Task Failure</div>
                <div style={{ wordBreak: 'break-word', fontFamily: 'monospace' }}>
                  {task.error_message}
                </div>
              </div>
            )}

            {/* Execution Metrics Grid */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '8px',
                padding: '10px',
                borderRadius: '8px',
                background: '#F8FAFC',
                border: '1px solid #E2E8F0',
                fontSize: '11.5px',
              }}
            >
              <div>
                <span style={{ color: '#64748B', display: 'block' }}>Duration</span>
                <strong style={{ color: '#0F172A' }}>{duration}</strong>
              </div>
              <div>
                <span style={{ color: '#64748B', display: 'block' }}>Tokens Spent</span>
                <strong style={{ color: '#0F172A' }}>{task.tokens_spent.toLocaleString()}</strong>
              </div>
              <div>
                <span style={{ color: '#64748B', display: 'block' }}>Provider Calls</span>
                <strong style={{ color: '#0F172A' }}>{task.provider_calls}</strong>
              </div>
              <div>
                <span style={{ color: '#64748B', display: 'block' }}>Retries</span>
                <strong style={{ color: '#0F172A' }}>
                  {task.retry_count || 0} / {task.max_retries || 0}
                </strong>
              </div>
            </div>

            {/* Dependencies */}
            <div>
              <span style={{ fontSize: '11px', fontWeight: 600, color: '#64748B', display: 'block', marginBottom: '6px' }}>
                DEPENDENCIES ({task.dependencies.length})
              </span>
              {task.dependencies.length === 0 ? (
                <span style={{ fontSize: '11.5px', color: '#94A3B8', fontStyle: 'italic' }}>
                  None (Root DAG Task)
                </span>
              ) : (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {task.dependencies.map((dep) => (
                    <span
                      key={dep}
                      style={{
                        fontSize: '11px',
                        fontFamily: 'monospace',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        background: '#EFF6FF',
                        border: '1px solid #BFDBFE',
                        color: '#1D4ED8',
                      }}
                    >
                      ← {dep}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Linked Execution Artifacts */}
            {artifactRefs.length > 0 && (
              <div>
                <span style={{ fontSize: '11px', fontWeight: 600, color: '#64748B', display: 'block', marginBottom: '6px' }}>
                  LINKED EXECUTION ARTIFACTS
                </span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {artifactRefs.map((ref) => {
                    const parts = ref.split('/');
                    const refId = parts[parts.length - 1];
                    return (
                      <div
                        key={ref}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '6px 10px',
                          borderRadius: '6px',
                          background: '#F0FDF4',
                          border: '1px solid #BBF7D0',
                        }}
                      >
                        <span style={{ fontSize: '11px', fontFamily: 'monospace', color: '#166534' }}>
                          📄 {ref}
                        </span>
                        {onOpenArtifact && (
                          <button
                            onClick={() => onOpenArtifact(refId)}
                            style={{
                              fontSize: '10.5px',
                              fontWeight: 600,
                              color: '#15803D',
                              background: '#DCFCE7',
                              border: '1px solid #86EFAC',
                              borderRadius: '4px',
                              padding: '2px 6px',
                              cursor: 'pointer',
                            }}
                          >
                            Preview
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Timestamps */}
            <div
              style={{
                fontSize: '11px',
                color: '#64748B',
                display: 'flex',
                flexDirection: 'column',
                gap: '4px',
                borderTop: '1px solid #F1F5F9',
                paddingTop: '10px',
              }}
            >
              <div>Created: {new Date(task.created_at).toLocaleString()}</div>
              {task.completed_at && <div>Completed: {new Date(task.completed_at).toLocaleString()}</div>}
            </div>
          </div>
        )}

        {activeTab === 'input' && (
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => handleCopy(sanitizedInput, 'input')}
              style={{
                position: 'absolute',
                top: '6px',
                right: '6px',
                padding: '3px 8px',
                fontSize: '10.5px',
                borderRadius: '4px',
                border: '1px solid #CBD5E1',
                background: '#FFFFFF',
                color: '#334155',
                cursor: 'pointer',
              }}
            >
              {copiedPayload === 'input' ? '✓ Copied' : '📋 Copy JSON'}
            </button>
            <pre
              style={{
                background: '#0F172A',
                color: '#F8FAFC',
                padding: '12px',
                borderRadius: '6px',
                fontSize: '11px',
                fontFamily: 'Consolas, Monaco, monospace',
                overflowX: 'auto',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                margin: 0,
              }}
            >
              {JSON.stringify(sanitizedInput, null, 2)}
            </pre>
          </div>
        )}

        {activeTab === 'output' && (
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => handleCopy(sanitizedOutput, 'output')}
              style={{
                position: 'absolute',
                top: '6px',
                right: '6px',
                padding: '3px 8px',
                fontSize: '10.5px',
                borderRadius: '4px',
                border: '1px solid #CBD5E1',
                background: '#FFFFFF',
                color: '#334155',
                cursor: 'pointer',
              }}
            >
              {copiedPayload === 'output' ? '✓ Copied' : '📋 Copy JSON'}
            </button>
            <pre
              style={{
                background: '#0F172A',
                color: '#F8FAFC',
                padding: '12px',
                borderRadius: '6px',
                fontSize: '11px',
                fontFamily: 'Consolas, Monaco, monospace',
                overflowX: 'auto',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                margin: 0,
              }}
            >
              {JSON.stringify(sanitizedOutput, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

