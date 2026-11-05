import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  MarkerType,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { getWorkflowRun, WorkflowRunResponse, WorkflowTaskResponse } from '../api/workflows.ts';
import { ExecutionTaskNode, ExecutionTaskNodeData } from './ExecutionTaskNode.tsx';
import { TaskInspectorPanel } from './TaskInspectorPanel.tsx';
import { ArtifactPreviewModal } from './ArtifactPreviewModal.tsx';
import {
  formatDuration,
  getExecutionStatusBadge,
  isTerminalStatus,
} from './workflowStudio/executionUtils.ts';

export interface ExecutionCanvasModalProps {
  runId: string;
  token: string;
  onClose: () => void;
}

const nodeTypes = {
  executionTask: ExecutionTaskNode,
};

export function ExecutionCanvasModal({ runId, token, onClose }: ExecutionCanvasModalProps) {
  const [run, setRun] = useState<WorkflowRunResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTaskKey, setSelectedTaskKey] = useState<string | null>(null);
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'canvas' | 'table'>('canvas');

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const isMountedRef = useRef<boolean>(true);
  const isPollingRef = useRef<boolean>(false);

  // Compute DAG topological layers and layout
  const computeDagLayout = useCallback((tasks: WorkflowTaskResponse[], activeKey: string | null) => {
    if (!tasks || tasks.length === 0) {
      return { layoutNodes: [], layoutEdges: [] };
    }

    const taskMap = new Map<string, WorkflowTaskResponse>();
    for (const t of tasks) {
      taskMap.set(t.task_key, t);
    }

    // Calculate topological depth for each task
    const depthMemo = new Map<string, number>();

    const getDepth = (key: string, visited = new Set<string>()): number => {
      if (depthMemo.has(key)) return depthMemo.get(key)!;
      if (visited.has(key)) return 0; // Guard against cycles
      visited.add(key);

      const task = taskMap.get(key);
      if (!task || !task.dependencies || task.dependencies.length === 0) {
        depthMemo.set(key, 0);
        return 0;
      }

      let maxDepDepth = 0;
      for (const dep of task.dependencies) {
        maxDepDepth = Math.max(maxDepDepth, getDepth(dep, new Set(visited)) + 1);
      }
      depthMemo.set(key, maxDepDepth);
      return maxDepDepth;
    };

    // Group tasks by depth layer
    const layerMap = new Map<number, WorkflowTaskResponse[]>();
    for (const task of tasks) {
      const d = getDepth(task.task_key);
      if (!layerMap.has(d)) layerMap.set(d, []);
      layerMap.get(d)!.push(task);
    }

    const layoutNodes: Node[] = [];
    const layoutEdges: Edge[] = [];

    // Layout nodes column by column
    layerMap.forEach((layerTasks, layerIdx) => {
      layerTasks.forEach((task, rowIdx) => {
        const xPos = 60 + layerIdx * 320;
        const yPos = 80 + rowIdx * 140;

        layoutNodes.push({
          id: task.task_key,
          type: 'executionTask',
          position: { x: xPos, y: yPos },
          selected: task.task_key === activeKey,
          data: {
            taskKey: task.task_key,
            taskType: task.task_type,
            status: task.status,
            dependencies: task.dependencies || [],
            tokensSpent: task.tokens_spent || 0,
            providerCalls: task.provider_calls || 0,
            retryCount: task.retry_count,
            errorMessage: task.error_message,
            startedAt: task.created_at,
            completedAt: task.completed_at,
            isSelected: task.task_key === activeKey,
          } as ExecutionTaskNodeData,
        });

        // Add directed edges for dependencies
        if (task.dependencies && task.dependencies.length > 0) {
          for (const depKey of task.dependencies) {
            const isCompleted = taskMap.get(depKey)?.status === 'COMPLETED';
            const isRunning = task.status === 'RUNNING';
            layoutEdges.push({
              id: `edge-${depKey}-${task.task_key}`,
              source: depKey,
              target: task.task_key,
              animated: isRunning,
              style: {
                stroke: isCompleted ? '#10B981' : isRunning ? '#3B82F6' : '#94A3B8',
                strokeWidth: 2,
              },
              markerEnd: {
                type: MarkerType.ArrowClosed,
                color: isCompleted ? '#10B981' : isRunning ? '#3B82F6' : '#94A3B8',
              },
            });
          }
        }
      });
    });

    return { layoutNodes, layoutEdges };
  }, []);

  // Fetch workflow run and manage polling
  useEffect(() => {
    isMountedRef.current = true;

    async function fetchWorkflow() {
      if (isPollingRef.current) return;
      isPollingRef.current = true;
      try {
        const data = await getWorkflowRun(runId, token);
        if (!isMountedRef.current) return;
        setRun(data);
        setLoading(false);
        setError(null);

        // Update nodes and edges
        const { layoutNodes, layoutEdges } = computeDagLayout(data.tasks || [], selectedTaskKey);
        setNodes(layoutNodes);
        setEdges(layoutEdges);

        // Auto-select first task if none selected
        if (!selectedTaskKey && data.tasks && data.tasks.length > 0) {
          setSelectedTaskKey(data.tasks[0].task_key);
        }
      } catch (err: any) {
        if (!isMountedRef.current) return;
        if (!run) {
          setError(err.message || 'Failed to load execution run.');
        }
      } finally {
        isPollingRef.current = false;
      }
    }

    fetchWorkflow();

    const intervalId = setInterval(() => {
      if (!isMountedRef.current) return;
      if (run && (isTerminalStatus(run.status) || run.status === 'AWAITING_APPROVAL')) {
        clearInterval(intervalId);
        return;
      }
      fetchWorkflow();
    }, 2500);

    return () => {
      isMountedRef.current = false;
      clearInterval(intervalId);
    };
  }, [runId, token, run?.status, computeDagLayout, selectedTaskKey, setNodes, setEdges]);

  // Handle node selection in canvas
  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedTaskKey(node.id);
      if (run?.tasks) {
        const { layoutNodes, layoutEdges } = computeDagLayout(run.tasks, node.id);
        setNodes(layoutNodes);
        setEdges(layoutEdges);
      }
    },
    [run, computeDagLayout, setNodes, setEdges]
  );

  const selectedTask = useMemo(() => {
    if (!run?.tasks || !selectedTaskKey) return null;
    return run.tasks.find((t) => t.task_key === selectedTaskKey) || null;
  }, [run?.tasks, selectedTaskKey]);

  const runBadge = run ? getExecutionStatusBadge(run.status, false) : null;
  const duration = run ? formatDuration(run.startedAt || run.created_at, run.completed_at) : '—';

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.7)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9990,
        padding: '16px',
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: '#FFFFFF',
          borderRadius: '12px',
          width: '95vw',
          maxWidth: '1350px',
          height: '90vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          border: '1px solid #E2E8F0',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top Header */}
        <div
          style={{
            padding: '12px 20px',
            borderBottom: '1px solid #E2E8F0',
            background: '#F8FAFC',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '12px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '20px' }}>📊</span>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 700, color: '#0F172A' }}>
                  Execution Canvas & DAG Inspector
                </h3>
                {runBadge && (
                  <span
                    style={{
                      fontSize: '11px',
                      fontWeight: 700,
                      padding: '2px 8px',
                      borderRadius: '4px',
                      border: `1px solid ${runBadge.borderColor}`,
                    }}
                    className={`${runBadge.badgeBg} ${runBadge.badgeText}`}
                  >
                    {runBadge.icon} {runBadge.label}
                  </span>
                )}
                {run?.scope_type && (
                  <span
                    style={{
                      fontSize: '10.5px',
                      fontFamily: 'monospace',
                      padding: '1px 6px',
                      borderRadius: '4px',
                      background: '#E2E8F0',
                      color: '#475569',
                      fontWeight: 600,
                    }}
                  >
                    {run.scope_type}
                  </span>
                )}
              </div>
              <div style={{ fontSize: '11px', color: '#64748B', fontFamily: 'monospace', marginTop: '2px' }}>
                Run ID: {runId} · Duration: {duration}
              </div>
            </div>
          </div>

          {/* Quick Metrics & View Toggle */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {run && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '11.5px', color: '#475569' }}>
                <div>
                  <span style={{ color: '#64748B' }}>Tokens: </span>
                  <strong>{run.tokens_spent.toLocaleString()}</strong>
                  <span style={{ color: '#94A3B8' }}> / {run.max_tokens_budget.toLocaleString()}</span>
                </div>
                <div>
                  <span style={{ color: '#64748B' }}>Calls: </span>
                  <strong>{run.provider_calls_spent}</strong>
                  <span style={{ color: '#94A3B8' }}> / {run.max_provider_calls_budget}</span>
                </div>
              </div>
            )}

            {/* View Switcher */}
            <div style={{ display: 'flex', background: '#E2E8F0', borderRadius: '6px', padding: '2px' }}>
              <button
                onClick={() => setActiveTab('canvas')}
                style={{
                  padding: '4px 10px',
                  fontSize: '11.5px',
                  fontWeight: 600,
                  borderRadius: '4px',
                  border: 'none',
                  cursor: 'pointer',
                  background: activeTab === 'canvas' ? '#FFFFFF' : 'transparent',
                  color: activeTab === 'canvas' ? '#1E293B' : '#64748B',
                  boxShadow: activeTab === 'canvas' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
                }}
              >
                DAG Canvas
              </button>
              <button
                onClick={() => setActiveTab('table')}
                style={{
                  padding: '4px 10px',
                  fontSize: '11.5px',
                  fontWeight: 600,
                  borderRadius: '4px',
                  border: 'none',
                  cursor: 'pointer',
                  background: activeTab === 'table' ? '#FFFFFF' : 'transparent',
                  color: activeTab === 'table' ? '#1E293B' : '#64748B',
                  boxShadow: activeTab === 'table' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
                }}
              >
                Task Table ({run?.tasks?.length || 0})
              </button>
            </div>

            <button
              onClick={onClose}
              style={{
                background: 'transparent',
                border: 'none',
                fontSize: '20px',
                color: '#64748B',
                cursor: 'pointer',
                padding: '2px 8px',
                borderRadius: '4px',
              }}
              title="Close Canvas"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Modal Main Area */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden', position: 'relative' }}>
          {loading && (
            <div
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'rgba(255, 255, 255, 0.8)',
                zIndex: 10,
                fontSize: '13px',
                color: '#64748B',
              }}
            >
              ⏳ Loading workflow execution DAG...
            </div>
          )}

          {error && (
            <div style={{ padding: '24px', color: '#DC2626', fontSize: '13px', width: '100%' }}>
              ⚠️ {error}
            </div>
          )}

          {!loading && activeTab === 'canvas' && (
            <>
              {/* ReactFlow Canvas Area */}
              <div style={{ flex: 1, height: '100%', position: 'relative', background: '#F8FAFC' }}>
                <ReactFlow
                  nodes={nodes}
                  edges={edges}
                  onNodesChange={onNodesChange}
                  onEdgesChange={onEdgesChange}
                  onNodeClick={handleNodeClick}
                  nodeTypes={nodeTypes}
                  fitView
                  fitViewOptions={{ padding: 0.2 }}
                  minZoom={0.2}
                  maxZoom={1.8}
                >
                  <Background color="#CBD5E1" gap={16} />
                  <Controls />
                  <MiniMap
                    nodeColor={(n) => {
                      const status = (n.data as any)?.status;
                      if (status === 'COMPLETED') return '#10B981';
                      if (status === 'RUNNING') return '#3B82F6';
                      if (status === 'FAILED') return '#EF4444';
                      if (status === 'AWAITING_APPROVAL') return '#F59E0B';
                      return '#94A3B8';
                    }}
                  />
                </ReactFlow>
              </div>

              {/* Task Inspector Sidebar */}
              <div style={{ width: '380px', height: '100%', display: 'flex', flexDirection: 'column' }}>
                <TaskInspectorPanel
                  task={selectedTask}
                  run={run!}
                  token={token}
                  onOpenArtifact={(id) => setSelectedArtifactId(id)}
                />
              </div>
            </>
          )}

          {!loading && activeTab === 'table' && run && (
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                <thead>
                  <tr style={{ background: '#F1F5F9', borderBottom: '2px solid #E2E8F0', textAlign: 'left' }}>
                    <th style={{ padding: '10px 12px' }}>Task Key</th>
                    <th style={{ padding: '10px 12px' }}>Type</th>
                    <th style={{ padding: '10px 12px' }}>Status</th>
                    <th style={{ padding: '10px 12px' }}>Dependencies</th>
                    <th style={{ padding: '10px 12px' }}>Duration</th>
                    <th style={{ padding: '10px 12px' }}>Tokens</th>
                    <th style={{ padding: '10px 12px' }}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {(run.tasks || []).map((t) => {
                    const tBadge = getExecutionStatusBadge(t.status, false);
                    const isSel = t.task_key === selectedTaskKey;
                    return (
                      <tr
                        key={t.id}
                        style={{
                          borderBottom: '1px solid #E2E8F0',
                          background: isSel ? '#EFF6FF' : 'transparent',
                        }}
                      >
                        <td style={{ padding: '10px 12px', fontFamily: 'monospace', fontWeight: 600 }}>
                          {t.task_key}
                        </td>
                        <td style={{ padding: '10px 12px' }}>
                          <span
                            style={{
                              fontSize: '10.5px',
                              fontFamily: 'monospace',
                              padding: '2px 6px',
                              borderRadius: '3px',
                              background: '#F1F5F9',
                            }}
                          >
                            {t.task_type}
                          </span>
                        </td>
                        <td style={{ padding: '10px 12px' }}>
                          <span
                            style={{
                              fontSize: '10.5px',
                              fontWeight: 700,
                              padding: '2px 6px',
                              borderRadius: '4px',
                            }}
                            className={`${tBadge.badgeBg} ${tBadge.badgeText}`}
                          >
                            {tBadge.icon} {tBadge.label}
                          </span>
                        </td>
                        <td style={{ padding: '10px 12px', color: '#64748B' }}>
                          {t.dependencies.length > 0 ? t.dependencies.join(', ') : '—'}
                        </td>
                        <td style={{ padding: '10px 12px', color: '#64748B' }}>
                          {formatDuration(t.created_at, t.completed_at)}
                        </td>
                        <td style={{ padding: '10px 12px', color: '#64748B' }}>
                          {t.tokens_spent.toLocaleString()}
                        </td>
                        <td style={{ padding: '10px 12px' }}>
                          <button
                            onClick={() => {
                              setSelectedTaskKey(t.task_key);
                              setActiveTab('canvas');
                            }}
                            style={{
                              padding: '3px 8px',
                              fontSize: '11px',
                              background: '#EFF6FF',
                              border: '1px solid #BFDBFE',
                              borderRadius: '4px',
                              color: '#1D4ED8',
                              cursor: 'pointer',
                            }}
                          >
                            Inspect in Canvas
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Embedded Secure Artifact Preview Modal */}
      {selectedArtifactId && (
        <ArtifactPreviewModal
          artifactId={selectedArtifactId}
          token={token}
          onClose={() => setSelectedArtifactId(null)}
        />
      )}
    </div>
  );
}

