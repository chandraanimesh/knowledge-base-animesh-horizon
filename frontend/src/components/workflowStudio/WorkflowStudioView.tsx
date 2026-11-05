import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  ReactFlowProvider,
  useReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import {
  WorkflowDefinitionResponse,
  WorkflowVersionResponse,
  WorkflowActionType,
  WorkflowTriggerType,
  WorkflowTriggerSpec,
  listWorkflowDefinitions,
  createWorkflowDefinition,
  listWorkflowVersions,
  getWorkflowVersion,
  createDraftVersion,
  updateDraftVersion,
  validateWorkflowDag,
  publishWorkflowVersion,
} from '../../api/workflowDefinitions.ts';
import {
  listWorkflowRuns,
  getWorkflowRun,
  type WorkflowRunResponse,
} from '../../api/workflows.ts';
import { WorkflowCanvasNode } from './WorkflowCanvasNode.tsx';
import { NodeLibrarySidebar } from './NodeLibrarySidebar.tsx';
import { RunHistorySidebar } from './RunHistorySidebar.tsx';
import { NodeConfigPanel } from './NodeConfigPanel.tsx';
import { WorkflowApprovalModal } from './WorkflowApprovalModal.tsx';
import { WorkflowTestRunModal } from './WorkflowTestRunModal.tsx';
import { ACTION_CATALOG, TRIGGER_CATALOG, StudioNodeData } from './types.ts';
import {
  checkCycle,
  validateNodeInputs,
  serializeCanvasToBackend,
  deserializeBackendToCanvas,
} from './dagUtils.ts';
import {
  isTerminalStatus,
  formatDuration,
  formatMetric,
  getExecutionStatusBadge,
} from './executionUtils.ts';

const nodeTypes = {
  workflowNode: WorkflowCanvasNode,
};

interface WorkflowStudioViewProps {
  workspaceId: string;
  token: string;
  userRole?: string; // 'ADMIN' | 'OWNER' | 'EDITOR' | 'VIEWER'
}

/**
 * Pure helper that maps task execution states onto canvas nodes without mutating original specs.
 */
function applyExecutionOverlay(
  baseNodes: any[],
  run: WorkflowRunResponse | null
): any[] {
  if (!run) {
    return baseNodes.map((n) => ({
      ...n,
      data: {
        ...n.data,
        isInspectionMode: false,
        executionStatus: undefined,
        taskData: undefined,
      },
    }));
  }

  return baseNodes.map((n) => {
    if (n.data?.isTrigger) {
      return {
        ...n,
        data: {
          ...n.data,
          isInspectionMode: true,
          executionStatus: undefined,
          taskData: undefined,
        },
      };
    }

    const task = run.tasks.find((t) => t.task_key === n.id);
    let execStatus = task?.status || 'PENDING';
    if (
      run.status === 'AWAITING_APPROVAL' &&
      n.data?.node_type === 'APPROVAL_GATE' &&
      execStatus !== 'COMPLETED'
    ) {
      execStatus = 'AWAITING_APPROVAL';
    }

    return {
      ...n,
      data: {
        ...n.data,
        isInspectionMode: true,
        executionStatus: execStatus,
        taskData: task || null,
      },
    };
  });
}

function StudioCanvasInner({
  workspaceId,
  token,
  userRole = 'EDITOR',
}: WorkflowStudioViewProps) {
  const isViewer = userRole === 'VIEWER';
  const reactFlowInstance = useReactFlow();

  // Studio Mode: 'design' (Draft & Edit) vs 'execution' (Runs Timeline & Inspector)
  const [studioMode, setStudioMode] = useState<'design' | 'execution'>('design');

  // Workflow & Version Selection State
  const [workflows, setWorkflows] = useState<WorkflowDefinitionResponse[]>([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState<WorkflowDefinitionResponse | null>(null);
  const [versions, setVersions] = useState<WorkflowVersionResponse[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<WorkflowVersionResponse | null>(null);

  // Phase 4B & 4C: Run History & Approval State
  const [runs, setRuns] = useState<WorkflowRunResponse[]>([]);
  const [selectedRun, setSelectedRun] = useState<WorkflowRunResponse | null>(null);
  const [showApprovalModal, setShowApprovalModal] = useState<boolean>(false);
  const [showTestModal, setShowTestModal] = useState<boolean>(false);
  const [isPolling, setIsPolling] = useState<boolean>(false);
  const pollingTimerRef = useRef<any>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Loading & Feedback State
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [validating, setValidating] = useState<boolean>(false);
  const [publishing, setPublishing] = useState<boolean>(false);
  const [toast, setToast] = useState<{ type: 'success' | 'error' | 'warning' | 'info'; message: string } | null>(null);

  // New Workflow Modal
  const [showNewWfModal, setShowNewWfModal] = useState<boolean>(false);
  const [newWfName, setNewWfName] = useState<string>('');
  const [newWfDesc, setNewWfDesc] = useState<string>('');

  // Canvas State (Nodes & Edges)
  const [nodes, setNodes, onNodesChange] = useNodesState<any>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [triggerSpec, setTriggerSpec] = useState<WorkflowTriggerSpec>({
    trigger_type: 'MANUAL',
    config: {},
  });
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  const showToast = useCallback((type: 'success' | 'error' | 'warning' | 'info', message: string) => {
    setToast({ type, message });
    setTimeout(() => setToast(null), 4000);
  }, []);

  // Is the currently loaded version immutable? (ACTIVE or ARCHIVED)
  const isVersionImmutable = selectedVersion ? selectedVersion.status !== 'DRAFT' : false;
  const isReadOnly = isViewer || isVersionImmutable || studioMode === 'execution';

  // Cleanup polling timers and in-flight abort controllers on unmount
  useEffect(() => {
    return () => {
      if (pollingTimerRef.current) {
        clearInterval(pollingTimerRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // ---------------------------------------------------------------------------
  // 1. Initial Load: Workflows in Workspace
  // ---------------------------------------------------------------------------
  const loadWorkflows = useCallback(async () => {
    if (!workspaceId || !token) return;
    try {
      setLoading(true);
      const list = await listWorkflowDefinitions(workspaceId, token);
      setWorkflows(list);
      if (list.length > 0 && !selectedWorkflow) {
        setSelectedWorkflow(list[0]);
      }
    } catch (err: any) {
      const msg = err.message || String(err);
      if (msg.includes('relation "workflow_definitions" does not exist') || msg.includes('500')) {
        showToast('warning', 'Database tables missing: Run "alembic upgrade head" to enable workflow persistence.');
      } else {
        showToast('error', `Failed to load workflows: ${msg}`);
      }
    } finally {
      setLoading(false);
    }
  }, [workspaceId, token, selectedWorkflow, showToast]);

  useEffect(() => {
    loadWorkflows();
  }, [loadWorkflows]);

  // ---------------------------------------------------------------------------
  // 2. Load Versions for Selected Workflow
  // ---------------------------------------------------------------------------
  const loadWorkflowDetails = useCallback(
    async (workflowId: string) => {
      if (!workspaceId || !token) return;
      try {
        setLoading(true);
        const vList = await listWorkflowVersions(workspaceId, workflowId, token);
        setVersions(vList);

        // Pick preferred version: latest DRAFT, or ACTIVE, or first
        let targetVersion =
          vList.find((v) => v.status === 'DRAFT') ||
          vList.find((v) => v.status === 'ACTIVE') ||
          vList[0];

        if (targetVersion) {
          const fullVersion = await getWorkflowVersion(workspaceId, workflowId, targetVersion.id, token);
          setSelectedVersion(fullVersion);

          const { nodes: loadedNodes, edges: loadedEdges, triggerSpec: loadedTrig } =
            deserializeBackendToCanvas(fullVersion);

          setNodes(loadedNodes);
          setEdges(loadedEdges);
          setTriggerSpec(loadedTrig);
          setSelectedNodeId(null);
        }

        // Also fetch runs list in background
        const rList = await listWorkflowRuns(token, {
          workflow_id: workflowId,
          workspace_id: workspaceId,
        });
        setRuns(rList);
      } catch (err: any) {
        showToast('error', `Failed to load workflow details: ${err.message || err}`);
      } finally {
        setLoading(false);
      }
    },
    [workspaceId, token, setNodes, setEdges, showToast]
  );

  useEffect(() => {
    if (selectedWorkflow) {
      loadWorkflowDetails(selectedWorkflow.id);
    }
  }, [selectedWorkflow, loadWorkflowDetails]);

  // ---------------------------------------------------------------------------
  // 3. Switch Selected Version (In Design Mode)
  // ---------------------------------------------------------------------------
  const handleSelectVersion = async (versionId: string) => {
    if (!selectedWorkflow || !workspaceId || !token) return;
    try {
      setLoading(true);
      const fullVersion = await getWorkflowVersion(workspaceId, selectedWorkflow.id, versionId, token);
      setSelectedVersion(fullVersion);

      const { nodes: loadedNodes, edges: loadedEdges, triggerSpec: loadedTrig } =
        deserializeBackendToCanvas(fullVersion);

      setNodes(loadedNodes);
      setEdges(loadedEdges);
      setTriggerSpec(loadedTrig);
      setSelectedNodeId(null);
      showToast('info', `Switched to version ${fullVersion.version_number} (${fullVersion.status})`);
    } catch (err: any) {
      showToast('error', `Failed to switch version: ${err.message || err}`);
    } finally {
      setLoading(false);
    }
  };

  // ---------------------------------------------------------------------------
  // 4. Phase 4B: Run Selection, Version Synchronization & Polling
  // ---------------------------------------------------------------------------
  const startPollingRun = useCallback(
    (runId: string) => {
      if (pollingTimerRef.current) {
        clearInterval(pollingTimerRef.current);
      }

      pollingTimerRef.current = setInterval(async () => {
        try {
          const refreshedRun = await getWorkflowRun(runId, token);
          setSelectedRun(refreshedRun);

          // Update run in runs array
          setRuns((prev) => prev.map((r) => (r.id === refreshedRun.id ? refreshedRun : r)));

          // Update execution badge overlay on canvas
          setNodes((currentNodes) => applyExecutionOverlay(currentNodes, refreshedRun));

          // Automatically stop polling on terminal states
          if (isTerminalStatus(refreshedRun.status)) {
            if (pollingTimerRef.current) {
              clearInterval(pollingTimerRef.current);
              pollingTimerRef.current = null;
            }
            setIsPolling(false);
          }
        } catch (err: any) {
          console.warn('Live poll error for workflow run:', runId, err);
        }
      }, 2500);
    },
    [token, setNodes]
  );

  const handleSelectRun = useCallback(
    async (runId: string) => {
      if (!selectedWorkflow || !workspaceId || !token) return;

      // Abort any in-flight request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      // Clear existing polling loop
      if (pollingTimerRef.current) {
        clearInterval(pollingTimerRef.current);
        pollingTimerRef.current = null;
      }

      try {
        setLoading(true);
        const fullRun = await getWorkflowRun(runId, token);
        if (abortController.signal.aborted) return;

        setSelectedRun(fullRun);

        // Ensure canvas reflects the EXACT immutable version that this run executed
        let canvasNodes = nodes;
        if (
          fullRun.workflow_version_id &&
          (!selectedVersion || selectedVersion.id !== fullRun.workflow_version_id)
        ) {
          const executedVersion = await getWorkflowVersion(
            workspaceId,
            selectedWorkflow.id,
            fullRun.workflow_version_id,
            token
          );
          if (abortController.signal.aborted) return;
          setSelectedVersion(executedVersion);

          const deserialized = deserializeBackendToCanvas(executedVersion);
          canvasNodes = deserialized.nodes;
          setEdges(deserialized.edges);
          setTriggerSpec(deserialized.triggerSpec);
        }

        // Apply task execution overlay
        setNodes(applyExecutionOverlay(canvasNodes, fullRun));

        // Start bounded polling if run is active
        if (!isTerminalStatus(fullRun.status)) {
          setIsPolling(true);
          startPollingRun(fullRun.id);
        } else {
          setIsPolling(false);
        }
      } catch (err: any) {
        if (!abortController.signal.aborted) {
          showToast('error', `Failed to load execution run: ${err.message || err}`);
        }
      } finally {
        if (!abortController.signal.aborted) {
          setLoading(false);
        }
      }
    },
    [selectedWorkflow, workspaceId, token, nodes, selectedVersion, setEdges, setNodes, showToast, startPollingRun]
  );

  const handleRefreshRuns = async () => {
    if (!selectedWorkflow || !workspaceId || !token) return;
    try {
      setLoading(true);
      const rList = await listWorkflowRuns(token, {
        workflow_id: selectedWorkflow.id,
        workspace_id: workspaceId,
      });
      setRuns(rList);
      if (selectedRun) {
        const updatedSelected = await getWorkflowRun(selectedRun.id, token);
        setSelectedRun(updatedSelected);
        setNodes((current) => applyExecutionOverlay(current, updatedSelected));
      } else if (rList.length > 0) {
        handleSelectRun(rList[0].id);
      }
      showToast('info', 'Refreshed run history.');
    } catch (err: any) {
      showToast('error', `Refresh failed: ${err.message || err}`);
    } finally {
      setLoading(false);
    }
  };

  // Switch between Design and Execution modes
  const handleSwitchMode = async (mode: 'design' | 'execution') => {
    if (mode === studioMode) return;

    if (mode === 'execution') {
      setStudioMode('execution');
      if (runs.length > 0) {
        handleSelectRun(runs[0].id);
      } else if (selectedWorkflow) {
        const rList = await listWorkflowRuns(token, {
          workflow_id: selectedWorkflow.id,
          workspace_id: workspaceId,
        });
        setRuns(rList);
        if (rList.length > 0) {
          handleSelectRun(rList[0].id);
        } else {
          showToast('info', 'No recorded execution runs found for this workflow.');
        }
      }
    } else {
      // Switching to Design mode: stop polling & restore clean canvas
      if (pollingTimerRef.current) {
        clearInterval(pollingTimerRef.current);
        pollingTimerRef.current = null;
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      setIsPolling(false);
      setSelectedRun(null);
      setStudioMode('design');

      // Reload active or draft version
      if (selectedWorkflow) {
        loadWorkflowDetails(selectedWorkflow.id);
      }
    }
  };

  const handleTestRunSuccess = useCallback((testRun: WorkflowRunResponse) => {
    setRuns((prev) => [testRun, ...prev.filter((r) => r.id !== testRun.id)]);
    setSelectedRun(testRun);
    setStudioMode('execution');
    showToast('success', `Test execution finished: Status ${testRun.status}`);
  }, [showToast]);

  // ---------------------------------------------------------------------------
  // 5. Create New Workflow Definition
  // ---------------------------------------------------------------------------
  const handleCreateWorkflow = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newWfName.trim()) return;

    try {
      setSaving(true);
      const viewport = reactFlowInstance?.getViewport() || { x: 0, y: 0, zoom: 1 };
      const currentSerialized = serializeCanvasToBackend(nodes, edges, triggerSpec, viewport);
      const hasCustomSteps = currentSerialized.steps_spec.length > 0;

      const created = await createWorkflowDefinition(
        {
          workspace_id: workspaceId,
          name: newWfName.trim(),
          description: newWfDesc.trim() || null,
          ...(hasCustomSteps
            ? {
                initial_trigger: currentSerialized.trigger_spec,
                initial_steps: currentSerialized.steps_spec,
                initial_canvas_layout: currentSerialized.canvas_layout,
              }
            : {}),
        },
        token
      );
      setWorkflows((prev) => [created, ...prev]);
      setSelectedWorkflow(created);
      setShowNewWfModal(false);
      setNewWfName('');
      setNewWfDesc('');
      showToast('success', `Created workflow "${created.name}" and saved canvas steps!`);
    } catch (err: any) {
      const msg = err.message || String(err);
      if (
        msg.includes('relation "workflow_definitions" does not exist') ||
        msg.includes('500') ||
        msg.includes('Internal Server Error')
      ) {
        showToast('error', 'Database tables missing: Pending migration. Please run "alembic upgrade head".');
      } else {
        showToast('error', `Creation failed: ${msg}`);
      }
    } finally {
      setSaving(false);
    }
  };

  // ---------------------------------------------------------------------------
  // 6. Connect Edges & Cycle Detection
  // ---------------------------------------------------------------------------
  const onConnect = useCallback(
    (connection: Connection) => {
      if (isReadOnly) return;
      if (!connection.source || !connection.target) return;

      if (connection.source === connection.target) {
        showToast('warning', 'A step cannot connect to itself.');
        return;
      }

      if (connection.target === 'trigger') {
        showToast('warning', 'Trigger node cannot have upstream dependencies.');
        return;
      }

      const cycleCheck = checkCycle(nodes, edges, {
        source: connection.source,
        target: connection.target,
      });

      if (cycleCheck.hasCycle) {
        showToast('error', cycleCheck.message || 'Cycle detected! Connection rejected.');
        return;
      }

      setEdges((eds) =>
        addEdge(
          {
            ...connection,
            animated: true,
            style: { stroke: '#0d9488', strokeWidth: 2 },
          },
          eds
        )
      );

      setNodes((nds) =>
        nds.map((n) => {
          if (n.id === connection.target && connection.source !== 'trigger') {
            const deps = new Set(n.data?.dependencies || []);
            deps.add(connection.source);
            return {
              ...n,
              data: {
                ...n.data,
                dependencies: Array.from(deps),
              },
            };
          }
          return n;
        })
      );
    },
    [isReadOnly, nodes, edges, setEdges, setNodes, showToast]
  );

  // ---------------------------------------------------------------------------
  // 7. Node Drag & Drop Creation
  // ---------------------------------------------------------------------------
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      if (isReadOnly) return;

      const actionType = event.dataTransfer.getData('application/reactflow/type') as WorkflowActionType;
      if (!actionType || !ACTION_CATALOG[actionType]) return;

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      handleAddNode(actionType, position);
    },
    [isReadOnly, reactFlowInstance]
  );

  const handleAddNode = (actionType: WorkflowActionType, customPos?: { x: number; y: number }) => {
    if (isReadOnly) return;
    const actionMeta = ACTION_CATALOG[actionType];
    const stepCount = nodes.filter((n) => !n.data?.isTrigger).length;
    const newId = `step_${actionType.toLowerCase()}_${stepCount + 1}`;

    const pos = customPos || {
      x: 350 + (stepCount % 3) * 260,
      y: 120 + Math.floor(stepCount / 3) * 160,
    };

    const initialInputs = { ...actionMeta.defaultInputs };
    const validationErrors = validateNodeInputs(actionType, initialInputs);

    const newNode = {
      id: newId,
      type: 'workflowNode',
      position: pos,
      data: {
        id: newId,
        name: actionMeta.name,
        node_type: actionType,
        inputs: initialInputs,
        dependencies: [],
        isValid: validationErrors.length === 0,
        validationErrors,
      },
    };

    setNodes((nds) => [...nds, newNode]);
    setSelectedNodeId(newId);
    showToast('info', `Added "${actionMeta.name}" step`);
  };

  // ---------------------------------------------------------------------------
  // 8. Node Data Update & Deletion
  // ---------------------------------------------------------------------------
  const handleUpdateNodeData = (nodeId: string, updatedData: Partial<StudioNodeData>) => {
    if (isReadOnly) return;

    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === nodeId) {
          const merged = { ...n.data, ...updatedData };
          if (merged.node_type && merged.inputs) {
            merged.validationErrors = validateNodeInputs(merged.node_type, merged.inputs);
            merged.isValid = merged.validationErrors.length === 0;
          }
          return {
            ...n,
            id: updatedData.id || n.id,
            data: merged,
          };
        }
        return n;
      })
    );
  };

  const handleDeleteNode = (nodeId: string) => {
    if (isReadOnly) return;
    if (nodeId === 'trigger') {
      showToast('warning', 'Trigger node cannot be deleted.');
      return;
    }

    setNodes((nds) => nds.filter((n) => n.id !== nodeId));
    setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
    setSelectedNodeId(null);
    showToast('info', `Deleted step "${nodeId}"`);
  };

  const handleUpdateTriggerSpec = (newSpec: WorkflowTriggerSpec) => {
    if (isReadOnly) return;
    setTriggerSpec(newSpec);
    setNodes((nds) =>
      nds.map((n) => {
        if (n.data?.isTrigger) {
          return {
            ...n,
            data: {
              ...n.data,
              name: `${newSpec.trigger_type} Trigger`,
              triggerType: newSpec.trigger_type,
              triggerConfig: newSpec.config,
            },
          };
        }
        return n;
      })
    );
  };

  // ---------------------------------------------------------------------------
  // 9. Save Draft
  // ---------------------------------------------------------------------------
  const handleSaveDraft = async () => {
    if (!selectedWorkflow) {
      setShowNewWfModal(true);
      showToast('info', 'Please name your workflow to save this canvas design.');
      return;
    }
    if (!selectedVersion) {
      showToast('warning', 'No version loaded to save.');
      return;
    }
    if (isReadOnly) {
      showToast(
        'warning',
        'Cannot edit: This version is published (ACTIVE) and immutable. Click "+ New Draft from Version" to create an editable copy.'
      );
      return;
    }

    try {
      setSaving(true);
      const viewport = reactFlowInstance.getViewport();
      const payload = serializeCanvasToBackend(nodes, edges, triggerSpec, viewport);

      const updated = await updateDraftVersion(
        workspaceId,
        selectedWorkflow.id,
        selectedVersion.id,
        payload,
        token
      );

      setSelectedVersion(updated);
      showToast('success', `Draft v${updated.version_number} saved successfully!`);
    } catch (err: any) {
      showToast('error', `Save failed: ${err.message || err}`);
    } finally {
      setSaving(false);
    }
  };

  // ---------------------------------------------------------------------------
  // 10. Validate DAG
  // ---------------------------------------------------------------------------
  const handleValidateDAG = async () => {
    if (!selectedWorkflow) {
      setShowNewWfModal(true);
      showToast('info', 'Please name and save your workflow definition before validating.');
      return;
    }
    if (!selectedVersion) {
      showToast('warning', 'No version loaded to validate.');
      return;
    }

    try {
      setValidating(true);
      if (!isReadOnly) {
        const viewport = reactFlowInstance.getViewport();
        const payload = serializeCanvasToBackend(nodes, edges, triggerSpec, viewport);
        await updateDraftVersion(
          workspaceId,
          selectedWorkflow.id,
          selectedVersion.id,
          payload,
          token
        );
      }

      const res = await validateWorkflowDag(
        workspaceId,
        selectedWorkflow.id,
        selectedVersion.id,
        token
      );

      if (res.is_valid) {
        showToast('success', `DAG Valid! (${res.node_count} nodes, 0 cycles detected)`);
      } else {
        showToast('warning', `Validation Errors: ${res.errors.join('; ')}`);
      }
    } catch (err: any) {
      showToast('error', `Validation failed: ${err.message || err}`);
    } finally {
      setValidating(false);
    }
  };

  // ---------------------------------------------------------------------------
  // 11. Publish Version
  // ---------------------------------------------------------------------------
  const handlePublish = async () => {
    if (!selectedWorkflow) {
      setShowNewWfModal(true);
      showToast('info', 'Please name and save your workflow definition before publishing.');
      return;
    }
    if (!selectedVersion) {
      showToast('warning', 'No version loaded to publish.');
      return;
    }
    if (isReadOnly) {
      showToast('warning', 'Cannot publish: This version is already published (ACTIVE) and immutable.');
      return;
    }

    if (
      !window.confirm(
        `Are you sure you want to publish v${selectedVersion.version_number}? Once published, this version becomes immutable and active.`
      )
    ) {
      return;
    }

    try {
      setPublishing(true);
      const viewport = reactFlowInstance.getViewport();
      const payload = serializeCanvasToBackend(nodes, edges, triggerSpec, viewport);
      await updateDraftVersion(
        workspaceId,
        selectedWorkflow.id,
        selectedVersion.id,
        payload,
        token
      );

      const published = await publishWorkflowVersion(
        workspaceId,
        selectedWorkflow.id,
        selectedVersion.id,
        token
      );

      setSelectedVersion(published);
      await loadWorkflowDetails(selectedWorkflow.id);
      showToast('success', `Version v${published.version_number} published successfully!`);
    } catch (err: any) {
      showToast('error', `Publish failed: ${err.message || err}`);
    } finally {
      setPublishing(false);
    }
  };

  // ---------------------------------------------------------------------------
  // 12. Create New Draft Fork
  // ---------------------------------------------------------------------------
  const handleCreateDraftFork = async () => {
    if (!selectedWorkflow || !selectedVersion) return;
    try {
      setSaving(true);
      const newDraft = await createDraftVersion(
        workspaceId,
        selectedWorkflow.id,
        { fork_from_version_id: selectedVersion.id },
        token
      );

      setVersions((prev) => [newDraft, ...prev]);
      setSelectedVersion(newDraft);

      const { nodes: loadedNodes, edges: loadedEdges, triggerSpec: loadedTrig } =
        deserializeBackendToCanvas(newDraft);

      setNodes(loadedNodes);
      setEdges(loadedEdges);
      setTriggerSpec(loadedTrig);
      setSelectedNodeId(null);

      showToast('success', `Created Draft v${newDraft.version_number} for editing.`);
    } catch (err: any) {
      showToast('error', `Failed to create draft: ${err.message || err}`);
    } finally {
      setSaving(false);
    }
  };

  // Pre-load RAG Research Template shortcut
  const handleLoadTemplate = () => {
    if (isReadOnly) return;
    const trigId = 'trigger';
    const s1 = 'step_retrieval';
    const s2 = 'step_web_search';
    const s3 = 'step_grounding';
    const s4 = 'step_synthesis';

    const templateNodes = [
      {
        id: trigId,
        type: 'workflowNode',
        position: { x: 50, y: 180 },
        data: {
          id: trigId,
          name: 'Manual Trigger',
          isTrigger: true,
          triggerType: 'MANUAL',
          triggerConfig: {},
          node_type: 'RETRIEVAL',
          inputs: {},
          dependencies: [],
        },
      },
      {
        id: s1,
        type: 'workflowNode',
        position: { x: 340, y: 80 },
        data: {
          id: s1,
          name: 'Internal Document RAG',
          node_type: 'RETRIEVAL',
          inputs: { query: '{{trigger.query}}', limit: 3, scope: 'workspace' },
          dependencies: [],
          isValid: true,
          validationErrors: [],
        },
      },
      {
        id: s2,
        type: 'workflowNode',
        position: { x: 340, y: 280 },
        data: {
          id: s2,
          name: 'Web Competitor Search',
          node_type: 'WEB_SEARCH',
          inputs: { query: '{{trigger.query}}', limit: 5 },
          dependencies: [],
          isValid: true,
          validationErrors: [],
        },
      },
      {
        id: s3,
        type: 'workflowNode',
        position: { x: 650, y: 180 },
        data: {
          id: s3,
          name: 'Evidence Grounding',
          node_type: 'GROUNDING',
          inputs: { claim: '{{steps.step_synthesis.output.synthesis_text}}', confidence_threshold: 0.75 },
          dependencies: [s1, s2],
          isValid: true,
          validationErrors: [],
        },
      },
      {
        id: s4,
        type: 'workflowNode',
        position: { x: 960, y: 180 },
        data: {
          id: s4,
          name: 'Executive Synthesis',
          node_type: 'SYNTHESIS',
          inputs: {
            prompt_template: 'Synthesize research brief based on retrieved and verified evidence.\nQuery: {{trigger.query}}',
            focus: 'Brief',
            max_length: 2000,
          },
          dependencies: [s3],
          isValid: true,
          validationErrors: [],
        },
      },
    ];

    const templateEdges = [
      { id: 'e-trig-s1', source: trigId, target: s1, animated: true, style: { stroke: '#0d9488', strokeWidth: 2 } },
      { id: 'e-trig-s2', source: trigId, target: s2, animated: true, style: { stroke: '#0d9488', strokeWidth: 2 } },
      { id: 'e-s1-s3', source: s1, target: s3, animated: true, style: { stroke: '#0d9488', strokeWidth: 2 } },
      { id: 'e-s2-s3', source: s2, target: s3, animated: true, style: { stroke: '#0d9488', strokeWidth: 2 } },
      { id: 'e-s3-s4', source: s3, target: s4, animated: true, style: { stroke: '#0d9488', strokeWidth: 2 } },
    ];

    setNodes(templateNodes);
    setEdges(templateEdges);
    showToast('info', 'Loaded RAG Research Template!');
  };

  const selectedNode = nodes.find((n) => n.id === selectedNodeId) || null;
  const actionStepCount = nodes.filter((n) => !n.data?.isTrigger).length;

  return (
    <div className="flex flex-col flex-1 h-full w-full bg-[#f0f0eb] select-none overflow-hidden min-h-0 overscroll-none">
      {/* ── TOP TOOLBAR ──────────────────────────────────────────────────────── */}
      <header className="h-14 bg-[#fafaf7] border-b border-[#d4d4cc] px-4 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          {/* Mode Switcher */}
          <div className="flex bg-slate-200/80 p-0.5 rounded-lg border border-slate-300">
            <button
              type="button"
              onClick={() => handleSwitchMode('design')}
              className={`text-xs font-semibold px-3 py-1 rounded-md transition-all ${
                studioMode === 'design'
                  ? 'bg-white text-slate-800 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
            >
              ✏ Studio Editor
            </button>
            <button
              type="button"
              onClick={() => handleSwitchMode('execution')}
              className={`text-xs font-semibold px-3 py-1 rounded-md transition-all flex items-center gap-1.5 ${
                studioMode === 'execution'
                  ? 'bg-white text-slate-800 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
            >
              <span>⚡ Executions</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-slate-100 text-slate-700 font-mono">
                {runs.length}
              </span>
            </button>
          </div>

          {/* Workflow Selector */}
          <div className="flex items-center gap-1.5 pl-3 border-l border-slate-300">
            <span className="text-xs font-bold text-slate-500 uppercase">Workflow:</span>
            <select
              value={selectedWorkflow?.id || ''}
              onChange={(e) => {
                const wf = workflows.find((w) => w.id === e.target.value);
                if (wf) setSelectedWorkflow(wf);
              }}
              className="text-xs font-semibold px-2 py-1 rounded border border-[#d4d4cc] bg-white text-slate-800"
            >
              {workflows.length === 0 ? (
                <option value="">No Workflows (Click + New)</option>
              ) : (
                workflows.map((wf) => (
                  <option key={wf.id} value={wf.id}>
                    {wf.name} {wf.is_active ? '' : '(Inactive)'}
                  </option>
                ))
              )}
            </select>
            {!isViewer && studioMode === 'design' && (
              <button
                type="button"
                onClick={() => setShowNewWfModal(true)}
                className="text-xs font-semibold px-2 py-1 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded border border-[#d4d4cc]"
              >
                + New
              </button>
            )}
          </div>

          {/* DESIGN MODE: Version Selector & Status Badge */}
          {studioMode === 'design' && versions.length > 0 && selectedVersion && (
            <div className="flex items-center gap-2 pl-3 border-l border-slate-300">
              <span className="text-xs font-bold text-slate-500 uppercase">Version:</span>
              <select
                value={selectedVersion.id}
                onChange={(e) => handleSelectVersion(e.target.value)}
                className="text-xs font-mono px-2 py-1 rounded border border-[#d4d4cc] bg-white text-slate-800"
              >
                {versions.map((v) => (
                  <option key={v.id} value={v.id}>
                    v{v.version_number} — {v.status}
                  </option>
                ))}
              </select>

              <span
                className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider ${
                  selectedVersion.status === 'ACTIVE'
                    ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                    : selectedVersion.status === 'ARCHIVED'
                    ? 'bg-slate-200 text-slate-600 border border-slate-300'
                    : 'bg-amber-100 text-amber-800 border border-amber-300'
                }`}
              >
                {selectedVersion.status}
              </span>
            </div>
          )}

          {/* EXECUTION MODE: Active Run Selector Dropdown */}
          {studioMode === 'execution' && (
            <div className="flex items-center gap-2 pl-3 border-l border-slate-300">
              <span className="text-xs font-bold text-slate-500 uppercase">Run:</span>
              <select
                value={selectedRun?.id || ''}
                onChange={(e) => handleSelectRun(e.target.value)}
                className="text-xs font-mono px-2 py-1 rounded border border-[#d4d4cc] bg-white text-slate-800 max-w-[200px]"
              >
                {runs.length === 0 ? (
                  <option value="">No runs</option>
                ) : (
                  runs.map((r, idx) => (
                    <option key={r.id} value={r.id}>
                      Run #{runs.length - idx} ({r.status})
                    </option>
                  ))
                )}
              </select>

              {selectedRun && (
                <div className="flex items-center gap-1.5">
                  {(() => {
                    const badge = getExecutionStatusBadge(selectedRun.status);
                    return (
                      <span
                        className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider flex items-center gap-1 ${
                          badge.badgeBg
                        } ${badge.badgeText} ${badge.pulse ? 'animate-pulse' : ''}`}
                      >
                        <span>{badge.icon}</span>
                        <span>{badge.label}</span>
                      </span>
                    );
                  })()}

                  <span className="text-[10px] font-mono text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
                    v-pinned
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Action Buttons */}
        <div className="flex items-center gap-2">
          {studioMode === 'execution' ? (
            <>
              {isPolling && (
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-teal-50 border border-teal-200 text-teal-800 text-xs font-medium">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-teal-600"></span>
                  </span>
                  <span>Polling every 2.5s</span>
                </div>
              )}

              <button
                type="button"
                onClick={handleRefreshRuns}
                disabled={loading}
                className="px-3 py-1.5 text-xs font-semibold bg-white hover:bg-slate-50 text-slate-700 border border-[#d4d4cc] rounded-lg shadow-xs transition-colors"
              >
                ↻ Refresh
              </button>

              {!isViewer && selectedVersion && (
                <button
                  type="button"
                  onClick={() => setShowTestModal(true)}
                  disabled={loading}
                  className="px-3 py-1.5 text-xs font-semibold bg-purple-600 hover:bg-purple-700 text-white rounded-lg shadow-xs transition-colors flex items-center gap-1.5"
                >
                  <span>🧪</span>
                  <span>New Test Run</span>
                </button>
              )}

              <button
                type="button"
                onClick={() => handleSwitchMode('design')}
                className="px-3 py-1.5 text-xs font-semibold bg-teal-700 hover:bg-teal-800 text-white rounded-lg shadow-xs transition-colors"
              >
                ← Back to Editor
              </button>
            </>
          ) : (
            <>
              {isViewer && (
                <span className="text-[11px] font-semibold text-slate-500 bg-slate-100 px-2.5 py-1 rounded border border-slate-200">
                  👁️ Viewer (Read-Only)
                </span>
              )}

              {!isViewer && selectedVersion && (
                <button
                  type="button"
                  onClick={async () => {
                    if (!isVersionImmutable && !isViewer && selectedWorkflow && selectedVersion) {
                      try {
                        const viewport = reactFlowInstance.getViewport();
                        const payload = serializeCanvasToBackend(nodes, edges, triggerSpec, viewport);
                        await updateDraftVersion(workspaceId, selectedWorkflow.id, selectedVersion.id, payload, token);
                      } catch (err: any) {
                        console.warn('Auto-save before test run:', err);
                      }
                    }
                    setShowTestModal(true);
                  }}
                  disabled={loading || saving}
                  className="px-3 py-1.5 text-xs font-semibold bg-purple-600 hover:bg-purple-700 text-white rounded-lg shadow-xs transition-colors flex items-center gap-1.5"
                >
                  <span>🧪</span>
                  <span>Test Run</span>
                </button>
              )}

              {isVersionImmutable && !isViewer && (
                <button
                  type="button"
                  onClick={handleCreateDraftFork}
                  disabled={saving}
                  className="px-3 py-1.5 text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white rounded-lg shadow-xs transition-colors"
                >
                  + New Draft from Version
                </button>
              )}

              {!isVersionImmutable && !isViewer && (
                <>
                  <button
                    type="button"
                    onClick={handleSaveDraft}
                    disabled={saving}
                    className="px-3 py-1.5 text-xs font-semibold bg-white hover:bg-slate-50 text-slate-700 border border-[#d4d4cc] rounded-lg shadow-xs transition-colors"
                  >
                    {saving ? 'Saving...' : 'Save Draft'}
                  </button>

                  <button
                    type="button"
                    onClick={handleValidateDAG}
                    disabled={validating}
                    className="px-3 py-1.5 text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-800 border border-slate-300 rounded-lg shadow-xs transition-colors"
                  >
                    {validating ? 'Validating...' : 'Validate DAG'}
                  </button>

                  <button
                    type="button"
                    onClick={handlePublish}
                    disabled={publishing}
                    className="px-3 py-1.5 text-xs font-semibold bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg shadow-xs transition-colors"
                  >
                    {publishing ? 'Publishing...' : 'Publish Version'}
                  </button>
                </>
              )}
            </>
          )}
        </div>
      </header>

      {/* Toast Alert */}
      {toast && (
        <div
          className={`px-4 py-2 text-xs font-medium border-b flex items-center justify-between ${
            toast.type === 'success'
              ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
              : toast.type === 'error'
              ? 'bg-red-50 text-red-800 border-red-200'
              : toast.type === 'warning'
              ? 'bg-amber-50 text-amber-800 border-amber-200'
              : 'bg-blue-50 text-blue-800 border-blue-200'
          }`}
        >
          <span>{toast.message}</span>
          <button onClick={() => setToast(null)} className="text-xs ml-4">
            ✕
          </button>
        </div>
      )}

      {/* ── 3-COLUMN STUDIO LAYOUT ─────────────────────────────────────────── */}
      <div className="flex-1 flex w-full overflow-hidden relative min-h-0">
        {/* Left Column: Node Library Sidebar in Design mode OR Run History in Execution mode */}
        {studioMode === 'execution' ? (
          <RunHistorySidebar
            runs={runs}
            selectedRunId={selectedRun?.id || null}
            onSelectRun={(rId) => handleSelectRun(rId)}
            onRefreshRuns={handleRefreshRuns}
            loading={loading}
            isPolling={isPolling}
            onOpenTestModal={() => setShowTestModal(true)}
          />
        ) : (
          <NodeLibrarySidebar
            onAddNode={(type) => handleAddNode(type)}
            onChangeTrigger={(tType) =>
              handleUpdateTriggerSpec({ trigger_type: tType, config: TRIGGER_CATALOG[tType].defaultConfig })
            }
            activeTriggerType={triggerSpec.trigger_type}
            isReadOnly={isReadOnly}
          />
        )}

        {/* Center: Interactive React Flow Canvas */}
        <div
          ref={reactFlowWrapper}
          className="flex-1 w-full bg-[#f4f4ef] relative flex flex-col min-h-0 overflow-hidden"
          onDragOver={onDragOver}
          onDrop={onDrop}
        >
          {/* Phase 4B: Execution Run Status Banner */}
          {studioMode === 'execution' && selectedRun && (
            <div className="bg-[#fcfcf9] border-b border-[#d4d4cc] px-4 py-2 flex items-center justify-between z-10 shrink-0">
              <div className="flex items-center gap-3 text-xs">
                <span className="font-bold text-slate-700">Execution Overview:</span>
                <span className="font-mono text-slate-500 text-[11px]">
                  ID: <span className="text-slate-800">{selectedRun.id.slice(0, 8)}...</span>
                </span>
                <span className="text-slate-400">|</span>
                <span className="text-slate-600 text-[11px]">
                  Duration: <strong className="font-mono">{formatDuration(selectedRun.created_at, selectedRun.completed_at || selectedRun.failed_at)}</strong>
                </span>
                <span className="text-slate-400">|</span>
                <span className="text-slate-600 text-[11px]">
                  Tokens: <strong className="font-mono">{formatMetric(selectedRun.tokens_spent)}</strong> / {formatMetric(selectedRun.max_tokens_budget)}
                </span>
                <span className="text-slate-400">|</span>
                <span className="text-slate-600 text-[11px]">
                  Calls: <strong className="font-mono">{formatMetric(selectedRun.provider_calls_spent)}</strong> / {formatMetric(selectedRun.max_provider_calls_budget)}
                </span>
              </div>

              {selectedRun.status === 'AWAITING_APPROVAL' && (
                <button
                  type="button"
                  onClick={() => setShowApprovalModal(true)}
                  className="text-[11px] font-bold text-amber-900 bg-amber-100 hover:bg-amber-200 px-3 py-1 rounded-full border border-amber-300 animate-pulse flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
                >
                  <span>⏸</span>
                  <span>Awaiting Human Approval — Review &amp; Decide</span>
                </button>
              )}
            </div>
          )}

          <div className="flex-1 w-full relative min-h-0">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              style={{ width: '100%', height: '100%' }}
              onNodesChange={isReadOnly ? undefined : onNodesChange}
              onEdgesChange={isReadOnly ? undefined : onEdgesChange}
              onConnect={isReadOnly ? undefined : onConnect}
              nodeTypes={nodeTypes}
              onNodeClick={(_, node) => setSelectedNodeId(node.id)}
              onPaneClick={() => setSelectedNodeId(null)}
              fitView
              minZoom={0.2}
              maxZoom={1.8}
              deleteKeyCode={isReadOnly ? null : ['Backspace', 'Delete']}
            >
              <Background gap={18} size={1} color="#cbd5e1" />
              <Controls className="!bg-white !border !border-slate-200 !shadow-sm" />
              <MiniMap
                className="!bg-white/80 !border !border-slate-200 !rounded-lg"
                nodeStrokeColor="#94a3b8"
                nodeColor="#f8fafc"
              />
            </ReactFlow>

            {/* Empty Canvas Onboarding Overlay */}
            {actionStepCount === 0 && studioMode === 'design' && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="bg-white/95 p-6 rounded-2xl border border-slate-300 shadow-xl max-w-md text-center pointer-events-auto backdrop-blur-xs">
                  <div className="w-12 h-12 bg-teal-50 text-teal-700 rounded-full flex items-center justify-center text-xl mx-auto mb-3 border border-teal-200 font-bold">
                    ✦
                  </div>
                  <h3 className="text-sm font-bold text-slate-800">Visual Workflow Studio</h3>
                  <p className="text-xs text-slate-500 mt-1">
                    Drag action components from the left sidebar onto the canvas, or start immediately with our standard research template.
                  </p>
                  {!isReadOnly && (
                    <div className="mt-4 flex justify-center gap-2">
                      <button
                        type="button"
                        onClick={handleLoadTemplate}
                        className="px-3.5 py-2 text-xs font-semibold bg-teal-700 hover:bg-teal-800 text-white rounded-lg shadow-xs transition-colors"
                      >
                        Load RAG Research Template
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right: Node Configuration & Variable Inspector Panel */}
        <NodeConfigPanel
          selectedNode={selectedNode}
          triggerSpec={triggerSpec}
          allNodes={nodes}
          onUpdateNodeData={handleUpdateNodeData}
          onUpdateTriggerSpec={handleUpdateTriggerSpec}
          onDeleteNode={handleDeleteNode}
          onClose={() => setSelectedNodeId(null)}
          isReadOnly={isReadOnly}
          isInspectionMode={studioMode === 'execution'}
          activeRun={selectedRun}
          onOpenApprovalModal={() => setShowApprovalModal(true)}
        />
      </div>

      {/* ── MODAL: HUMAN APPROVAL (PHASE 4C) ──────────────────────────────── */}
      {showApprovalModal && selectedRun && (
        <WorkflowApprovalModal
          run={selectedRun}
          workflowName={selectedWorkflow?.name}
          versionNumber={selectedVersion?.version_number}
          token={token}
          isOpen={showApprovalModal}
          onClose={() => setShowApprovalModal(false)}
          onDecisionCompleted={(res) => {
            setShowApprovalModal(false);
            showToast(
              res.status === 'COMPLETED' ? 'success' : res.status === 'REJECTED' ? 'warning' : 'info',
              `Approval decision "${res.status}" recorded. Side-effect: ${res.side_effect_executed ? 'Executed' : 'None'}`
            );
            handleSelectRun(res.workflow_run_id);
          }}
        />
      )}

      {/* ── MODAL: TEST RUN (PHASE 4D) ────────────────────────────────────── */}
      {showTestModal && selectedWorkflow && selectedVersion && (
        <WorkflowTestRunModal
          isOpen={showTestModal}
          onClose={() => setShowTestModal(false)}
          workspaceId={workspaceId}
          workflowId={selectedWorkflow.id}
          workflowName={selectedWorkflow.name}
          versionId={selectedVersion.id}
          versionNumber={selectedVersion.version_number}
          versionStatus={selectedVersion.status}
          triggerType={triggerSpec.trigger_type}
          token={token}
          onTestRunSuccess={handleTestRunSuccess}
        />
      )}

      {/* ── MODAL: CREATE WORKFLOW ─────────────────────────────────────────── */}
      {showNewWfModal && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-xs flex items-center justify-center z-50">
          <div className="bg-white rounded-xl border border-slate-300 shadow-2xl w-full max-w-sm p-5">
            <h3 className="text-sm font-bold text-slate-800 mb-3">Create Workflow Definition</h3>
            <form onSubmit={handleCreateWorkflow} className="space-y-3">
              <div>
                <label className="block text-[11px] font-bold text-slate-600 uppercase mb-1">
                  Workflow Name *
                </label>
                <input
                  type="text"
                  required
                  value={newWfName}
                  onChange={(e) => setNewWfName(e.target.value)}
                  placeholder="e.g. Due Diligence Brief"
                  className="w-full text-xs px-2.5 py-2 rounded border border-slate-300 focus:outline-none focus:ring-1 focus:ring-teal-600"
                />
              </div>

              <div>
                <label className="block text-[11px] font-bold text-slate-600 uppercase mb-1">
                  Description
                </label>
                <textarea
                  rows={2}
                  value={newWfDesc}
                  onChange={(e) => setNewWfDesc(e.target.value)}
                  placeholder="Optional workflow goal or summary..."
                  className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-300 focus:outline-none focus:ring-1 focus:ring-teal-600"
                />
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowNewWfModal(false)}
                  className="px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-100 rounded border border-slate-200"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={saving || !newWfName.trim()}
                  className="px-3.5 py-1.5 text-xs font-semibold bg-teal-700 hover:bg-teal-800 text-white rounded shadow-xs disabled:opacity-50"
                >
                  {saving ? 'Creating...' : 'Create Definition'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export function WorkflowStudioView(props: WorkflowStudioViewProps) {
  return (
    <ReactFlowProvider>
      <StudioCanvasInner {...props} />
    </ReactFlowProvider>
  );
}
