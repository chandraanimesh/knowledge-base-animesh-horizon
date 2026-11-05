import type {
  WorkflowNodeDefinition,
  WorkflowTriggerSpec,
  WorkflowCanvasLayout,
  WorkflowVersionResponse,
  WorkflowActionType,
} from '../../api/workflowDefinitions.ts';
import type { StudioNodeData } from './types.ts';
import { ACTION_CATALOG } from './types.ts';

export const NODE_ID_REGEX = /^[a-zA-Z0-9_\-]{1,64}$/;

export interface CycleCheckResult {
  hasCycle: boolean;
  message?: string;
}

/**
 * Validates a node ID against backend constraints:
 * 1-64 alphanumeric characters, underscores, or hyphens.
 */
export function isValidNodeId(id: string): boolean {
  return NODE_ID_REGEX.test(id.trim());
}

/**
 * Checks if adding an edge (or the current set of edges) would form a cycle in the DAG.
 * Uses Kahn's algorithm (topological sort / in-degree reduction).
 */
export function checkCycle(
  nodes: { id: string }[],
  edges: { source: string; target: string }[],
  proposedEdge?: { source: string; target: string }
): CycleCheckResult {
  if (proposedEdge) {
    if (proposedEdge.source === proposedEdge.target) {
      return {
        hasCycle: true,
        message: 'Self-referencing edges are not allowed: a step cannot depend on itself.',
      };
    }
  }

  const allEdges = proposedEdge
    ? [...edges.filter((e) => !(e.source === proposedEdge.source && e.target === proposedEdge.target)), proposedEdge]
    : edges;

  const nodeIds = new Set(nodes.map((n) => n.id));
  if (proposedEdge) {
    nodeIds.add(proposedEdge.source);
    nodeIds.add(proposedEdge.target);
  }

  const inDegree: Record<string, number> = {};
  const adj: Record<string, string[]> = {};

  for (const id of nodeIds) {
    inDegree[id] = 0;
    adj[id] = [];
  }

  for (const edge of allEdges) {
    if (nodeIds.has(edge.source) && nodeIds.has(edge.target)) {
      adj[edge.source].push(edge.target);
      inDegree[edge.target] = (inDegree[edge.target] || 0) + 1;
    }
  }

  // Queue nodes with in-degree 0
  const queue: string[] = [];
  for (const id of nodeIds) {
    if (inDegree[id] === 0) {
      queue.push(id);
    }
  }

  let visitedCount = 0;
  while (queue.length > 0) {
    const curr = queue.shift()!;
    visitedCount++;

    for (const neighbor of adj[curr]) {
      inDegree[neighbor]--;
      if (inDegree[neighbor] === 0) {
        queue.push(neighbor);
      }
    }
  }

  if (visitedCount < nodeIds.size) {
    return {
      hasCycle: true,
      message: 'Cycle detected: adding this connection would create a closed loop dependency.',
    };
  }

  return { hasCycle: false };
}

/**
 * Validates inputs for a specific action type against required fields.
 */
export function validateNodeInputs(
  nodeType: WorkflowActionType,
  inputs: Record<string, any>
): string[] {
  const errors: string[] = [];

  switch (nodeType) {
    case 'RETRIEVAL':
      if (!inputs.query || String(inputs.query).trim().length === 0) {
        errors.push("Retrieval requires a non-empty 'query' parameter.");
      }
      break;

    case 'WEB_SEARCH':
      if (!inputs.query || String(inputs.query).trim().length === 0) {
        errors.push("Web search requires a non-empty 'query' parameter.");
      }
      break;

    case 'EXTRACTION':
      if (inputs.fields && !Array.isArray(inputs.fields)) {
        errors.push("'fields' must be an array of string field names.");
      }
      break;

    case 'GROUNDING':
      if (inputs.confidence_threshold !== undefined) {
        const conf = Number(inputs.confidence_threshold);
        if (isNaN(conf) || conf < 0 || conf > 1) {
          errors.push('Confidence threshold must be a number between 0.0 and 1.0.');
        }
      }
      break;

    case 'SYNTHESIS':
      if (
        (!inputs.prompt_template || String(inputs.prompt_template).trim().length === 0) &&
        (!inputs.focus || String(inputs.focus).trim().length === 0)
      ) {
        errors.push("Synthesis requires either a 'prompt_template' or a 'focus' query.");
      }
      break;

    case 'IF_ELSE':
      if (inputs.left_value === undefined || inputs.left_value === null || String(inputs.left_value).trim().length === 0) {
        errors.push("IF/ELSE requires a 'left_value' operand.");
      }
      if (!inputs.operator) {
        errors.push("IF/ELSE requires a comparison 'operator'.");
      }
      break;

    case 'APPROVAL_GATE':
      if (!inputs.gate_name || String(inputs.gate_name).trim().length === 0) {
        errors.push("Approval Gate requires a 'gate_name'.");
      }
      break;

    case 'EXECUTION':
      if (!inputs.action || String(inputs.action).trim().length === 0) {
        errors.push("Execution action requires a specific 'action' name.");
      }
      break;
  }

  return errors;
}

/**
 * Serializes React Flow nodes and edges into separated execution graph (steps_spec)
 * and presentation layout (canvas_layout) for backend persistence.
 */
export function serializeCanvasToBackend(
  rfNodes: any[],
  rfEdges: any[],
  triggerSpec: WorkflowTriggerSpec,
  viewport?: { x: number; y: number; zoom: number }
): {
  steps_spec: WorkflowNodeDefinition[];
  canvas_layout: WorkflowCanvasLayout;
  trigger_spec: WorkflowTriggerSpec;
} {
  // 1. Build dependency map from edges (source -> target means target depends on source)
  const dependencyMap: Record<string, Set<string>> = {};
  for (const node of rfNodes) {
    if (!node.data?.isTrigger) {
      dependencyMap[node.id] = new Set<string>();
    }
  }

  for (const edge of rfEdges) {
    // If target is an action step and source is an action step, record dependency
    if (dependencyMap[edge.target] && edge.source !== 'trigger') {
      dependencyMap[edge.target].add(edge.source);
    }
  }

  // 2. Build execution graph (steps_spec) - strictly execution metadata
  const steps_spec: WorkflowNodeDefinition[] = [];
  const canvasNodesLayout: any[] = [];

  for (const node of rfNodes) {
    if (node.data?.isTrigger) {
      // Save trigger position in canvas_layout
      canvasNodesLayout.push({
        id: node.id,
        position: { x: node.position.x, y: node.position.y },
        data: { isTrigger: true, triggerType: triggerSpec.trigger_type },
      });
      continue;
    }

    const data: StudioNodeData = node.data || {};
    const dependencies = Array.from(dependencyMap[node.id] || []);

    steps_spec.push({
      id: node.id,
      name: data.name || node.id,
      node_type: data.node_type || 'RETRIEVAL',
      inputs: data.inputs || {},
      dependencies,
    });

    // Save presentation position in canvas_layout
    canvasNodesLayout.push({
      id: node.id,
      position: { x: node.position.x, y: node.position.y },
      data: {
        name: data.name,
        node_type: data.node_type,
      },
    });
  }

  const canvasEdgesLayout = rfEdges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
  }));

  const canvas_layout: WorkflowCanvasLayout = {
    nodes: canvasNodesLayout,
    edges: canvasEdgesLayout,
    viewport: viewport || { x: 0, y: 0, zoom: 1 },
  };

  return {
    steps_spec,
    canvas_layout,
    trigger_spec: triggerSpec,
  };
}

/**
 * Deserializes backend WorkflowVersionResponse into React Flow nodes and edges.
 * Auto-arranges nodes if canvas_layout is empty or missing coordinates.
 */
export function deserializeBackendToCanvas(version: WorkflowVersionResponse): {
  nodes: any[];
  edges: any[];
  triggerSpec: WorkflowTriggerSpec;
  viewport?: { x: number; y: number; zoom: number };
} {
  const triggerSpec = version.trigger_spec || {
    trigger_type: 'MANUAL',
    config: {},
  };

  const layoutNodeMap = new Map<string, { x: number; y: number }>();
  if (version.canvas_layout?.nodes) {
    for (const n of version.canvas_layout.nodes) {
      if (n.position) {
        layoutNodeMap.set(n.id, n.position);
      }
    }
  }

  const nodes: any[] = [];
  const edges: any[] = [];

  // 1. Add Trigger Node
  const triggerPos = layoutNodeMap.get('trigger') || { x: 50, y: 200 };
  nodes.push({
    id: 'trigger',
    type: 'workflowNode',
    position: triggerPos,
    data: {
      id: 'trigger',
      name: `${triggerSpec.trigger_type} Trigger`,
      isTrigger: true,
      triggerType: triggerSpec.trigger_type,
      triggerConfig: triggerSpec.config || {},
      node_type: 'RETRIEVAL' as WorkflowActionType,
      inputs: {},
      dependencies: [],
    },
  });

  // 2. Add Step Nodes
  const steps = version.steps_spec || [];
  steps.forEach((step, idx) => {
    const existingPos = layoutNodeMap.get(step.id);
    const pos = existingPos || {
      x: 350 + (idx % 3) * 280,
      y: 100 + Math.floor(idx / 3) * 180,
    };

    const actionMeta = ACTION_CATALOG[step.node_type];
    const validationErrors = validateNodeInputs(step.node_type, step.inputs || {});

    nodes.push({
      id: step.id,
      type: 'workflowNode',
      position: pos,
      data: {
        id: step.id,
        name: step.name,
        node_type: step.node_type,
        inputs: step.inputs || actionMeta?.defaultInputs || {},
        dependencies: step.dependencies || [],
        isValid: validationErrors.length === 0,
        validationErrors,
      },
    });

    // 3. Add Edges based on step dependencies
    if (step.dependencies && step.dependencies.length > 0) {
      for (const depId of step.dependencies) {
        edges.push({
          id: `e-${depId}-${step.id}`,
          source: depId,
          target: step.id,
          animated: true,
          style: { stroke: '#94a3b8', strokeWidth: 2 },
        });
      }
    } else {
      // If no step dependencies, optionally connect from trigger
      edges.push({
        id: `e-trigger-${step.id}`,
        source: 'trigger',
        target: step.id,
        animated: true,
        style: { stroke: '#94a3b8', strokeWidth: 2 },
      });
    }
  });

  // If canvas_layout already had explicitly saved edges, use those if matching node IDs
  if (version.canvas_layout?.edges && version.canvas_layout.edges.length > 0) {
    const validNodeIds = new Set(nodes.map((n) => n.id));
    const layoutEdges = version.canvas_layout.edges
      .filter((e) => validNodeIds.has(e.source) && validNodeIds.has(e.target))
      .map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        animated: true,
        style: { stroke: '#94a3b8', strokeWidth: 2 },
      }));

    if (layoutEdges.length > 0) {
      return {
        nodes,
        edges: layoutEdges,
        triggerSpec,
        viewport: version.canvas_layout.viewport,
      };
    }
  }

  return {
    nodes,
    edges,
    triggerSpec,
    viewport: version.canvas_layout?.viewport,
  };
}
