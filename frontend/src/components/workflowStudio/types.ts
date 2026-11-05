import type { WorkflowActionType, WorkflowTriggerType } from '../../api/workflowDefinitions.ts';
import type { WorkflowTaskResponse } from '../../api/workflows.ts';

export interface ActionMetadata {
  type: WorkflowActionType;
  name: string;
  category: 'knowledge' | 'control' | 'actions';
  description: string;
  icon: string;
  color: {
    bg: string;
    border: string;
    text: string;
    badgeBg: string;
  };
  defaultInputs: Record<string, any>;
  outputSchema: Record<string, string>;
}

export interface TriggerMetadata {
  type: WorkflowTriggerType;
  name: string;
  description: string;
  icon: string;
  defaultConfig: Record<string, any>;
}

export const TRIGGER_CATALOG: Record<WorkflowTriggerType, TriggerMetadata> = {
  MANUAL: {
    type: 'MANUAL',
    name: 'Manual / API Trigger',
    description: 'Triggered on-demand via direct user action or authenticated API call.',
    icon: 'Play',
    defaultConfig: {},
  },
  SCHEDULE: {
    type: 'SCHEDULE',
    name: 'Cron / Interval Schedule',
    description: 'Autonomous execution on periodic cron schedule with timezone & misfire policies.',
    icon: 'Clock',
    defaultConfig: {
      cron: '0 9 * * *',
      timezone: 'UTC',
      misfire_policy: 'RUN_ONCE',
    },
  },
  WEBHOOK: {
    type: 'WEBHOOK',
    name: 'Twenty CRM Webhook',
    description: 'HMAC-SHA256 signed inbound webhook from Twenty CRM or external services.',
    icon: 'Webhook',
    defaultConfig: {
      secret_configured: true,
      allowed_events: ['company.created', 'person.updated'],
    },
  },
  EVENT: {
    type: 'EVENT',
    name: 'Internal System Event',
    description: 'Triggered by internal workspace events (document uploaded, curation updated).',
    icon: 'Bell',
    defaultConfig: {},
  },
};

export const ACTION_CATALOG: Record<WorkflowActionType, ActionMetadata> = {
  RETRIEVAL: {
    type: 'RETRIEVAL',
    name: 'RAG Retrieval',
    category: 'knowledge',
    description: 'Hybrid vector & lexical search across internal workspace documents.',
    icon: 'Database',
    color: {
      bg: '#f0fdf4',
      border: '#86efac',
      text: '#166534',
      badgeBg: '#dcfce7',
    },
    defaultInputs: {
      query: '{{trigger.query}}',
      limit: 3,
      scope: 'workspace',
    },
    outputSchema: {
      status: 'string (success)',
      retrieved_count: 'number',
      chunk_ids: 'string[]',
    },
  },
  WEB_SEARCH: {
    type: 'WEB_SEARCH',
    name: 'Web Research',
    category: 'knowledge',
    description: 'External search and scraping via TinyFish / Firecrawl providers.',
    icon: 'Globe',
    color: {
      bg: '#f0fdfa',
      border: '#5eead4',
      text: '#115e59',
      badgeBg: '#ccfbf1',
    },
    defaultInputs: {
      query: '{{trigger.query}}',
      limit: 5,
    },
    outputSchema: {
      status: 'string (success)',
      results_count: 'number',
      urls: 'string[]',
    },
  },
  EXTRACTION: {
    type: 'EXTRACTION',
    name: 'Fact Extraction',
    category: 'knowledge',
    description: 'Extracts structured key facts, entities, and data points from upstream evidence.',
    icon: 'FileText',
    color: {
      bg: '#eff6ff',
      border: '#93c5fd',
      text: '#1e40af',
      badgeBg: '#dbeafe',
    },
    defaultInputs: {
      query: 'Extract key business details and numerical metrics',
      fields: ['revenue', 'competitors', 'key_dates'],
      source_tasks: [],
    },
    outputSchema: {
      status: 'string (success)',
      extracted_facts_count: 'number',
      data: 'Record<string, any>',
    },
  },
  GROUNDING: {
    type: 'GROUNDING',
    name: 'Grounding Verification',
    category: 'knowledge',
    description: 'Anti-hallucination verification checking evidence coverage and confidence.',
    icon: 'ShieldCheck',
    color: {
      bg: '#fdf4ff',
      border: '#f0abfc',
      text: '#86198f',
      badgeBg: '#fae8ff',
    },
    defaultInputs: {
      claim: '{{steps.synthesis.output.synthesis_text}}',
      confidence_threshold: 0.7,
    },
    outputSchema: {
      status: 'string (grounding_verified)',
      is_grounded: 'boolean',
      confidence: 'number (0.0 - 1.0)',
      unresolved_gaps: 'string[]',
    },
  },
  SYNTHESIS: {
    type: 'SYNTHESIS',
    name: 'LLM Synthesis',
    category: 'knowledge',
    description: 'Evidence-grounded text, report, or brief synthesis with citations.',
    icon: 'Sparkles',
    color: {
      bg: '#fffbeb',
      border: '#fde68a',
      text: '#92400e',
      badgeBg: '#fef3c7',
    },
    defaultInputs: {
      prompt_template: 'Synthesize comprehensive research report based on retrieved evidence.\nFocus: {{trigger.query}}',
      focus: 'Executive Brief',
      max_length: 2000,
    },
    outputSchema: {
      status: 'string (synthesized)',
      synthesis_text: 'string',
      citations_count: 'number',
    },
  },
  IF_ELSE: {
    type: 'IF_ELSE',
    name: 'IF / ELSE Branch',
    category: 'control',
    description: 'Evaluates conditional logic to route graph execution branches.',
    icon: 'GitBranch',
    color: {
      bg: '#fff7ed',
      border: '#fdba74',
      text: '#9a3412',
      badgeBg: '#ffedd5',
    },
    defaultInputs: {
      left_value: '{{steps.grounding.output.is_grounded}}',
      operator: 'EQUALS',
      right_value: true,
      true_branch: 'branch_verified',
      false_branch: 'branch_review',
    },
    outputSchema: {
      result: 'boolean',
      branch: 'string',
      operator: 'string',
      evaluated_left: 'any',
      evaluated_right: 'any',
    },
  },
  APPROVAL_GATE: {
    type: 'APPROVAL_GATE',
    name: 'Approval Gate (HITL)',
    category: 'control',
    description: 'Cryptographic human-in-the-loop approval barrier for high-risk actions.',
    icon: 'UserCheck',
    color: {
      bg: '#fef2f2',
      border: '#fca5a5',
      text: '#991b1b',
      badgeBg: '#fee2e2',
    },
    defaultInputs: {
      gate_name: 'Manager Approval Gate',
      required_role: 'ADMIN',
    },
    outputSchema: {
      gate_status: 'string (APPROVED | PENDING)',
      approval_id: 'string',
    },
  },
  EXECUTION: {
    type: 'EXECUTION',
    name: 'Restricted CRM Write',
    category: 'actions',
    description: 'Side-effecting mutating action protected by server-side approval and idempotency.',
    icon: 'Send',
    color: {
      bg: '#faf5ff',
      border: '#d8b4fe',
      text: '#6b21a8',
      badgeBg: '#f3e8ff',
    },
    defaultInputs: {
      target_system: 'crm',
      action: 'CRM_CREATE_NOTE',
      payload: {
        note_body: '{{steps.synthesis.output.synthesis_text}}',
      },
    },
    outputSchema: {
      status: 'string (executed)',
      executed: 'boolean',
      mutation_id: 'string',
      details: 'Record<string, any>',
    },
  },
};

export interface StudioNodeData {
  id: string;
  name: string;
  node_type: WorkflowActionType;
  inputs: Record<string, any>;
  dependencies: string[];
  isTrigger?: boolean;
  triggerType?: WorkflowTriggerType;
  triggerConfig?: Record<string, any>;
  isValid?: boolean;
  validationErrors?: string[];
  // Phase 4B: Execution Inspection
  isInspectionMode?: boolean;
  executionStatus?:
    | 'PENDING'
    | 'READY'
    | 'RUNNING'
    | 'COMPLETED'
    | 'FAILED'
    | 'CANCELLED'
    | 'SKIPPED'
    | 'AWAITING_APPROVAL';
  taskData?: WorkflowTaskResponse | null;
  durationMs?: number | null;
  [key: string]: any;
}
