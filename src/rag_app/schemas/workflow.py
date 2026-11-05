import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class WorkflowTaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"


class ScopeType(str, Enum):
    GLOBAL = "GLOBAL"
    WORKSPACE = "WORKSPACE"


class TaskType(str, Enum):
    RETRIEVAL = "RETRIEVAL"
    WEB_SEARCH = "WEB_SEARCH"
    EXTRACTION = "EXTRACTION"
    GROUNDING = "GROUNDING"
    SYNTHESIS = "SYNTHESIS"
    EXECUTION = "EXECUTION"
    IF_ELSE = "IF_ELSE"
    APPROVAL_GATE = "APPROVAL_GATE"


def compute_canonical_request_hash(
    query: str | dict[str, Any],
    workspace_id: Optional[UUID] = None,
    department_id: Optional[UUID] = None,
    budget_limits: Optional[dict[str, Any]] = None,
) -> str:
    """
    Computes deterministic SHA-256 hash of sorted canonical workflow request parameters.
    Used for engine-level idempotency and payload mutation conflict detection.
    """
    if isinstance(query, dict):
        query_str = json.dumps(query, sort_keys=True, separators=(",", ":"))
    else:
        query_str = str(query).strip()

    canonical_payload = {
        "query": query_str,
        "workspace_id": str(workspace_id) if workspace_id else None,
        "department_id": str(department_id) if department_id else None,
        "budget_limits": budget_limits or {},
    }
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_canonical_draft_hash(
    content_markdown: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """
    Computes cryptographic SHA-256 hash of draft content.
    Normalizes whitespace and line endings, and appends sorted metadata.
    """
    normalized_text = content_markdown.strip().replace("\r\n", "\n")
    hasher = hashlib.sha256(normalized_text.encode("utf-8"))
    if metadata:
        meta_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        hasher.update(b"\x00" + meta_bytes)
    return hasher.hexdigest()


class BudgetLimits(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    max_tokens: int = Field(default=25000, ge=1000, le=200000)
    max_provider_calls: int = Field(default=15, ge=1, le=100)
    wall_clock_timeout_seconds: int = Field(default=180, ge=10, le=3600)


class BudgetUsage(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    tokens_spent: int = Field(default=0, ge=0)
    tokens_reserved: int = Field(default=0, ge=0)
    provider_calls_spent: int = Field(default=0, ge=0)
    provider_calls_reserved: int = Field(default=0, ge=0)


class WorkflowTaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    workflow_run_id: UUID
    task_key: str
    task_type: str
    status: WorkflowTaskStatus
    dependencies: list[str] = Field(default_factory=list)
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_result: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    tokens_spent: int = 0
    provider_calls: int = 0
    retry_count: int = 0
    max_retries: int = 3
    locked_by: Optional[str] = None
    fencing_token: int = 0
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


class WorkflowRunCreate(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    workspace_id: Optional[UUID] = None
    department_id: Optional[UUID] = None
    conversation_id: Optional[UUID] = None
    idempotency_key: Optional[str] = Field(None, min_length=4, max_length=255)
    budget_limits: BudgetLimits = Field(default_factory=BudgetLimits)


class WorkflowRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    workspace_id: Optional[UUID] = None
    department_id: Optional[UUID] = None
    conversation_id: Optional[UUID] = None
    workflow_id: Optional[UUID] = None
    workflow_version_id: Optional[UUID] = None
    scope_type: ScopeType
    thread_id: Optional[str] = None
    status: WorkflowStatus
    idempotency_key: Optional[str] = None
    request_hash: str
    draft_hash: Optional[str] = None
    max_tokens_budget: int
    tokens_spent: int
    tokens_reserved: int
    max_provider_calls_budget: int
    provider_calls_spent: int
    provider_calls_reserved: int
    wall_clock_timeout_seconds: int
    failure_category: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    awaiting_approval_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    executing_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    tasks: list[WorkflowTaskResponse] = Field(default_factory=list)
    draft_text: Optional[str] = None
    evidence_items: list[dict[str, Any]] = Field(default_factory=list)
    approval_id: Optional[UUID] = None
    intended_action: Optional[str] = None
    target_record_id: Optional[str] = None
    approval_status: Optional[str] = None
    approval_expires_at: Optional[datetime] = None
    approval_requested_by: Optional[UUID] = None
    target_object: Optional[str] = None
    is_test: bool = Field(default=False, description="Whether this execution was run in TEST mode")
    execution_mode: str = Field(default="PRODUCTION", description="'TEST' or 'PRODUCTION'")
    provider_snapshot: Optional[dict[str, Any]] = Field(default=None, description="Immutable runtime configuration snapshot captured at run initialization")


class WorkflowApprovalRequest(BaseModel):
    decision: str = Field(..., description="Approval decision: 'APPROVED', 'REJECTED', or 'CANCELLED'")
    draft_hash: str = Field(..., min_length=64, max_length=64, description="Canonical SHA-256 draft hash issued by server")
    rejection_reason: Optional[str] = Field(None, max_length=1000, description="Optional feedback explanation if rejected")
    proposed_action: str = Field(default="CRM_CREATE_NOTE", max_length=100)
    proposed_target: Optional[str] = Field(None, max_length=255)


class WorkflowApprovalResponse(BaseModel):
    workflow_run_id: UUID
    approval_id: UUID
    status: WorkflowStatus
    side_effect_executed: bool = False
    error_message: Optional[str] = None
    writeback_result: Optional[dict[str, Any]] = None


# ============================================================================
# Milestone 2: Adaptive Task Planner & Typed DAG Execution Contracts
# ============================================================================

class ExecutionTask(BaseModel):
    """
    Normalized, validated unit of work within an AdaptiveTaskPlan DAG.
    Specifies dependencies, expected inputs, resource estimates, and safety flags.
    """
    model_config = ConfigDict(from_attributes=True)

    task_key: str = Field(..., min_length=1, max_length=100, description="Stable, unique task key within the plan")
    task_type: TaskType = Field(..., description="Allowed task type from strict server-side allowlist")
    description: str = Field(..., min_length=1, max_length=1000, description="Human-readable purpose of the task")
    dependencies: list[str] = Field(default_factory=list, description="Keys of prerequisite tasks that must complete before this task")
    input_payload: dict[str, Any] = Field(default_factory=dict, description="Input parameters/payload for task execution")
    estimated_tokens: int = Field(default=1000, ge=0, le=100000, description="Estimated token reservation requirement")
    estimated_provider_calls: int = Field(default=1, ge=0, le=50, description="Estimated external provider calls")
    timeout_seconds: int = Field(default=60, ge=1, le=1800, description="Task execution timeout in seconds")
    is_read_only: bool = Field(default=True, description="Whether the task has no external side effects and can be safely parallelized")
    capability_id: Optional[str] = Field(default=None, description="Resolved capability identifier in CapabilityRegistry")
    expected_output_schema: dict[str, Any] = Field(default_factory=dict, description="Expected schema descriptor for task output")
    side_effect_level: str = Field(default="READ_ONLY", description="Side effect level (READ_ONLY, EXTERNAL_READ, DRAFT, WRITE, HIGH_IMPACT)")
    required_permissions: list[str] = Field(default_factory=list, description="Permissions required to execute this task")


class AdaptiveTaskPlan(BaseModel):
    """
    A typed, validated execution plan forming a Directed Acyclic Graph (DAG).
    Generated by AdaptiveTaskPlanner and validated by DAGValidator prior to execution.
    """
    model_config = ConfigDict(from_attributes=True)

    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    workflow_run_id: Optional[UUID] = Field(default=None, description="Bound WorkflowRun ID in durable storage")
    goal: str = Field(..., min_length=1, max_length=5000, description="High-level goal or complex query being planned")
    tasks: list[ExecutionTask] = Field(..., min_length=1, max_length=25, description="Topological list of tasks in the DAG")
    iteration: int = Field(default=1, ge=1, le=3, description="Adaptive replan iteration (hard maximum 3)")
    total_estimated_tokens: int = Field(default=0, ge=0)
    total_estimated_provider_calls: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskEvidenceRef(BaseModel):
    """
    Typed, provenance-preserving evidence extracted by an ExecutionTask.
    Preserves source identity, location, confidence, and conflicting claims.
    """
    model_config = ConfigDict(from_attributes=True)

    evidence_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    task_key: str = Field(..., description="Task that produced this evidence item")
    source_type: str = Field(..., description="Source origin: 'internal_rag', 'web_search', 'crm', etc.")
    source_id: str = Field(..., description="Stable identifier of document, URL, record, or chunk")
    source_uri: Optional[str] = Field(default=None, description="Resolvable URI or URL where applicable")
    snippet: str = Field(..., min_length=1, description="Verbatim or summarized evidence snippet")
    facts: list[dict[str, Any]] = Field(default_factory=list, description="Extracted structured facts if applicable")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    contradictions: list[str] = Field(default_factory=list, description="Contradictions noted against other evidence")
    unresolved_gaps: list[str] = Field(default_factory=list, description="Unresolved gaps identified during research")
    provenance_metadata: dict[str, Any] = Field(default_factory=dict)


class TaskExecutionResult(BaseModel):
    """
    Standardized result contract returned upon completion of an ExecutionTask.
    """
    model_config = ConfigDict(from_attributes=True)

    task_key: str
    status: WorkflowTaskStatus
    output_result: Optional[dict[str, Any]] = None
    evidence_items: list[TaskEvidenceRef] = Field(default_factory=list)
    tokens_spent: int = Field(default=0, ge=0)
    provider_calls_spent: int = Field(default=0, ge=0)
    error_message: Optional[str] = None
    fencing_token: int = Field(default=0, ge=0)


class DAGValidationResult(BaseModel):
    """
    Deterministic validation summary for an AdaptiveTaskPlan DAG.
    """
    model_config = ConfigDict(from_attributes=True)

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    topological_order: list[str] = Field(default_factory=list)
    max_concurrency: int = 0

