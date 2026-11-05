"""
Typed, validated DAG schemas and DTOs for Persistent Workflow Definitions & Versioning.
Enforces graph integrity, cycle detection, dependency completeness, action allowlists,
presentation metadata decoupling, and security sanitization.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from enum import Enum
import re
from typing import Any, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkflowVersionStatus(str, Enum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"


class WorkflowActionType(str, Enum):
    RETRIEVAL = "RETRIEVAL"
    WEB_SEARCH = "WEB_SEARCH"
    EXTRACTION = "EXTRACTION"
    GROUNDING = "GROUNDING"
    SYNTHESIS = "SYNTHESIS"
    EXECUTION = "EXECUTION"
    IF_ELSE = "IF_ELSE"
    APPROVAL_GATE = "APPROVAL_GATE"


class WorkflowTriggerType(str, Enum):
    MANUAL = "MANUAL"
    SCHEDULE = "SCHEDULE"
    WEBHOOK = "WEBHOOK"
    EVENT = "EVENT"


# Node ID validation pattern: alphanumeric, hyphen, underscore (1 to 64 chars)
NODE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
FORBIDDEN_KEY_PATTERNS = ["__class__", "__subclasses__", "__globals__", "eval(", "exec(", "import "]


class WorkflowNodeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique node identifier in the workflow graph")
    name: str = Field(..., min_length=1, max_length=255, description="Human-readable node label")
    node_type: WorkflowActionType = Field(..., description="Allowlisted task action type")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Typed node input parameters")
    dependencies: list[str] = Field(default_factory=list, description="IDs of upstream parent nodes")

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        clean = v.strip()
        if not clean or not NODE_ID_REGEX.match(clean):
            raise ValueError(f"Invalid node id '{v}'. Must be 1-64 alphanumeric chars, underscores or hyphens.")
        return clean

    @field_validator("inputs")
    @classmethod
    def sanitize_inputs(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Prevents code injection / prototype pollution in JSON specifications."""
        serialized = str(v).lower()
        for forbidden in FORBIDDEN_KEY_PATTERNS:
            if forbidden in serialized:
                raise ValueError(f"Security violation: Forbidden pattern '{forbidden}' detected in node inputs.")
        return v


class WorkflowTriggerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger_type: WorkflowTriggerType = Field(default=WorkflowTriggerType.MANUAL)
    config: dict[str, Any] = Field(default_factory=dict)


class CanvasPosition(BaseModel):
    x: float
    y: float


class CanvasNodeLayout(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    position: CanvasPosition
    data: Optional[dict[str, Any]] = None


class CanvasEdgeLayout(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    source: str
    target: str


class WorkflowCanvasLayout(BaseModel):
    model_config = ConfigDict(extra="ignore")

    nodes: list[CanvasNodeLayout] = Field(default_factory=list)
    edges: list[CanvasEdgeLayout] = Field(default_factory=list)
    viewport: Optional[dict[str, Any]] = None


def validate_workflow_dag(
    steps: list[WorkflowNodeDefinition] | list[dict[str, Any]],
    max_nodes: int = 25,
) -> list[str]:
    """
    Deterministically validates a Workflow DAG specification.
    Returns a list of error messages. Empty list indicates a valid DAG.

    Invariants enforced:
    1. Node count must be between 1 and max_nodes (bounded complexity).
    2. Node IDs must be unique and properly formatted.
    3. Node types must belong to WorkflowActionType allowlist.
    4. No self-dependencies.
    5. All declared dependencies must reference declared nodes.
    6. No cycles (topological sort via Kahn's algorithm).
    7. Presentation metadata (canvas coordinates) must not alter execution logic.
    """
    errors: list[str] = []

    if not steps:
        errors.append("Workflow specification must contain at least one step.")
        return errors

    if len(steps) > max_nodes:
        errors.append(f"Workflow contains {len(steps)} steps, exceeding the maximum allowed limit of {max_nodes}.")

    nodes: list[WorkflowNodeDefinition] = []
    for idx, item in enumerate(steps):
        if isinstance(item, WorkflowNodeDefinition):
            nodes.append(item)
        elif isinstance(item, dict):
            try:
                nodes.append(WorkflowNodeDefinition.model_validate(item))
            except Exception as e:
                errors.append(f"Step at index {idx} failed schema validation: {e}")
        else:
            errors.append(f"Step at index {idx} has invalid format.")

    if errors:
        return errors

    node_ids: set[str] = set()
    for node in nodes:
        if node.id in node_ids:
            errors.append(f"Duplicate node id '{node.id}' detected.")
        else:
            node_ids.add(node.id)

        if node.id in node.dependencies:
            errors.append(f"Node '{node.id}' cannot depend on itself (self-dependency detected).")

    # Dependency existence
    for node in nodes:
        for dep in node.dependencies:
            if dep not in node_ids:
                errors.append(f"Node '{node.id}' references undeclared dependency '{dep}'.")

    # Cycle detection via Kahn's algorithm
    if not errors:
        in_degree: dict[str, int] = {node.id: 0 for node in nodes}
        adj_list: dict[str, list[str]] = {node.id: [] for node in nodes}

        for node in nodes:
            for dep in node.dependencies:
                adj_list[dep].append(node.id)
                in_degree[node.id] += 1

        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            curr = queue.popleft()
            visited_count += 1
            for neighbor in adj_list[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count < len(nodes):
            errors.append("Cycle detected in workflow DAG. Workflow dependencies must form a Directed Acyclic Graph.")

    return errors


# ---------------------------------------------------------------------------
# API DTOs
# ---------------------------------------------------------------------------

class WorkflowDefinitionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: uuid.UUID
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    initial_trigger: Optional[WorkflowTriggerSpec] = None
    initial_steps: Optional[list[WorkflowNodeDefinition]] = None
    initial_canvas_layout: Optional[WorkflowCanvasLayout] = None


class WorkflowDefinitionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    is_active: Optional[bool] = None


class WorkflowDefinitionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: Optional[str] = None
    created_by_id: uuid.UUID
    is_active: bool
    active_version_id: Optional[uuid.UUID] = None
    active_version_number: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class WorkflowVersionCreateDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fork_from_version_id: Optional[uuid.UUID] = None
    trigger_spec: Optional[WorkflowTriggerSpec] = None
    steps_spec: Optional[list[WorkflowNodeDefinition]] = None
    canvas_layout: Optional[WorkflowCanvasLayout] = None


class WorkflowVersionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger_spec: Optional[WorkflowTriggerSpec] = None
    steps_spec: Optional[list[WorkflowNodeDefinition]] = None
    canvas_layout: Optional[WorkflowCanvasLayout] = None


class WorkflowVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workflow_id: uuid.UUID
    version_number: int
    status: WorkflowVersionStatus
    trigger_spec: dict[str, Any]
    steps_spec: list[dict[str, Any]]
    canvas_layout: Optional[dict[str, Any]] = None
    schema_version: int
    created_by_id: uuid.UUID
    created_at: datetime
    published_at: Optional[datetime] = None


class WorkflowDAGValidationResponse(BaseModel):
    is_valid: bool
    errors: list[str]
    node_count: int


class WorkflowTestRunRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Sample trigger input payload passed to workflow execution",
    )
    budget_limits: Optional[Any] = Field(
        default=None,
        description="Optional overrides for execution token and provider call bounds",
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional idempotency key for test run deduplication",
    )


