"""
Unified Capability Registry & Tool Discovery Engine (Milestone 10B).
Defines typed capability descriptors, permission boundaries, side-effect classifications,
and authorization-aware discovery across Knowledge, Memory, Research, CRM, and Artifacts.
"""

from __future__ import annotations

from enum import Enum
import logging
from typing import Any, Callable, Coroutine, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class CapabilityCategory(str, Enum):
    KNOWLEDGE = "KNOWLEDGE"
    MEMORY = "MEMORY"
    RESEARCH = "RESEARCH"
    CRM = "CRM"
    WORKFLOW = "WORKFLOW"
    VALIDATION = "VALIDATION"
    ARTIFACT = "ARTIFACT"
    SYNTHESIS = "SYNTHESIS"


class SideEffectLevel(str, Enum):
    READ_ONLY = "READ_ONLY"             # Authorized direct execution
    EXTERNAL_READ = "EXTERNAL_READ"     # Outbound read to external provider (CRM/Web)
    DRAFT = "DRAFT"                     # Draft proposal created, no remote mutation
    WRITE = "WRITE"                     # Remote state mutation (requires allowlist / HITL)
    HIGH_IMPACT = "HIGH_IMPACT"         # Sensitive mutation (strict approval required)


class ExecutionMode(str, Enum):
    DIRECT = "DIRECT"
    ASYNC_OUTBOX = "ASYNC_OUTBOX"
    HITL_APPROVAL = "HITL_APPROVAL"


class AvailabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    CONFIG_REQUIRED = "CONFIG_REQUIRED"


class CapabilityError(Exception):
    """Base exception for all capability operations."""
    pass


class CapabilityNotFoundError(CapabilityError, KeyError):
    """Raised when an unregistered or unknown capability is requested."""
    pass


class CapabilitySchemaValidationError(CapabilityError, ValueError):
    """Raised when input or output parameters fail capability schema validation."""
    pass


class CapabilityPermissionDeniedError(CapabilityError, PermissionError):
    """Raised when the caller lacks required permissions for a capability."""
    pass


class CapabilityUnavailableError(CapabilityError, RuntimeError):
    """Raised when a capability is disabled, unavailable, or requires unconfigured credentials."""
    pass


class UnifiedCapabilityDescriptor(BaseModel):
    """Normalized, typed metadata descriptor for an authorized platform capability."""
    model_config = ConfigDict(from_attributes=True)

    capability_id: str = Field(..., description="Unique, stable capability identifier (e.g. 'rag.retrieve_internal')")
    category: CapabilityCategory
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    required_permissions: list[str] = Field(default_factory=list)
    side_effect_level: SideEffectLevel = SideEffectLevel.READ_ONLY
    supported_execution_mode: ExecutionMode = ExecutionMode.DIRECT
    timeout_seconds: int = 60
    max_retries: int = 3
    availability_status: AvailabilityStatus = AvailabilityStatus.AVAILABLE
    provider_requirement: Optional[str] = None


class CapabilityRegistry:
    """
    Unified Capability Registry (Milestone 10B).
    Controls discovery, input/output validation, and authorization for all executable actions.
    Rejects unknown or unauthorized tools before execution.
    """

    def __init__(self):
        self._capabilities: dict[str, UnifiedCapabilityDescriptor] = {}
        self._executors: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._register_default_capabilities()

    def register(
        self,
        descriptor: UnifiedCapabilityDescriptor,
        executor: Optional[Callable[..., Coroutine[Any, Any, Any]]] = None,
    ) -> None:
        """Registers a typed capability descriptor and optional executor."""
        self._capabilities[descriptor.capability_id] = descriptor
        if executor:
            self._executors[descriptor.capability_id] = executor
        logger.debug(f"Registered capability '{descriptor.capability_id}' ({descriptor.category.value})")

    def get(self, capability_id: str) -> Optional[UnifiedCapabilityDescriptor]:
        return self._capabilities.get(capability_id)

    def resolve(self, capability_id: str) -> UnifiedCapabilityDescriptor:
        cap = self.get(capability_id)
        if not cap:
            raise CapabilityNotFoundError(f"Unknown or unregistered capability: '{capability_id}'")
        return cap

    def set_availability(self, capability_id: str, status: AvailabilityStatus) -> None:
        """Sets the runtime availability status of a capability."""
        cap = self.resolve(capability_id)
        cap.availability_status = status
        logger.info(f"Capability '{capability_id}' availability status updated to {status.value}")

    def check_authorization(self, capability_id: str, user_permissions: Optional[list[str]] = None) -> bool:
        """
        Verifies that the caller holds all required permissions for the capability.
        Fails closed with CapabilityPermissionDeniedError.
        """
        cap = self.resolve(capability_id)
        if cap.availability_status != AvailabilityStatus.AVAILABLE:
            raise CapabilityUnavailableError(
                f"Capability '{capability_id}' is currently {cap.availability_status.value}."
            )
        if not cap.required_permissions:
            return True
        user_perms_set = set(user_permissions or [])
        if "ADMIN" in user_perms_set:
            return True
        missing = [p for p in cap.required_permissions if p not in user_perms_set]
        if missing:
            raise CapabilityPermissionDeniedError(
                f"Caller lacks required permissions for '{capability_id}': missing {missing}"
            )
        return True

    def validate_capability_inputs(self, capability_id: str, inputs: dict[str, Any]) -> bool:
        """
        Validates that required input fields are present and conform to declared schema.
        Fails closed on missing or invalid required arguments.
        """
        cap = self.resolve(capability_id)
        if not isinstance(inputs, dict):
            raise CapabilitySchemaValidationError(
                f"Capability '{capability_id}' input must be a dictionary, got {type(inputs).__name__}"
            )

        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
        }

        for field_name, expected_type_str in cap.input_schema.items():
            if field_name not in inputs:
                raise CapabilitySchemaValidationError(
                    f"Capability '{capability_id}' missing required input field: '{field_name}'"
                )
            val = inputs[field_name]
            expected_type = type_map.get(str(expected_type_str).lower())
            if expected_type and not isinstance(val, expected_type):
                raise CapabilitySchemaValidationError(
                    f"Capability '{capability_id}' field '{field_name}' expected type {expected_type_str}, got {type(val).__name__}"
                )
        return True

    def validate_capability_outputs(self, capability_id: str, outputs: dict[str, Any]) -> bool:
        """
        Validates that output fields conform to declared output schema.
        """
        cap = self.resolve(capability_id)
        if not isinstance(outputs, dict):
            raise CapabilitySchemaValidationError(
                f"Capability '{capability_id}' output must be a dictionary, got {type(outputs).__name__}"
            )

        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
        }

        for field_name, expected_type_str in cap.output_schema.items():
            if field_name not in outputs:
                raise CapabilitySchemaValidationError(
                    f"Capability '{capability_id}' output missing required field: '{field_name}'"
                )
            val = outputs[field_name]
            expected_type = type_map.get(str(expected_type_str).lower())
            if expected_type and not isinstance(val, expected_type):
                raise CapabilitySchemaValidationError(
                    f"Capability '{capability_id}' output field '{field_name}' expected type {expected_type_str}, got {type(val).__name__}"
                )
        return True

    def discover_capabilities(
        self,
        intent: Optional[str] = None,
        category: Optional[CapabilityCategory] = None,
        user_permissions: Optional[list[str]] = None,
        workspace_id: Optional[uuid.UUID] = None,
        max_side_effect: Optional[SideEffectLevel] = None,
    ) -> list[UnifiedCapabilityDescriptor]:
        """
        Discovers capabilities matching criteria while filtering out unauthorized or unavailable tools.
        """
        user_perms_set = set(user_permissions or [])
        allowed: list[UnifiedCapabilityDescriptor] = []

        for cap in self._capabilities.values():
            if cap.availability_status != AvailabilityStatus.AVAILABLE:
                continue

            if category is not None and cap.category != category:
                continue

            if intent is not None and (intent.lower() not in cap.capability_id.lower() and intent.lower() not in cap.description.lower()):
                continue

            # Permission check: User must hold all required permissions for this capability
            if cap.required_permissions:
                if not all(p in user_perms_set or "ADMIN" in user_perms_set for p in cap.required_permissions):
                    continue

            # Side-effect filter: If max_side_effect specified, filter out higher-risk actions
            if max_side_effect:
                levels_order = [
                    SideEffectLevel.READ_ONLY,
                    SideEffectLevel.EXTERNAL_READ,
                    SideEffectLevel.DRAFT,
                    SideEffectLevel.WRITE,
                    SideEffectLevel.HIGH_IMPACT,
                ]
                if levels_order.index(cap.side_effect_level) > levels_order.index(max_side_effect):
                    continue

            allowed.append(cap)

        return allowed

    def _register_default_capabilities(self) -> None:
        # 1. Internal RAG Retrieval
        self.register(
            UnifiedCapabilityDescriptor(
                capability_id="rag.retrieve_internal",
                category=CapabilityCategory.KNOWLEDGE,
                description="Retrieve authoritative internal corporate documents and knowledge",
                input_schema={"query": "str", "limit": "int", "scope": "str"},
                output_schema={"retrieved_count": "int", "results": "list"},
                required_permissions=["DOCUMENTS_READ"],
                side_effect_level=SideEffectLevel.READ_ONLY,
                supported_execution_mode=ExecutionMode.DIRECT,
            )
        )

        # 2. Personal Memory Retrieval
        self.register(
            UnifiedCapabilityDescriptor(
                capability_id="memory.retrieve_context",
                category=CapabilityCategory.MEMORY,
                description="Retrieve user personal preferences, living profile, and prior decisions",
                input_schema={"query": "str", "limit": "int"},
                output_schema={"memories": "list", "profile_summary": "dict"},
                required_permissions=["MEMORY_READ"],
                side_effect_level=SideEffectLevel.READ_ONLY,
                supported_execution_mode=ExecutionMode.DIRECT,
            )
        )

        # 3. Web Search / Research
        self.register(
            UnifiedCapabilityDescriptor(
                capability_id="research.web_search",
                category=CapabilityCategory.RESEARCH,
                description="Perform external web research for external context and industry patterns",
                input_schema={"query": "str", "limit": "int"},
                output_schema={"urls": "list", "results": "list"},
                required_permissions=["RESEARCH_READ"],
                side_effect_level=SideEffectLevel.EXTERNAL_READ,
                supported_execution_mode=ExecutionMode.DIRECT,
            )
        )

        # 4. Structured Fact Extraction
        self.register(
            UnifiedCapabilityDescriptor(
                capability_id="research.extract_facts",
                category=CapabilityCategory.RESEARCH,
                description="Extract structured facts, dates, numbers, and key findings from evidence",
                input_schema={"fields": "list"},
                output_schema={"extracted_facts": "list"},
                required_permissions=[],
                side_effect_level=SideEffectLevel.READ_ONLY,
                supported_execution_mode=ExecutionMode.DIRECT,
            )
        )

        # 5. Evidence Grounding & Contradiction Check
        self.register(
            UnifiedCapabilityDescriptor(
                capability_id="validation.verify_grounding",
                category=CapabilityCategory.VALIDATION,
                description="Cross-verify findings against citations and detect contradictory claims",
                input_schema={"claims": "list"},
                output_schema={"is_grounded": "bool", "contradictions": "list"},
                required_permissions=[],
                side_effect_level=SideEffectLevel.READ_ONLY,
                supported_execution_mode=ExecutionMode.DIRECT,
            )
        )

        # 6. Technical Synthesis & Report
        self.register(
            UnifiedCapabilityDescriptor(
                capability_id="synthesis.generate_findings",
                category=CapabilityCategory.SYNTHESIS,
                description="Synthesize grounded report, technical recommendation, or gap analysis",
                input_schema={"format": "str"},
                output_schema={"report_markdown": "str", "citations": "list"},
                required_permissions=[],
                side_effect_level=SideEffectLevel.READ_ONLY,
                supported_execution_mode=ExecutionMode.DIRECT,
            )
        )

        # 7. CRM Query / Read
        self.register(
            UnifiedCapabilityDescriptor(
                capability_id="crm.query_records",
                category=CapabilityCategory.CRM,
                description="Query authorized CRM entities (companies, contacts, notes) via MCP",
                input_schema={"entity_type": "str", "query": "str"},
                output_schema={"records": "list"},
                required_permissions=["CRM_READ"],
                side_effect_level=SideEffectLevel.EXTERNAL_READ,
                supported_execution_mode=ExecutionMode.DIRECT,
            )
        )

        # 8. CRM Note / Mutation (Requires HITL)
        self.register(
            UnifiedCapabilityDescriptor(
                capability_id="crm.create_note",
                category=CapabilityCategory.CRM,
                description="Create or attach note to CRM company/contact (Requires human approval)",
                input_schema={"target_id": "str", "note_text": "str"},
                output_schema={"approval_id": "str", "status": "str"},
                required_permissions=["CRM_WRITE"],
                side_effect_level=SideEffectLevel.WRITE,
                supported_execution_mode=ExecutionMode.HITL_APPROVAL,
            )
        )

        # 9. Artifact Offloading & Storage
        self.register(
            UnifiedCapabilityDescriptor(
                capability_id="artifact.offload_output",
                category=CapabilityCategory.ARTIFACT,
                description="Persist large execution output (> 1000 tokens) to symbolic storage",
                input_schema={"content": "str", "artifact_type": "str"},
                output_schema={"artifact_uri": "str", "size_bytes": "int", "content_hash": "str"},
                required_permissions=[],
                side_effect_level=SideEffectLevel.READ_ONLY,
                supported_execution_mode=ExecutionMode.DIRECT,
            )
        )

        # 10. Artifact Retrieval by Symbolic Reference
        self.register(
            UnifiedCapabilityDescriptor(
                capability_id="artifact.retrieve_output",
                category=CapabilityCategory.ARTIFACT,
                description="Retrieve authorized execution artifact by symbolic reference with scope check",
                input_schema={"artifact_id": "str"},
                output_schema={"content": "str", "metadata": "dict"},
                required_permissions=[],
                side_effect_level=SideEffectLevel.READ_ONLY,
                supported_execution_mode=ExecutionMode.DIRECT,
            )
        )


_GLOBAL_REGISTRY: Optional[CapabilityRegistry] = None

def get_capability_registry() -> CapabilityRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = CapabilityRegistry()
    return _GLOBAL_REGISTRY

