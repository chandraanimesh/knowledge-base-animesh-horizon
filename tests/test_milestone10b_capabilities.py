"""
Milestone 10B Verification Test Suite: Capability Discovery & Schema Validation.

Verifies:
1. Registration and resolution of typed capabilities.
2. Input schema validation with fail-closed enforcement of missing or mistyped fields.
3. Output schema validation for compliant outputs.
4. Authorization enforcement: callers lacking required permissions are rejected.
5. Capability discovery filtered by permissions, side-effect boundaries, and category.
6. Availability status filtering: disabled or config-required capabilities excluded safely.
7. Unknown capability rejection before execution.
"""

from __future__ import annotations

import pytest

from rag_app.orchestration.capability_registry import (
    CapabilityCategory,
    CapabilityRegistry,
    SideEffectLevel,
    ExecutionMode,
    AvailabilityStatus,
    UnifiedCapabilityDescriptor,
    CapabilityNotFoundError,
    CapabilitySchemaValidationError,
    CapabilityPermissionDeniedError,
    CapabilityUnavailableError,
    get_capability_registry,
)


@pytest.fixture
def registry():
    # Fresh registry with default capabilities
    return CapabilityRegistry()


def test_default_capabilities_registered(registry):
    # Verify core built-in capabilities exist
    rag_cap = registry.resolve("rag.retrieve_internal")
    assert rag_cap.category == CapabilityCategory.KNOWLEDGE
    assert rag_cap.side_effect_level == SideEffectLevel.READ_ONLY
    assert "DOCUMENTS_READ" in rag_cap.required_permissions

    crm_write = registry.resolve("crm.create_note")
    assert crm_write.category == CapabilityCategory.CRM
    assert crm_write.side_effect_level == SideEffectLevel.WRITE
    assert crm_write.supported_execution_mode == ExecutionMode.HITL_APPROVAL


def test_resolve_unknown_capability_fails_closed(registry):
    with pytest.raises(CapabilityNotFoundError) as exc_info:
        registry.resolve("unauthorized.malicious_tool")
    assert "unregistered capability" in str(exc_info.value)


def test_input_schema_validation_success(registry):
    valid_inputs = {"query": "Search query", "limit": 10, "scope": "internal"}
    assert registry.validate_capability_inputs("rag.retrieve_internal", valid_inputs) is True


def test_input_schema_validation_missing_field(registry):
    invalid_inputs = {"query": "Search query"}  # missing 'limit' and 'scope'
    with pytest.raises(CapabilitySchemaValidationError) as exc:
        registry.validate_capability_inputs("rag.retrieve_internal", invalid_inputs)
    assert "missing required input field" in str(exc.value)


def test_input_schema_validation_invalid_type(registry):
    invalid_inputs = {"query": 12345, "limit": 10, "scope": "internal"}  # query must be str
    with pytest.raises(CapabilitySchemaValidationError) as exc:
        registry.validate_capability_inputs("rag.retrieve_internal", invalid_inputs)
    assert "expected type str" in str(exc.value)


def test_output_schema_validation(registry):
    valid_outputs = {"retrieved_count": 2, "results": ["doc1", "doc2"]}
    assert registry.validate_capability_outputs("rag.retrieve_internal", valid_outputs) is True

    invalid_outputs = {"retrieved_count": "two"}  # missing 'results' and wrong type
    with pytest.raises(CapabilitySchemaValidationError):
        registry.validate_capability_outputs("rag.retrieve_internal", invalid_outputs)


def test_check_authorization_enforces_permissions(registry):
    # User with DOCUMENTS_READ permission
    assert registry.check_authorization("rag.retrieve_internal", ["DOCUMENTS_READ"]) is True

    # User without DOCUMENTS_READ permission fails
    with pytest.raises(CapabilityPermissionDeniedError) as exc:
        registry.check_authorization("rag.retrieve_internal", ["SOME_OTHER_PERM"])
    assert "Caller lacks required permissions" in str(exc.value)

    # ADMIN bypasses specific permission checks
    assert registry.check_authorization("rag.retrieve_internal", ["ADMIN"]) is True


def test_discover_capabilities_filters_by_permissions(registry):
    # Empty permissions can only discover permissionless tools (Extraction, Grounding, Synthesis, Artifact)
    open_caps = registry.discover_capabilities(user_permissions=[])
    cap_ids = [c.capability_id for c in open_caps]
    assert "rag.retrieve_internal" not in cap_ids
    assert "crm.create_note" not in cap_ids
    assert "research.extract_facts" in cap_ids
    assert "validation.verify_grounding" in cap_ids

    # Adding CRM_READ reveals CRM query but NOT CRM write
    crm_read_caps = registry.discover_capabilities(user_permissions=["CRM_READ"])
    crm_read_ids = [c.capability_id for c in crm_read_caps]
    assert "crm.query_records" in crm_read_ids
    assert "crm.create_note" not in crm_read_ids


def test_discover_capabilities_filters_by_side_effect(registry):
    # Restrict to READ_ONLY
    ro_caps = registry.discover_capabilities(
        user_permissions=["ADMIN"],
        max_side_effect=SideEffectLevel.READ_ONLY,
    )
    for cap in ro_caps:
        assert cap.side_effect_level == SideEffectLevel.READ_ONLY

    ro_ids = [c.capability_id for c in ro_caps]
    assert "crm.create_note" not in ro_ids  # WRITE
    assert "research.web_search" not in ro_ids  # EXTERNAL_READ


def test_unavailable_or_unconfigured_capability_handling(registry):
    # Disable a capability
    registry.set_availability("research.web_search", AvailabilityStatus.CONFIG_REQUIRED)

    # Should not appear in discovery
    discovered = registry.discover_capabilities(user_permissions=["ADMIN"])
    discovered_ids = [c.capability_id for c in discovered]
    assert "research.web_search" not in discovered_ids

    # Direct authorization check raises CapabilityUnavailableError
    with pytest.raises(CapabilityUnavailableError):
        registry.check_authorization("research.web_search", ["ADMIN"])

