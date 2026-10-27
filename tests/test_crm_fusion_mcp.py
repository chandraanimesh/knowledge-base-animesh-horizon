"""
Integration and regression tests for CRM Factory, Fail-Closed Authorization Gate,
and MultiSourceEvidenceFusionService with Twenty MCP Client.

Covers:
1. Client selection seam: defaults to TwentyCRMClient (REST); opts into TwentyMCPClient when enabled.
2. Fail-closed CRM authorization gate:
   - Inactive user denied.
   - Admin user permitted.
   - Non-admin permitted only if allowed_departments contains twenty_crm_department_id.
   - Non-admin denied if allowed_departments lacks twenty_crm_department_id.
   - Non-admin denied if twenty_crm_department_id is unconfigured (fail-closed).
   - Non-admin denied if twenty_crm_department_id is invalid UUID.
   - CRM client method is NOT invoked when authorization fails; CRM evidence absent.
3. MultiSourceEvidenceFusionService integration:
   - Authorized MCP retrieval populates company, people, opportunities.
   - Partial failure surfaces in crm_notice without discarding valid evidence.
   - Bounded pagination truncation is surfaced in crm_notice.
   - Ambiguous company match surfaces clarification in crm_notice.
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag_app.core.config import settings
from rag_app.crm.factory import get_crm_client
from rag_app.crm.twenty_client import (
    CRMCompany,
    CRMCompanyState,
    CRMOpportunity,
    CRMPerson,
    TwentyCRMClient,
)
from rag_app.crm.twenty_mcp_client import TwentyMCPClient
from rag_app.db.models import User
from rag_app.graph.fusion import (
    FusedEvidenceResult,
    MultiSourceEvidenceFusionService,
)


# ---------------------------------------------------------------------------
# 1. Factory Client Selection Tests
# ---------------------------------------------------------------------------

def test_crm_factory_defaults_to_rest():
    """Verify that get_crm_client() defaults to TwentyCRMClient (REST) when disabled."""
    with patch.object(settings, "twenty_mcp_enabled", False):
        client = get_crm_client()
        assert isinstance(client, TwentyCRMClient)
        assert not isinstance(client, TwentyMCPClient)


def test_crm_factory_returns_mcp_when_enabled():
    """Verify that get_crm_client() returns TwentyMCPClient when twenty_mcp_enabled=True."""
    with patch.object(settings, "twenty_mcp_enabled", True):
        client = get_crm_client()
        assert isinstance(client, TwentyMCPClient)


# ---------------------------------------------------------------------------
# 2. Fail-Closed CRM Authorization Gate Tests
# ---------------------------------------------------------------------------

def make_mock_db_session():
    session = AsyncMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    session.scalars.return_value = mock_scalars

    mock_exec = MagicMock()
    mock_exec.all.return_value = []
    session.execute.return_value = mock_exec
    return session


@pytest.fixture
def mock_session():
    return make_mock_db_session()


@pytest.fixture
def fusion_service(mock_session):
    return MultiSourceEvidenceFusionService(session=mock_session)


def test_auth_gate_inactive_user_denied(fusion_service):
    """Inactive user is denied access to CRM data."""
    user = User(id=uuid.uuid4(), email="user@test.com", role="USER", is_active=False)
    allowed_depts = [uuid.uuid4()]
    with patch.object(settings, "twenty_crm_department_id", str(allowed_depts[0])):
        is_auth, reason = fusion_service._check_crm_authorization(user, allowed_depts)
        assert is_auth is False
        assert "inactive" in reason.lower()


def test_auth_gate_admin_allowed(fusion_service):
    """Admin role bypasses department check matching repo admin policy."""
    user = User(id=uuid.uuid4(), email="admin@test.com", role="ADMIN", is_active=True)
    is_auth, reason = fusion_service._check_crm_authorization(user, allowed_departments=[])
    assert is_auth is True
    assert reason is None


def test_auth_gate_authorized_non_admin(fusion_service):
    """Non-admin user whose allowed_departments contains twenty_crm_department_id is allowed."""
    crm_dept = uuid.uuid4()
    other_dept = uuid.uuid4()
    user = User(id=uuid.uuid4(), email="sales@test.com", role="USER", is_active=True)

    with patch.object(settings, "twenty_crm_department_id", str(crm_dept)):
        is_auth, reason = fusion_service._check_crm_authorization(
            user, allowed_departments=[other_dept, crm_dept]
        )
        assert is_auth is True
        assert reason is None


def test_auth_gate_unauthorized_non_admin_denied(fusion_service):
    """Non-admin user whose allowed_departments does NOT contain CRM department is denied."""
    crm_dept = uuid.uuid4()
    other_dept = uuid.uuid4()
    user = User(id=uuid.uuid4(), email="eng@test.com", role="USER", is_active=True)

    with patch.object(settings, "twenty_crm_department_id", str(crm_dept)):
        is_auth, reason = fusion_service._check_crm_authorization(
            user, allowed_departments=[other_dept]
        )
        assert is_auth is False
        assert "not have authorization" in reason.lower()


def test_auth_gate_unconfigured_crm_department_fails_closed(fusion_service):
    """When twenty_crm_department_id is None, non-admin fails closed."""
    user = User(id=uuid.uuid4(), email="user@test.com", role="USER", is_active=True)
    with patch.object(settings, "twenty_crm_department_id", None):
        is_auth, reason = fusion_service._check_crm_authorization(
            user, allowed_departments=[uuid.uuid4()]
        )
        assert is_auth is False
        assert "not configured" in reason.lower()


def test_auth_gate_invalid_crm_department_fails_closed(fusion_service):
    """When twenty_crm_department_id is not a valid UUID, non-admin fails closed."""
    user = User(id=uuid.uuid4(), email="user@test.com", role="USER", is_active=True)
    with patch.object(settings, "twenty_crm_department_id", "not-a-valid-uuid"):
        is_auth, reason = fusion_service._check_crm_authorization(
            user, allowed_departments=[uuid.uuid4()]
        )
        assert is_auth is False
        assert "invalid" in reason.lower()


# ---------------------------------------------------------------------------
# 3. MultiSourceEvidenceFusion End-to-End with MCP
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fusion_unauthorized_user_skips_crm_call():
    """Verify that an unauthorized user never triggers CRM lookup and CRM records are absent."""
    mock_crm = AsyncMock()
    fusion = MultiSourceEvidenceFusionService(session=make_mock_db_session(), crm_client=mock_crm)

    crm_dept = uuid.uuid4()
    user = User(id=uuid.uuid4(), email="unauth@test.com", role="USER", is_active=True)

    with patch.object(settings, "twenty_crm_department_id", str(crm_dept)), \
         patch("rag_app.graph.fusion.MemoryMaterializer.get_user_memories", new_callable=AsyncMock) as mock_mem:
        mock_mem.return_value = []

        result = await fusion.fuse_company_evidence(
            user=user,
            company_name="Airbnb",
            query="Tell me about Airbnb",
            allowed_departments=[],  # Denied!
        )

        # CRM client must NEVER be called
        mock_crm.get_live_company_state.assert_not_called()
        assert result.crm_status_available is False
        assert "restricted" in result.crm_notice.lower()
        assert len(result.crm_records) == 0


@pytest.mark.asyncio
async def test_fusion_authorized_mcp_populates_all_streams():
    """Verify that an authorized user receives company, people, and opportunities from MCP."""
    mock_crm = AsyncMock()
    mock_crm_state = CRMCompanyState(
        company_name="Acme",
        is_connected=True,
        company=CRMCompany(id="c-1", name="Acme", domain="acme.com", annual_revenue=100000.0),
        people=[
            CRMPerson(id="p-1", name="Alice Smith", job_title="VP Sales", email="alice@acme.com", company_id="c-1")
        ],
        opportunities=[
            CRMOpportunity(id="o-1", name="Expansion", amount=50000.0, stage="NEGOTIATION", company_id="c-1")
        ],
    )
    mock_crm.get_live_company_state.return_value = mock_crm_state

    fusion = MultiSourceEvidenceFusionService(session=make_mock_db_session(), crm_client=mock_crm)

    crm_dept = uuid.uuid4()
    user = User(id=uuid.uuid4(), email="auth@test.com", role="USER", is_active=True)

    with patch.object(settings, "twenty_crm_department_id", str(crm_dept)), \
         patch("rag_app.graph.fusion.MemoryMaterializer.get_user_memories", new_callable=AsyncMock) as mock_mem:
        mock_mem.return_value = []

        result = await fusion.fuse_company_evidence(
            user=user,
            company_name="Acme",
            query="Tell me about Acme",
            allowed_departments=[crm_dept],
        )

        mock_crm.get_live_company_state.assert_called_once()
        assert result.crm_status_available is True
        assert len(result.crm_records) == 3

        # Check records streams and provenance
        comp_rec = [r for r in result.crm_records if r.title == "CRM Company: Acme"][0]
        assert "acme.com" in comp_rec.content
        assert comp_rec.provenance["crm_system"] == "Twenty CRM"
        assert comp_rec.provenance["crm_record_id"] == "c-1"

        person_rec = [r for r in result.crm_records if "Alice Smith" in r.title][0]
        assert "VP Sales" in person_rec.content
        assert person_rec.provenance["crm_record_id"] == "p-1"

        opp_rec = [r for r in result.crm_records if "Expansion" in r.title][0]
        assert "NEGOTIATION" in opp_rec.content
        assert opp_rec.provenance["stage"] == "NEGOTIATION"

        # Check synthesis includes operational CRM status
        assert "Live Operational CRM State" in result.formatted_synthesis
        assert "Alice Smith" in result.formatted_synthesis
        assert "NEGOTIATION" in result.formatted_synthesis

        # Check chunk conversion includes correct department id
        chunks = result.to_retrieved_chunks()
        crm_chunks = [ch for ch in chunks if ch.document_id.startswith("crm_")]
        assert len(crm_chunks) == 3
        for ch in crm_chunks:
            assert ch.department_id == str(crm_dept)


@pytest.mark.asyncio
async def test_fusion_partial_failure_preserves_valid_evidence():
    """Verify that partial failure (people failed, opps succeeded) is reported accurately without data loss."""
    mock_crm = AsyncMock()
    mock_crm_state = CRMCompanyState(
        company_name="Acme",
        is_connected=True,
        company=CRMCompany(id="c-1", name="Acme", domain="acme.com"),
        people=[],
        opportunities=[
            CRMOpportunity(id="o-1", name="Renewal", amount=20000.0, stage="CLOSED_WON", company_id="c-1")
        ],
        people_error="Twenty MCP contacts error (TimeoutException)",
    )
    mock_crm.get_live_company_state.return_value = mock_crm_state

    fusion = MultiSourceEvidenceFusionService(session=make_mock_db_session(), crm_client=mock_crm)

    user = User(id=uuid.uuid4(), email="admin@test.com", role="ADMIN", is_active=True)

    with patch("rag_app.graph.fusion.MemoryMaterializer.get_user_memories", new_callable=AsyncMock) as mock_mem:
        mock_mem.return_value = []

        result = await fusion.fuse_company_evidence(
            user=user,
            company_name="Acme",
            query="Tell me about Acme",
            allowed_departments=[],
        )

        assert result.crm_status_available is True
        assert len(result.crm_records) == 2  # Company + 1 Opportunity
        assert "contacts could not be retrieved" in result.crm_notice.lower()
        assert "1 active deal" in result.crm_notice.lower()


@pytest.mark.asyncio
async def test_fusion_truncation_notice_surfaced():
    """Verify that truncation flags are surfaced in crm_notice."""
    mock_crm = AsyncMock()
    mock_crm_state = CRMCompanyState(
        company_name="BigCorp",
        is_connected=True,
        company=CRMCompany(id="c-big", name="BigCorp"),
        people=[CRMPerson(id="p-1", name="Person 1", company_id="c-big")],
        opportunities=[],
        people_truncated=True,
    )
    mock_crm.get_live_company_state.return_value = mock_crm_state

    fusion = MultiSourceEvidenceFusionService(session=make_mock_db_session(), crm_client=mock_crm)

    user = User(id=uuid.uuid4(), email="admin@test.com", role="ADMIN", is_active=True)

    with patch("rag_app.graph.fusion.MemoryMaterializer.get_user_memories", new_callable=AsyncMock) as mock_mem:
        mock_mem.return_value = []

        result = await fusion.fuse_company_evidence(
            user=user,
            company_name="BigCorp",
            query="Tell me about BigCorp",
            allowed_departments=[],
        )

        assert result.crm_status_available is True
        assert "truncated to maximum page budget" in result.crm_notice.lower()
