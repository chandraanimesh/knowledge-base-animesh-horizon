"""
Unit tests for isolated Twenty CRM MCP Client Adapter.

Covers:
1. Successful company lookup with adaptation to CRMCompany and CRMCompanyState.
2. Company not found (empty records -> company=None, is_connected=True).
3. Tool-level error response (isError=True or success=False).
4. Authentication failure (HTTP 401/403).
5. Timeout / transport failure.
6. Malformed JSON / unexpected response payload.
7. Strict client-side allowlist enforcement rejecting write tools.
8. Explicit disabled-by-default behavior verification.
"""

import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch

from rag_app.crm.twenty_client import CRMCompany, CRMCompanyState
from rag_app.crm.twenty_mcp_client import (
    TwentyMCPClient,
    TwentyMCPAllowlistViolation,
    TwentyMCPError,
    READ_ONLY_ALLOWED_TOOLS,
)


def _build_sse_mcp_response(inner_payload: dict, is_error: bool = False) -> str:
    """Builds standard Streamable HTTP SSE payload as returned by Twenty MCP."""
    data = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(inner_payload),
                }
            ],
            "isError": is_error,
        },
    }
    return f"event: message\ndata: {json.dumps(data)}\n\n"


@pytest.fixture
def mcp_client():
    """Returns an enabled TwentyMCPClient with mock credentials."""
    return TwentyMCPClient(
        mcp_url="http://localhost:3000/mcp",
        api_key="mock_secret_key",
        timeout=2.0,
        enabled=True,
    )


# ---------------------------------------------------------------------------
# 1. Disabled-by-Default Opt-In Verification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_client_disabled_by_default():
    """Verify that client fails closed when enabled=False (default behavior)."""
    client = TwentyMCPClient(enabled=False, api_key="test_key")
    state = await client.get_live_company_state("Airbnb")
    assert state.is_connected is False
    assert state.company is None
    assert "disabled" in (state.error_message or "").lower()


@pytest.mark.asyncio
async def test_mcp_client_missing_api_key():
    """Verify that client fails closed when API key is empty."""
    client = TwentyMCPClient(enabled=True, api_key="")
    state = await client.get_live_company_state("Airbnb")
    assert state.is_connected is False
    assert state.company is None
    assert "api_key" in (state.error_message or "").lower()


# ---------------------------------------------------------------------------
# 2. Strict Client-Side Allowlist Enforcement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_allowlist_enforces_read_only_tools(mcp_client):
    """Verify that only approved read-only tools are allowed."""
    assert "find_many_companies" in READ_ONLY_ALLOWED_TOOLS
    assert "find_one_company" in READ_ONLY_ALLOWED_TOOLS
    assert "find_many_people" in READ_ONLY_ALLOWED_TOOLS
    assert "find_many_opportunities" in READ_ONLY_ALLOWED_TOOLS
    assert "create_one_company" not in READ_ONLY_ALLOWED_TOOLS
    assert "delete_one_company" not in READ_ONLY_ALLOWED_TOOLS

    # Calling an unauthorized write tool directly raises TwentyMCPAllowlistViolation
    for write_tool in ["create_one_company", "update_one_company", "delete_one_company", "create_many_companies"]:
        with pytest.raises(TwentyMCPAllowlistViolation) as exc_info:
            await mcp_client.call_tool(write_tool, {"name": "Test"})
        assert "not in the read-only MCP allowlist" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. Successful Company Lookup and Contract Adaptation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_company_lookup(mcp_client):
    """Verify company lookup parses Twenty MCP payload into CRMCompany contract."""
    mock_payload = {
        "success": True,
        "message": "Found 1 company records",
        "result": {
            "records": [
                {
                    "id": "c776ee49-f608-4a77-8cc8-6fe96ae1e43f",
                    "name": "Airbnb",
                    "domainName": {
                        "primaryLinkLabel": "",
                        "primaryLinkUrl": "airbnb.com",
                        "secondaryLinks": [],
                    },
                    "address": {
                        "addressStreet1": "888 Brannan St",
                        "addressCity": "San Francisco",
                        "addressState": "CA",
                        "addressCountry": "United States",
                    },
                    "annualRevenue": {
                        "amountMicros": 50000000000,
                        "currencyCode": "USD",
                    },
                    "createdAt": "2026-03-10T16:26:00.000Z",
                    "updatedAt": "2026-03-10T16:26:00.000Z",
                }
            ],
            "count": "1",
            "hasNextPage": False,
        },
    }

    comp_resp = httpx.Response(200, text=_build_sse_mcp_response(mock_payload), request=httpx.Request("POST", "http://localhost:3000/mcp"))
    empty_people = httpx.Response(200, text=_build_sse_mcp_response({"success": True, "result": {"records": [], "count": "0", "hasNextPage": False}}), request=httpx.Request("POST", "http://localhost:3000/mcp"))
    empty_opps = httpx.Response(200, text=_build_sse_mcp_response({"success": True, "result": {"records": [], "count": "0", "hasNextPage": False}}), request=httpx.Request("POST", "http://localhost:3000/mcp"))

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [comp_resp, empty_people, empty_opps]

        state = await mcp_client.get_live_company_state("Airbnb")

        assert state.is_connected is True
        assert state.error_message is None
        assert state.company is not None
        assert state.company_name == "Airbnb"
        assert state.has_crm_records is True

        # Verify adapted CRMCompany contract fields
        comp: CRMCompany = state.company
        assert comp.id == "c776ee49-f608-4a77-8cc8-6fe96ae1e43f"
        assert comp.name == "Airbnb"
        assert comp.domain == "airbnb.com"
        assert "San Francisco" in (comp.address or "")
        assert comp.annual_revenue == 50000.0
        assert comp.created_at == "2026-03-10T16:26:00.000Z"

        # People and opportunities cleanly empty
        assert state.people == []
        assert state.opportunities == []
        assert state.people_error is None
        assert state.opportunities_error is None


# ---------------------------------------------------------------------------
# 4. Company Not Found (Clean Empty State)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_company_not_found_returns_clean_state(mcp_client):
    """Verify that an empty records response cleanly sets company=None without error."""
    mock_payload = {
        "success": True,
        "message": "Found 0 company records",
        "result": {
            "records": [],
            "count": "0",
            "hasNextPage": False,
        },
    }

    sse_response_text = _build_sse_mcp_response(mock_payload)
    mock_response = httpx.Response(200, text=sse_response_text, request=httpx.Request("POST", "http://localhost:3000/mcp"))

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        state = await mcp_client.get_live_company_state("NonExistentCo")

        assert state.is_connected is True
        assert state.error_message is None
        assert state.company is None
        assert state.has_crm_records is False


# ---------------------------------------------------------------------------
# 5. Authentication Failure (HTTP 401 / 403)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_authentication_failure(mcp_client):
    """Verify HTTP 401/403 fails closed with distinct error message and is_connected=False."""
    mock_response = httpx.Response(
        401,
        text="Unauthorized",
        request=httpx.Request("POST", "http://localhost:3000/mcp"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        state = await mcp_client.get_live_company_state("Airbnb")

        assert state.is_connected is False
        assert state.company is None
        assert "authentication failed" in (state.error_message or "").lower()


# ---------------------------------------------------------------------------
# 6. Timeout / Network Transport Failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_failure(mcp_client):
    """Verify timeout does not masquerade as successful empty result."""
    with patch.object(httpx.AsyncClient, "post", side_effect=httpx.TimeoutException("Read timeout")):
        state = await mcp_client.get_live_company_state("Airbnb")

        assert state.is_connected is False
        assert state.company is None
        assert "timed out" in (state.error_message or "").lower()


@pytest.mark.asyncio
async def test_connect_failure(mcp_client):
    """Verify connection error fails gracefully with clear error state."""
    with patch.object(httpx.AsyncClient, "post", side_effect=httpx.ConnectError("Connection refused")):
        state = await mcp_client.get_live_company_state("Airbnb")

        assert state.is_connected is False
        assert state.company is None
        assert "unavailable" in (state.error_message or "").lower()


# ---------------------------------------------------------------------------
# 7. Tool-Level Error Response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_level_error_response(mcp_client):
    """Verify tool execution error (isError=True) does not masquerade as empty result."""
    mock_error_payload = {
        "success": False,
        "message": "Select at least one field in select parameter",
        "error": "Select is required",
    }
    sse_response_text = _build_sse_mcp_response(mock_error_payload, is_error=True)
    mock_response = httpx.Response(200, text=sse_response_text, request=httpx.Request("POST", "http://localhost:3000/mcp"))

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        state = await mcp_client.get_live_company_state("Airbnb")

        assert state.is_connected is False
        assert state.company is None
        assert "tool error" in (state.error_message or "").lower() or "error" in (state.error_message or "").lower()


# ---------------------------------------------------------------------------
# 8. Malformed or Unexpected Payload Structure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_malformed_json_response(mcp_client):
    """Verify non-JSON response produces a distinct error state."""
    mock_response = httpx.Response(
        200,
        text="<html><body>502 Bad Gateway</body></html>",
        request=httpx.Request("POST", "http://localhost:3000/mcp"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        state = await mcp_client.get_live_company_state("Airbnb")

        assert state.is_connected is False
        assert state.company is None
        assert "malformed" in (state.error_message or "").lower() or "error" in (state.error_message or "").lower()


# ---------------------------------------------------------------------------
# 9. Successful People and Opportunity Contract Adaptation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_company_with_people_and_opportunities(mcp_client):
    """Verify that contacts and deals are parsed into CRMPerson and CRMOpportunity contracts."""
    comp_payload = {
        "success": True,
        "result": {
            "records": [
                {
                    "id": "comp-123",
                    "name": "Stripe",
                    "domainName": {"primaryLinkUrl": "stripe.com"},
                }
            ],
            "count": "1",
            "hasNextPage": False,
        },
    }
    people_payload = {
        "success": True,
        "result": {
            "records": [
                {
                    "id": "person-456",
                    "name": {"firstName": "Patrick", "lastName": "Collison"},
                    "jobTitle": "CEO",
                    "emails": {"primaryEmail": "patrick@stripe.com"},
                    "phones": {"primaryPhoneNumber": "+1-555-0100"},
                    "companyId": "comp-123",
                }
            ],
            "count": "1",
            "hasNextPage": False,
        },
    }
    opps_payload = {
        "success": True,
        "result": {
            "records": [
                {
                    "id": "opp-789",
                    "name": "Global Enterprise Expansion",
                    "amount": {"amountMicros": 250000000000, "currencyCode": "USD"},
                    "stage": "CONTRACT_SENT",
                    "closeDate": "2026-12-31T00:00:00.000Z",
                    "companyId": "comp-123",
                }
            ],
            "count": "1",
            "hasNextPage": False,
        },
    }

    comp_resp = httpx.Response(200, text=_build_sse_mcp_response(comp_payload), request=httpx.Request("POST", "http://localhost:3000/mcp"))
    people_resp = httpx.Response(200, text=_build_sse_mcp_response(people_payload), request=httpx.Request("POST", "http://localhost:3000/mcp"))
    opps_resp = httpx.Response(200, text=_build_sse_mcp_response(opps_payload), request=httpx.Request("POST", "http://localhost:3000/mcp"))

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [comp_resp, people_resp, opps_resp]

        state = await mcp_client.get_live_company_state("Stripe")

        assert state.is_connected is True
        assert state.company is not None
        assert state.company.name == "Stripe"
        assert len(state.people) == 1
        assert state.people[0].id == "person-456"
        assert state.people[0].name == "Patrick Collison"
        assert state.people[0].job_title == "CEO"
        assert state.people[0].email == "patrick@stripe.com"
        assert state.people[0].phone == "+1-555-0100"
        assert state.people[0].company_id == "comp-123"

        assert len(state.opportunities) == 1
        assert state.opportunities[0].id == "opp-789"
        assert state.opportunities[0].name == "Global Enterprise Expansion"
        assert state.opportunities[0].amount == 250000.0
        assert state.opportunities[0].currency == "USD"
        assert state.opportunities[0].stage == "CONTRACT_SENT"
        assert state.opportunities[0].close_date == "2026-12-31T00:00:00.000Z"
        assert state.opportunities[0].company_id == "comp-123"


# ---------------------------------------------------------------------------
# 10. Subresource Company ID Filter & Query Verification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subresource_company_id_filter(mcp_client):
    """Verify that people and opportunities queries filter strictly by companyId eq."""
    people_payload = {
        "success": True,
        "result": {"records": [], "count": "0", "hasNextPage": False},
    }
    resp = httpx.Response(200, text=_build_sse_mcp_response(people_payload), request=httpx.Request("POST", "http://localhost:3000/mcp"))

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = resp

        people, err, trunc = await mcp_client.get_company_people("target-uuid-456")
        assert people == []
        assert err is None
        assert trunc is False

        # Verify arguments passed to call_tool
        call_args = mock_post.call_args[1]["json"]
        assert call_args["params"]["name"] == "execute_tool"
        assert call_args["params"]["arguments"]["toolName"] == "find_many_people"
        tool_arguments = call_args["params"]["arguments"]["arguments"]
        assert tool_arguments["companyId"] == {"eq": "target-uuid-456"}
        assert "select" in tool_arguments


# ---------------------------------------------------------------------------
# 11. Pagination Across Multiple Pages and Bounded Truncation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pagination_and_truncation(mcp_client):
    """Verify that client paginates when hasNextPage is True and flags truncation when max_pages reached."""
    client = TwentyMCPClient(
        mcp_url="http://localhost:3000/mcp",
        api_key="mock_secret_key",
        enabled=True,
        max_pages=2,
        page_size=2,
    )

    page1_payload = {
        "success": True,
        "result": {
            "records": [
                {"id": "p1", "name": "Person 1"},
                {"id": "p2", "name": "Person 2"},
            ],
            "count": "4",
            "hasNextPage": True,
        },
    }
    page2_payload = {
        "success": True,
        "result": {
            "records": [
                {"id": "p3", "name": "Person 3"},
                {"id": "p4", "name": "Person 4"},
            ],
            "count": "4",
            "hasNextPage": True,  # Still more pages exist beyond max_pages=2
        },
    }

    resp1 = httpx.Response(200, text=_build_sse_mcp_response(page1_payload), request=httpx.Request("POST", "http://localhost:3000/mcp"))
    resp2 = httpx.Response(200, text=_build_sse_mcp_response(page2_payload), request=httpx.Request("POST", "http://localhost:3000/mcp"))

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [resp1, resp2]

        people, err, is_trunc = await client.get_company_people("comp-1")

        assert len(people) == 4
        assert err is None
        assert is_trunc is True  # Flagged truncated because max_pages=2 was hit while hasNextPage remained True


# ---------------------------------------------------------------------------
# 12. Partial-Failure Subresource Isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_partial_failure_subresource_isolation(mcp_client):
    """Verify that people failure does not discard valid company or opportunities evidence."""
    comp_payload = {
        "success": True,
        "result": {
            "records": [{"id": "c1", "name": "Acme Corp"}],
            "count": "1",
            "hasNextPage": False,
        },
    }
    people_error_payload = {
        "success": False,
        "message": "Internal error retrieving people",
    }
    opps_payload = {
        "success": True,
        "result": {
            "records": [{"id": "o1", "name": "Deal 1", "amount": 1000}],
            "count": "1",
            "hasNextPage": False,
        },
    }

    comp_resp = httpx.Response(200, text=_build_sse_mcp_response(comp_payload), request=httpx.Request("POST", "http://localhost:3000/mcp"))
    people_resp = httpx.Response(200, text=_build_sse_mcp_response(people_error_payload, is_error=True), request=httpx.Request("POST", "http://localhost:3000/mcp"))
    opps_resp = httpx.Response(200, text=_build_sse_mcp_response(opps_payload), request=httpx.Request("POST", "http://localhost:3000/mcp"))

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [comp_resp, people_resp, opps_resp]

        state = await mcp_client.get_live_company_state("Acme Corp")

        assert state.is_connected is True
        assert state.company is not None
        assert state.company.name == "Acme Corp"
        assert state.people == []
        assert state.people_error is not None
        assert "error" in state.people_error.lower()
        assert len(state.opportunities) == 1
        assert state.opportunities[0].name == "Deal 1"
        assert state.opportunities_error is None


# ---------------------------------------------------------------------------
# 13. Ambiguous Company Match Handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ambiguous_company_match(mcp_client):
    """Verify that ambiguous matches cleanly set is_ambiguous=True and candidate list."""
    ambig_payload = {
        "success": True,
        "result": {
            "records": [
                {"id": "c1", "name": "Acme Software Inc"},
                {"id": "c2", "name": "Acme Global Corp"},
            ],
            "count": "2",
            "hasNextPage": False,
        },
    }
    resp = httpx.Response(200, text=_build_sse_mcp_response(ambig_payload), request=httpx.Request("POST", "http://localhost:3000/mcp"))

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = resp

        state = await mcp_client.get_live_company_state("Acme")

        assert state.is_connected is True
        assert state.company is None
        assert state.is_ambiguous is True
        assert set(state.ambiguous_candidates) == {"Acme Software Inc", "Acme Global Corp"}
        assert "Multiple CRM companies match" in (state.error_message or "")
