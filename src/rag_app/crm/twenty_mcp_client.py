"""
Read-Only Twenty CRM MCP Client Adapter.

Provides bounded, read-only company, contact, and opportunity retrieval through
Twenty CRM's native MCP server (Streamable HTTP transport at /mcp).

Guarantees:
- Enforces strict client-side allowlist: rejects all write tools (create, update, delete).
- Explicitly disabled by default (opt-in configuration).
- Preserves distinct failure states (timeouts, auth errors, and tool errors do NOT
  masquerade as successful empty results).
- Partial-failure transparency: distinguishes company found with failed contacts
  vs. clean zero contacts.
- Bounded pagination: respects maximum page count and page size limits, recording
  explicit truncation state.
- Adapts Twenty MCP records into existing CRMCompany, CRMPerson, CRMOpportunity,
  and CRMCompanyState contracts.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

from rag_app.core.config import settings
from rag_app.crm.twenty_client import (
    CRMCompany,
    CRMOpportunity,
    CRMPerson,
    CRMCompanyState,
)

logger = logging.getLogger(__name__)

# Strict client-side allowlist for verified read-only operations
READ_ONLY_ALLOWED_TOOLS = frozenset({
    "find_many_companies",
    "find_one_company",
    "find_many_people",
    "find_many_opportunities",
})

# Bounded client-side allowlist for verified, approved write-back operations ONLY
APPROVED_WRITE_BACK_TOOLS = frozenset({
    "create_one_note",
    "create_one_note_target",
})


class TwentyMCPError(Exception):
    """Base exception for Twenty MCP client operations."""


class TwentyMCPAllowlistViolation(TwentyMCPError):
    """Raised when an unapproved or write-capable tool name is requested."""


class TwentyMCPClient:
    """
    Isolated, read-only MCP client adapter communicating with Twenty CRM
    via Streamable HTTP JSON-RPC at /mcp.
    """

    def __init__(
        self,
        mcp_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        enabled: Optional[bool] = None,
        max_pages: int = 3,
        page_size: int = 20,
    ):
        base_url = (settings.twenty_base_url or "http://localhost:3000").rstrip("/")
        default_mcp_url = getattr(settings, "twenty_mcp_url", None) or f"{base_url}/mcp"
        self.mcp_url = mcp_url or default_mcp_url
        self.api_key = api_key if api_key is not None else settings.twenty_api_key
        self.timeout = timeout if timeout is not None else getattr(settings, "twenty_timeout_seconds", 5.0)
        self.enabled = (
            enabled
            if enabled is not None
            else getattr(settings, "twenty_mcp_enabled", False)
        )
        self.max_pages = max_pages
        self.page_size = page_size

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _execute_rpc(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        client: Optional[httpx.AsyncClient] = None,
    ) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "execute_tool",
                "arguments": {
                    "toolName": tool_name,
                    "arguments": arguments,
                },
            },
        }

        async def _do_post(c: httpx.AsyncClient) -> httpx.Response:
            return await c.post(
                self.mcp_url,
                headers=self._get_headers(),
                json=payload,
            )

        if client is not None:
            res = await _do_post(client)
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                res = await _do_post(c)

        if res.status_code in (401, 403):
            raise httpx.HTTPStatusError(
                f"Authentication failed ({res.status_code})",
                request=res.request,
                response=res,
            )
        elif res.status_code == 429:
            raise httpx.HTTPStatusError(
                "Rate limit exceeded",
                request=res.request,
                response=res,
            )
        elif res.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP error {res.status_code}",
                request=res.request,
                response=res,
            )

        return self._parse_mcp_response(res.text)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        client: Optional[httpx.AsyncClient] = None,
    ) -> dict[str, Any]:
        """
        Executes a verified read-only tool via the MCP server's execute_tool meta-tool.
        Enforces strict allowlist rejection on write-capable or unknown tools before
        initiating any network request.
        """
        if tool_name not in READ_ONLY_ALLOWED_TOOLS:
            raise TwentyMCPAllowlistViolation(
                f"Tool '{tool_name}' is not in the read-only MCP allowlist. "
                f"Allowed tools: {sorted(READ_ONLY_ALLOWED_TOOLS)}"
            )

        return await self._execute_rpc(tool_name, arguments, client=client)

    async def call_write_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        client: Optional[httpx.AsyncClient] = None,
    ) -> dict[str, Any]:
        """
        Executes an explicitly approved write-back tool via the MCP server's execute_tool meta-tool.
        Strictly rejects any tool outside APPROVED_WRITE_BACK_TOOLS.
        """
        if tool_name not in APPROVED_WRITE_BACK_TOOLS:
            raise TwentyMCPAllowlistViolation(
                f"Tool '{tool_name}' is not in the approved write-back allowlist. "
                f"Approved write tools: {sorted(APPROVED_WRITE_BACK_TOOLS)}"
            )

        return await self._execute_rpc(tool_name, arguments, client=client)

    async def create_note(
        self,
        title: str,
        markdown_body: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> str:
        """
        Creates a new Note in Twenty CRM and returns its unique UUID.
        """
        arguments = {
            "position": "first",
            "title": title,
            "bodyV2": {
                "markdown": markdown_body,
            },
        }
        res = await self.call_write_tool("create_one_note", arguments, client=client)
        result_payload = res.get("result", {}) if isinstance(res, dict) else {}
        record = result_payload.get("record") or result_payload
        note_id = record.get("id") if isinstance(record, dict) else None
        if not note_id:
            raise TwentyMCPError("Twenty CRM did not return a valid note ID.")
        return str(note_id)

    async def create_note_target_for_company(
        self,
        note_id: str,
        company_id: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> str:
        """
        Links a Note to a Company via NoteTarget and returns the target UUID.
        """
        arguments = {
            "position": "first",
            "noteId": note_id,
            "targetCompanyId": company_id,
        }
        res = await self.call_write_tool("create_one_note_target", arguments, client=client)
        result_payload = res.get("result", {}) if isinstance(res, dict) else {}
        record = result_payload.get("record") or result_payload
        target_id = record.get("id") if isinstance(record, dict) else None
        if not target_id:
            raise TwentyMCPError("Twenty CRM did not return a valid noteTarget ID.")
        return str(target_id)

    def _parse_mcp_response(self, raw_text: str) -> dict[str, Any]:
        """
        Parses Streamable HTTP event-stream lines or direct JSON payload,
        extracting inner tool execution results and validating error flags.
        """
        data: Optional[dict[str, Any]] = None
        for line in raw_text.splitlines():
            line_str = line.strip()
            if line_str.startswith("data: "):
                try:
                    data = json.loads(line_str[6:])
                    break
                except json.JSONDecodeError:
                    continue

        if data is None:
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as jde:
                raise TwentyMCPError(f"Malformed MCP response payload: {jde}") from jde

        if not isinstance(data, dict):
            raise TwentyMCPError("MCP response must be a JSON object.")

        if "error" in data:
            err = data["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise TwentyMCPError(f"MCP protocol error: {msg}")

        result_obj = data.get("result")
        if not isinstance(result_obj, dict):
            raise TwentyMCPError("MCP response missing valid 'result' object.")

        if result_obj.get("isError") is True:
            content_items = result_obj.get("content", [])
            err_msg = "Tool execution reported error."
            if content_items and isinstance(content_items[0], dict):
                inner_text = content_items[0].get("text", "")
                try:
                    inner_json = json.loads(inner_text)
                    err_msg = inner_json.get("message") or inner_json.get("error") or inner_text
                except Exception:
                    err_msg = inner_text or err_msg
            raise TwentyMCPError(f"Twenty MCP tool error: {err_msg}")

        content = result_obj.get("content", [])
        if not content or not isinstance(content[0], dict):
            raise TwentyMCPError("MCP tool response missing content payload.")

        raw_inner = content[0].get("text", "")
        try:
            inner_data = json.loads(raw_inner)
        except json.JSONDecodeError as jde:
            raise TwentyMCPError(f"Malformed tool inner result JSON: {jde}") from jde

        if isinstance(inner_data, dict) and inner_data.get("success") is False:
            err_msg = inner_data.get("message") or inner_data.get("error") or "Operation failed"
            raise TwentyMCPError(f"Twenty tool reported failure: {err_msg}")

        return inner_data

    async def get_company_by_name(
        self,
        name: str,
        aliases: Optional[list[str]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> tuple[Optional[CRMCompany], dict[str, Any]]:
        """
        Looks up a company by name or aliases using Twenty's find_many_companies tool.
        Validates unique company identity and returns ambiguity indicators when multiple
        plausible matches exist.
        """
        clean_name = name.strip()
        if not clean_name:
            return None, {"is_ambiguous": False, "candidates": []}

        search_names = [clean_name]
        if aliases:
            search_names.extend(a.strip() for a in aliases if a and a.strip() != clean_name)

        select_fields = [
            "id",
            "name",
            "domainName",
            "address",
            "annualRevenue",
            "createdAt",
            "updatedAt",
        ]

        exact_matches: list[dict[str, Any]] = []
        substring_matches: list[dict[str, Any]] = []

        for sname in search_names:
            if not sname:
                continue

            tool_args = {
                "name": {"ilike": f"%{sname}%"},
                "limit": 10,
                "select": select_fields,
            }

            inner_result = await self.call_tool("find_many_companies", tool_args, client=client)
            result_payload = inner_result.get("result", {}) if isinstance(inner_result, dict) else {}
            records = result_payload.get("records", [])

            for rec in records:
                c_name = (rec.get("name") or "").strip()
                if c_name.lower() == sname.lower():
                    if rec not in exact_matches:
                        exact_matches.append(rec)
                elif sname.lower() in c_name.lower() or c_name.lower() in sname.lower():
                    if rec not in substring_matches:
                        substring_matches.append(rec)

        # 1. Exact match takes precedence
        if len(exact_matches) == 1:
            return self._parse_company(exact_matches[0]), {"is_ambiguous": False, "candidates": []}
        elif len(exact_matches) > 1:
            candidates = sorted(list({(r.get("name") or "").strip() for r in exact_matches}))
            return None, {"is_ambiguous": True, "candidates": candidates}

        # 2. Substring matches
        if len(substring_matches) == 1:
            return self._parse_company(substring_matches[0]), {"is_ambiguous": False, "candidates": []}
        elif len(substring_matches) > 1:
            candidates = sorted(list({(r.get("name") or "").strip() for r in substring_matches}))
            return None, {"is_ambiguous": True, "candidates": candidates}

        return None, {"is_ambiguous": False, "candidates": []}

    async def get_company_people(
        self,
        company_id: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> tuple[list[CRMPerson], Optional[str], bool]:
        """
        Fetches people/contacts associated with companyId using find_many_people.
        Supports bounded pagination across multiple pages and reports truncation.
        Returns: (people_list, error_message, is_truncated)
        """
        if not company_id:
            return [], None, False

        select_fields = [
            "id",
            "name",
            "jobTitle",
            "emails",
            "phones",
            "companyId",
            "createdAt",
            "updatedAt",
        ]

        all_people: list[CRMPerson] = []
        offset = 0
        is_truncated = False

        for page in range(self.max_pages):
            tool_args = {
                "companyId": {"eq": company_id},
                "limit": self.page_size,
                "offset": offset,
                "select": select_fields,
            }

            try:
                inner_result = await self.call_tool("find_many_people", tool_args, client=client)
                result_payload = inner_result.get("result", {}) if isinstance(inner_result, dict) else {}
                records = result_payload.get("records", [])

                for rec in records:
                    all_people.append(self._parse_person(rec))

                has_next = result_payload.get("hasNextPage", False)
                if not has_next:
                    break

                offset += len(records)
                if page == self.max_pages - 1 and has_next:
                    is_truncated = True

            except Exception as ex:
                logger.warning("Twenty MCP get_company_people failed on page %d: %s", page, type(ex).__name__)
                err_msg = str(ex) if isinstance(ex, TwentyMCPError) else f"Twenty MCP contacts error ({type(ex).__name__})"
                return all_people, err_msg, is_truncated

        return all_people, None, is_truncated

    async def get_company_opportunities(
        self,
        company_id: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> tuple[list[CRMOpportunity], Optional[str], bool]:
        """
        Fetches deals/opportunities associated with companyId using find_many_opportunities.
        Supports bounded pagination across multiple pages and reports truncation.
        Returns: (opportunities_list, error_message, is_truncated)
        """
        if not company_id:
            return [], None, False

        select_fields = [
            "id",
            "name",
            "amount",
            "stage",
            "closeDate",
            "companyId",
            "createdAt",
            "updatedAt",
        ]

        all_opps: list[CRMOpportunity] = []
        offset = 0
        is_truncated = False

        for page in range(self.max_pages):
            tool_args = {
                "companyId": {"eq": company_id},
                "limit": self.page_size,
                "offset": offset,
                "select": select_fields,
            }

            try:
                inner_result = await self.call_tool("find_many_opportunities", tool_args, client=client)
                result_payload = inner_result.get("result", {}) if isinstance(inner_result, dict) else {}
                records = result_payload.get("records", [])

                for rec in records:
                    all_opps.append(self._parse_opportunity(rec))

                has_next = result_payload.get("hasNextPage", False)
                if not has_next:
                    break

                offset += len(records)
                if page == self.max_pages - 1 and has_next:
                    is_truncated = True

            except Exception as ex:
                logger.warning("Twenty MCP get_company_opportunities failed on page %d: %s", page, type(ex).__name__)
                err_msg = str(ex) if isinstance(ex, TwentyMCPError) else f"Twenty MCP opportunities error ({type(ex).__name__})"
                return all_opps, err_msg, is_truncated

        return all_opps, None, is_truncated

    async def get_live_company_state(
        self,
        company_name: str,
        aliases: Optional[list[str]] = None,
    ) -> CRMCompanyState:
        """
        Unified live CRM state retrieval through MCP.
        Reuses a single httpx.AsyncClient session across company, contacts, and opportunities queries.
        Preserves distinct partial-failure states and truncation metadata.
        """
        if not self.enabled:
            return CRMCompanyState(
                company_name=company_name,
                is_connected=False,
                error_message="Twenty MCP integration is disabled in configuration (opt-in only).",
            )

        if not self.api_key:
            return CRMCompanyState(
                company_name=company_name,
                is_connected=False,
                error_message="TWENTY_API_KEY is not configured.",
            )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 1. Company lookup
                company, ambig_info = await self.get_company_by_name(company_name, aliases=aliases, client=client)

                if ambig_info and ambig_info.get("is_ambiguous"):
                    candidates = ambig_info.get("candidates", [])
                    return CRMCompanyState(
                        company_name=company_name,
                        company=None,
                        is_connected=True,
                        is_ambiguous=True,
                        ambiguous_candidates=candidates,
                        error_message=f"Multiple CRM companies match '{company_name}': {', '.join(candidates)}.",
                    )

                if not company:
                    return CRMCompanyState(
                        company_name=company_name,
                        company=None,
                        is_connected=True,
                        error_message=None,
                    )

                # 2. Associated people & opportunities with subresource error isolation
                people, people_err, people_trunc = await self.get_company_people(company.id, client=client)
                opps, opps_err, opps_trunc = await self.get_company_opportunities(company.id, client=client)

                return CRMCompanyState(
                    company_name=company_name,
                    company=company,
                    people=people,
                    opportunities=opps,
                    is_connected=True,
                    error_message=None,
                    people_error=people_err,
                    opportunities_error=opps_err,
                    people_truncated=people_trunc,
                    opportunities_truncated=opps_trunc,
                )

        except TwentyMCPAllowlistViolation as av:
            return CRMCompanyState(
                company_name=company_name,
                is_connected=False,
                error_message=f"MCP security violation: {av}",
            )
        except httpx.TimeoutException:
            return CRMCompanyState(
                company_name=company_name,
                is_connected=False,
                error_message="Twenty MCP request timed out.",
            )
        except httpx.HTTPStatusError as hse:
            status = hse.response.status_code if hse.response else 500
            if status in (401, 403):
                err = "Twenty MCP authentication failed."
            elif status == 429:
                err = "Twenty MCP rate limit exceeded."
            else:
                err = f"Twenty MCP HTTP error ({status})."
            return CRMCompanyState(
                company_name=company_name,
                is_connected=False,
                error_message=err,
            )
        except TwentyMCPError as te:
            return CRMCompanyState(
                company_name=company_name,
                is_connected=False,
                error_message=str(te),
            )
        except Exception as ex:
            logger.warning("Twenty MCP query error: %s", type(ex).__name__)
            return CRMCompanyState(
                company_name=company_name,
                is_connected=False,
                error_message=f"Twenty MCP unavailable ({type(ex).__name__}).",
            )

    async def fetch_companies_page(
        self,
        limit: int = 10,
        offset: int = 0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> tuple[list[CRMCompany], bool, int]:
        """
        Fetches a bounded page of companies using find_many_companies.
        Returns: (companies, has_next_page, total_count)
        """
        select_fields = [
            "id",
            "name",
            "domainName",
            "address",
            "annualRevenue",
            "createdAt",
            "updatedAt",
        ]
        tool_args = {
            "select": select_fields,
            "limit": min(max(1, limit), 100),
            "offset": max(0, offset),
            "orderBy": [{"createdAt": "AscNullsFirst"}],
        }

        inner_result = await self.call_tool("find_many_companies", tool_args, client=client)
        result_payload = inner_result.get("result", {}) if isinstance(inner_result, dict) else {}
        records = result_payload.get("records", [])
        has_next = bool(result_payload.get("hasNextPage", False))
        total_count = int(result_payload.get("totalCount") or result_payload.get("count") or len(records))

        companies = [self._parse_company(r) for r in records if isinstance(r, dict)]
        return companies, has_next, total_count

    async def get_company_by_id(
        self,
        company_id: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Optional[CRMCompany]:
        """
        Retrieves a single company by its unique UUID using find_one_company.
        """
        if not company_id:
            return None
        select_fields = [
            "id",
            "name",
            "domainName",
            "address",
            "annualRevenue",
            "createdAt",
            "updatedAt",
        ]
        tool_args = {
            "id": company_id,
            "select": select_fields,
        }
        try:
            inner_result = await self.call_tool("find_one_company", tool_args, client=client)
            result_payload = inner_result.get("result", {}) if isinstance(inner_result, dict) else {}
            record = result_payload.get("record") or (result_payload.get("records") or [None])[0] or result_payload
            if isinstance(record, dict) and record.get("id"):
                return self._parse_company(record)
            return None
        except TwentyMCPError:
            return None

    async def get_live_company_state_by_id(
        self,
        company_id: str,
    ) -> CRMCompanyState:
        """
        Retrieves full company state (company, contacts, deals) by company UUID.
        """
        if not self.enabled:
            return CRMCompanyState(
                company_name=company_id,
                is_connected=False,
                error_message="Twenty MCP client is disabled.",
            )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                company = await self.get_company_by_id(company_id, client=client)
                if not company:
                    return CRMCompanyState(
                        company_name=company_id,
                        company=None,
                        is_connected=True,
                        error_message=f"Company with ID '{company_id}' not found.",
                    )

                people, people_err, people_trunc = await self.get_company_people(company.id, client=client)
                opps, opps_err, opps_trunc = await self.get_company_opportunities(company.id, client=client)

                return CRMCompanyState(
                    company_name=company.name,
                    company=company,
                    people=people,
                    opportunities=opps,
                    is_connected=True,
                    error_message=None,
                    people_error=people_err,
                    opportunities_error=opps_err,
                    people_truncated=people_trunc,
                    opportunities_truncated=opps_trunc,
                )
        except Exception as ex:
            logger.warning("Twenty MCP query error by id: %s", type(ex).__name__)
            return CRMCompanyState(
                company_name=company_id,
                is_connected=False,
                error_message=f"Twenty MCP unavailable ({type(ex).__name__}).",
            )

    @staticmethod
    def _parse_company(data: dict[str, Any]) -> CRMCompany:
        domain_obj = data.get("domainName") or {}
        domain = domain_obj.get("primaryLinkUrl") if isinstance(domain_obj, dict) else str(domain_obj)

        addr_obj = data.get("address") or {}
        address_str = None
        if isinstance(addr_obj, dict):
            parts = [
                addr_obj.get("addressStreet1"),
                addr_obj.get("addressCity"),
                addr_obj.get("addressState"),
                addr_obj.get("addressCountry"),
            ]
            parts = [str(p).strip() for p in parts if p and str(p).strip()]
            if parts:
                address_str = ", ".join(parts)

        rev_obj = data.get("annualRevenue") or {}
        revenue = None
        if isinstance(rev_obj, dict) and rev_obj.get("amountMicros") is not None:
            try:
                revenue = float(rev_obj["amountMicros"]) / 1_000_000.0
            except (ValueError, TypeError):
                pass

        return CRMCompany(
            id=str(data.get("id")),
            name=str(data.get("name") or "Unknown"),
            domain=domain,
            address=address_str,
            annual_revenue=revenue,
            created_at=data.get("createdAt"),
            updated_at=data.get("updatedAt"),
        )

    @staticmethod
    def _parse_person(data: dict[str, Any]) -> CRMPerson:
        name_obj = data.get("name") or {}
        if isinstance(name_obj, dict):
            first = name_obj.get("firstName") or ""
            last = name_obj.get("lastName") or ""
            full_name = f"{first} {last}".strip()
        else:
            full_name = str(name_obj or "Unknown Contact").strip()

        emails_obj = data.get("emails") or {}
        email = (
            emails_obj.get("primaryEmail") or emails_obj.get("email")
            if isinstance(emails_obj, dict)
            else (str(emails_obj) if emails_obj else None)
        )

        phones_obj = data.get("phones") or {}
        phone = (
            phones_obj.get("primaryPhoneNumber") or phones_obj.get("primaryPhone") or phones_obj.get("number")
            if isinstance(phones_obj, dict)
            else (str(phones_obj) if phones_obj else None)
        )

        return CRMPerson(
            id=str(data.get("id")),
            name=full_name or "Unknown Contact",
            job_title=data.get("jobTitle"),
            email=email,
            phone=phone,
            company_id=data.get("companyId"),
        )

    @staticmethod
    def _parse_opportunity(data: dict[str, Any]) -> CRMOpportunity:
        amt_obj = data.get("amount") or {}
        amount = None
        currency = "USD"
        if isinstance(amt_obj, dict):
            if amt_obj.get("amountMicros") is not None:
                try:
                    amount = float(amt_obj["amountMicros"]) / 1_000_000.0
                except (ValueError, TypeError):
                    pass
            currency = amt_obj.get("currencyCode") or "USD"

        return CRMOpportunity(
            id=str(data.get("id")),
            name=str(data.get("name") or "Unnamed Opportunity"),
            amount=amount,
            currency=currency,
            stage=str(data.get("stage") or "UNKNOWN"),
            close_date=data.get("closeDate"),
            company_id=data.get("companyId"),
            updated_at=data.get("updatedAt"),
        )
