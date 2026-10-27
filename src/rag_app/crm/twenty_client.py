"""
Read-Only Twenty CRM Connector.

Provides safe, bounded, read-only access to Twenty CRM objects:
- Companies (/rest/companies)
- People / Contacts (/rest/people)
- Opportunities / Deals (/rest/opportunities)

Enforces strict timeouts, safe error handling, no credential logging,
and zero external mutations/writes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from rag_app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CRMCompany:
    id: str
    name: str
    domain: Optional[str] = None
    address: Optional[str] = None
    annual_revenue: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class CRMPerson:
    id: str
    name: str
    job_title: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company_id: Optional[str] = None


@dataclass
class CRMOpportunity:
    id: str
    name: str
    amount: Optional[float] = None
    currency: Optional[str] = None
    stage: str = "UNKNOWN"
    close_date: Optional[str] = None
    company_id: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class CRMCompanyState:
    company_name: str
    company: Optional[CRMCompany] = None
    people: list[CRMPerson] = field(default_factory=list)
    opportunities: list[CRMOpportunity] = field(default_factory=list)
    is_connected: bool = False
    error_message: Optional[str] = None
    retrieval_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    people_error: Optional[str] = None
    opportunities_error: Optional[str] = None
    people_truncated: bool = False
    opportunities_truncated: bool = False
    is_ambiguous: bool = False
    ambiguous_candidates: list[str] = field(default_factory=list)

    @property
    def has_crm_records(self) -> bool:
        return self.company is not None


class TwentyCRMClient:
    """
    Minimal, secure, read-only HTTP client for Twenty CRM.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.twenty_base_url or "http://localhost:3000").rstrip("/")
        self.api_key = api_key if api_key is not None else settings.twenty_api_key
        self.timeout = timeout if timeout is not None else getattr(settings, "twenty_timeout_seconds", 5.0)

    def _get_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def check_health(self) -> bool:
        """Verifies if the CRM instance is reachable."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(f"{self.base_url}/healthz")
                return res.status_code == 200
        except Exception as e:
            logger.warning("Twenty CRM health check failed: %s", type(e).__name__)
            return False

    async def get_company_by_name(
        self, name: str, aliases: Optional[list[str]] = None
    ) -> Optional[CRMCompany]:
        """
        Looks up a company by canonical name or verified aliases using
        Twenty's REST API filtering.
        """
        if not self.api_key:
            logger.debug("Twenty CRM API key is missing; cannot fetch company.")
            return None

        search_names = [name]
        if aliases:
            search_names.extend(a for a in aliases if a and a != name)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for sname in search_names:
                clean_name = sname.strip()
                if not clean_name:
                    continue

                # Query exact match or case-insensitive pattern
                for filter_expr in [f'name[eq]:"{clean_name}"', f'name[ilike]:"%{clean_name}%"']:
                    try:
                        res = await client.get(
                            f"{self.base_url}/rest/companies",
                            headers=self._get_headers(),
                            params={"filter": filter_expr, "limit": 10},
                        )

                        if res.status_code == 200:
                            data = res.json().get("data", {}).get("companies", [])
                            for c in data:
                                if c.get("deletedAt") is not None:
                                    continue
                                c_name = (c.get("name") or "").strip()
                                # Verify match similarity
                                if (
                                    c_name.lower() == clean_name.lower()
                                    or clean_name.lower() in c_name.lower()
                                ):
                                    return self._parse_company(c)

                        elif res.status_code in (401, 403, 429) or res.status_code >= 400:
                            res.raise_for_status()

                    except httpx.TimeoutException:
                        logger.warning("Twenty CRM request timed out for query: %s", clean_name)
                        raise
                    except httpx.HTTPStatusError:
                        raise
                    except Exception as ex:
                        logger.warning("Twenty CRM query error: %s", type(ex).__name__)
                        raise

        return None

    async def get_company_people(self, company_id: str) -> list[CRMPerson]:
        """Fetches contacts associated with the company."""
        if not self.api_key or not company_id:
            return []

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(
                    f"{self.base_url}/rest/people",
                    headers=self._get_headers(),
                    params={"filter": f'companyId[eq]:"{company_id}"', "limit": 20},
                )
                if res.status_code == 200:
                    people_data = res.json().get("data", {}).get("people", [])
                    return [self._parse_person(p) for p in people_data if p.get("deletedAt") is None]
        except Exception as ex:
            logger.warning("Failed to fetch people for CRM company %s: %s", company_id, type(ex).__name__)
        return []

    async def get_company_opportunities(self, company_id: str) -> list[CRMOpportunity]:
        """Fetches deals and sales opportunities for the company."""
        if not self.api_key or not company_id:
            return []

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(
                    f"{self.base_url}/rest/opportunities",
                    headers=self._get_headers(),
                    params={"filter": f'companyId[eq]:"{company_id}"', "limit": 20},
                )
                if res.status_code == 200:
                    opps_data = res.json().get("data", {}).get("opportunities", [])
                    return [self._parse_opportunity(o) for o in opps_data if o.get("deletedAt") is None]
        except Exception as ex:
            logger.warning("Failed to fetch opportunities for CRM company %s: %s", company_id, type(ex).__name__)
        return []

    async def get_live_company_state(
        self, company_name: str, aliases: Optional[list[str]] = None
    ) -> CRMCompanyState:
        """
        Retrieves the unified live CRM state for a company, including
        contacts and active pipeline opportunities.
        """
        if not getattr(settings, "enable_crm_integration", True):
            return CRMCompanyState(
                company_name=company_name,
                is_connected=False,
                error_message="CRM integration is disabled in configuration.",
            )

        if not self.api_key:
            return CRMCompanyState(
                company_name=company_name,
                is_connected=False,
                error_message="TWENTY_API_KEY is not configured.",
            )

        try:
            # 1. Look up company
            company = await self.get_company_by_name(company_name, aliases=aliases)
            if not company:
                return CRMCompanyState(
                    company_name=company_name,
                    company=None,
                    is_connected=True,
                    error_message=None,
                )

            # 2. Parallel/sequential retrieval of associated people & opportunities
            people = await self.get_company_people(company.id)
            opps = await self.get_company_opportunities(company.id)

            return CRMCompanyState(
                company_name=company_name,
                company=company,
                people=people,
                opportunities=opps,
                is_connected=True,
                error_message=None,
            )

        except httpx.TimeoutException:
            return CRMCompanyState(
                company_name=company_name,
                is_connected=False,
                error_message="Twenty CRM request timed out.",
            )
        except httpx.HTTPStatusError as hse:
            status = hse.response.status_code
            if status in (401, 403):
                err = "Twenty CRM authentication failed."
            elif status == 429:
                err = "Twenty CRM rate limit exceeded."
            else:
                err = f"Twenty CRM HTTP error ({status})."
            return CRMCompanyState(
                company_name=company_name,
                is_connected=False,
                error_message=err,
            )
        except Exception as ex:
            return CRMCompanyState(
                company_name=company_name,
                is_connected=False,
                error_message=f"Twenty CRM unavailable ({type(ex).__name__}).",
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
            full_name = f"{name_obj.get('firstName', '')} {name_obj.get('lastName', '')}".strip()
        else:
            full_name = str(name_obj or "Unknown")

        emails = data.get("emails") or {}
        email = emails.get("primaryEmail") if isinstance(emails, dict) else None

        phones = data.get("phones") or {}
        phone = phones.get("primaryPhone") if isinstance(phones, dict) else None

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
