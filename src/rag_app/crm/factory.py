"""
CRM Client Factory.

Provides a unified factory seam selecting between:
- TwentyMCPClient (opt-in when settings.twenty_mcp_enabled is True)
- TwentyCRMClient (default REST baseline when twenty_mcp_enabled is False)
"""

from __future__ import annotations

from typing import Union

from rag_app.core.config import settings
from rag_app.crm.twenty_client import TwentyCRMClient
from rag_app.crm.twenty_mcp_client import TwentyMCPClient

AnyCRMClient = Union[TwentyCRMClient, TwentyMCPClient]


def get_crm_client() -> AnyCRMClient:
    """
    Returns the configured CRM client instance adhering to the unified CRMCompanyState contract.
    Defaults to TwentyCRMClient (REST baseline) unless twenty_mcp_enabled is explicitly True.
    """
    if getattr(settings, "twenty_mcp_enabled", False):
        return TwentyMCPClient()
    return TwentyCRMClient()
