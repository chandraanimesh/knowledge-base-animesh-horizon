"""
Security utilities for Outbound Generic MCP Client:
- SSRF prevention (cloud metadata, private IP protection, unsafe schemes)
- Header and credential sanitization for logging
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Any, Optional
from urllib.parse import urlparse

from rag_app.mcp.client.models import MCPSSRFError

logger = logging.getLogger(__name__)

# Cloud metadata addresses to strictly block across all environments
PROHIBITED_METADATA_HOSTS = frozenset({
    "169.254.169.254",
    "169.254.170.2",
    "metadata.google.internal",
    "metadata.azure.internal",
    "fd00:ec2::254",
})

SENSITIVE_HEADER_KEYS = frozenset({
    "authorization",
    "proxy-authorization",
    "cookie",
    "x-api-key",
    "api-key",
    "apikey",
    "secret",
    "token",
    "password",
})


def validate_mcp_url(url: str, allow_private_hosts: bool = False) -> None:
    """
    Validates that the remote endpoint URL conforms to SSRF safety requirements.
    
    1. Only http and https schemes are permitted.
    2. Cloud metadata endpoints are strictly rejected regardless of configuration.
    3. Loopback, private network, and link-local addresses are rejected unless
       allow_private_hosts is explicitly set to True (strictly permitted only in dev/test environments).
    4. Domain resolution check: Resolves hostname via socket.getaddrinfo to detect domains pointing
       to prohibited private or cloud-metadata IPs.

    DNS-Rebinding Limitation Notice:
    Resolving hostname at validation time provides strong pre-flight protection against private-network
    and cloud-metadata resolution. However, standard HTTP transports re-resolve DNS on socket creation,
    leaving a theoretical Time-of-Check to Time-of-Use (TOCTOU) DNS-rebinding window if an attacker
    controls the authoritative nameserver and returns short TTLs. In high-assurance environments,
    transport-level IP pinning, egress firewalls, or forward proxying should be combined with this check.
    Endpoint revalidation occurs on registration, update, sync, and tool invocation.
    """
    if not url or not isinstance(url, str):
        raise MCPSSRFError("MCP endpoint URL must be a non-empty string.")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise MCPSSRFError(f"Malformed endpoint URL: {e}") from e

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise MCPSSRFError(
            f"Unsupported URL scheme '{scheme}'. Only 'http' and 'https' are permitted for MCP endpoints."
        )

    host = parsed.hostname
    if not host:
        raise MCPSSRFError("MCP endpoint URL missing valid hostname.")

    host_lower = host.lower()

    # 1. Strictly block known cloud metadata services
    if host_lower in PROHIBITED_METADATA_HOSTS:
        raise MCPSSRFError(
            f"Access to prohibited cloud metadata endpoint '{host}' is forbidden."
        )

    # 2. Check if host is a literal IP address
    try:
        ip = ipaddress.ip_address(host_lower)
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
            if not allow_private_hosts:
                raise MCPSSRFError(
                    f"Access to private/local IP address '{host}' is forbidden without allow_private_hosts=True."
                )
        return
    except ValueError:
        # Host is a domain name, proceed with domain and DNS checks
        pass

    # 3. Check for obvious private / internal domain names
    if not allow_private_hosts:
        if (
            host_lower == "localhost"
            or host_lower.endswith(".localhost")
            or host_lower.endswith(".local")
            or host_lower.endswith(".internal")
        ):
            raise MCPSSRFError(
                f"Access to local domain '{host}' is forbidden without allow_private_hosts=True."
            )

        # 4. Resolve domain name to check for DNS rebinding / private resolution
        try:
            addr_info = socket.getaddrinfo(host, None)
            for family, _, _, _, sockaddr in addr_info:
                ip_str = sockaddr[0]
                try:
                    resolved_ip = ipaddress.ip_address(ip_str)
                    if (
                        resolved_ip.is_loopback
                        or resolved_ip.is_private
                        or resolved_ip.is_link_local
                        or resolved_ip.is_reserved
                    ):
                        raise MCPSSRFError(
                            f"Domain '{host}' resolves to prohibited private/local IP '{ip_str}'."
                        )
                    if str(resolved_ip) in PROHIBITED_METADATA_HOSTS:
                        raise MCPSSRFError(
                            f"Domain '{host}' resolves to prohibited cloud metadata IP '{ip_str}'."
                        )
                except ValueError:
                    continue
        except (socket.gaierror, socket.herror) as dns_err:
            # If DNS resolution fails, raise connection-level error
            raise MCPSSRFError(f"Failed to resolve endpoint domain '{host}': {dns_err}") from dns_err


def sanitize_headers_for_logging(headers: Optional[dict[str, Any]]) -> dict[str, str]:
    """
    Returns a copy of the headers dictionary with sensitive credential values redacted.
    Never exposes bearer tokens, api keys, or authorization headers in logs.
    """
    if not headers:
        return {}
    sanitized: dict[str, str] = {}
    for k, v in headers.items():
        k_lower = str(k).lower()
        if any(sens in k_lower for sens in SENSITIVE_HEADER_KEYS):
            sanitized[str(k)] = "[REDACTED]"
        else:
            sanitized[str(k)] = str(v)
    return sanitized
