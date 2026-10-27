"""
Pydantic schemas for Workspace-Scoped MCP Connections (Milestone 6C).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field


class MCPConnectionCreate(BaseModel):
    """Payload to register a new MCP connection in a workspace."""
    name: str = Field(..., min_length=1, max_length=255, description="Display name for connection")
    endpoint_url: str = Field(..., min_length=1, description="Remote MCP endpoint URL (http/https)")
    transport: str = Field(default="streamable-http", description="Transport type: 'streamable-http' or 'sse'")
    credential_mode: str = Field(
        default="env_ref",
        description="Explicit credential mode: 'default' (platform-managed), 'custom' (workspace encrypted secret), or 'env_ref' (approved server env var)",
    )
    credential_ref: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional server-managed environment variable reference name when credential_mode='env_ref'",
    )
    custom_credential: Optional[str] = Field(
        default=None,
        description="Plaintext custom secret to encrypt at rest via Fernet when credential_mode='custom' (never stored plaintext)",
    )
    configuration: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-secret runtime settings (e.g. timeout_seconds, page_size)",
    )


class MCPConnectionUpdate(BaseModel):
    """Payload to update an existing MCP connection."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    endpoint_url: Optional[str] = Field(default=None, min_length=1)
    transport: Optional[str] = Field(default=None)
    credential_mode: Optional[str] = Field(
        default=None,
        description="Update credential mode: 'default', 'custom', or 'env_ref'",
    )
    credential_ref: Optional[str] = Field(default=None, max_length=255)
    custom_credential: Optional[str] = Field(
        default=None,
        description="New custom secret to encrypt. Omit or null to preserve existing credential",
    )
    clear_custom_credential: Optional[bool] = Field(
        default=False,
        description="If True, securely clears the stored encrypted custom credential",
    )
    is_enabled: Optional[bool] = None
    configuration: Optional[dict[str, Any]] = None


class MCPToolToggleRequest(BaseModel):
    """Payload to enable or disable a single discovered tool."""
    tool_name: str = Field(..., min_length=1, max_length=255)
    enabled: bool


class MCPToolAllowlistUpdate(BaseModel):
    """Payload to replace the active enabled tool allowlist."""
    enabled_tools: list[str] = Field(..., description="List of approved tool names to allow")


class MCPToolInvokeRequest(BaseModel):
    """Payload to execute an approved workspace tool."""
    tool_name: str = Field(..., min_length=1, max_length=255)
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPConnectionResponse(BaseModel):
    """Read model for a workspace MCP connection."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    endpoint_url: str
    transport: str
    credential_mode: str = Field(
        default="env_ref",
        description="Active credential mode: 'default', 'custom', or 'env_ref'",
    )
    has_custom_credential: bool = Field(
        default=False,
        description="True if an encrypted custom credential is configured for this connection",
    )
    credential_ref: Optional[str] = None
    is_enabled: bool
    status: str
    error_message: Optional[str] = None
    last_connected_at: Optional[datetime] = None
    last_synced_at: Optional[datetime] = None
    server_info: dict[str, Any] = Field(default_factory=dict)
    discovered_tools: list[dict[str, Any]] = Field(default_factory=list)
    enabled_tools: list[str] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[uuid.UUID] = None
    created_at: datetime
    updated_at: datetime


class MCPToolInvokeResponse(BaseModel):
    """Execution result for an approved workspace tool invocation."""
    tool_name: str
    is_error: bool
    text: str
    structured_content: Optional[Any] = None
    error_message: Optional[str] = None


class MCPProposalCreateRequest(BaseModel):
    """Payload to propose an external MCP mutation."""
    arguments: dict[str, Any] = Field(default_factory=dict)
    run_id: Optional[uuid.UUID] = None
    task_id: Optional[uuid.UUID] = None


class MCPApprovalResponse(BaseModel):
    """Details of a human approval ticket for an MCP mutation."""
    id: uuid.UUID
    workspace_id: uuid.UUID
    connection_id: Optional[uuid.UUID] = None
    tool_name: str
    target_record_id: str
    status: str
    approved_draft_hash: Optional[str] = None
    expires_at: datetime
    created_at: datetime


class MCPApprovalDecisionRequest(BaseModel):
    """Decision submitted by human reviewer."""
    decision: str = Field(..., description="'APPROVED' or 'REJECTED'")
    rejection_reason: Optional[str] = None
    arguments: Optional[dict[str, Any]] = None

