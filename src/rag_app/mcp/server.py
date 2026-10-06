"""
FastMCP Server for Production RAG Backend.
Exposes the authoritative 51-tool capability surface structured by domain:
- Identity (2 tools)
- RAG (6 tools)
- Conversations (4 tools)
- Workspaces (12 tools)
- Access Requests (5 tools)
- Admin Users (9 tools)
- Admin Permissions (4 tools)
- Admin Documents (5 tools)
- Admin Departments (3 tools)
- Diagnostics (1 tool)

All tools are protected by MCP Gateway, enforcing authenticated context, scope checks,
and RBAC authorization while preserving the existing application services.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional
from fastmcp import FastMCP

from rag_app.mcp.gateway import mcp_gateway
from rag_app.mcp.policies import TOOL_POLICIES
from rag_app.schemas.users import PermissionPreset

# Import domain tool functions
from rag_app.mcp.tools import (
    identity,
    rag,
    conversations,
    workspaces,
    access_requests,
    admin_users,
    admin_permissions,
    admin_documents,
    admin_departments,
    diagnostics,
    discovery,
)

logger = logging.getLogger(__name__)

SERVER_INSTRUCTIONS = """
FastMCP interface for the Enterprise Production RAG Backend.

CRITICAL OPERATIONAL & SECURITY INSTRUCTIONS:
1. All tools operate using the authenticated user's context (resolved via Bearer JWT token).
2. Tool arguments NEVER grant authorization. Caller identity and roles are cryptographically verified.
3. Never attempt to impersonate another user or override caller identity in tool parameters.
4. RAG retrieval and response generation are strictly constrained by existing RBAC and department boundaries.
5. Administrative tools require appropriate administrative privilege ('ADMIN' role) and MCP scope.
6. Use 'rag_search' to retrieve grounded evidence chunks and 'rag_ask' for complete grounded answers.
7. Only ingest documents when explicitly requested by the user.
8. Never request, log, or expose raw credentials or secrets.
9. Use 'search_tools' to discover relevant tools and parameter schemas by natural language or domain based on your access level.
"""

# Initialize FastMCP Server
mcp = FastMCP(
    name="Production RAG",
    instructions=SERVER_INSTRUCTIONS.strip(),
)


# ── 1. IDENTITY DOMAIN (2 tools) ─────────────────────────────────────────────

@mcp.tool(
    name="get_current_user_profile",
    description=TOOL_POLICIES["get_current_user_profile"].description,
    tags=set(TOOL_POLICIES["get_current_user_profile"].tags),
)
async def get_current_user_profile(
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Get the authenticated caller's profile (ID, email, role, active status)."""
    return await identity.get_current_user_profile(auth_token=auth_token)


@mcp.tool(
    name="get_current_user_access",
    description=TOOL_POLICIES["get_current_user_access"].description,
    tags=set(TOOL_POLICIES["get_current_user_access"].tags),
)
async def get_current_user_access(
    auth_token: Optional[str] = None,
) -> dict[str, list[str]]:
    """Inspect the current user's active department memberships and permissions."""
    return await identity.get_current_user_access(auth_token=auth_token)


# ── 2. RAG DOMAIN (6 tools) ──────────────────────────────────────────────────

@mcp.tool(
    name="list_documents",
    description=TOOL_POLICIES["list_documents"].description,
    tags=set(TOOL_POLICIES["list_documents"].tags),
)
async def list_documents(
    department_code: Optional[str] = None,
    workspace_id: Optional[str] = None,
    limit: int = 50,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """List authorized documents accessible in the RAG system with chunk counts and statuses."""
    return await rag.list_documents(
        department_code=department_code,
        workspace_id=workspace_id,
        limit=limit,
        auth_token=auth_token,
    )


@mcp.tool(
    name="preview_document",
    description=TOOL_POLICIES["preview_document"].description,
    tags=set(TOOL_POLICIES["preview_document"].tags),
)
async def preview_document(
    document_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Read-only preview of document content without re-ingesting."""
    return await rag.preview_document(
        document_id=document_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="ingest_document",
    description=TOOL_POLICIES["ingest_document"].description,
    tags=set(TOOL_POLICIES["ingest_document"].tags),
)
async def ingest_document(
    filename: str,
    content_base64: Optional[str] = None,
    content_text: Optional[str] = None,
    department_code: Optional[str] = None,
    workspace_id: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Ingest a document into PostgreSQL / pgvector and BM25 index."""
    return await rag.ingest_document(
        filename=filename,
        content_base64=content_base64,
        content_text=content_text,
        department_code=department_code,
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="rag_search",
    description=TOOL_POLICIES["rag_search"].description,
    tags=set(TOOL_POLICIES["rag_search"].tags),
)
async def rag_search(
    query: str,
    top_k: int = 5,
    rerank: bool = True,
    workspace_id: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Retrieve authorized evidence chunks from the RAG pipeline without generating an LLM response."""
    return await rag.rag_search(
        query=query,
        top_k=top_k,
        rerank=rerank,
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="rag_ask",
    description=TOOL_POLICIES["rag_ask"].description,
    tags=set(TOOL_POLICIES["rag_ask"].tags),
)
async def rag_ask(
    query: str,
    top_k: int = 5,
    rerank: bool = True,
    workspace_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Ask a question and receive an authorized, grounded response with source citations."""
    return await rag.rag_ask(
        query=query,
        top_k=top_k,
        rerank=rerank,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="list_active_departments",
    description=TOOL_POLICIES["list_active_departments"].description,
    tags=set(TOOL_POLICIES["list_active_departments"].tags),
)
async def list_active_departments(
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List active departments for authenticated users."""
    return await rag.list_active_departments(auth_token=auth_token)


# ── 3. CONVERSATIONS DOMAIN (4 tools) ────────────────────────────────────────

@mcp.tool(
    name="list_conversations",
    description=TOOL_POLICIES["list_conversations"].description,
    tags=set(TOOL_POLICIES["list_conversations"].tags),
)
async def list_conversations(
    workspace_id: Optional[str] = None,
    limit: int = 50,
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List conversations for current authenticated user."""
    return await conversations.list_conversations(
        workspace_id=workspace_id,
        limit=limit,
        auth_token=auth_token,
    )


@mcp.tool(
    name="create_conversation",
    description=TOOL_POLICIES["create_conversation"].description,
    tags=set(TOOL_POLICIES["create_conversation"].tags),
)
async def create_conversation(
    title: Optional[str] = None,
    workspace_id: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Create a new conversation for current authenticated user."""
    return await conversations.create_conversation(
        title=title,
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="get_conversation",
    description=TOOL_POLICIES["get_conversation"].description,
    tags=set(TOOL_POLICIES["get_conversation"].tags),
)
async def get_conversation(
    conversation_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Get full conversation message history."""
    return await conversations.get_conversation(
        conversation_id=conversation_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="delete_conversation",
    description=TOOL_POLICIES["delete_conversation"].description,
    tags=set(TOOL_POLICIES["delete_conversation"].tags),
)
async def delete_conversation(
    conversation_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    """Delete a conversation and all its messages."""
    return await conversations.delete_conversation(
        conversation_id=conversation_id,
        auth_token=auth_token,
    )


# ── 4. WORKSPACES DOMAIN (12 tools) ──────────────────────────────────────────

@mcp.tool(
    name="create_workspace",
    description=TOOL_POLICIES["create_workspace"].description,
    tags=set(TOOL_POLICIES["create_workspace"].tags),
)
async def create_workspace(
    name: str,
    code: str,
    department_id: str,
    description: Optional[str] = None,
    is_active: bool = True,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Create a new workspace within a department."""
    return await workspaces.create_workspace(
        name=name,
        code=code,
        department_id=department_id,
        description=description,
        is_active=is_active,
        auth_token=auth_token,
    )


@mcp.tool(
    name="list_workspaces",
    description=TOOL_POLICIES["list_workspaces"].description,
    tags=set(TOOL_POLICIES["list_workspaces"].tags),
)
async def list_workspaces(
    department_id: Optional[str] = None,
    include_inactive: bool = False,
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List workspaces accessible by the caller."""
    return await workspaces.list_workspaces(
        department_id=department_id,
        include_inactive=include_inactive,
        auth_token=auth_token,
    )


@mcp.tool(
    name="get_workspace",
    description=TOOL_POLICIES["get_workspace"].description,
    tags=set(TOOL_POLICIES["get_workspace"].tags),
)
async def get_workspace(
    workspace_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Get metadata and status of a workspace."""
    return await workspaces.get_workspace(
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="update_workspace",
    description=TOOL_POLICIES["update_workspace"].description,
    tags=set(TOOL_POLICIES["update_workspace"].tags),
)
async def update_workspace(
    workspace_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    is_active: Optional[bool] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Update workspace metadata (Admin or Workspace Owner only)."""
    return await workspaces.update_workspace(
        workspace_id=workspace_id,
        name=name,
        description=description,
        is_active=is_active,
        auth_token=auth_token,
    )


@mcp.tool(
    name="delete_workspace",
    description=TOOL_POLICIES["delete_workspace"].description,
    tags=set(TOOL_POLICIES["delete_workspace"].tags),
)
async def delete_workspace(
    workspace_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    """Soft delete a workspace (Admin or Workspace Owner only)."""
    return await workspaces.delete_workspace(
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="list_workspace_members",
    description=TOOL_POLICIES["list_workspace_members"].description,
    tags=set(TOOL_POLICIES["list_workspace_members"].tags),
)
async def list_workspace_members(
    workspace_id: str,
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List members of a workspace."""
    return await workspaces.list_workspace_members(
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="add_workspace_member",
    description=TOOL_POLICIES["add_workspace_member"].description,
    tags=set(TOOL_POLICIES["add_workspace_member"].tags),
)
async def add_workspace_member(
    workspace_id: str,
    user_id: str,
    role: str = "MEMBER",
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Add a member to a workspace (Admin or Workspace Owner only)."""
    return await workspaces.add_workspace_member(
        workspace_id=workspace_id,
        user_id=user_id,
        role=role,
        auth_token=auth_token,
    )


@mcp.tool(
    name="update_workspace_member_role",
    description=TOOL_POLICIES["update_workspace_member_role"].description,
    tags=set(TOOL_POLICIES["update_workspace_member_role"].tags),
)
async def update_workspace_member_role(
    workspace_id: str,
    user_id: str,
    role: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Update role of a workspace member (Admin or Workspace Owner only)."""
    return await workspaces.update_workspace_member_role(
        workspace_id=workspace_id,
        user_id=user_id,
        role=role,
        auth_token=auth_token,
    )


@mcp.tool(
    name="remove_workspace_member",
    description=TOOL_POLICIES["remove_workspace_member"].description,
    tags=set(TOOL_POLICIES["remove_workspace_member"].tags),
)
async def remove_workspace_member(
    workspace_id: str,
    user_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    """Remove a member from a workspace (Admin or Workspace Owner only)."""
    return await workspaces.remove_workspace_member(
        workspace_id=workspace_id,
        user_id=user_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="get_workspace_provider_profile",
    description=TOOL_POLICIES["get_workspace_provider_profile"].description,
    tags=set(TOOL_POLICIES["get_workspace_provider_profile"].tags),
)
async def get_workspace_provider_profile(
    workspace_id: str,
    auth_token: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Get active provider profile for a workspace."""
    return await workspaces.get_workspace_provider_profile(
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="upsert_workspace_provider_profile",
    description=TOOL_POLICIES["upsert_workspace_provider_profile"].description,
    tags=set(TOOL_POLICIES["upsert_workspace_provider_profile"].tags),
)
async def upsert_workspace_provider_profile(
    workspace_id: str,
    chat_provider: Optional[str] = None,
    chat_model: Optional[str] = None,
    chat_credential_ref: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_credential_ref: Optional[str] = None,
    rerank_provider: Optional[str] = None,
    rerank_model: Optional[str] = None,
    rerank_credential_ref: Optional[str] = None,
    notes: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Create or update provider profile for a workspace (Admin or Owner only)."""
    return await workspaces.upsert_workspace_provider_profile(
        workspace_id=workspace_id,
        chat_provider=chat_provider,
        chat_model=chat_model,
        chat_credential_ref=chat_credential_ref,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_credential_ref=embedding_credential_ref,
        rerank_provider=rerank_provider,
        rerank_model=rerank_model,
        rerank_credential_ref=rerank_credential_ref,
        notes=notes,
        auth_token=auth_token,
    )


@mcp.tool(
    name="list_provider_capabilities",
    description=TOOL_POLICIES["list_provider_capabilities"].description,
    tags=set(TOOL_POLICIES["list_provider_capabilities"].tags),
)
async def list_provider_capabilities(
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """List registered provider capabilities and models."""
    return await workspaces.list_provider_capabilities(auth_token=auth_token)


# ── 5. ACCESS REQUESTS DOMAIN (5 tools) ──────────────────────────────────────

@mcp.tool(
    name="create_access_request",
    description=TOOL_POLICIES["create_access_request"].description,
    tags=set(TOOL_POLICIES["create_access_request"].tags),
)
async def create_access_request(
    department_code: str,
    requested_permissions: list[str],
    reason: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Submit a request for access to a department."""
    return await access_requests.create_access_request(
        department_code=department_code,
        requested_permissions=requested_permissions,
        reason=reason,
        auth_token=auth_token,
    )


@mcp.tool(
    name="list_my_access_requests",
    description=TOOL_POLICIES["list_my_access_requests"].description,
    tags=set(TOOL_POLICIES["list_my_access_requests"].tags),
)
async def list_my_access_requests(
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List all access requests submitted by current user."""
    return await access_requests.list_my_access_requests(auth_token=auth_token)


@mcp.tool(
    name="list_all_access_requests",
    description=TOOL_POLICIES["list_all_access_requests"].description,
    tags=set(TOOL_POLICIES["list_all_access_requests"].tags),
)
async def list_all_access_requests(
    status_filter: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List all access requests (Admin only)."""
    return await access_requests.list_all_access_requests(
        status_filter=status_filter,
        auth_token=auth_token,
    )


@mcp.tool(
    name="approve_access_request",
    description=TOOL_POLICIES["approve_access_request"].description,
    tags=set(TOOL_POLICIES["approve_access_request"].tags),
)
async def approve_access_request(
    request_id: str,
    approved_permissions: Optional[list[str]] = None,
    review_note: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Approve a pending access request (Admin only)."""
    return await access_requests.approve_access_request(
        request_id=request_id,
        approved_permissions=approved_permissions,
        review_note=review_note,
        auth_token=auth_token,
    )


@mcp.tool(
    name="reject_access_request",
    description=TOOL_POLICIES["reject_access_request"].description,
    tags=set(TOOL_POLICIES["reject_access_request"].tags),
)
async def reject_access_request(
    request_id: str,
    review_note: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Reject a pending access request (Admin only)."""
    return await access_requests.reject_access_request(
        request_id=request_id,
        review_note=review_note,
        auth_token=auth_token,
    )


# ── 6. ADMIN USERS DOMAIN (9 tools) ──────────────────────────────────────────

@mcp.tool(
    name="admin_create_user",
    description=TOOL_POLICIES["admin_create_user"].description,
    tags=set(TOOL_POLICIES["admin_create_user"].tags),
)
async def admin_create_user(
    email: str,
    password: str,
    role: Literal["ADMIN", "USER"] = "USER",
    department_code: Optional[str] = None,
    permissions: Optional[list[str]] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Create a new user with optional initial department and permissions (Admin only)."""
    return await admin_users.admin_create_user(
        email=email,
        password=password,
        role=role,
        department_code=department_code,
        permissions=permissions,
        auth_token=auth_token,
    )


@mcp.tool(
    name="admin_list_users",
    description=TOOL_POLICIES["admin_list_users"].description,
    tags=set(TOOL_POLICIES["admin_list_users"].tags),
)
async def admin_list_users(
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List all registered users (Admin only)."""
    return await admin_users.admin_list_users(auth_token=auth_token)


@mcp.tool(
    name="admin_list_users_with_access",
    description=TOOL_POLICIES["admin_list_users_with_access"].description,
    tags=set(TOOL_POLICIES["admin_list_users_with_access"].tags),
)
async def admin_list_users_with_access(
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List all users with aggregated department memberships and permissions (Admin only)."""
    return await admin_users.admin_list_users_with_access(auth_token=auth_token)


@mcp.tool(
    name="admin_list_all_departments_lookup",
    description=TOOL_POLICIES["admin_list_all_departments_lookup"].description,
    tags=set(TOOL_POLICIES["admin_list_all_departments_lookup"].tags),
)
async def admin_list_all_departments_lookup(
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List all departments for admin user management dropdowns (Admin only)."""
    return await admin_users.admin_list_all_departments_lookup(auth_token=auth_token)


@mcp.tool(
    name="admin_get_user",
    description=TOOL_POLICIES["admin_get_user"].description,
    tags=set(TOOL_POLICIES["admin_get_user"].tags),
)
async def admin_get_user(
    user_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Get user details by ID (Admin only)."""
    return await admin_users.admin_get_user(
        user_id=user_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="admin_update_user_status",
    description=TOOL_POLICIES["admin_update_user_status"].description,
    tags=set(TOOL_POLICIES["admin_update_user_status"].tags),
)
async def admin_update_user_status(
    user_id: str,
    is_active: bool,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Activate or deactivate a user account (Admin only)."""
    return await admin_users.admin_update_user_status(
        user_id=user_id,
        is_active=is_active,
        auth_token=auth_token,
    )


@mcp.tool(
    name="admin_list_user_departments",
    description=TOOL_POLICIES["admin_list_user_departments"].description,
    tags=set(TOOL_POLICIES["admin_list_user_departments"].tags),
)
async def admin_list_user_departments(
    user_id: str,
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List departments assigned to a specific user (Admin only)."""
    return await admin_users.admin_list_user_departments(
        user_id=user_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="admin_assign_user_department",
    description=TOOL_POLICIES["admin_assign_user_department"].description,
    tags=set(TOOL_POLICIES["admin_assign_user_department"].tags),
)
async def admin_assign_user_department(
    user_id: str,
    department_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Assign a user to a department (Admin only)."""
    return await admin_users.admin_assign_user_department(
        user_id=user_id,
        department_code=department_code,
        auth_token=auth_token,
    )


@mcp.tool(
    name="admin_remove_user_department",
    description=TOOL_POLICIES["admin_remove_user_department"].description,
    tags=set(TOOL_POLICIES["admin_remove_user_department"].tags),
)
async def admin_remove_user_department(
    user_id: str,
    department_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    """Remove a user from a department (Admin only)."""
    return await admin_users.admin_remove_user_department(
        user_id=user_id,
        department_code=department_code,
        auth_token=auth_token,
    )


# ── 7. ADMIN PERMISSIONS DOMAIN (4 tools) ────────────────────────────────────

@mcp.tool(
    name="admin_list_user_permissions",
    description=TOOL_POLICIES["admin_list_user_permissions"].description,
    tags=set(TOOL_POLICIES["admin_list_user_permissions"].tags),
)
async def admin_list_user_permissions(
    user_id: str,
    department_code: str,
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List permissions granted to a user in a department (Admin only)."""
    return await admin_permissions.admin_list_user_permissions(
        user_id=user_id,
        department_code=department_code,
        auth_token=auth_token,
    )


@mcp.tool(
    name="admin_grant_user_permission",
    description=TOOL_POLICIES["admin_grant_user_permission"].description,
    tags=set(TOOL_POLICIES["admin_grant_user_permission"].tags),
)
async def admin_grant_user_permission(
    user_id: str,
    department_code: str,
    permission_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    """Grant a permission to a user in a department (Admin only)."""
    return await admin_permissions.admin_grant_user_permission(
        user_id=user_id,
        department_code=department_code,
        permission_code=permission_code,
        auth_token=auth_token,
    )


@mcp.tool(
    name="admin_revoke_user_permission",
    description=TOOL_POLICIES["admin_revoke_user_permission"].description,
    tags=set(TOOL_POLICIES["admin_revoke_user_permission"].tags),
)
async def admin_revoke_user_permission(
    user_id: str,
    department_code: str,
    permission_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    """Revoke a permission from a user in a department (Admin only)."""
    return await admin_permissions.admin_revoke_user_permission(
        user_id=user_id,
        department_code=department_code,
        permission_code=permission_code,
        auth_token=auth_token,
    )


@mcp.tool(
    name="admin_apply_permission_preset",
    description=TOOL_POLICIES["admin_apply_permission_preset"].description,
    tags=set(TOOL_POLICIES["admin_apply_permission_preset"].tags),
)
async def admin_apply_permission_preset(
    user_id: str,
    department_code: str,
    preset: PermissionPreset,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    """Apply a standardized permission preset to a user in a department (Admin only)."""
    return await admin_permissions.admin_apply_permission_preset(
        user_id=user_id,
        department_code=department_code,
        preset=preset,
        auth_token=auth_token,
    )


# ── 8. ADMIN DOCUMENTS DOMAIN (5 tools) ──────────────────────────────────────

@mcp.tool(
    name="admin_update_document_department",
    description=TOOL_POLICIES["admin_update_document_department"].description,
    tags=set(TOOL_POLICIES["admin_update_document_department"].tags),
)
async def admin_update_document_department(
    document_id: str,
    department_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Reassign a document to a different department (Admin only)."""
    return await admin_documents.admin_update_document_department(
        document_id=document_id,
        department_code=department_code,
        auth_token=auth_token,
    )


@mcp.tool(
    name="admin_get_document_permissions",
    description=TOOL_POLICIES["admin_get_document_permissions"].description,
    tags=set(TOOL_POLICIES["admin_get_document_permissions"].tags),
)
async def admin_get_document_permissions(
    document_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Get document-level permissions and effective user access for a document (Admin only)."""
    return await admin_documents.admin_get_document_permissions(
        document_id=document_id,
        auth_token=auth_token,
    )


@mcp.tool(
    name="admin_grant_document_permission",
    description=TOOL_POLICIES["admin_grant_document_permission"].description,
    tags=set(TOOL_POLICIES["admin_grant_document_permission"].tags),
)
async def admin_grant_document_permission(
    document_id: str,
    user_id: str,
    permission_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    """Grant document-level permission to a user (Admin only)."""
    return await admin_documents.admin_grant_document_permission(
        document_id=document_id,
        user_id=user_id,
        permission_code=permission_code,
        auth_token=auth_token,
    )


@mcp.tool(
    name="admin_revoke_document_permission",
    description=TOOL_POLICIES["admin_revoke_document_permission"].description,
    tags=set(TOOL_POLICIES["admin_revoke_document_permission"].tags),
)
async def admin_revoke_document_permission(
    document_id: str,
    user_id: str,
    permission_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    """Revoke document-level permission from a user (Admin only)."""
    return await admin_documents.admin_revoke_document_permission(
        document_id=document_id,
        user_id=user_id,
        permission_code=permission_code,
        auth_token=auth_token,
    )


@mcp.tool(
    name="admin_set_document_permissions",
    description=TOOL_POLICIES["admin_set_document_permissions"].description,
    tags=set(TOOL_POLICIES["admin_set_document_permissions"].tags),
)
async def admin_set_document_permissions(
    document_id: str,
    user_id: str,
    permissions: list[str],
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    """Set exact list of document-level permissions for a user (Admin only)."""
    return await admin_documents.admin_set_document_permissions(
        document_id=document_id,
        user_id=user_id,
        permissions=permissions,
        auth_token=auth_token,
    )


# ── 9. ADMIN DEPARTMENTS DOMAIN (3 tools) ────────────────────────────────────

@mcp.tool(
    name="admin_list_departments",
    description=TOOL_POLICIES["admin_list_departments"].description,
    tags=set(TOOL_POLICIES["admin_list_departments"].tags),
)
async def admin_list_departments(
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List all departments including active and inactive (Admin only)."""
    return await admin_departments.admin_list_departments(auth_token=auth_token)


@mcp.tool(
    name="admin_create_department",
    description=TOOL_POLICIES["admin_create_department"].description,
    tags=set(TOOL_POLICIES["admin_create_department"].tags),
)
async def admin_create_department(
    code: str,
    name: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Create a new department (Admin only)."""
    return await admin_departments.admin_create_department(
        code=code,
        name=name,
        auth_token=auth_token,
    )


@mcp.tool(
    name="admin_update_department_status",
    description=TOOL_POLICIES["admin_update_department_status"].description,
    tags=set(TOOL_POLICIES["admin_update_department_status"].tags),
)
async def admin_update_department_status(
    code: str,
    is_active: bool,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Activate or deactivate a department (Admin only)."""
    return await admin_departments.admin_update_department_status(
        code=code,
        is_active=is_active,
        auth_token=auth_token,
    )


# ── 10. DIAGNOSTICS DOMAIN (1 tool) ──────────────────────────────────────────

@mcp.tool(
    name="run_diagnostics",
    description=TOOL_POLICIES["run_diagnostics"].description,
    tags=set(TOOL_POLICIES["run_diagnostics"].tags),
)
async def run_diagnostics(
    include_rag_smoketest: bool = False,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Run safe, non-destructive platform health checks."""
    return await diagnostics.run_diagnostics(
        include_rag_smoketest=include_rag_smoketest,
        auth_token=auth_token,
    )


# ── 11. TOOL DISCOVERY & SEARCH (1 tool) ─────────────────────────────────────

@mcp.tool(
    name="search_tools",
    description=TOOL_POLICIES["search_tools"].description,
    tags=set(TOOL_POLICIES["search_tools"].tags),
)
async def search_tools(
    query: str,
    domain: Optional[str] = None,
    limit: int = 5,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Search and discover available MCP tools using natural language or domain filtering."""
    return await discovery.search_tools(
        query=query,
        domain=domain,
        limit=limit,
        auth_token=auth_token,
    )


# ── Server Helpers ────────────────────────────────────────────────────────────

def get_mcp_server() -> FastMCP:
    """Return the FastMCP server instance."""
    return mcp


def get_mcp_asgi_app(transport: str = "streamable-http"):
    """Return the ASGI application for Streamable HTTP or SSE mounting."""
    return mcp.http_app(transport=transport)


if __name__ == "__main__":
    import uvicorn
    asgi_app = get_mcp_asgi_app(transport="streamable-http")
    uvicorn.run(asgi_app, host="127.0.0.1", port=8001)
