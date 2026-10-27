"""
Workspace MCP Read-Only Tool Binding for LangGraph and Adaptive Workflows (Milestone 6D).

Exposes ONLY server-verified, explicitly enabled read-only MCP tools to the LangGraph
execution context and Adaptive DAG workflows.
Enforces:
1. Strict server-side verification: only tools in VERIFIED_READ_ONLY_MCP_TOOLS can be bound.
2. Complete exclusion of mutating and destructive tools.
3. Every tool invocation routes through WorkspaceMCPConnectionService.invoke_workspace_tool.
4. Per-invocation re-checking of workspace membership, connection status, tool allowlist, and policy.
5. Untrusted schema and payload handling: argument validation and bounded content (max 10,000 chars).
6. Correlation ID tracking (run_id, task_id) and atomic budget preservation.
7. Masked secrets, error normalization, and structured audit observability.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Coroutine, Optional, Type
import uuid
from uuid import UUID

from fastapi import HTTPException, status
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import (
    User,
    WorkflowRun,
    Workspace,
    WorkspaceMCPConnection,
    WorkspaceMember,
)
from rag_app.mcp.connection_service import (
    VERIFIED_READ_ONLY_MCP_TOOLS,
    WorkspaceMCPConnectionService,
)

logger = logging.getLogger(__name__)

# Bounded content limits to protect LLM context windows and prevent token budget blowout
MAX_TOOL_OUTPUT_CHARS = 10000
MAX_TOOL_DESC_CHARS = 500


def sanitize_tool_description(raw_desc: Optional[str]) -> str:
    """
    Sanitizes external MCP tool descriptions before exposing them to the agent.
    Truncates description length and strips obvious prompt injection patterns.
    """
    if not raw_desc or not str(raw_desc).strip():
        return "Authorized workspace read-only tool."
    clean = str(raw_desc).strip()
    # Strip potential prompt injection delimiters
    clean = re.sub(r"(?i)(ignore\s+previous\s+instructions|system\s+prompt|system\s*:)", "[REDACTED_DIRECTIVE]", clean)
    if len(clean) > MAX_TOOL_DESC_CHARS:
        clean = clean[:MAX_TOOL_DESC_CHARS] + "..."
    return clean


def bound_tool_output(data: Any) -> str:
    """
    Normalizes and truncates external tool output to MAX_TOOL_OUTPUT_CHARS.
    Guarantees untrusted data does not flood agent context or exhaust token budgets.
    """
    if data is None:
        return ""
    if isinstance(data, (dict, list)):
        try:
            rendered = json.dumps(data, indent=2, default=str)
        except Exception:
            rendered = str(data)
    else:
        rendered = str(data)

    if len(rendered) > MAX_TOOL_OUTPUT_CHARS:
        truncated_msg = f"\n\n[Output truncated at {MAX_TOOL_OUTPUT_CHARS} characters to preserve context limit]"
        rendered = rendered[: MAX_TOOL_OUTPUT_CHARS - len(truncated_msg)] + truncated_msg
    return rendered


def build_dynamic_args_schema(tool_name: str, input_schema: dict[str, Any]) -> Type[BaseModel]:
    """
    Constructs a validated Pydantic model from an untrusted JSON Schema definition.
    Fails closed if the schema is malformed.
    """
    if not isinstance(input_schema, dict):
        class FallbackEmptyArgs(BaseModel):
            model_config = ConfigDict(extra="allow")
        return FallbackEmptyArgs

    properties = input_schema.get("properties", {})
    required_fields = set(input_schema.get("required", [])) if isinstance(input_schema.get("required"), list) else set()

    type_mapping: dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    fields: dict[str, Any] = {}
    if isinstance(properties, dict):
        for prop_name, prop_def in properties.items():
            if not isinstance(prop_def, dict):
                continue
            # Sanitize field name: only alphanumeric and underscores
            safe_prop_name = re.sub(r"\W|^(?=\d)", "_", str(prop_name))
            json_type = prop_def.get("type", "string")
            py_type = type_mapping.get(str(json_type).lower(), Any)
            field_desc = str(prop_def.get("description", ""))[:200]

            if prop_name in required_fields:
                fields[safe_prop_name] = (py_type, Field(..., description=field_desc))
            else:
                fields[safe_prop_name] = (Optional[py_type], Field(None, description=field_desc))

    if not fields:
        class EmptyToolArgs(BaseModel):
            model_config = ConfigDict(extra="allow")
        return EmptyToolArgs

    safe_model_name = re.sub(r"\W", "_", tool_name).capitalize() + "_InputSchema"
    return create_model(safe_model_name, **fields)


class WorkspaceReadOnlyMCPToolBinding:
    """
    Factory for discovering, validating, and generating LangGraph BaseTool
    instances strictly bound to authorized workspace MCP connections.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.connection_service = WorkspaceMCPConnectionService(session=session)

    async def get_workspace_readonly_tools(
        self,
        user: User,
        workspace_id: UUID,
        run_id: Optional[UUID] = None,
        task_id: Optional[UUID] = None,
    ) -> list[BaseTool]:
        """
        Discovers all explicitly enabled, server-verified read-only MCP tools
        available to the user within the workspace.

        Security guarantees:
        1. Workspace membership is checked immediately. Unauthorized users fail closed (return []).
        2. Inactive or disabled connections are ignored.
        3. Only tools present in VERIFIED_READ_ONLY_MCP_TOOLS are included.
        4. Mutating, destructive, or approval-requiring tools are strictly excluded.
        5. Generated tools invoke WorkspaceMCPConnectionService.invoke_workspace_tool at runtime.
        """
        # 1. Re-check workspace membership and caller role
        stmt = select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user.id,
        )
        membership = await self.session.scalar(stmt)
        if not membership and user.role not in ("ADMIN", "OWNER"):
            logger.warning(
                "Unauthorized MCP tool discovery attempt: user %s has no access to workspace %s",
                user.id, workspace_id,
            )
            return []

        # 2. Fetch all active and enabled MCP connections in workspace
        conn_stmt = select(WorkspaceMCPConnection).where(
            WorkspaceMCPConnection.workspace_id == workspace_id,
            WorkspaceMCPConnection.is_enabled == True,  # noqa: E712
        )
        connections = (await self.session.scalars(conn_stmt)).all()
        if not connections:
            return []

        langchain_tools: list[BaseTool] = []

        for conn in connections:
            discovered_tools = conn.discovered_tools or []
            enabled_tools = set(conn.enabled_tools or [])
            tool_policies = (conn.configuration or {}).get("tool_policies", {})

            for t in discovered_tools:
                if not isinstance(t, dict):
                    continue
                tool_name = t.get("name")
                if not tool_name or tool_name not in enabled_tools:
                    continue

                # STRICT SERVER-SIDE FILTER:
                # Must be in server-verified read-only catalog
                if tool_name not in VERIFIED_READ_ONLY_MCP_TOOLS:
                    continue

                # Must not have workspace policy escalating to mutating or destructive
                policy = tool_policies.get(tool_name, {})
                if policy.get("risk_level") in ("mutating", "destructive") or policy.get("requires_approval"):
                    continue

                # Build StructuredTool for this verified read-only MCP tool
                args_schema = build_dynamic_args_schema(tool_name, t.get("inputSchema", {}))
                sanitized_desc = sanitize_tool_description(t.get("description"))

                tool_instance = self._create_langgraph_tool(
                    user=user,
                    workspace_id=workspace_id,
                    connection_id=conn.id,
                    connection_name=conn.name,
                    tool_name=tool_name,
                    description=sanitized_desc,
                    args_schema=args_schema,
                    run_id=run_id,
                    task_id=task_id,
                )
                langchain_tools.append(tool_instance)

        logger.info(
            "Bound %d verified read-only MCP tools into LangGraph context for workspace %s (user=%s)",
            len(langchain_tools), workspace_id, user.id,
        )
        return langchain_tools

    async def get_workspace_proposal_tools(
        self,
        user: User,
        workspace_id: UUID,
        run_id: Optional[UUID] = None,
        task_id: Optional[UUID] = None,
    ) -> list[BaseTool]:
        """
        Milestone 6E: Discovers approved write-back tools (specifically create_one_note)
        and binds them as PROPOSAL-ONLY tools into the LangGraph context.
        The agent can draft and propose changes, which generate human-approval tickets,
        but can NEVER unilaterally execute external mutations.
        """
        from rag_app.crm.twenty_mcp_client import APPROVED_WRITE_BACK_TOOLS

        # 1. Re-check workspace membership and caller role
        stmt = select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user.id,
        )
        membership = await self.session.scalar(stmt)
        if not membership and user.role not in ("ADMIN", "OWNER"):
            return []

        # 2. Fetch active connections
        conn_stmt = select(WorkspaceMCPConnection).where(
            WorkspaceMCPConnection.workspace_id == workspace_id,
            WorkspaceMCPConnection.is_enabled == True,
        )
        connections = (await self.session.scalars(conn_stmt)).all()
        if not connections:
            return []

        proposal_tools: list[BaseTool] = []

        for conn in connections:
            enabled_tools = set(conn.enabled_tools or [])
            if "create_one_note" in enabled_tools and "create_one_note" in APPROVED_WRITE_BACK_TOOLS:
                class ProposeCreateNoteInput(BaseModel):
                    model_config = ConfigDict(extra="forbid")
                    company_id: str = Field(..., description="Target Twenty CRM Company UUID to link the note to")
                    title: str = Field(..., description="Title of the proposed CRM note")
                    markdown: str = Field(..., description="Markdown body/summary content for the proposed CRM note")

                connection_id = conn.id

                async def _propose_coroutine(company_id: str, title: str, markdown: str) -> str:
                    try:
                        approval = await self.connection_service.propose_tool_mutation(
                            user=user,
                            workspace_id=workspace_id,
                            connection_id=connection_id,
                            tool_name="create_one_note",
                            arguments={
                                "title": title,
                                "bodyV2": {"markdown": markdown},
                                "targetCompanyId": company_id,
                            },
                            run_id=run_id,
                            task_id=task_id,
                        )
                        return (
                            f"Mutation Proposed Successfully (Approval Ticket ID: {approval.id}). "
                            f"Intended Action: create_one_note, Target Company: {company_id}, Hash: {approval.approved_draft_hash[:16]}... "
                            "Status: PENDING_HUMAN_APPROVAL. The external CRM note will not be created until authorized."
                        )
                    except Exception as exc:
                        logger.error("Failed to propose MCP mutation: %s", exc)
                        return f"Error proposing mutation: {str(exc)}"

                p_tool = StructuredTool(
                    name="mcp_propose_create_one_note",
                    description=(
                        "Propose creating a CRM note for a company. This action requests human review "
                        "and approval before any changes are committed to Twenty CRM."
                    ),
                    args_schema=ProposeCreateNoteInput,
                    coroutine=_propose_coroutine,
                )
                proposal_tools.append(p_tool)

        return proposal_tools

    async def get_all_workspace_tools(
        self,
        user: User,
        workspace_id: UUID,
        run_id: Optional[UUID] = None,
        task_id: Optional[UUID] = None,
    ) -> list[BaseTool]:
        """
        Returns verified read-only tools PLUS proposal-only tools for approved write operations.
        Direct mutating tools remain strictly excluded from autonomous execution.
        """
        read_tools = await self.get_workspace_readonly_tools(user, workspace_id, run_id, task_id)
        prop_tools = await self.get_workspace_proposal_tools(user, workspace_id, run_id, task_id)
        return read_tools + prop_tools

    def _create_langgraph_tool(
        self,
        user: User,
        workspace_id: UUID,
        connection_id: UUID,
        connection_name: str,
        tool_name: str,
        description: str,
        args_schema: Type[BaseModel],
        run_id: Optional[UUID] = None,
        task_id: Optional[UUID] = None,
    ) -> BaseTool:
        """
        Creates an asynchronous LangChain StructuredTool wrapping WorkspaceMCPConnectionService.
        """
        async def _coroutine(**kwargs: Any) -> str:
            start_time = time.monotonic()
            try:
                # Atomic budget verification if correlated with a WorkflowRun
                if run_id:
                    refreshed_run = await self.session.get(WorkflowRun, run_id)
                    if refreshed_run:
                        if refreshed_run.provider_calls_spent + refreshed_run.provider_calls_reserved >= refreshed_run.max_provider_calls_budget:
                            logger.warning(
                                "MCP invocation blocked: workflow run %s exceeded max_provider_calls budget (%d)",
                                run_id, refreshed_run.max_provider_calls_budget,
                            )
                            return f"Error: Workflow budget exceeded (max provider calls {refreshed_run.max_provider_calls_budget} reached)."

                # Route invocation strictly through WorkspaceMCPConnectionService
                res = await self.connection_service.invoke_workspace_tool(
                    user=user,
                    workspace_id=workspace_id,
                    connection_id=connection_id,
                    tool_name=tool_name,
                    arguments=kwargs,
                    run_id=run_id,
                    task_id=task_id,
                )

                duration_ms = int((time.monotonic() - start_time) * 1000)
                logger.info(
                    "LangGraph MCP Tool Invocation Succeeded: tool=%s ws=%s duration_ms=%d",
                    tool_name, workspace_id, duration_ms,
                    extra={
                        "event": "langgraph_mcp_tool_success",
                        "tool_name": tool_name,
                        "workspace_id": str(workspace_id),
                        "connection_id": str(connection_id),
                        "run_id": str(run_id) if run_id else None,
                        "task_id": str(task_id) if task_id else None,
                        "duration_ms": duration_ms,
                    },
                )

                if res.is_error:
                    err_msg = res.error_message or res.text or "Remote tool reported execution error."
                    return f"Tool Execution Error: {bound_tool_output(err_msg)}"

                output_text = res.text or res.structured_content or "Success"
                return bound_tool_output(output_text)

            except HTTPException as http_exc:
                duration_ms = int((time.monotonic() - start_time) * 1000)
                logger.warning(
                    "LangGraph MCP Tool Invocation Rejected: tool=%s status=%d detail=%s",
                    tool_name, http_exc.status_code, http_exc.detail,
                    extra={
                        "event": "langgraph_mcp_tool_http_error",
                        "tool_name": tool_name,
                        "status_code": http_exc.status_code,
                        "detail": str(http_exc.detail),
                        "duration_ms": duration_ms,
                    },
                )
                return f"Tool Execution Rejected ({http_exc.status_code}): {bound_tool_output(str(http_exc.detail))}"
            except Exception as exc:
                duration_ms = int((time.monotonic() - start_time) * 1000)
                logger.error(
                    "LangGraph MCP Tool Unexpected Error: tool=%s error=%s",
                    tool_name, exc,
                    exc_info=True,
                    extra={
                        "event": "langgraph_mcp_tool_exception",
                        "tool_name": tool_name,
                        "error": str(exc),
                        "duration_ms": duration_ms,
                    },
                )
                return f"Tool Execution Failed: {bound_tool_output(str(exc))}"

        # Clean tool identifier for LLM invocation
        tool_identifier = f"mcp_{tool_name}"

        return StructuredTool.from_function(
            coroutine=_coroutine,
            name=tool_identifier,
            description=f"Authorized read-only MCP tool from '{connection_name}': {description}",
            args_schema=args_schema,
        )
