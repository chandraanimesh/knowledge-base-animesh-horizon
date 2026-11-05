"""
Pydantic Schemas for Workflow Triggers, Webhooks, Outbox Events, and Schedules.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
import uuid
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from rag_app.schemas.workflow import BudgetLimits, WorkflowRunResponse


class TriggerSource(str, Enum):
    API = "api"
    WEBHOOK = "webhook"
    TWENTY_CRM = "twenty_crm"
    SCHEDULE = "schedule"


class OutboxStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DISPATCHED = "DISPATCHED"
    FAILED = "FAILED"


class MisfirePolicy(str, Enum):
    RUN_ONCE = "RUN_ONCE"
    SKIP = "SKIP"


class WorkflowTriggerRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Trigger payload variables passed to workflow DAG (e.g. query, record_id)",
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        min_length=4,
        max_length=255,
        description="Client-supplied idempotency key for replay deduplication",
    )
    budget_limits: Optional[BudgetLimits] = Field(
        default=None,
        description="Optional override for execution token and provider call budgets",
    )


class WorkflowTriggerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    workflow_id: UUID
    workflow_version_id: UUID
    workflow_run_id: Optional[UUID] = None
    status: str
    is_replay: bool = False
    created_at: datetime
    dispatched_at: Optional[datetime] = None
    run_details: Optional[WorkflowRunResponse] = None


class WorkflowScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cron: Optional[str] = Field(
        default=None,
        description="Standard 5-field cron string (minute hour day_of_month month day_of_week)",
    )
    interval_seconds: Optional[int] = Field(
        default=None,
        ge=60,
        le=2592000,
        description="Interval schedule in seconds (minimum 60s)",
    )
    timezone: str = Field(
        default="UTC",
        description="IANA timezone (e.g. 'UTC', 'America/New_York', 'Asia/Kolkata')",
    )
    misfire_policy: MisfirePolicy = Field(
        default=MisfirePolicy.RUN_ONCE,
        description="Recovery behavior when a schedule window was missed during downtime",
    )


class WorkflowScheduleStateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    workflow_id: UUID
    last_scheduled_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    last_window_key: Optional[str] = None
    misfire_count: int = 0
    status: str = "IDLE"
    updated_at: datetime


class WorkflowWebhookResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    event_id: str
    workflow_run_id: Optional[str] = None
    message: str

