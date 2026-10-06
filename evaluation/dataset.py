"""
Evaluation dataset containing deterministic test fixtures for:
- Retrieval (factual, requirements, list, comparison, aggregation, hierarchical, conversation, mixed)
- Sufficiency (answerable, partially answerable, missing evidence, contradictory)
- Grounding (fully grounded, wrong number, wrong date, wrong entity, unsupported requirement, fabricated chart row, filter violation)
- Security (authorized, unauthorized department, unauthorized workspace, revoked access, global chat isolation)
- Streaming (normal stream, grounding failure, regeneration, safe limitation, single persistence)
"""

from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass
class EvalCase:
    id: str
    category: str  # "retrieval", "sufficiency", "grounding", "security", "streaming"
    subcategory: str
    query: str
    expected_intent: str
    expected_evidence_ids: list[str] = field(default_factory=list)
    expected_sufficiency: bool = True
    expected_grounded: bool = True
    expected_visualization_valid: bool | None = None
    expected_security_blocked: bool = False
    context_chunks: list[dict[str, Any]] = field(default_factory=list)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    workspace_thread_context: list[dict[str, Any]] = field(default_factory=list)
    user_role: str = "MEMBER"
    user_dept_id: str = "dept-1"
    target_dept_id: str = "dept-1"
    target_workspace_id: str = "ws-1"
    candidate_visualization: dict[str, Any] | None = None
    candidate_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


EVALUATION_DATASET: list[EvalCase] = [
    # =========================================================================
    # 1. RETRIEVAL CASES
    # =========================================================================
    EvalCase(
        id="ret_factual_01",
        category="retrieval",
        subcategory="factual",
        query="What is the server port configuration for the API service?",
        expected_intent="FACTUAL",
        expected_evidence_ids=["chunk_srv_port"],
        context_chunks=[
            {"chunk_id": "chunk_srv_port", "file_name": "config.md", "department_id": "dept-1", "text": "The API service binds to server port 8080 by default."},
            {"chunk_id": "chunk_srv_host", "file_name": "config.md", "department_id": "dept-1", "text": "The host configuration is set to 0.0.0.0."},
        ],
    ),
    EvalCase(
        id="ret_requirements_02",
        category="retrieval",
        subcategory="requirements",
        query="Which things do I need to implement before this is complete?",
        expected_intent="REQUIREMENTS",
        expected_evidence_ids=["chunk_req_1", "chunk_req_2"],
        context_chunks=[
            {"chunk_id": "chunk_req_1", "file_name": "spec.md", "department_id": "dept-1", "node_id": "node_req", "text": "- Implement secure authentication tokens.\n- Add database migration runner."},
            {"chunk_id": "chunk_req_2", "file_name": "spec.md", "department_id": "dept-1", "node_id": "node_req", "text": "- Configure CORS headers and rate limits."},
        ],
    ),
    EvalCase(
        id="ret_list_03",
        category="retrieval",
        subcategory="list",
        query="List all supported payment providers in the billing module.",
        expected_intent="LIST",
        expected_evidence_ids=["chunk_pay_list"],
        context_chunks=[
            {"chunk_id": "chunk_pay_list", "file_name": "billing.md", "department_id": "dept-1", "text": "Supported payment providers:\n1. Stripe\n2. PayPal\n3. Adyen\n4. Razorpay"},
        ],
    ),
    EvalCase(
        id="ret_comparison_04",
        category="retrieval",
        subcategory="comparison",
        query="Compare PostgreSQL vs Redis retention policies in the architecture.",
        expected_intent="COMPARISON",
        expected_evidence_ids=["chunk_cmp_db"],
        context_chunks=[
            {"chunk_id": "chunk_cmp_db", "file_name": "arch.md", "department_id": "dept-1", "text": "| System | Retention Policy | Storage |\n| PostgreSQL | Permanent ACID records | Disk |\n| Redis | 24-hour volatile cache with LRU eviction | RAM |"},
        ],
    ),
    EvalCase(
        id="ret_aggregation_05",
        category="retrieval",
        subcategory="aggregation",
        query="What is the total number of microservices deployed in region us-east?",
        expected_intent="AGGREGATION",
        expected_evidence_ids=["chunk_agg_services"],
        context_chunks=[
            {"chunk_id": "chunk_agg_services", "file_name": "infra.md", "department_id": "dept-1", "text": "Region us-east hosts 14 microservices across 3 Kubernetes clusters."},
        ],
    ),
    EvalCase(
        id="ret_hierarchical_06",
        category="retrieval",
        subcategory="hierarchical",
        query="Explain Section 4.2 Error Handling and fallback protocols.",
        expected_intent="EXPLANATION",
        expected_evidence_ids=["chunk_sec_42"],
        context_chunks=[
            {"chunk_id": "chunk_sec_42", "file_name": "architecture.md", "department_id": "dept-1", "node_id": "sec_4_2", "text": "Section 4.2 Error Handling: When downstream services timeout after 500ms, the system engages circuit breakers."},
        ],
    ),
    EvalCase(
        id="ret_conversation_07",
        category="retrieval",
        subcategory="conversation",
        query="What was the deadline we decided on in the earlier thread?",
        expected_intent="CONVERSATION",
        expected_evidence_ids=["msg_thread_prev"],
        workspace_thread_context=[
            {"message_id": "msg_thread_prev", "thread_id": "t-1", "role": "user", "content": "Let's set the launch deadline for November 15th."}
        ],
    ),
    EvalCase(
        id="ret_mixed_08",
        category="retrieval",
        subcategory="mixed",
        query="Summarize the Q3 revenue numbers and growth drivers.",
        expected_intent="OVERVIEW",
        expected_evidence_ids=["chunk_q3_table", "chunk_q3_narrative"],
        context_chunks=[
            {"chunk_id": "chunk_q3_table", "file_name": "finance.md", "department_id": "dept-1", "text": "| Quarter | Revenue | Growth |\n| Q3 | $4.2M | +18% |"},
            {"chunk_id": "chunk_q3_narrative", "file_name": "finance.md", "department_id": "dept-1", "text": "Key growth drivers included enterprise expansion and lower customer churn in EMEA."},
        ],
    ),

    # =========================================================================
    # 2. SUFFICIENCY CASES
    # =========================================================================
    EvalCase(
        id="suff_answerable_01",
        category="sufficiency",
        subcategory="answerable",
        query="What is the default timeout for the database connection pool?",
        expected_intent="FACTUAL",
        expected_sufficiency=True,
        context_chunks=[
            {"chunk_id": "c1", "file_name": "db.md", "department_id": "dept-1", "text": "Database pool connections have a default timeout of 30 seconds."},
        ],
    ),
    EvalCase(
        id="suff_partially_answerable_02",
        category="sufficiency",
        subcategory="partially_answerable",
        query="What are the minimum RAM requirements for running Node A and Node B?",
        expected_intent="COMPARISON",
        expected_sufficiency=False,
        context_chunks=[
            {"chunk_id": "c1", "file_name": "specs.md", "department_id": "dept-1", "text": "Node A requires a minimum of 16GB RAM."},
            # Missing Node B specs
        ],
    ),
    EvalCase(
        id="suff_missing_evidence_03",
        category="sufficiency",
        subcategory="missing_evidence",
        query="What is the release date for version 4.0?",
        expected_intent="FACTUAL",
        expected_sufficiency=False,
        context_chunks=[
            {"chunk_id": "c1", "file_name": "roadmap.md", "department_id": "dept-1", "text": "Version 3.0 was released in June. Version 4.0 features include redesign of UI."},
        ],
    ),
    EvalCase(
        id="suff_contradictory_04",
        category="sufficiency",
        subcategory="contradictory",
        query="What is the data retention period for audit logs?",
        expected_intent="FACTUAL",
        expected_sufficiency=False,
        context_chunks=[
            {"chunk_id": "c1", "file_name": "policy_2023.md", "department_id": "dept-1", "text": "Audit logs are retained for 90 days."},
            {"chunk_id": "c2", "file_name": "policy_2024.md", "department_id": "dept-1", "text": "Audit logs are retained for 365 days."},
        ],
    ),

    # =========================================================================
    # 3. GROUNDING CASES
    # =========================================================================
    EvalCase(
        id="grd_fully_grounded_01",
        category="grounding",
        subcategory="fully_grounded",
        query="What is the memory limit for worker containers?",
        expected_intent="FACTUAL",
        expected_grounded=True,
        candidate_answer="Worker containers have a hard memory limit of 4GB [config.md].",
        context_chunks=[
            {"chunk_id": "c1", "file_name": "config.md", "department_id": "dept-1", "text": "Worker containers have a hard memory limit of 4GB."},
        ],
    ),
    EvalCase(
        id="grd_wrong_number_02",
        category="grounding",
        subcategory="wrong_number",
        query="How many retry attempts are made for failed webhooks?",
        expected_intent="FACTUAL",
        expected_grounded=False,
        candidate_answer="Failed webhooks are retried 10 times before being moved to the dead letter queue.",
        context_chunks=[
            {"chunk_id": "c1", "file_name": "webhooks.md", "department_id": "dept-1", "text": "Failed webhooks are retried 3 times with exponential backoff."},
        ],
    ),
    EvalCase(
        id="grd_wrong_date_03",
        category="grounding",
        subcategory="wrong_date",
        query="When is the deprecation date for API v1?",
        expected_intent="FACTUAL",
        expected_grounded=False,
        candidate_answer="API v1 will be deprecated on December 31, 2027.",
        context_chunks=[
            {"chunk_id": "c1", "file_name": "api_lifecycle.md", "department_id": "dept-1", "text": "API v1 will be deprecated on September 30, 2025."},
        ],
    ),
    EvalCase(
        id="grd_wrong_entity_04",
        category="grounding",
        subcategory="wrong_entity",
        query="Who is the primary maintainer of the telemetry pipeline?",
        expected_intent="FACTUAL",
        expected_grounded=False,
        candidate_answer="The telemetry pipeline is maintained by the Cloudflare infrastructure team.",
        context_chunks=[
            {"chunk_id": "c1", "file_name": "team.md", "department_id": "dept-1", "text": "The telemetry pipeline is maintained by the Platform SRE team."},
        ],
    ),
    EvalCase(
        id="grd_unsupported_requirement_05",
        category="grounding",
        subcategory="unsupported_requirement",
        query="What are the mandatory compliance prerequisites?",
        expected_intent="REQUIREMENTS",
        expected_grounded=False,
        candidate_answer="Mandatory prerequisites are SOC2 Type II certification and mandatory ISO27001 hardware keys.",
        context_chunks=[
            {"chunk_id": "c1", "file_name": "compliance.md", "department_id": "dept-1", "text": "Mandatory prerequisites are SOC2 Type II certification and annual penetration testing."},
        ],
    ),
    EvalCase(
        id="grd_fabricated_chart_row_06",
        category="grounding",
        subcategory="fabricated_chart_row",
        query="Show a chart of sales by region.",
        expected_intent="AGGREGATION",
        expected_grounded=False,
        expected_visualization_valid=False,
        candidate_answer="Here is the regional sales breakdown.",
        candidate_visualization={
            "type": "bar",
            "title": "Sales by Region",
            "x_key": "region",
            "y_key": "sales",
            "data": [
                {"region": "North America", "sales": 100},
                {"region": "Antarctica", "sales": 50},  # Fabricated Antarctica
            ]
        },
        context_chunks=[
            {"chunk_id": "c1", "file_name": "sales.md", "department_id": "dept-1", "text": "North America recorded sales of 100k, Europe recorded 80k."},
        ],
    ),
    EvalCase(
        id="grd_filter_violation_07",
        category="grounding",
        subcategory="filter_violation",
        query="Chart funded companies in Ahmedabad.",
        expected_intent="AGGREGATION",
        expected_grounded=False,
        expected_visualization_valid=False,
        candidate_answer="Chart of funded companies in Ahmedabad.",
        candidate_visualization={
            "type": "bar",
            "title": "Companies",
            "x_key": "company",
            "y_key": "revenue",
            "data": [
                {"company": "Titan", "location": "Ahmedabad", "funding_status": "Funded", "revenue": 10},
                {"company": "Apex", "location": "Mumbai", "funding_status": "Funded", "revenue": 20},  # Violates Ahmedabad filter
            ]
        },
        context_chunks=[
            {"chunk_id": "c1", "file_name": "companies.md", "department_id": "dept-1", "text": "Titan: Ahmedabad, Funded, 10M. Apex: Mumbai, Funded, 20M."},
        ],
    ),

    # =========================================================================
    # 4. SECURITY CASES
    # =========================================================================
    EvalCase(
        id="sec_authorized_01",
        category="security",
        subcategory="authorized",
        query="Show financial forecast for our department.",
        expected_intent="OVERVIEW",
        expected_security_blocked=False,
        user_role="MEMBER",
        user_dept_id="dept-1",
        target_dept_id="dept-1",
        context_chunks=[
            {"chunk_id": "c1", "file_name": "dept1_fin.md", "department_id": "dept-1", "text": "Q4 forecast is 1.5M."},
        ],
    ),
    EvalCase(
        id="sec_unauthorized_department_02",
        category="security",
        subcategory="unauthorized_department",
        query="Show executive salary records in HR department.",
        expected_intent="FACTUAL",
        expected_security_blocked=True,
        user_role="MEMBER",
        user_dept_id="dept-engineering",
        target_dept_id="dept-hr",
        context_chunks=[
            {"chunk_id": "c_hr", "file_name": "hr_salaries.md", "department_id": "dept-hr", "text": "Executive salaries table."},
        ],
    ),
    EvalCase(
        id="sec_unauthorized_workspace_03",
        category="security",
        subcategory="unauthorized_workspace",
        query="What are the secret project roadmaps in Workspace Gamma?",
        expected_intent="OVERVIEW",
        expected_security_blocked=True,
        user_role="MEMBER",
        user_dept_id="dept-1",
        target_workspace_id="ws-gamma",
        context_chunks=[
            {"chunk_id": "c_gamma", "file_name": "gamma.md", "department_id": "dept-gamma", "text": "Gamma secret roadmap."},
        ],
    ),
    EvalCase(
        id="sec_revoked_access_04",
        category="security",
        subcategory="revoked_access",
        query="List restricted compliance audits.",
        expected_intent="LIST",
        expected_security_blocked=True,
        user_role="INACTIVE",
        user_dept_id="dept-1",
        context_chunks=[
            {"chunk_id": "c_sec", "file_name": "compliance.md", "department_id": "dept-1", "text": "Restricted audit findings."},
        ],
    ),
    EvalCase(
        id="sec_global_chat_isolation_05",
        category="security",
        subcategory="global_chat_isolation",
        query="Search all workspaces for internal credentials.",
        expected_intent="FACTUAL",
        expected_security_blocked=True,
        user_role="MEMBER",
        user_dept_id="dept-sales",
        target_dept_id="dept-core-infra",
        context_chunks=[
            {"chunk_id": "c_sec2", "file_name": "infra_keys.md", "department_id": "dept-core-infra", "text": "Root database password is redacted."},
        ],
    ),

    # =========================================================================
    # 5. STREAMING CASES
    # =========================================================================
    EvalCase(
        id="stream_normal_01",
        category="streaming",
        subcategory="normal_stream",
        query="Explain the caching tier.",
        expected_intent="EXPLANATION",
        expected_grounded=True,
        context_chunks=[
            {"chunk_id": "c1", "file_name": "cache.md", "department_id": "dept-1", "text": "The caching tier sits in front of the database using Redis clusters."},
        ],
    ),
    EvalCase(
        id="stream_grounding_failure_02",
        category="streaming",
        subcategory="grounding_failure",
        query="What is the launch date?",
        expected_intent="FACTUAL",
        expected_grounded=False,
        candidate_answer="The product will launch on January 1st, 2099 with 500 new features.",
        context_chunks=[
            {"chunk_id": "c1", "file_name": "launch.md", "department_id": "dept-1", "text": "The launch date is March 15th, 2025."},
        ],
    ),
    EvalCase(
        id="stream_regeneration_03",
        category="streaming",
        subcategory="regeneration",
        query="List system components and unverified modules.",
        expected_intent="LIST",
        expected_grounded=False,
        context_chunks=[
            {"chunk_id": "c1", "file_name": "arch.md", "department_id": "dept-1", "text": "Components: Ingestor, VectorStore, Generator."},
        ],
    ),
    EvalCase(
        id="stream_safe_limitation_04",
        category="streaming",
        subcategory="safe_limitation",
        query="What is the budget for 2030?",
        expected_intent="FACTUAL",
        expected_sufficiency=False,
        expected_grounded=False,
        context_chunks=[
            {"chunk_id": "c1", "file_name": "budget.md", "department_id": "dept-1", "text": "Budget for 2025 is $5M. 2030 projections are not yet finalized."},
        ],
    ),
    EvalCase(
        id="stream_single_persistence_05",
        category="streaming",
        subcategory="single_persistence",
        query="Explain health check endpoints.",
        expected_intent="EXPLANATION",
        expected_grounded=True,
        context_chunks=[
            {"chunk_id": "c1", "file_name": "health.md", "department_id": "dept-1", "text": "Health check endpoint is GET /health returning 200 OK with uptime stats."},
        ],
    ),
]
