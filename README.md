# Enterprise Adaptive RAG, Agentic Orchestration & Research Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.0-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16_with_pgvector-336791.svg?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![MCP](https://img.shields.io/badge/Model_Context_Protocol-FastMCP_4+-8A2BE2.svg)](https://modelcontextprotocol.io/)
[![Package Manager](https://img.shields.io/badge/uv-Astral-FF69B4.svg)](https://astral.sh/uv)

An enterprise-grade, extensible **Retrieval-Augmented Generation (RAG) and Multi-Agent Orchestration Platform**. It pairs **Hybrid Multi-Strategy Retrieval** (Dense Vector + BM25 + RRF + Cross-Encoder) and **Hierarchical Section-Aware Context Compilation (OpenViking)** with an **Adaptive DAG Task Orchestrator**, **Human-in-the-Loop Cryptographic Approval Gates**, **Model Context Protocol (MCP) Hub & Spoke Architecture**, **Signed Inbound Webhooks & Event Automation (Twenty CRM & n8n)**, **Deep Web Research (TinyFish & Firecrawl)**, and **Multi-Format Publication-Ready Artifact Generation (PDF, DOCX, CSV, JSON, Markdown)**.

---

## Table of Contents

- [1. System Requirements & Architecture](#1-system-requirements--architecture)
- [2. 1-Click Docker Compose Deployment (Recommended)](#2-1-click-docker-compose-deployment-recommended)
- [3. Quickstart on Windows (Local Dev)](#3-quickstart-on-windows-local-dev)
- [4. Quickstart on macOS (Local Dev)](#4-quickstart-on-macos-local-dev)
- [5. Complete Environment Configuration (`.env`)](#5-complete-environment-configuration-env)
- [5. Model Context Protocol (MCP) Guide](#5-model-context-protocol-mcp-guide)
  - [Connecting Cursor IDE](#1-cursor-ide-setup)
  - [Connecting ChatGPT](#2-chatgpt-web--desktop-setup)
  - [Connecting Claude Desktop](#3-claude-desktop-setup)
  - [Outbound MCP (Twenty CRM)](#4-outbound-mcp-twenty-crm)
- [6. Inbound Webhooks & n8n / Twenty CRM Automation](#6-inbound-webhooks--n8n--twenty-crm-automation)
  - [Cryptographic Verification Contract](#webhook-cryptographic-contract)
  - [Configuring n8n Webhook Workflow](#configuring-n8n-workflow)
  - [Testing Webhooks Locally via Terminal](#testing-webhooks-locally)
- [7. Web Search & Deep Research Guide (TinyFish & Firecrawl)](#7-web-search--deep-research-guide)
- [8. Human-in-the-Loop (HITL) Mutation Approval & Two-Step Write Recovery](#8-human-in-the-loop-hitl-approval--write-recovery)
- [9. Testing & Quality Verification](#9-testing--quality-verification)
- [10. Troubleshooting & Common Issues (macOS & Windows)](#10-troubleshooting--common-issues)
- [11. Repository Structure](#11-repository-structure)

---

## 1. System Requirements & Architecture

### Prerequisites Matrix

| Component | Minimum Version | Recommended Version | Purpose / Notes |
| :--- | :--- | :--- | :--- |
| **Operating System** | Windows 10/11 or macOS 13+ (Ventura/Sonoma/Sequoia) | Windows 11 / macOS 14+ | Full cross-platform support (Intel & Apple Silicon M1/M2/M3) |
| **Python** | `>= 3.11` | `3.12+` | Managed strictly via `uv` for reproducible builds and fast dependency resolution |
| **Node.js** | `>= 18.x` | `20.x` or `22.x LTS` | Powers React 19, TypeScript, and Vite 8 frontend |
| **PostgreSQL** | `>= 15.x` | `16.x` | **Mandatory:** `pgvector` extension enabled (`v0.5.0+`) |
| **Package Managers**| `uv` + `npm` | Latest `uv` + `npm` | `uv` manages Python lockfile (`uv.lock`); `npm` manages frontend |
| **Docker** *(Optional)* | Docker Desktop | Docker Desktop / Docker Engine | Recommended for running PostgreSQL + `pgvector` with zero local configuration |

### System Architecture Overview

```mermaid
flowchart TD
    UserQuery([User Prompt / Workflow Goal]) --> Router{Intent & Task Classifier}

    Router -->|Conversational / Memory| DirectLLM[Direct LLM Handler + Episodic Memory]
    Router -->|Document / Enterprise RAG| HybridRetrieval[Multi-Strategy Hybrid Retrieval]
    Router -->|Complex / Multi-Step Goal| AdaptiveDAG[Adaptive DAG Task Planner]
    Router -->|Inbound Webhook Event| WebhookTrigger[Signed Webhook Receiver]

    subgraph HybridEngine [Hybrid Retrieval Pipeline]
        HybridRetrieval --> VectorSearch[Dense Vector HNSW pgvector 384-dim]
        HybridRetrieval --> BM25[BM25 / ts_rank_cd Lexical Search]
        HybridRetrieval --> HierarchicalTree[OpenViking Tree L0/L1/L2 Summaries]
    end

    VectorSearch & BM25 & HierarchicalTree --> RRF[Reciprocal Rank Fusion RRF]
    RRF --> CrossEncoder[CrossEncoder Re-ranker ms-marco-MiniLM]
    CrossEncoder --> SufficiencyGate{Sufficiency Evaluator}

    SufficiencyGate -->|Sufficient| ContextCompiler[Hierarchical Context Compiler]
    SufficiencyGate -->|Insufficient / Web Trigger| WebResearch[TinyFish / Firecrawl Deep Web Search]
    WebResearch --> ContextCompiler

    ContextCompiler --> GroundedGen[Grounded LLM Generation Service]
    GroundedGen --> ProvenanceEngine[Citation & Grounding Verification]

    subgraph AgentOrchestration [Adaptive DAG Execution Engine]
        AdaptiveDAG --> DAGValidator[DAG Validator & Budget Guardrails]
        DAGValidator --> TaskLeasing[Task Fencing & Safe Worker Leasing]
        TaskLeasing --> ReadTools[Read-Only Knowledge & MCP Query Tools]
        TaskLeasing --> MutationProposal[HITL Mutation Proposal Tool]
        MutationProposal --> ApprovalTicket{Single-Use Approval Gate}
        ApprovalTicket -->|Rejected| Terminate([Aborted / Zero Writes])
        ApprovalTicket -->|Approved| TwoStepWrite[Two-Step Idempotent Write Recovery]
        TwoStepWrite --> ExternalSvc[(Twenty CRM / Remote MCP Server)]
    end

    WebhookTrigger --> AdaptiveDAG
    GroundedGen & AgentOrchestration --> ArtifactEngine[Multi-Format Artifact Renderer PDF/DOCX/CSV/JSON]
    ArtifactEngine --> ProtectedStorage[(storage/artifacts Directory)]
    ProvenanceEngine --> FinalOutput([Verified Interactive Chat & Downloadable Reports])
```

---

## 2. 1-Click Docker Compose Deployment (Recommended)

The entire platform (PostgreSQL `pgvector`, FastAPI backend, and React 19 / Vite production frontend) can be spun up in a single command with zero manual environment configuration.

### Step 1: Configure Environment Variables
Copy the `.env.example` file to `.env` and fill in your API keys (e.g. `GROQ_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `TINYFISH_API_KEY`):

```bash
# Windows PowerShell
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

### Step 2: Build and Start All Services
```bash
docker compose up -d --build
```

### Step 3: Access the Applications

| Component | URL | Notes |
| :--- | :--- | :--- |
| **Frontend Web App** | `http://localhost:8443` or `http://localhost:5173` | React 19 + Vite SPA served via Nginx reverse proxy |
| **Backend REST API** | `http://localhost:8000` | FastAPI ASGI server |
| **Interactive API Docs**| `http://localhost:8000/docs` | Swagger UI documentation |
| **PostgreSQL pgvector** | `localhost:5433` | Database: `ragdb`, User: `rag`, Pass: `rag` |

Default administrative credentials seeded on first startup:
* **Email:** `admin@enterprise-rag.internal`
* **Password:** `AdminSecretPass123!`

To view logs or shut down:
```bash
# View live streaming logs across all services
docker compose logs -f

# Stop all containers
docker compose down
```

---

## 3. Quickstart on Windows (Local Dev)

Follow these steps on a clean Windows machine using **PowerShell**:

### Step 1: Install Required Tools
Open **PowerShell as Administrator** and install `uv`, `Node.js LTS`, and `Git`:

```powershell
# 1. Install uv (Fast Python Package Manager)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install Node.js LTS and Git using winget
winget install OpenJS.NodeJS.LTS
winget install Git.Git

# 3. (Optional but Recommended) Install Docker Desktop
winget install Docker.DockerDesktop
```

> [!NOTE]
> Close and reopen your PowerShell window after running `winget` so your system `PATH` updates.

### Step 2: Start PostgreSQL with pgvector

**Method 1: Using Docker (Fastest & Recommended)**
```powershell
docker run -d `
  --name rag-postgres `
  -p 5432:5432 `
  -e POSTGRES_USER=rag `
  -e POSTGRES_PASSWORD=rag `
  -e POSTGRES_DB=ragdb `
  pgvector/pgvector:pg16
```

**Method 2: Using Native Windows PostgreSQL**
1. Install PostgreSQL 16 from [EnterpriseDB](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads).
2. Download precompiled `pgvector` binaries for PostgreSQL 16 from [pgvector releases](https://github.com/pgvector/pgvector/releases).
3. Copy `vector.dll` into `C:\Program Files\PostgreSQL\16\lib` and extension control/sql files into `C:\Program Files\PostgreSQL\16\share\extension`.
4. Open `psql` and execute:
   ```sql
   CREATE DATABASE ragdb;
   \c ragdb
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

### Step 3: Clone Project & Set Up Backend

```powershell
# 1. Clone repository and navigate to root
git clone <repo-url>
cd ragdemo1

# 2. Copy the environment template
copy .env.example .env

# 3. Synchronize Python virtual environment and dependencies via uv
uv sync

# 4. Run database migrations
uv run alembic upgrade head

# 5. Seed default administrator and canonical departments
uv run python scripts/seed_admin.py
```

### Step 4: Start Backend Server
```powershell
uv run uvicorn rag_app.main:app --reload --port 8000
```
*Backend API and Swagger Documentation will be live at `http://localhost:8000/docs`.*

### Step 5: Start Frontend Application
Open a **new PowerShell terminal window**:

```powershell
cd ragdemo1\frontend

# Install frontend dependencies
npm install

# Start Vite development server
npm run dev
```
*The frontend is accessible at `http://localhost:8443`.*

---

## 4. Quickstart on macOS (Local Dev)

Follow these steps on macOS (compatible with both **Apple Silicon M1/M2/M3/M4** and **Intel** Macs):

### Step 1: Install Required Tools
Open **Terminal** and install `Homebrew`, `uv`, `Node.js`, and `Git`:

```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install uv, Node.js LTS, and Git
brew install uv node git
```

### Step 2: Start PostgreSQL with pgvector

**Method 1: Using Docker Desktop (Recommended)**
```bash
docker run -d \
  --name rag-postgres \
  -p 5432:5432 \
  -e POSTGRES_USER=rag \
  -e POSTGRES_PASSWORD=rag \
  -e POSTGRES_DB=ragdb \
  pgvector/pgvector:pg16
```

**Method 2: Using Native Homebrew**
```bash
brew install postgresql@16 pgvector
brew services start postgresql@16

# Create database and activate pgvector extension
createdb ragdb
psql -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Step 3: Clone Project & Set Up Backend

```bash
# 1. Clone repository and navigate to root
git clone <repo-url>
cd ragdemo1

# 2. Create local .env from template
cp .env.example .env

# 3. Install virtualenv & all dependencies with uv
uv sync

# 4. Run database migrations
uv run alembic upgrade head

# 5. Seed default administrator and canonical departments
uv run python scripts/seed_admin.py
```

### Step 4: Start Backend Server
```bash
uv run uvicorn rag_app.main:app --reload --port 8000
```

### Step 5: Start Frontend Application
Open a **new Terminal window**:

```bash
cd ragdemo1/frontend

# Install frontend packages
npm install

# Start Vite development server
npm run dev
```
*Access the React application at `http://localhost:8443`.*

---

## 4. Complete Environment Configuration (`.env`)

Configure your `.env` file based on your requirements. Below is the full annotated configuration:

```ini
# ==============================================================================
# 1. CORE APPLICATION & DATABASE SETTINGS
# ==============================================================================
APP_NAME=production-rag-backend
ENV=dev
LOG_LEVEL=INFO

# PostgreSQL connection string with asyncpg driver and pgvector enabled
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/ragdb

# ==============================================================================
# 2. EMBEDDING PROVIDER CONFIGURATION
# ==============================================================================
# Options: 'sentence_transformers' (Local/Free) | 'openai' | 'fake'
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

# ==============================================================================
# 3. LLM INFERENCE PROVIDERS
# ==============================================================================
# Options: 'groq' (Recommended for fast inference) | 'openai' | 'gemini' | 'fake'
LLM_PROVIDER=groq

# Groq Configuration (Free high-speed API keys at console.groq.com)
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b

# OpenAI Configuration (Optional)
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o-mini

# Google Gemini Configuration (Optional)
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-1.5-flash

# ==============================================================================
# 4. WEB SEARCH & DEEP RESEARCH (TinyFish & Firecrawl)
# ==============================================================================
ENABLE_WEB_RESEARCH=true

# Choose provider: 'tinyfish' | 'firecrawl' | 'fake'
WEB_SEARCH_PROVIDER=tinyfish

# TinyFish Search & Fetch APIs
TINYFISH_API_KEY=your_tinyfish_api_key_here
TINYFISH_SEARCH_URL=https://api.search.tinyfish.ai
TINYFISH_FETCH_URL=https://api.fetch.tinyfish.ai

# Firecrawl Configuration (Alternative)
FIRECRAWL_API_KEY=
FIRECRAWL_BASE_URL=https://api.firecrawl.dev

# Web Research Guardrails & Budget Bounds
WEB_RESEARCH_TARGET_PAGES=2
WEB_RESEARCH_MAX_SCRAPES=4
WEB_RESEARCH_MAX_CANDIDATES_PER_DOMAIN=1

# ==============================================================================
# 5. SECURITY, ENCRYPTION & AUTHENTICATION
# ==============================================================================
# Mandatory 32+ char secret for HS256 JWT access tokens
JWT_SECRET_KEY=super_secure_jwt_secret_key_change_in_production_min_32_chars
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# Dedicated Fernet key for workspace credential encryption (MultiFernet rotation support)
APP_ENCRYPTION_KEY=

# Default Seed Administrator Account
ADMIN_EMAIL=admin@local.dev
ADMIN_PASSWORD=Admin@12345

# ==============================================================================
# 6. SIGNED INBOUND WEBHOOKS & EVENT AUTOMATION (Twenty CRM & n8n)
# ==============================================================================
# Shared secret for HMAC-SHA256 signature verification on inbound webhooks
TWENTY_WEBHOOK_SECRET=your_webhook_hmac_secret_key_here
TWENTY_WEBHOOK_MAX_BYTES=1048576

# Server-side mapping of Webhook IDs to internal workspace/department codes
TWENTY_WEBHOOK_MAPPINGS={"default-webhook-id":{"department_id":"SALES"}}

# Background Scheduler Loop Interval (in seconds)
SCHEDULER_INTERVAL_SECONDS=60

# ==============================================================================
# 7. OUTBOUND MCP & TWENTY CRM INTEGRATION
# ==============================================================================
ENABLE_CRM_INTEGRATION=true
TWENTY_BASE_URL=http://localhost:3000
TWENTY_API_KEY=
TWENTY_MCP_ENABLED=false
TWENTY_MCP_URL=http://localhost:3000/mcp

# ==============================================================================
# 8. RAG HYPERPARAMETERS & ARTIFACT ENGINE
# ==============================================================================
CHUNK_SIZE=900
CHUNK_OVERLAP=150
DEFAULT_TOP_K=5
MAX_CHUNKS_PER_DOCUMENT=4
MIN_VECTOR_SIMILARITY=0.25
RETRIEVAL_CANDIDATE_K=20
RRF_K=60
VECTOR_SEARCH_WEIGHT=1.0
KEYWORD_SEARCH_WEIGHT=1.2
STRUCTURED_SEARCH_WEIGHT=2.0
HIERARCHICAL_SEARCH_WEIGHT=0.8
HIERARCHICAL_MAX_DESCENDANT_DEPTH=3

NODE_L0_MAX_CHARS=300
NODE_L1_MAX_CHARS=600

ARTIFACT_STORAGE_DIR=storage/artifacts
ARTIFACT_RETENTION_HOURS=48
ARTIFACT_MAX_BYTES=26214400
ARTIFACT_MAX_ROWS=500
```

---

## 5. Model Context Protocol (MCP) Guide

The platform functions as both an **Inbound MCP Server** (exposing authoritative enterprise tools to AI IDEs and clients) and an **Outbound MCP Client** (connecting to external business tools like Twenty CRM).

### MCP Server Endpoints
- **Streamable-HTTP Transport**: `http://localhost:8000/mcp`
- **Server-Sent Events (SSE) Transport**: `http://localhost:8000/mcp/sse`

### Generating an MCP Authentication Token
To connect authenticated external clients, generate a 30-day JWT token:
```bash
uv run python scripts/generate_mcp_token.py
```
This prints the raw Bearer token and connection details.

---

### 1. Cursor IDE Setup
Cursor connects directly to the backend over SSE:

1. Open Cursor and navigate to **Settings** (`Cmd + ,` or `Ctrl + ,`).
2. Go to **Features** $\to$ **MCP**.
3. Click **"+ Add New MCP Server"**:
   - **Name**: `rag-platform`
   - **Type**: `sse`
   - **Server URL**: `http://localhost:8000/mcp/sse`
4. Click **Add**. The status indicator will turn green, exposing the platform's tools to Cursor's Composer and Agent.

---

### 2. ChatGPT (Web & Desktop) Setup
ChatGPT requires an encrypted public HTTPS tunnel:

1. Start a Cloudflare tunnel pointing to your backend:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
2. In ChatGPT, go to **Settings** $\to$ **Connected Apps** $\to$ **+ Add custom MCP server**.
3. Set URL to:
   ```text
   https://<your-subdomain>.trycloudflare.com/mcp
   ```
4. Choose **No authentication** (or configure Custom Header `Authorization: Bearer <token>`).

---

### 3. Claude Desktop Setup
Configure Claude Desktop by editing your configuration file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add the following entry:
```json
{
  "mcpServers": {
    "enterprise-rag": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/mcp/sse"]
    }
  }
}
```
Restart Claude Desktop. The hammer icon will show all available tools.

---

### 4. Outbound MCP (Twenty CRM)
The orchestrator connects to external MCP servers to execute approved actions:
1. Set `TWENTY_MCP_ENABLED=true` in `.env`.
2. Configure `TWENTY_MCP_URL=http://localhost:3000/mcp`.
3. The platform will dynamically discover tools, classify them into **Read** vs **Approval-Gated Write**, and enforce cryptographic approval before execution.

---

## 6. Inbound Webhooks & n8n / Twenty CRM Automation

The platform provides hardened webhook ingestion endpoints to automate tasks triggered by external platforms like **Twenty CRM**, **n8n**, or **GitHub**.

### Inbound Endpoints
- `POST /triggers/webhook` — General event trigger for workflow execution.
- `POST /crm/webhooks/twenty` — Dedicated Twenty CRM entity synchronization.

---

### Webhook Cryptographic Contract

Inbound webhook requests enforce a fail-closed cryptographic contract:

1. **HMAC-SHA256 Signature Header (`X-Twenty-Webhook-Signature`)**:
   Computed using the secret key configured in `TWENTY_WEBHOOK_SECRET`:
   $$\text{signature} = \text{HMAC-SHA256}(\text{key} = \text{TWENTY\_WEBHOOK\_SECRET}, \text{msg} = \text{raw\_body})$$
2. **Timestamp Verification (`X-Twenty-Webhook-Timestamp`)**:
   Enforces a maximum **300-second (5-minute)** clock drift window to prevent replay attacks.
3. **Nonce Header (`X-Twenty-Webhook-Nonce`)**:
   Unique event identifier checked against a durable database replay table (`crm_webhook_events`).
4. **Server-Side Tenant Routing**:
   The `webhookId` inside the payload is validated against `TWENTY_WEBHOOK_MAPPINGS`. Payloads with unknown webhook IDs are rejected.

---

### Configuring n8n Workflow

To trigger automated workflows from [n8n](https://n8n.io/):

1. **Webhook Trigger / Schedule Node**: Collect the event payload in n8n.
2. **Crypto Node / Code Node**: Compute the HMAC-SHA256 signature of the JSON body.
   ```javascript
   const crypto = require('crypto');
   const secret = "your_webhook_hmac_secret_key_here";
   const body = JSON.stringify($input.all()[0].json);
   const timestamp = Math.floor(Date.now() / 1000).toString();
   const nonce = crypto.randomUUID();
   const signature = crypto.createHmac('sha256', secret).update(body).digest('hex');

   return {
     headers: {
       'Content-Type': 'application/json',
       'X-Twenty-Webhook-Signature': signature,
       'X-Twenty-Webhook-Timestamp': timestamp,
       'X-Twenty-Webhook-Nonce': nonce
     },
     body: body
   };
   ```
3. **HTTP Request Node**:
   - **Method**: `POST`
   - **URL**: `http://localhost:8000/triggers/webhook`
   - **Headers**: Map headers from the Crypto/Code step.
   - **Body**: Send the raw JSON payload.

---

### Testing Webhooks Locally

You can test signed webhook ingestion using Python or terminal commands.

#### Using Python Test Script:
```python
import hmac, hashlib, time, uuid, httpx

secret = "your_webhook_hmac_secret_key_here"
payload = '{"webhookId": "default-webhook-id", "event": "deal.created", "data": {"amount": 50000}}'
timestamp = str(int(time.time()))
nonce = str(uuid.uuid4())

sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

headers = {
    "Content-Type": "application/json",
    "X-Twenty-Webhook-Signature": sig,
    "X-Twenty-Webhook-Timestamp": timestamp,
    "X-Twenty-Webhook-Nonce": nonce,
}

response = httpx.post("http://localhost:8000/triggers/webhook", content=payload, headers=headers)
print("Status:", response.status_code)
print("Response:", response.json())
```

---

## 7. Web Search & Deep Research Guide

When the internal knowledge base does not contain sufficient facts to answer a prompt, the platform activates the **Deep Web Research** subsystem.

### Supported Providers
1. **TinyFish (`tinyfish`)**: Default, high-speed, structured research and full-page markdown scraper.
2. **Firecrawl (`firecrawl`)**: Deep web crawler with anti-bot bypass and clean markdown conversion.
3. **Fake (`fake`)**: Zero-cost simulated search engine for testing.

### How It Operates
1. **Intent & Sufficiency Check**: The Cross-Encoder evaluates retrieval relevance score. If average relevance is below `MIN_VECTOR_SIMILARITY` (default `0.25`) or explicit search keywords are used, web research is triggered.
2. **Query Formulation**: LLM transforms the user goal into targeted search queries.
3. **Domain Diversity Guardrail**: Limits candidate pages to `WEB_RESEARCH_MAX_CANDIDATES_PER_DOMAIN=1` to prevent domain concentration.
4. **Scraping & Grounding**: Downloads sanitized markdown, splits into semantic chunks, and merges into the hierarchical context compiler.
5. **Citations & Provenance**: Every web statement is referenced with full URL citations in the generated output.

---

## 8. Human-in-the-Loop (HITL) Approval & Write Recovery

Autonomous agents must never perform destructive or mutating operations without human supervision.

```mermaid
sequenceDiagram
    autonumber
    actor User as Operator / Admin
    participant Agent as Autonomous Agent
    participant Gate as Approval Gate
    participant Recovery as Two-Step Write Recovery
    participant Ext as External System (CRM/MCP)

    Agent->>Gate: Propose Action (Draft Parameters)
    Gate-->>Agent: Ticket Created (Status: PENDING, SHA-256 Hash)
    Gate->>User: Surface Interactive Approval Card in UI
    alt Operator Approves
        User->>Gate: Approve Ticket with Canonical Draft Hash
        Gate->>Recovery: Transition: IN_FLIGHT + Idempotency Key
        Recovery->>Ext: Dispatch Remote API Mutation
        Ext-->>Recovery: Success 200 OK
        Recovery-->>Gate: Status: CONFIRMED
        Gate-->>User: Mutation Applied Successfully
    else Operator Rejects
        User->>Gate: Reject Ticket with Reason
        Gate-->>Agent: Ticket CANCELLED (0 Writes Performed)
    end
```

### Safety Guarantees
- **Draft Hash Integrity**: The approval payload must contain the exact SHA-256 hash of the proposed mutation. Any tampering invalidates the request with a `409 Conflict`.
- **Two-Step Idempotency**: If a remote call times out, the durable state transitions to `UNKNOWN`. The recovery engine queries the remote system before attempting any retry, ensuring zero accidental double writes.

---

## 9. Testing & Quality Verification

The repository contains automated unit, integration, and security test suites for both backend and frontend.

### Running Backend Tests
Execute test suites across RAG retrieval, DAG orchestration, and MCP security:

```bash
# Run the entire backend test suite
uv run pytest tests/ -v

# Run Adaptive DAG Orchestrator tests
uv run pytest tests/test_milestone2_adaptive_dag.py -v

# Run MCP Security and Two-Step Write Recovery tests
uv run pytest tests/test_mcp_security_and_resilience.py tests/test_mcp_two_step_write_recovery.py -v

# Run Inbound Webhook and Trigger tests
uv run pytest tests/test_workflow_triggers.py -v
```

### Running Frontend Tests & Production Build
```bash
cd frontend

# Run TypeScript unit tests (110+ passing tests)
npm test

# Test production build bundling
npm run build
```

### API Health Verification
Verify that backend services and database connections are operational:
```bash
curl http://localhost:8000/health
```
Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "version": "1.0.0"
}
```

---

## 10. Troubleshooting & Common Issues

### 1. `pgvector` Extension Error on Database Migration
**Symptom**: `alembic upgrade head` fails with `type "vector" does not exist`.  
**Cause**: The PostgreSQL instance does not have the `pgvector` extension installed.  
**Fix**: Ensure you are using the `pgvector/pgvector:pg16` Docker image, or run `CREATE EXTENSION IF NOT EXISTS vector;` in your PostgreSQL database as superuser.

### 2. Port Collisions (5432, 8000, 8443)
**Symptom**: `bind: address already in use`.  
**Fix**:
- **Port 5432**: If local Postgres is already running, run the Docker container on port `5433:5432` and update `DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5433/ragdb` in `.env`.
- **Port 8000**: Run uvicorn on another port: `uv run uvicorn rag_app.main:app --port 8001`.
- **Port 8443**: Set `PORT=3000 npm run dev` in the `frontend` folder.

### 3. Missing `JWT_SECRET_KEY` on Startup
**Symptom**: `ValidationError: Field required: jwt_secret_key`.  
**Fix**: Ensure `.env` is created from `.env.example`. The `JWT_SECRET_KEY` variable must be present and at least 32 characters long.

### 4. Webhook Rejection (`401 Unauthorized` / `Invalid HMAC Signature`)
**Symptom**: Inbound webhooks return `Invalid webhook HMAC-SHA256 signature`.  
**Fix**:
1. Verify `TWENTY_WEBHOOK_SECRET` matches in both your sender (n8n/script) and `.env`.
2. Ensure the signature is computed over the **exact raw byte stream** of the request body before any JSON formatting or whitespace alterations.
3. Check system clocks: if your timestamp deviates by more than 300 seconds from server UTC, the request is rejected.

### 5. First-Time HuggingFace Model Download Delay
**Symptom**: First retrieval query hangs for 15–30 seconds.  
**Explanation**: On the initial run, `sentence-transformers` automatically downloads `all-MiniLM-L6-v2` (~90 MB) and caches it in `~/.cache/huggingface`. Subsequent executions run instantaneously offline.

---

## 11. Repository Structure

```text
ragdemo1/
├── .env.example               # Complete environment variables template
├── docker-compose.yml         # Container definitions (PostgreSQL + pgvector + Twenty)
├── pyproject.toml             # Python package configuration and dependencies
├── uv.lock                    # Deterministic uv lockfile for Python
├── alembic.ini                # Alembic migration configuration
├── src/
│   └── rag_app/
│       ├── api/               # FastAPI route controllers (chat, workflows, artifacts, MCP, webhooks)
│       ├── context/           # Hierarchical OpenViking context compiler & token budgets
│       ├── core/              # Global application settings and lifespan management
│       ├── crm/               # Twenty CRM client, sync service, and reconciliation loop
│       ├── db/                # SQLAlchemy models, session engine, and pgvector schemas
│       ├── generation/        # Grounded LLM response generation and citation validation
│       ├── mcp/               # FastMCP dual-transport server, tool bindings, and risk gates
│       ├── orchestration/     # Adaptive DAG planner, task leasing, and trigger service
│       ├── providers/         # Pluggable capability adapters (LLMs, embeddings, web search)
│       ├── reports/           # Multi-format artifact generation (PDF, DOCX, CSV, JSON)
│       ├── research/          # LangGraph state machine, query planning, and TinyFish integration
│       ├── retrieval/         # Vector, BM25, RRF, and CrossEncoder hybrid retrieval pipeline
│       └── security/          # JWT authentication, RBAC authorization, and MultiFernet encryption
├── frontend/
│   ├── src/                   # React 19 UI (Chat, Workflow Studio, Execution Canvas, Artifacts)
│   ├── package.json           # Frontend dependencies and test runner scripts
│   └── vite.config.ts         # Vite configuration with backend API proxying
├── migrations/                # Alembic database migration scripts
├── scripts/                   # Administration and utility scripts (seed_admin, generate_mcp_token)
├── storage/                   # Protected local artifact and report storage
└── tests/                     # 100+ pytest test suites covering all platform capabilities
```

---

## License

This project is licensed under the Apache 2.0 License.
