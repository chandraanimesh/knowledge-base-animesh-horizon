# Enterprise Adaptive RAG, Agentic Orchestration & Research Platform

A production-grade, extensible **Retrieval-Augmented Generation (RAG) and Agentic Workflow Platform**. It integrates **Hybrid Multi-Strategy Retrieval (Dense Vector + BM25 + RRF + Cross-Encoder)**, **Hierarchical Section-Aware Context Compilation (OpenViking)**, **Adaptive DAG-Based Task Orchestration**, **Human-in-the-Loop Cryptographic Approval Gates**, **Model Context Protocol (MCP) Hub & Spoke Architecture**, **Live CRM Write-Back & Two-Step Write Recovery**, **Deep Web Research (TinyFish)**, and **Multi-Format Artifact Generation (PDF, DOCX, CSV, JSON)**.

---

## 1. System Requirements & Tech Stack

### Core System Requirements
| Component | Supported Version | Notes |
| :--- | :--- | :--- |
| **Python** | `>= 3.11` (Recommended `3.12+`) | Managed via `uv` |
| **Node.js** | `>= 18.x` (Recommended `20.x` / `22.x LTS`) | For React 19 / Vite Frontend |
| **PostgreSQL** | `>= 15.x` / `16.x` | With `pgvector` extension enabled (`v0.5.0+`) |
| **Package Managers** | `uv` (Python) & `npm` (Node) | High-speed dependency resolution and lockfile enforcement |
| **Docker** *(Optional)* | Docker Desktop / Docker Engine | Recommended for running PostgreSQL + `pgvector` seamlessly |

### Key Backend Libraries
- **API & Core**: `fastapi>=0.115.0`, `uvicorn[standard]>=0.30.0`, `pydantic>=2.7.0`, `pydantic-settings>=2.3.0`
- **Database & Vectors**: `sqlalchemy[asyncio]>=2.0.30`, `asyncpg>=0.29.0`, `pgvector>=0.3.0`, `alembic>=1.13.0`
- **Retrieval & Reranking**: `sentence-transformers>=5.6.0`, `rank-bm25>=0.2.2`, `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`)
- **Document Parsing & OCR**: `docling[rapidocr]>=2.121.0`, `pymupdf>=1.24.0`, `pytesseract>=0.3.13`, `paddleocr>=2.10.0`
- **Workflow & Orchestration**: `langgraph>=1.2.12`, `langchain-core`, `langchain-groq>=1.1.3`
- **LLM & Embeddings Providers**: `groq`, `openai>=1.35.0`, `google-generativeai>=0.7.0`
- **MCP Protocols**: `fastmcp>=4.0.11`, SSE & Streamable-HTTP dual transport
- **Web Research**: `firecrawl-py>=4.44.0`, TinyFish search and scrape adapters
- **Security & Auth**: `PyJWT>=2.8.0`, `passlib[bcrypt]>=1.7.4`, `bcrypt==4.0.1`

### Frontend Stack
- **Framework**: React 19 (`react`, `react-dom`, `react-router-dom 7`)
- **Bundler & Styling**: `vite 8`, `tailwindcss 4`
- **Language**: TypeScript 5.7+
- **Test Runner**: Node test runner with `--experimental-strip-types`

---

## 2. OS Setup & Installation Guide

### Option A: Setup on Windows

#### 1. Install Prerequisites on Windows
Open **PowerShell as Administrator** or run in your terminal:

```powershell
# 1. Install uv (Python Package Manager)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install Node.js LTS (v20+) and Git (via Windows Package Manager winget)
winget install OpenJS.NodeJS.LTS
winget install Git.Git
```
*Restart your PowerShell session after installation so environment paths update.*

#### 2. Setup PostgreSQL with pgvector on Windows

**Recommended (via Docker Desktop):**
```powershell
docker run -d `
  --name rag-postgres `
  -p 5432:5432 `
  -e POSTGRES_USER=rag `
  -e POSTGRES_PASSWORD=rag `
  -e POSTGRES_DB=ragdb `
  pgvector/pgvector:pg16
```

**Alternative (Native Windows Installer):**
1. Download and run the PostgreSQL 16 installer from [EnterpriseDB](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads).
2. Download precompiled `pgvector` binaries for your PostgreSQL version from [pgvector releases](https://github.com/pgvector/pgvector/releases).
3. Copy `vector.dll` into `C:\Program Files\PostgreSQL\16\lib` and extension files into `C:\Program Files\PostgreSQL\16\share\extension`.
4. Run in `psql`:
   ```sql
   CREATE DATABASE ragdb;
   \c ragdb
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

#### 3. Setup Project & Environment
```powershell
# Clone the repository
git clone <repo-url>
cd ragdemo1

# Create .env from template
copy .env.example .env

# Install backend dependencies via uv
uv sync

# Run database migrations
uv run alembic upgrade head
```

#### 4. Run Backend & Frontend on Windows
- **Start Backend**:
  ```powershell
  uv run uvicorn rag_app.main:app --reload --port 8000
  ```
- **Start Frontend** (in a separate terminal):
  ```powershell
  cd frontend
  npm install
  npm run dev
  ```
  *Frontend runs at `http://localhost:8443` (or `http://localhost:5173`).*

---

### Option B: Setup on macOS

#### 1. Install Prerequisites on macOS
Open your terminal:

```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install uv, Node.js, and Git
brew install uv node git
```

#### 2. Setup PostgreSQL with pgvector on macOS

**Method 1 (Native Homebrew):**
```bash
# Install PostgreSQL 16 and pgvector
brew install postgresql@16 pgvector
brew services start postgresql@16

# Create database and enable pgvector
createdb ragdb
psql -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Method 2 (via Docker):**
```bash
docker run -d \
  --name rag-postgres \
  -p 5432:5432 \
  -e POSTGRES_USER=rag \
  -e POSTGRES_PASSWORD=rag \
  -e POSTGRES_DB=ragdb \
  pgvector/pgvector:pg16
```

#### 3. Setup Project & Environment
```bash
# Clone the repository
git clone <repo-url>
cd ragdemo1

# Create .env file
cp .env.example .env

# Install backend dependencies via uv
uv sync

# Run database migrations
uv run alembic upgrade head
```

#### 4. Run Backend & Frontend on macOS
- **Start Backend**:
  ```bash
  uv run uvicorn rag_app.main:app --reload --port 8000
  ```
- **Start Frontend** (in a separate terminal):
  ```bash
  cd frontend
  npm install
  npm run dev
  ```

---

## 3. Environment Configuration (`.env`)

Create `.env` in the root folder with the following configuration:

```ini
# Core Application Configuration
APP_NAME=production-rag-backend
ENV=dev
LOG_LEVEL=INFO

# Database Configuration (PostgreSQL with pgvector)
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/ragdb

# Embedding Provider Configuration
# Supported: sentence_transformers | openai | fake
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

# LLM Provider Configuration
# Supported: groq | openai | gemini | fake
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
OPENAI_API_KEY=
GOOGLE_API_KEY=

# Web Research & External Data Fetching (TinyFish / Firecrawl)
ENABLE_WEB_RESEARCH=true
WEB_SEARCH_PROVIDER=tinyfish
TINYFISH_API_KEY=your_tinyfish_api_key_here
TINYFISH_SEARCH_URL=https://api.search.tinyfish.ai
TINYFISH_FETCH_URL=https://api.fetch.tinyfish.ai

# Retrieval & Hybrid Ranking Hyperparameters
DEFAULT_TOP_K=5
MIN_VECTOR_SIMILARITY=0.25
RETRIEVAL_CANDIDATE_K=20
RRF_K=60
VECTOR_SEARCH_WEIGHT=1.0
KEYWORD_SEARCH_WEIGHT=1.2
STRUCTURED_SEARCH_WEIGHT=2.0
HIERARCHICAL_SEARCH_WEIGHT=0.8
HIERARCHICAL_MAX_DESCENDANT_DEPTH=3

# Hierarchical Context Bounds (OpenViking L0/L1)
NODE_L0_MAX_CHARS=300
NODE_L1_MAX_CHARS=600
CHUNK_SIZE=900
CHUNK_OVERLAP=150

# Security & JWT Authentication
JWT_SECRET_KEY=super_secure_jwt_secret_key_change_in_production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# Default Seed Administrator
ADMIN_EMAIL=admin@local.dev
ADMIN_PASSWORD=Admin@12345

# Twenty CRM Integration (Optional Local Service)
TWENTY_BASE_URL=http://localhost:3000
TWENTY_API_KEY=
ENABLE_CRM_INTEGRATION=true
```

---

## 4. Current Architecture & Key Subsystems

```mermaid
flowchart TD
    UserQuery([User Query / Workflow Goal]) --> Router{Intent & Task Router}
    
    Router -->|Conversational / Greeting| DirectLLM[Direct Conversational Handler]
    Router -->|Document / Factual RAG| MultiRetrieval[Multi-Strategy Hybrid Retrieval]
    Router -->|Multi-Step Deep Research| AdaptiveDAG[Adaptive DAG Task Planner]
    
    subgraph MultiRetrievalEngine [Hybrid Retrieval Pipeline]
        MultiRetrieval --> VectorSearch[Dense Vector HNSW pgvector]
        MultiRetrieval --> BM25[BM25 / Full-Text Search ts_rank_cd]
        MultiRetrieval --> GraphTriples[Graph Entity Triples 1-Hop / 2-Hop]
        MultiRetrieval --> HierarchicalTree[Hierarchical Node Tree L0 / L1 / L2]
    end
    
    VectorSearch & BM25 & GraphTriples & HierarchicalTree --> RRF[Reciprocal Rank Fusion RRF]
    RRF --> CrossEncoder[CrossEncoder Re-ranker ms-marco]
    CrossEncoder --> ACLFilter[Department & Workspace ACL Gate]
    
    ACLFilter --> SufficiencyCheck{Sufficiency Gate}
    SufficiencyCheck -->|Sufficient| ContextCompiler[Hierarchical Context Compiler]
    SufficiencyCheck -->|Insufficient / Web Trigger| WebResearch[TinyFish Deep Web Research]
    WebResearch --> ContextCompiler
    
    ContextCompiler --> GroundedGen[Grounded LLM Generation Service]
    GroundedGen --> ProvenanceEngine[Citation & Span Verification Engine]
    
    subgraph OrchestrationEngine [Adaptive Workflow & Mutation Engine]
        AdaptiveDAG --> DAGValidator[DAG Validator & Budget Enforcer]
        DAGValidator --> Executor[DAG Parallel / Sequential Execution Engine]
        Executor --> ReadMCP[Read-only MCP Tool Binding]
        Executor --> ProposalTool[Mutation Proposal Tool]
        ProposalTool --> ApprovalGate{Human-in-the-Loop Approval Ticket}
        ApprovalGate -->|Rejected| Terminate([Workflow Terminated / 0 Writes])
        ApprovalGate -->|Approved| WriteRecovery[Two-Step Idempotent Write-Back]
        WriteRecovery --> TwentyCRM[(Twenty CRM / External MCP Server)]
    end
    
    GroundedGen & Executor --> ArtifactGenerator[Multi-Format Artifact Renderer PDF/DOCX/CSV]
    ArtifactGenerator --> ProtectedStorage[(storage/artifacts Directory)]
    
    ProvenanceEngine --> FinalAnswer([Verified Answer + Interactive Citations])
```

### Detailed Subsystems

#### 1. Multi-Strategy Hybrid Retrieval Engine
- **Dense Vector Search**: Powered by `pgvector` HNSW indexes using Cosine distance.
- **Full-Text Keyword Search**: PostgreSQL English tsvector dictionary with `ts_rank_cd` scoring.
- **Reciprocal Rank Fusion (RRF)**: Merges rank lists with weighted multipliers:
  $$RRF(d) = \sum_{m \in M} \frac{w_m}{k + \text{rank}_m(d)}$$
- **Cross-Encoder Reranking**: Re-scores top candidate pairs using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **OpenViking Hierarchical Compilation**: Section-aware hierarchical context parsing (L0 Abstract $\le 300$ chars, L1 Overview $\le 600$ chars, L2 Content Chunks).

#### 2. Adaptive DAG Task Orchestrator (Milestone 2)
- **Dynamic Task Planning**: Breaks down high-level research questions into a Directed Acyclic Graph (DAG) with explicit dependencies (`depends_on`).
- **Safety & Guardrails**:
  - Cycle detection & topological validation.
  - Fail-closed execution with bounded concurrency, runtime limits, and provider token/call budgets.
  - Task identity leasing with fencing tokens: stale `RUNNING` tasks are recovered safely to `READY` on node failure.
  - External tool outputs treated as untrusted evidence, detecting and resolving contradictory findings.

#### 3. Model Context Protocol (MCP) Hub & Spoke (Milestone 6)
- **Inbound MCP Server**: Dual-transport endpoint (`/mcp` Streamable-HTTP and `/mcp/sse`) exposing 52 authoritative tools to AI hosts like **ChatGPT**, **Cursor IDE**, and **Claude Desktop**.
- **Outbound MCP Client**: Integrates with external servers such as **Twenty CRM** (`http://localhost:3000/mcp`) or custom business tools.
- **Server-Verified Tool Risk Classification**:
  - `Read`: Verified safe query tools (accessible to `VIEWER`+).
  - `Write (Approval Gated)`: Mutating tools requiring explicit operator review and tickets.
  - `Restricted`: Unverified or dangerous tools blocked by default-deny policy.

#### 4. Human-in-the-Loop Approval & Two-Step Write Recovery (Milestones 6E, 6E.1, 6F.1)
- **Proposal-Only Agent Binding**: Autonomous agents cannot directly invoke remote mutations; they emit deterministic proposal tickets (`mcp_propose_create_one_note`).
- **Cryptographic Draft Integrity**: Canonical SHA-256 draft hashing. Any modification of proposed parameters invalidates the ticket (`409 Conflict`).
- **Durable Outcome State Machine**:
  - Tracks operations through `PENDING` $\to$ `IN_FLIGHT` $\to$ `CONFIRMED` / `UNKNOWN` / `RECONCILIATION_REQUIRED`.
  - Ambiguous remote timeouts are marked `UNKNOWN` and block blind retries.
  - Automated reconciliation confirms committed writes before retrying.
  - Replays with identical idempotency keys return cached results with `is_replay: True`.

#### 5. Deep Web Research (TinyFish Integration)
- Autonomous web research triggered when internal knowledge base sufficiency is low.
- Multi-query formulation, domain-diversity filtering, and full-page content scraping with grounding validation.

#### 6. Multi-Format Generated Artifacts System
- Generates publication-ready artifacts in **PDF**, **DOCX**, **CSV**, **JSON**, and **Markdown**.
- Versioned drafts, cryptographic approval verification, and protected downloads via `/reports` endpoints.
- Real-time review and download directly in the frontend (`ArtifactList.tsx`).

---

## 5. Connecting External MCP Clients

### 1. Cursor IDE
Cursor connects directly to your local instance over SSE:
1. Open Cursor **Settings → Features → MCP**.
2. Click **"+ Add New MCP Server"**:
   - **Name**: `rag-backend`
   - **Type**: `sse`
   - **Server URL**: `http://localhost:8000/mcp/sse`
3. Click **Add**. The status dot will turn green and expose all 52 tools.

### 2. ChatGPT (Web & Desktop)
ChatGPT requires an encrypted public HTTPS endpoint:
1. Start a Cloudflare tunnel:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
2. In ChatGPT: Go to **Settings → Connected Apps → + Add custom MCP server**.
3. Set URL to `https://<your-tunnel-url>.trycloudflare.com/mcp` and select **No authentication**.

### 3. Claude Desktop
Add the following to your `claude_desktop_config.json`:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "production-rag": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/mcp/sse"]
    }
  }
}
```

---

## 6. Verification & Testing

### Running Backend Tests
The backend test suite covers RAG retrieval, context compilation, DAG orchestration, MCP security, and CRM write recovery:

```bash
# Run all backend tests
uv run pytest tests/ -v

# Run MCP and Workflow-specific test suites
uv run pytest tests/test_mcp_security_and_resilience.py tests/test_mcp_approval_writeback.py tests/test_mcp_two_step_write_recovery.py -v

# Run DAG Orchestrator and Vertical Slice tests
uv run pytest tests/test_milestone2_adaptive_dag.py tests/test_vertical_slice_workflow.py -v
```

### Running Frontend Tests & Production Build
```bash
cd frontend

# Run frontend unit tests
npm test

# Verify production build compilation
npm run build
```

---

## 7. Project Structure

```text
ragdemo1/
├── src/
│   └── rag_app/
│       ├── api/               # FastAPI REST & SSE routes (workspaces, workflows, artifacts, MCP)
│       ├── context/           # Hierarchical context compiler & tokenizer
│       ├── crm/               # Twenty CRM client, factory, and durable sync service
│       ├── db/                # SQLAlchemy async models & database session management
│       ├── generation/        # Grounded generation, citation validation, and prompt templates
│       ├── mcp/               # FastMCP dual-transport server, client security, and tool bindings
│       ├── orchestration/     # Adaptive DAG planner, DAG execution engine, evidence & budget gates
│       ├── providers/         # Pluggable provider architecture (LLMs, embeddings, web search)
│       ├── reports/           # Multi-format artifact renderers (PDF, DOCX, CSV, JSON)
│       ├── research/          # LangGraph state machine, query planning, and TinyFish web research
│       ├── retrieval/         # Vector, BM25, RRF, and CrossEncoder hybrid retrieval
│       └── security/          # Fernet credential encryption, JWT auth, and RBAC enforcement
├── frontend/
│   ├── src/
│   │   ├── api/               # API clients for chat, workspaces, workflows, and MCP connections
│   │   ├── components/        # React 19 UI (RAG view, MCP Integrations, Workflow Progress, Artifacts)
│   │   └── ...
│   ├── package.json
│   └── vite.config.ts         # Vite config with backend proxy and port 8443
├── migrations/                # Alembic database migrations (pgvector, MCP, workflows, artifacts)
├── tests/                     # 100+ pytest test suites covering RAG, MCP, DAG, and Security
├── pyproject.toml             # Python package definition and dependencies
└── README.md
```
