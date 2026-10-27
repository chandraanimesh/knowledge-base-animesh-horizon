# Enterprise Production RAG & Agentic Knowledge Assistant

A high-performance, production-grade Retrieval-Augmented Generation (RAG) platform featuring **Hybrid Multi-Strategy Retrieval (Vector + BM25 + RRF + CrossEncoder)**, **Hierarchical Section-Aware Context Compilation (OpenViking)**, **Graph/Relationship Traversal**, **Role-Based Multi-Department ACLs**, and **LangGraph Sufficiency & Generation Decision Flow**.

---

## 1. Tech Stack & Version Requirements

### Core Runtime & System Requirements
| Component | Supported Version | Notes |
| :--- | :--- | :--- |
| **Python** | `>= 3.11` (Recommended `3.12+`) | Managed via `uv` or standard virtual environments |
| **Node.js** | `>= 18.x` (Recommended `20.x` / `22.x`) | For React/Vite Frontend |
| **PostgreSQL** | `>= 15.x` / `16.x` | With `pgvector` extension enabled (`v0.5.0+`) |
| **Package Manager** | `uv` (Python) & `npm` (Node) | Ultra-fast dependency resolution |

### Key Backend Libraries
- **API & Core**: `fastapi>=0.115.0`, `uvicorn[standard]>=0.30.0`, `pydantic>=2.7.0`, `pydantic-settings>=2.3.0`
- **Database & Vectors**: `sqlalchemy[asyncio]>=2.0.30`, `asyncpg>=0.29.0`, `pgvector>=0.3.0`, `alembic>=1.13.0`
- **Retrieval & Reranking**: `sentence-transformers>=5.6.0`, `rank-bm25>=0.2.2`, `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`)
- **Document Parsing & OCR**: `docling[rapidocr]>=2.121.0`, `pymupdf>=1.24.0`, `pytesseract>=0.3.13`, `paddleocr>=2.10.0`
- **Orchestration & LLM**: `langgraph>=1.2.12`, `langchain-groq>=1.1.3`, `groq`, `openai>=1.35.0`, `google-generativeai>=0.7.0`
- **Security & Auth**: `PyJWT>=2.8.0`, `passlib[bcrypt]>=1.7.4`, `bcrypt==4.0.1`

### Frontend Stack
- **Framework**: React 19 (`react`, `react-dom`, `react-router-dom 7`)
- **Bundler & Styling**: `vite 8`, `tailwindcss 4`
- **Language**: TypeScript 5.7+

---

## 2. Environment Configuration (`.env.example`)

Create a `.env` file in the project root based on the following template:

```ini
# Application Configuration
APP_NAME=production-rag-backend
ENV=dev
LOG_LEVEL=INFO

# PostgreSQL Database with pgvector
DATABASE_URL=postgresql+asyncpg://rag:rag@127.0.0.1:5433/ragdb

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

# Ingestion Chunking Configuration
CHUNK_SIZE=900
CHUNK_OVERLAP=150

# Retrieval & Hybrid Ranking Hyperparameters
DEFAULT_TOP_K=5
MIN_VECTOR_SIMILARITY=0.10
RETRIEVAL_CANDIDATE_K=20
RRF_K=60
VECTOR_SEARCH_WEIGHT=1.0
KEYWORD_SEARCH_WEIGHT=1.2
STRUCTURED_SEARCH_WEIGHT=2.0
HIERARCHICAL_SEARCH_WEIGHT=0.8
HIERARCHICAL_MAX_DESCENDANT_DEPTH=3

# Hierarchical Document Node Bounds
NODE_L0_MAX_CHARS=300
NODE_L1_MAX_CHARS=600

# Security & JWT Authentication
JWT_SECRET_KEY=dev_secret_key_change_me_in_prod
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# Default Seed Administrator
ADMIN_EMAIL=admin@local.dev
ADMIN_PASSWORD=Admin@12345
```

---

## 3. Setup & Installation Guide

### Prerequisites

#### For macOS (Mac):
```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python package manager (uv), Node.js, and PostgreSQL
brew install uv node postgresql@16

# 3. Install pgvector for PostgreSQL
brew install pgvector

# 4. Start PostgreSQL service
brew services start postgresql@16

# 5. Create database & enable pgvector
createdb ragdb
psql -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### For Windows:
```powershell
# 1. Install uv (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Ensure PostgreSQL with pgvector is running on localhost (default 5432 or 5433)
```

---

### Backend Setup (macOS & Windows)

1. **Install Python Dependencies**:
   ```bash
   uv sync
   ```

2. **Run Database Migrations** (Creates all core RAG tables + MCP tables):
   ```bash
   uv run alembic upgrade head
   ```

3. **Start the Backend Dev Server** (Exposes API & MCP Server):
   ```bash
   uv run uvicorn rag_app.main:app --reload --port 8000
   ```
   *The backend API and Swagger docs will be live at `http://localhost:8000/docs`.*
   *The built-in MCP server will be active at `http://localhost:8000/mcp` and `http://localhost:8000/mcp/sse`.*

---

### Frontend Setup (macOS & Windows)

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node Dependencies**:
   ```bash
   npm install
   ```

3. **Start the Frontend Dev Server**:
   ```bash
   npm run dev
   ```
   *The application UI will be accessible at `http://localhost:5173`.*

---


## 4. Architecture, Algorithms & Retrieval Processes

```mermaid
flowchart TD
    UserQuery([User Query]) --> ClassifyQuery{Conversational Query Router}
    
    ClassifyQuery -->|Greeting / Casual| DirectConv[Conversational LLM Handler]
    ClassifyQuery -->|History / Prior Turn| HistoryHandler[Conversation History Resolver]
    ClassifyQuery -->|Knowledge / Document Query| QueryParser[Query Parser & Plan Builder]
    
    QueryParser --> MultiRetrieval[Multi-Strategy Retrieval Engine]
    
    subgraph MultiRetrievalEngine [Retrieval Pipelines]
        MultiRetrieval --> VectorSearch[Dense Vector Search pgvector HNSW]
        MultiRetrieval --> FTS[Full-Text Keyword Search BM25 / GIN]
        MultiRetrieval --> GraphSearch[Graph Knowledge & Entity Traversal]
        MultiRetrieval --> HierarchicalSearch[Hierarchical Section Search OpenViking]
    end
    
    VectorSearch & FTS & GraphSearch & HierarchicalSearch --> Fusion[Reciprocal Rank Fusion RRF]
    Fusion --> CrossEncoder[CrossEncoder Re-ranker ms-marco]
    CrossEncoder --> ACLFilter[Department & Workspace ACL Enforcer]
    
    ACLFilter --> SufficiencyEval{Sufficiency Evaluation Gate}
    
    SufficiencyEval -->|Sufficient = True| ContextCompiler[Hierarchical Context Compiler]
    ContextCompiler --> GenerationLLM[Grounded LLM Generation Service]
    GenerationLLM --> CitationEngine[Citation & Provenance Validator]
    CitationEngine --> FinalResponse([Answer + Interactive Source Citations])
    
    SufficiencyEval -->|Sufficient = False / 0 Chunks| Fallback[Standard Document Fallback]
```

---

### Key Subsystems & Algorithmic Process

#### 1. Ingestion, Parsing & Chunking Pipeline
- **Multi-Format Extraction**: Ingests PDFs, DOCX, XLSX, and TXT using `Docling`, `RapidOCR`, and `PyMuPDF`.
- **Hierarchical Node Tree**: Decomposes complex documents into hierarchical structural trees with:
  - **L0 Abstract**: Ultra-concise document/section summary ($\le 300$ chars).
  - **L1 Overview**: Contextual section summaries ($\le 600$ chars).
  - **L2 Chunks**: Concrete content chunks ($\approx 900$ chars with 150-char overlap).
- **HNSW Indexing**: Embeddings are stored in PostgreSQL using `pgvector` with Cosine distance indexing (`vector_cosine_ops`).

#### 2. Multi-Strategy Hybrid Retrieval
1. **Dense Vector Search**: Queries embedding space using cosine similarity ($1 - \text{distance}$) with configurable threshold filters.
2. **PostgreSQL Full-Text Search (BM25 Equivalent)**: Keyword matching using PostgreSQL English text search dictionaries and `ts_rank_cd`.
3. **Reciprocal Rank Fusion (RRF)**:
   Combines rank lists from dense vector search, full-text keyword search, and knowledge lookups:
   $$RRF(d) = \sum_{m \in M} \frac{w_m}{k + \text{rank}_m(d)}$$
   where $k = 60$, with weighted multipliers ($w_{\text{vector}} = 1.0$, $w_{\text{keyword}} = 1.2$, $w_{\text{structured}} = 2.0$, $w_{\text{hierarchical}} = 0.8$).
4. **Cross-Encoder Reranking**:
   Top $K$ candidate pairs `(query, chunk_text)` pass through `cross-encoder/ms-marco-MiniLM-L-6-v2` for cross-attention scoring.

#### 3. GraphRAG & Entity Provenance
- Extracts structured entity triples `(Subject, Predicate, Object)` and links them to underlying document chunks.
- Supports 1-hop and 2-hop graph relation traversal to resolve complex multi-document corporate entities and relationships.

#### 4. Security & Multi-Department Access Control (ACL)
- **Role-Based Scoping**: `Admin` (global cross-department access), `Lead` (department-level administrative access), and `Member` (assigned workspace access).
- **Hard Security Invariant**: Context compilers enforce strict department filtering so that unauthorized chunks can never reach the LLM context prompt.

#### 5. LangGraph Sufficiency & Generation Decision Flow
- **Conversational Router**: Accurately differentiates greetings, conversation history queries, and factual knowledge inquiries.
- **Sufficiency Gate**: Evaluates whether authorized retrieved chunks contain substantive factual guidelines to address the user query.
- **Citation & Provenance Engine**: Validates answer claims against verbatim chunk spans and attaches provenance traces (Strategy, Hop, Score, Document, Page).

---

## 5. Model Context Protocol (MCP) & Twenty CRM Integration

The backend exposes an authoritative **Model Context Protocol (MCP)** interface with **52 domain tools** (covering RAG retrieval, document ingestion, conversation history, workspace isolation, admin controls, and system diagnostics).

### Architecture Overview

MCP uses a **Hub & Spoke** architecture. External AI clients (Cline, Cursor, Claude Desktop, ChatGPT) act as the orchestrator/brain and connect simultaneously to multiple MCP servers:

```text
              ┌──────────────────────────────────────┐
              │  AI Client / Host (Cline / Cursor)   │
              └──────────────────┬───────────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
 ┌──────────────────────┐                  ┌──────────────────────┐
 │    Twenty CRM        │                  │  Production RAG      │
 │    (MCP Server 1)    │                  │  (MCP Server 2)      │
 │  localhost:3000/mcp  │                  │  localhost:8000/mcp  │
 └──────────────────────┘                  └──────────────────────┘
   (Leads & Contacts Data)                   (Documents & Policies)
```

---

### Architecture & Supported Protocols

The platform provides a **Unified Dual-Transport MCP Gateway** under `/mcp` supporting 52 authoritative tools:

| Endpoint | Protocol | Supported Clients | Purpose |
| :--- | :--- | :--- | :--- |
| `http://localhost:8000/mcp` | Streamable HTTP (JSON-RPC) & HTTP GET | **ChatGPT**, Modern MCP Clients, Web Browsers | Direct POST JSON-RPC execution & GET JSON discovery metadata. |
| `http://localhost:8000/mcp/sse` | Server-Sent Events (SSE) | **Cursor IDE**, **Claude Desktop**, Twenty CRM | Streaming event transport with unbuffered delivery. |
| `http://localhost:8000/mcp/messages/` | HTTP POST | Standard SSE clients | Inbound JSON-RPC session message dispatcher. |

---

### Step 1: Start the Backend Server

```bash
uv run uvicorn rag_app.main:app --host 127.0.0.1 --port 8000
```
Verify status:
```bash
curl http://localhost:8000/mcp
# Output: {"status":"ready","name":"Production RAG MCP Server","transport":"streamable-http+sse","tools_count":52,...}
```

---

### Step 2: Client Setup & Connection Guides

#### 1. ChatGPT (Web & Desktop)
ChatGPT runs on cloud servers, so it needs a public HTTPS URL to reach your local MCP server.

1. **Start a public tunnel** (using Cloudflare or ngrok):
   - **On macOS**:
     ```bash
     brew install cloudflared
     cloudflared tunnel --url http://localhost:8000
     ```
   - **On Windows**:
     ```powershell
     cloudflared tunnel --url http://localhost:8000
     ```
   Copy the assigned URL (e.g. `https://<random-name>.trycloudflare.com`).

2. **In ChatGPT**:
   - Go to **Settings → Connected Apps / Custom MCP Server** (or click **+ Add custom MCP server**).
   - **Name**: `rag` (or `Production RAG`)
   - **Connection**: Select **Server URL**
   - **URL**: `https://<your-tunnel-url>.trycloudflare.com/mcp` *(or `/mcp/sse` — both work seamlessly)*
   - **Authentication**: Select **No authentication**
   - Check **"I understand and want to continue"**
   - Click **Create as a plugin**.
3. **Usage**: In a new chat, mention `@rag` or ask:
   > *"Search our workspace documents for company policies and summarize them"*

---

#### 2. Cursor IDE (macOS & Windows)
Cursor runs locally on your machine, so you can connect directly to `localhost` without any tunnel.

1. In Cursor, open **Settings (Gear Icon) → Cursor Settings**.
2. In the left sidebar, navigate to **Features → MCP**.
3. Click **"+ Add New MCP Server"**:
   - **Name**: `rag-backend`
   - **Type**: `sse`
   - **Server URL**: `http://localhost:8000/mcp/sse`
4. Click **Add**.
5. The status indicator will turn green and expose all 52 tools.
6. **Usage**: In Cursor Composer (`Ctrl + I` on Windows / `Cmd + I` on macOS) or Agent Chat, ask questions about your documents. Cursor will autonomously call `rag_search` and `preview_document`.

---

#### 3. Claude Desktop (macOS & Windows)
Claude Desktop configures MCP servers via its central JSON configuration file.

1. **Locate the configuration file**:
   - **On macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **On Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
2. **Add the MCP Server configuration**:
   ```json
   {
     "mcpServers": {
       "production-rag": {
         "command": "npx",
         "args": [
           "-y",
           "mcp-remote",
           "http://localhost:8000/mcp/sse"
         ]
       }
     }
   }
   ```
3. Restart **Claude Desktop**.
4. In any chat, click the **Hammer (Tools)** icon at the bottom of the prompt box to inspect the 52 available RAG tools.

---

#### 4. Twenty CRM Integration (Outbound MCP Client)
Your RAG platform can also act as an **MCP Client** connecting to Twenty CRM running locally (`http://localhost:3000`).

1. Open your RAG frontend at `http://localhost:5173`.
2. Select your Workspace → navigate to **Settings → MCP Connections** tab.
3. Click **"Add Connection"**:
   - **Name**: `Twenty CRM`
   - **Endpoint URL**: `http://localhost:3000/mcp`
   - **Transport**: `streamable-http`
   - **Credential**: Your Twenty CRM API key (or *No Credential* for local testing).
4. Click **Test Connection** (Status will change to `CONNECTED`).
5. **Tool Allowlist**:
   Twenty CRM uses a **Meta-Tool Architecture**. Toggle **ON** the following key tools:
   - `execute_tool`: Required for creating, updating, and querying records (`create_one_company`, `create_one_person`, `create_one_opportunity`, etc.).
   - `learn_tools`: Allows the AI agent to inspect schema requirements before creating records.
   - `list_object_metadata_names`: Allows the AI agent to discover all available CRM objects.
6. Click **Save Connection**.

---

#### 5. Connecting Any Other CRM (HubSpot, Salesforce, Pipedrive)
The database schema (`workspace_mcp_connections` & `crm_sync_records`) is **vendor-agnostic and generic**:
1. In the RAG Frontend (**Settings → MCP Connections**), click **Add Connection**.
2. Enter the CRM's MCP Server URL (e.g. `https://mcp.hubspot.com/mcp`) and your API Key.
3. The platform automatically discovers the CRM's tools into a dynamic JSONB schema.
4. Toggle on the approved tools under the **Default-Deny Policy**.
5. All records are normalized server-side into canonical `CRMCompanyState` entities (`CRMCompany`, `CRMPerson`, `CRMOpportunity`), requiring **zero database schema migrations**.


---

## 6. Verification & Testing

### Running the Test Suite
The repository includes a comprehensive unit and integration test suite covering retrieval algorithms, sufficiency gates, provider registries, security ACLs, and network MCP endpoints:

```bash
# Run all backend tests
uv run pytest tests/ -v

# Run MCP-specific integration tests
uv run pytest tests/test_mcp_network_endpoint.py tests/test_mcp_discovery.py tests/test_mcp_auth_failclosed.py -v
```

### Production Build Validation
```bash
# Validate frontend production build
cd frontend
npm run build
```
