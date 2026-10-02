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
1. **PostgreSQL with pgvector**: Ensure PostgreSQL is running on your configured port (e.g. `5433` or `5432`) with the `pgvector` extension installed (`CREATE EXTENSION IF NOT EXISTS vector;`).
2. **Install `uv`** (Recommended Python Package Manager):
   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   # Linux/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

---

### Backend Setup

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Run Database Migrations**:
   ```bash
   uv run alembic upgrade head
   ```

3. **Start the Backend Dev Server**:
   ```bash
   uv run uvicorn rag_app.main:app --reload --port 8000
   ```
   *The backend API and Swagger docs will be live at `http://localhost:8000/docs`.*

---

### Frontend Setup

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

## 5. Verification & Testing

### Running the Test Suite
The repository includes a comprehensive unit and integration test suite covering retrieval algorithms, sufficiency gates, provider registries, and security ACLs:

```bash
# Run all backend tests
uv run pytest tests/ -v
```

### Production Build Validation
```bash
# Validate frontend production build
cd frontend
npm run build
```
