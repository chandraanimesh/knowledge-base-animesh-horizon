# System Architecture and Design

## System Overview
The system provides enterprise retrieval augmented generation with multi-strategy search.

## Runtime Query Flows
The runtime query flow orchestrates sequential strategy execution. When a query is received, the RetrievalController determines the execution plan across graph, knowledge, semantic, structured, and hierarchical routes.

### Query Parsing and Normalization
Queries are normalized and parsed into a QueryPlan containing semantic text, entity search terms, and structured filters.

### Candidate Fusion and Ranking
Candidates from vector search, keyword search, knowledge search, and hierarchical search enter Reciprocal Rank Fusion (RRF) with distinct weights.

### Cross-Encoder Reranking
Top candidates from RRF pass through CrossEncoder reranking for final contextual relevance scoring.
