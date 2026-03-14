# GitNexus: The Nervous System for Agent Context
**Technical Specification v1.4.0**
*March 2026*

## Abstract
GitNexus indexes any codebase into a knowledge graph—tracking every dependency, call chain, cluster, and execution flow. It exposes this structural intelligence via smart tools (MCP) so AI agents (Bio-Nodes) never miss code context.

---

## 1. Core Architecture
GitNexus precomputes relational intelligence at index time, enabling:
* **Impact Analysis:** Blast radius analysis for code changes.
* **Process-Grouped Search:** Tracing execution flows from entry points.
* **360-Degree Context:** Comprehensive symbol views including process participation.
* **Cypher Queries:** Raw graph queries over the codebase schema.

### Tech Stack
* **Runtime:** Node.js (Native) / Browser (WASM).
* **Parsing:** Tree-sitter.
* **Database:** KuzuDB (Embedded graph database with vector support).
* **Embeddings:** Transformers.js.

---

## 2. Smart Tools (MCP)
1. `list_repos`: Discover all indexed repositories.
2. `query`: Hybrid search (BM25 + Semantic + RRF).
3. `context`: Categorized refs and process participation.
4. `impact`: Upstream/Downstream blast radius analysis.
5. `detect_changes`: Mapping diffs to affected execution processes.
6. `rename`: Coordinated multi-file rename using graph intelligence.
7. `cypher`: Direct graph querying.

---

## 3. Arkhe(n) Integration Mapping
| GitNexus Concept | Arkhe(n) Mapping |
|------------------|------------------|
| Knowledge Graph  | Codebase Tzinor   |
| Call Chain       | Causal Trajectory |
| Cluster          | Functional Community |
| Impact Analysis  | Risk Assessment (XenoFirewall) |

---

## 4. Operational Commands
* `gitnexus analyze`: Generates the codebase knowledge graph.
* `gitnexus mcp`: Starts the Model Context Protocol server.
* `gitnexus serve`: Local HTTP server for multi-repo visual exploration.
