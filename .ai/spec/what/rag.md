# RAG (Retrieval-Augmented Generation)

The RAG subsystem augments LLM responses with relevant documentation so that answers are grounded in authoritative content rather than relying solely on the model's training data. Two retrieval architectures coexist: **OKP** retrieves OCP product documentation via a tool-based flow (Solr hybrid search through the `search_openshift_documentation` LangChain tool, invoked by the LLM during the tool-calling loop), while **BYOK** retrieves customer-supplied content via direct FAISS RAG (chunks merged into prompt context before generation).

## Behavioral Rules — OKP Retrieval (OCP Product Documentation)

1. OCP product documentation is retrieved via the `search_openshift_documentation` LangChain tool. The LLM decides when to invoke it during the tool-calling loop. This is tool-based retrieval, not direct RAG — passages are not merged into prompt context automatically.

2. The tool connects to an RHOKP sidecar (Solr HTTP on localhost:8080) deployed by the operator alongside the service pod. The sidecar is a self-contained single-image appliance containing Solr and the OCP documentation corpus.

3. Query normalization: before search, the query is normalized — stop words removed, IP/CIDR addresses and NIC names stripped, hyphenated terms quoted for stable Solr matching.

4. The tool embeds the normalized query using `ibm-granite/granite-embedding-30m-english` and submits a hybrid-search POST to Solr's `portal-rag` collection. The search uses lexical edismax as the primary query with KNN vector reranking. [PENDING: Ask OKP team if server-side embedding is supported — preferred approach that would eliminate the granite model from the service image.]

5. Results are deduped by parent document (`parent_id` or `id`), filtered by configurable score threshold, and capped at `max_results`.

6. The tool returns JSON passages with `text`, `score`, `title`, and `docs_url`. The LLM uses these to compose a grounded answer with citations.

7. When the OKP tool is active, supplementary prompt guidance (`SOLR_DOCS_TOOL_SUPPLEMENT`) is appended to agent instructions (ASK mode) directing the LLM to ground answers on retrieved passages and cite sources.

8. Solr failures degrade gracefully — the tool returns empty results or a structured error, and the request continues without OKP passages. The service never fails a user request due to Solr errors.

9. Tool calling is enabled when Solr hybrid is configured, even without MCP servers. This ensures OKP retrieval works in non-agentic deployments.

10. [OLS-1894] When the `OLS_ROSA_PRODUCT` environment variable is set (by the operator on ROSA clusters), the Solr `chunk_filter_query` includes the ROSA product alongside `openshift_container_platform`. The filter becomes a compound OR: `(product:openshift_container_platform AND product_version:<ocp_resolved>) OR (product:<rosa_product> AND product_version:<rosa_resolved>)`. ROSA product version resolution uses the same facet-query + clamp-to-nearest mechanism as OCP: extract the major version from `OCP_CLUSTER_VERSION`, query Solr for available versions of the ROSA product, and clamp to nearest. When `OLS_ROSA_PRODUCT` is absent, the filter is OCP-only (no change from current behavior). If the ROSA product is not found in Solr, the service logs a warning and falls back to OCP-only filtering.

## Behavioral Rules — BYOK Retrieval (Customer Content)

1. BYOK FAISS vector indexes are built offline from customer-supplied Markdown documentation. Indexes are loaded from the local filesystem at startup and are never built or modified at runtime.

2. Each index is identified by three properties: a filesystem path to the persisted FAISS vector store (`product_docs_index_path`), an optional index identifier used during deserialization (`product_docs_index_id`), and an optional human-readable origin label used in logging and result metadata (`product_docs_origin`).

3. Multiple indexes may be loaded simultaneously. This enables customer-specific content separation (e.g., internal runbooks, product-specific docs, organization knowledge bases).

4. Each index must be loaded independently. If one index fails to load, the remaining indexes must continue loading and the service must operate with whatever indexes succeeded. Partial load must be logged as a warning.

5. When a configured index path does not exist, validation must fail immediately with a hard configuration error. There is no fallback logic — BYOK users supply their own paths and are responsible for ensuring they exist.

6. The embedding model used for BYOK vector similarity is configurable via a filesystem path to a HuggingFace-compatible model (`embeddings_model_path`). If no path is configured, the service must fall back to the default model (`sentence-transformers/all-mpnet-base-v2`), which is bundled directly in the service image. The chosen model must be redistributable under an Apache 2.0 compatible license. [PLANNED: OLS-1812 — operator CRD support for per-index embedding model path]

7. Retrieval must use vector similarity search against the loaded FAISS indexes. The number of most-similar document chunks returned per query is controlled by a configurable content limit. Chunks scoring below a configurable similarity cutoff must be discarded, even if they are within the top-k.

8. When multiple indexes are loaded, results from all indexes must be merged into a single ranked list using score dilution. The first index receives no penalty. Subsequent indexes receive a progressively increasing score penalty, capped at a fixed dilution depth. After dilution, all results across all indexes must be sorted by weighted score in descending order and the top-k returned.

9. Each retrieved chunk must be annotated with the `index_id` and `index_origin` metadata from the index it came from. This metadata must flow through to logging, diagnostics, and referenced document output.

10. Retrieved document chunks must be converted to referenced document citations consisting of a URL (`docs_url` metadata) and a title (`title` metadata). These citations must be deduplicated by URL, preserving insertion order. The deduplicated list must be included in the API response so the UI can present citations to the user.

11. Each retrieved chunk must be checked against the available token budget. If a chunk does not fit within the minimum token threshold, it and all subsequent chunks must be skipped. Chunks that fit within the remaining budget may be truncated to fit.

12. RAG dependencies (LlamaIndex, FAISS, HuggingFace embeddings) are heavy. To avoid unnecessary memory overhead when RAG is not configured:
    - RAG library imports must be deferred until first use.
    - If no reference content is configured, RAG libraries must never be loaded.
    - Type annotations for RAG-specific types must be aliased to `Any` at module scope to avoid import-time dependencies.

13. The readiness probe must check whether the BYOK index has finished loading. If reference content is configured with a non-empty indexes list but the index has not yet loaded, the service must report not ready (HTTP 503) with cause "Index is not ready". If no reference content is configured, or if reference content is present but has no indexes (empty list or None), the index check must pass — BYOK RAG is optional. The service must not accept user queries until any configured BYOK index is fully loaded.

## Behavioral Rules — Tool & Skill Filtering (Hybrid RAG)

15. The hybrid retrieval system combines dense retrieval (cosine similarity via an in-memory Qdrant vector store) with sparse retrieval (BM25 keyword matching). Scores are fused using a weighted linear combination controlled by a configurable alpha parameter: alpha = 1.0 means pure dense retrieval, alpha = 0.0 means pure sparse retrieval. Results below a configurable similarity threshold must be discarded.

16. The tools hybrid RAG instance must only be created when tool filtering configuration is present and MCP servers are configured. The skills hybrid RAG instance must only be created when skills configuration is present and a skills directory containing valid skill definitions exists.

## Configuration Surface

### OKP (Solr Hybrid)

- `ols_config.solr_hybrid.solr_http_base` — Solr HTTP base URL (default `http://localhost:8080`, operator-generated).
- `ols_config.solr_hybrid.max_results` — Max passages returned after parent dedup (default `RAG_CONTENT_LIMIT`).
- `ols_config.solr_hybrid.chunk_filter_query` — Optional Solr `fq` for product/content filtering.
- `ols_config.solr_hybrid.hybrid_vector_boost` — Vector weight in reranker (default 8.0).
- `ols_config.solr_hybrid.hybrid_pool_docs` — `reRankDocs` pool size (default 100).
- `ols_config.solr_hybrid.hybrid_score_threshold` — Drop low-score hits (default 0.0).
- `ols_config.solr_hybrid.hybrid_solr_timeout_s` — HTTP timeout in seconds (default 60).

### BYOK (FAISS)

- `ols_config.reference_content.embeddings_model_path` — Filesystem path to a HuggingFace-compatible embedding model directory. Falls back to `sentence-transformers/all-mpnet-base-v2` if unset.
- `ols_config.reference_content.indexes[]` — List of index definitions, each containing:
  - `product_docs_index_path` — Filesystem path to the persisted FAISS vector store directory.
  - `product_docs_index_id` — Optional index identifier used during deserialization from the storage context.
  - `product_docs_origin` — Optional human-readable label for logging and result metadata (e.g., "custom").

### Tool & Skill Filtering

- `ols_config.tool_filtering` — Tool filtering via hybrid RAG (presence enables the feature):
  - `embed_model_path` — Optional path to sentence transformer model for embeddings.
  - `alpha` — Weight for dense vs. sparse retrieval (0.0–1.0, default 0.8).
  - `top_k` — Number of tools to retrieve (1–50, default 10).
  - `threshold` — Minimum similarity score for results (0.0–1.0, default 0.01).
- `ols_config.skills` — Skill selection via hybrid RAG (presence enables the feature):
  - `skills_dir` — Path to directory containing skill subdirectories.
  - `embed_model_path` — Optional path to sentence transformer model for embeddings.
  - `alpha` — Weight for dense vs. sparse retrieval (0.0–1.0, default 0.8).
  - `threshold` — Minimum similarity score to accept a skill match (0.0–1.0, default 0.35).

## Constraints

1. BYOK indexes must be pre-built offline and loaded read-only. The service must never create, modify, or rebuild an index at runtime.

2. The BYOK embedding model used for retrieval must be the same model used to create the index. A mismatch will produce meaningless similarity scores.

3. All embedding models shipped with the product must be redistributable under an Apache 2.0 compatible license.

4. The `product_docs_index_id` must not be set without a corresponding `product_docs_index_path`. This combination must be rejected at configuration validation time.

5. Score dilution is applied positionally: the first index in the configuration list is the primary index and receives no penalty. Index ordering in the configuration therefore determines retrieval priority.

6. Referenced document deduplication is by URL only. If two chunks from different indexes share the same URL but different titles, the first-seen title wins.

7. OKP configuration (`solr_hybrid`) is operator-generated and not user-facing. It is always present when the RHOKP sidecar is deployed.

8. When RAG libraries are lazily loaded, the `index_loader` module is excluded from static type checking (mypy) because its types are only available after the deferred import executes.

## BYOK (Bring Your Own Knowledge)

Customers can supply their own documentation as additional RAG indexes, so that responses incorporate organization-specific knowledge alongside standard product documentation.

**Phase 1 (shipped):** Customers manually import Markdown documentation by pre-building a FAISS index and configuring it as an additional entry in the indexes list.

**Phase 2 (not shipped):** Seamless, one-click import from knowledge sources such as Git repositories and Confluence. [PLANNED: OLS-1872 — internal web source integration]

## Planned Changes

- [DONE: OLS-1894] ROSA-aware OKP retrieval — service reads `OLS_ROSA_PRODUCT` env var and builds compound Solr filter. Operator-side detection pending.
- [PLANNED: OLS-2704] RAG as a service / MCP — externalize RAG retrieval behind an MCP interface.
- [PLANNED: OLS-1872] BYOK — internal web source integration (Git, Confluence).
- [PLANNED: OLS-1812] Add embedding model path to CRD for each index, enabling per-index embedding model configuration through the operator.
- [PENDING] OKP server-side embedding — if OKP team confirms, the granite model can be removed from the service image.
- [PLANNED] Multi-product OKP filtering — product-scoped retrieval for OpenStack and other layered products.
- [PLANNED] Multi-version OKP support — query specific OCP version documentation.
