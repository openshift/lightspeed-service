# RAG (Retrieval-Augmented Generation)

The RAG subsystem augments LLM responses with relevant documentation. OpenShift product documentation is retrieved through RHOKP's dedicated MCP search tool; customer-supplied content is retrieved from local FAISS indexes (BYOK). Hybrid RAG is also used to filter generic MCP tools and select skills.

## Behavioral Rules — OpenShift Documentation (RHOKP MCP)

1. When `ols_config.openshift_docs_mcp` is configured, OLS must expose a built-in documentation tool named `search` with a `query` argument. The dedicated RHOKP endpoint is separate from generic `mcp_servers` configuration.
2. The configured RHOKP URL must use HTTPS. `max_results` defaults to 5, is limited to 1–20, and is passed to RHOKP as the `rows` argument.
3. OLS must derive product/version filters in service code rather than asking the LLM to choose them. `OCP_CLUSTER_VERSION` supplies the major.minor version for `openshift_container_platform`; when `OLS_ROSA_PRODUCT` is set, OLS adds that product with the same major.minor version.
4. If `OCP_CLUSTER_VERSION` is missing or does not match the supported version format, documentation search must fail closed without calling RHOKP.
5. OLS must discover RHOKP's native `search` tool before exposing the local documentation tool. Discovery retries up to 36 times with a five-second backoff and uses a 60-second timeout for each Streamable HTTP request. A successful tool discovery is cached by endpoint for the process lifetime. If all attempts fail on the runner startup path, startup fails; if discovery is deferred to first-request tool resolution, that request fails.
6. In the normal runner path, RHOKP discovery occurs before Uvicorn starts. If startup takes the Pyroscope path that starts Uvicorn before `load_index()`, tool resolution must perform the same discovery on the first request before exposing `search`; subsequent requests use the cache.
7. When the dedicated docs MCP is configured, `search` is reserved for RHOKP. Generic MCP tools with that name must be omitted from the request's tool list to avoid duplicate tool names; other generic MCP tools remain available.
8. RHOKP results may identify source URLs as `online_source_url`. OLS must provide a `docs_url` alias to the LLM so the existing citation prompt contract remains valid, and must include source URLs and titles in the API's referenced-document metadata.
9. If an RHOKP search invocation fails after initialization, OLS must return a structured tool error. Product/version filters must not be omitted to recover from the failure.
10. When the docs MCP is configured, tool calling is enabled even if no generic MCP servers are configured. Ask-mode prompt guidance directs the LLM to ground relevant answers on retrieved passages; the BYOK variant prioritizes relevant customer context and uses product docs as supplementary information.

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

1. The hybrid retrieval system combines dense retrieval (cosine similarity via an in-memory Qdrant vector store) with sparse retrieval (BM25 keyword matching). Scores are fused using a weighted linear combination controlled by a configurable alpha parameter: alpha = 1.0 means pure dense retrieval, alpha = 0.0 means pure sparse retrieval. Results below a configurable similarity threshold must be discarded.
2. The tools hybrid RAG instance must only be created when tool filtering configuration is present and generic MCP servers are configured. The skills hybrid RAG instance must only be created when skills configuration is present and a skills directory containing valid skill definitions exists.

## Configuration Surface

### OpenShift Documentation (RHOKP MCP)

- `ols_config.openshift_docs_mcp.url` — HTTPS endpoint for the dedicated RHOKP MCP server.
- `ols_config.openshift_docs_mcp.max_results` — Maximum number of results sent as `rows` (default 5, range 1–20).
- `OCP_CLUSTER_VERSION` — Cluster version used to derive the OpenShift product filter.
- `OLS_ROSA_PRODUCT` — Optional ROSA product name added to the product filter.

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
7. OLS must not load or package a RHOKP documentation embedding model; RHOKP owns the embedding and retrieval implementation.
8. When RAG libraries are lazily loaded, the `index_loader` module is excluded from static type checking (mypy) because its types are only available after the deferred import executes.

## BYOK (Bring Your Own Knowledge)

Customers can supply their own documentation as additional RAG indexes, so that responses incorporate organization-specific knowledge alongside standard product documentation.

**Phase 1 (shipped):** Customers manually import Markdown documentation by pre-building a FAISS index and configuring it as an additional entry in the indexes list.

**Phase 2 (not shipped):** Seamless, one-click import from knowledge sources such as Git repositories and Confluence. [PLANNED: OLS-1872 — internal web source integration]

## Planned Changes

- [PLANNED: OLS-1872] BYOK — internal web source integration (Git, Confluence).
- [PLANNED: OLS-1812] Add embedding model path to CRD for each index, enabling per-index embedding model configuration through the operator.
