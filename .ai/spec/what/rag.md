# RAG (Retrieval-Augmented Generation)

The RAG subsystem augments LLM responses with relevant product documentation so that answers are grounded in authoritative content rather than relying solely on the model's training data. The service retrieves contextually relevant document chunks from pre-built vector indexes and injects them into the LLM prompt before generating a response.

## Behavioral Rules

1. The service must use FAISS vector indexes that are built offline from product documentation. Indexes are loaded from the local filesystem at startup and are never built or modified at runtime.

2. Each index is identified by three properties: a filesystem path to the persisted FAISS vector store (`product_docs_index_path`), an optional index identifier used during deserialization (`product_docs_index_id`), and an optional human-readable origin label used in logging and result metadata (`product_docs_origin`).

3. Multiple indexes may be loaded simultaneously. This enables version-specific documentation (e.g., OCP 4.17, 4.18, 4.19, 4.20), product-specific content (e.g., OpenShift Virtualization, OpenStack, ACM), and customer-supplied documentation (BYOK).

4. Each index must be loaded independently. If one index fails to load, the remaining indexes must continue loading and the service must operate with whatever indexes succeeded. Partial load must be logged as a warning.

5. When a configured index path does not exist, the system must attempt to locate a `latest` directory in the parent of the configured path. If found, the system must use that path and clear the configured `product_docs_index_id` (since the actual index identity is unknown). If neither the configured path nor the `latest` fallback exists, the index must fail to load.

6. The embedding model used for vector similarity is configurable via a filesystem path to a HuggingFace-compatible model (`embeddings_model_path`). If no path is configured, the service must fall back to the default model (sentence-transformers/all-mpnet-base-v2). The chosen model must be redistributable under an Apache 2.0 compatible license. [PLANNED: OLS-1812 -- operator CRD support for per-index embedding model path]

7. Retrieval must use vector similarity search against the loaded FAISS indexes. The number of most-similar document chunks returned per query is controlled by a configurable content limit. Chunks scoring below a configurable similarity cutoff must be discarded, even if they are within the top-k.

8. When multiple indexes are loaded, results from all indexes must be merged into a single ranked list using score dilution. The first index receives no penalty. Subsequent indexes receive a progressively increasing score penalty, capped at a fixed dilution depth. After dilution, all results across all indexes must be sorted by weighted score in descending order and the top-k returned.

9. Each retrieved chunk must be annotated with the `index_id` and `index_origin` metadata from the index it came from. This metadata must flow through to logging, diagnostics, and referenced document output.

10. Retrieved document chunks must be converted to referenced document citations consisting of a URL (`docs_url` metadata) and a title (`title` metadata). These citations must be deduplicated by URL, preserving insertion order. The deduplicated list must be included in the API response so the UI can present citations to the user.

11. Each retrieved chunk must be checked against the available token budget. If a chunk does not fit within the minimum token threshold, it and all subsequent chunks must be skipped. Chunks that fit within the remaining budget may be truncated to fit.

12. RAG dependencies (LlamaIndex, FAISS, HuggingFace embeddings) are heavy. To avoid unnecessary memory overhead when RAG is not configured:
    - RAG library imports must be deferred until first use.
    - If no reference content is configured, RAG libraries must never be loaded.
    - Type annotations for RAG-specific types must be aliased to `Any` at module scope to avoid import-time dependencies.

13. The readiness probe must check whether the RAG index has finished loading. If reference content is configured but the index has not yet loaded, the service must report not ready (HTTP 503) with cause "Index is not ready". If no reference content is configured, the index check must pass (RAG is optional). The service must not accept user queries until the RAG index is fully loaded.

14. **Hybrid RAG is NOT used for document retrieval.** Document retrieval uses pure FAISS vector similarity search as described above. A separate hybrid retrieval system (dense + sparse BM25) exists but is used exclusively for tool filtering and skill selection.

15. The hybrid retrieval system combines dense retrieval (cosine similarity via an in-memory Qdrant vector store) with sparse retrieval (BM25 keyword matching). Scores are fused using a weighted linear combination controlled by a configurable alpha parameter: alpha = 1.0 means pure dense retrieval, alpha = 0.0 means pure sparse retrieval. Results below a configurable similarity threshold must be discarded.

16. The tools hybrid RAG instance must only be created when tool filtering configuration is present and MCP servers are configured. The skills hybrid RAG instance must only be created when skills configuration is present and a skills directory containing valid skill definitions exists.

## Configuration Surface

- `ols_config.reference_content.embeddings_model_path` -- Filesystem path to a HuggingFace-compatible embedding model directory. Falls back to `sentence-transformers/all-mpnet-base-v2` if unset.
- `ols_config.reference_content.indexes[]` -- List of index definitions, each containing:
  - `product_docs_index_path` -- Filesystem path to the persisted FAISS vector store directory.
  - `product_docs_index_id` -- Optional index identifier used during deserialization from the storage context.
  - `product_docs_origin` -- Optional human-readable label for logging and result metadata (e.g., "ocp-4.18", "custom").
- `ols_config.tool_filtering` -- Tool filtering via hybrid RAG (presence enables the feature):
  - `embed_model_path` -- Optional path to sentence transformer model for embeddings.
  - `alpha` -- Weight for dense vs. sparse retrieval (0.0--1.0, default 0.8).
  - `top_k` -- Number of tools to retrieve (1--50, default 10).
  - `threshold` -- Minimum similarity score for results (0.0--1.0, default 0.01).
- `ols_config.skills` -- Skill selection via hybrid RAG (presence enables the feature):
  - `skills_dir` -- Path to directory containing skill subdirectories.
  - `embed_model_path` -- Optional path to sentence transformer model for embeddings.
  - `alpha` -- Weight for dense vs. sparse retrieval (0.0--1.0, default 0.8).
  - `threshold` -- Minimum similarity score to accept a skill match (0.0--1.0, default 0.35).

## Constraints

1. Indexes must be pre-built offline and loaded read-only. The service must never create, modify, or rebuild an index at runtime.

2. The embedding model used for retrieval must be the same model used to create the index. A mismatch will produce meaningless similarity scores.

3. All embedding models shipped with the product must be redistributable under an Apache 2.0 compatible license.

4. The `product_docs_index_id` must not be set without a corresponding `product_docs_index_path`. This combination must be rejected at configuration validation time.

5. Score dilution is applied positionally: the first index in the configuration list is the primary index and receives no penalty. Index ordering in the configuration therefore determines retrieval priority.

6. Referenced document deduplication is by URL only. If two chunks from different indexes share the same URL but different titles, the first-seen title wins.

7. The hybrid RAG system (dense + sparse BM25) must never be used for document retrieval. It is restricted to tool filtering and skill selection.

8. When RAG libraries are lazily loaded, the `index_loader` module is excluded from static type checking (mypy) because its types are only available after the deferred import executes.

## BYOK (Bring Your Own Knowledge)

Customers can supply their own documentation as additional RAG indexes, so that responses incorporate organization-specific knowledge alongside standard product documentation.

**Phase 1 (shipped):** Customers manually import Markdown documentation by pre-building a FAISS index and configuring it as an additional entry in the indexes list.

**Phase 2 (not shipped):** Seamless, one-click import from knowledge sources such as Git repositories and Confluence. [PLANNED: OLS-1872 -- internal web source integration]

## Planned Changes

- [PLANNED: OLS-2903] Propose plans to use OKP-based RAG with OLS.
- [PLANNED: OLS-2704] RAG as a service / MCP -- externalize RAG retrieval behind an MCP interface.
- [PLANNED: OLS-1872] BYOK -- internal web source integration (Git, Confluence).
- [PLANNED: OLS-1812] Add embedding model path to CRD for each index, enabling per-index embedding model configuration through the operator.
