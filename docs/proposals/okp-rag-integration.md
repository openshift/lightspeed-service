# OKP as a RAG content source for OpenShift LightSpeed (OLS)

## Executive summary

OpenShift LightSpeed (OLS) already supports **multiple native FAISS indexes** (`ols_config.reference_content.indexes[]`) for product documentation and **BYOK** corpora. **OKP** is expected to become the **source of truth** for **OpenShift product documentation**, while **BYOK** remains customer-owned.

This document frames the **problem**, compares **native vs MCP-based RAG**, and asks the program to **record an explicit approach decision** among **options A–C**—after spikes and evidence, not by assuming native FAISS first.

**Status:** Working proposal.

**Document map:** **Context** (OLS today) → **Native vs MCP** (industry + structured comparison) → **Corpora & problem** → **Recommendation** (options A–C) → **Implementation artefacts** → **Phasing** → **References** (spec + **code map** for this repo).

---

## Purpose and audience

Single alignment artifact for **engineering and product**: OKP impact on OpenShift docs RAG, distinction from BYOK, retrieval architecture choices, and implementation after an explicit approach decision.

---

## Context: how OLS uses RAG today

OLS augments the model with **retrieval-augmented generation (RAG)** over **pre-built FAISS indexes** loaded at startup. Configuration is a list of indexes under `ols_config.reference_content.indexes[]` (path, optional id, optional origin label). Multiple indexes load at once with **primary vs secondary ordering**, deduplication by URL, and partial load if one index fails (see `.ai/spec/what/rag.md`). **MCP** in this repo is for **tools** (configured MCP servers), not today a replacement for that `reference_content` pipeline. An OKP implementation must choose **Option A**, **B**, or **C** (native for **both** corpora, MCP for **both**, or **hybrid** with one native and one MCP), then define **ordering / grounding policy** and readiness accordingly.

---

## Native RAG vs MCP-based RAG (industry context)

**Definitions (important):**

- **Native / direct RAG** — The application loads or hosts retrieval itself (e.g. in-process vector store, embedded FAISS, fixed corpus on disk). Retrieval is not exposed to the model as an MCP tool unless you explicitly wrap it.
- **MCP-based RAG** — Documentation or knowledge is retrieved through the **Model Context Protocol**: an MCP server exposes search/read tools; the **model** (or agent runtime) chooses when to call them. The corpus may live remotely; latency and auth follow MCP semantics.

Public reporting in **2025–2026** shows **strong MCP momentum** for enterprise agents, tool catalogs, and integrations (e.g. large public MCP server directories, high SDK download volumes, and vendor “enterprise MCP” guides). Those figures mostly describe **MCP as an integration layer for tools and data**, not a standardized survey of “what percentage of production RAG is MCP-native vs embedded.” **There is no widely cited, methodologically transparent market study** that directly measures “Native RAG vs MCP RAG vs hybrid” as deployment categories. Treat percentage bands below as **directional practitioner estimates** for discussion, not audited facts.

| Approach | Directional share (illustrative) | Maturity / notes |
|----------|----------------------------------|------------------|
| Native RAG | ~75–80% | Mature; dominant in systems designed before MCP or with strict latency and determinism needs |
| MCP RAG | ~15–20% | Growing; attractive when knowledge is federated or when vendors ship **MCP servers** instead of custom SDKs |
| Hybrid | ~5% | Emerging: e.g. embedded RAG for core corpus + MCP tools for live systems |

Stereotypical **drivers** (legacy stacks, latency, single-service control vs federated sources, agent frameworks, vendor MCP surfaces) are expanded in the **structured comparison** tables below—avoid duplicating them here.

### Structured comparison: Native RAG vs RAG through MCP

**Verification (read first):** The tables below are **architectural stereotypes** useful for design discussion. They are **not** universal laws: e.g. MCP stacks can **force** retrieval every turn (prompt or router); native stacks can use a **separate model** to decide whether to retrieve; “one round trip” for native still includes embed + search work before the main generate call; MCP “token savings” can disappear if the model issues **many** tool calls or returns **large** tool payloads. Use the rows as **default tradeoffs**, then validate against a concrete OKP + OLS design.

#### Architecture overview

| Dimension | Native RAG | RAG through MCP |
|-----------|------------|-----------------|
| **Control** | Application decides when / what to retrieve (policy, router, or always-on). | Typically the **LLM** chooses whether / when to call a retrieval tool; orchestration can override. |
| **Coupling** | Tight: embedding, chunking, retrieval live in the app or same deployable unit. | Looser: retrieval is an **external** capability behind MCP; contract is tool schemas + JSON-RPC. |
| **Prompt flow** | App retrieves → injects chunks into system / user context → model generates. | Model may generate → **tool call** → tool result becomes message history → model continues. |

<a id="okp-structured-1-retrieval-control"></a>

#### 1. Retrieval control

| Native RAG | RAG through MCP |
|------------|-----------------|
| Developer or service encodes retrieval policy (e.g. always search before answer, or gated by intent classifier). | Model uses reasoning (and tool descriptions) to decide if / how often to retrieve. |
| Strong fit for **deterministic** domains and compliance (“always ground on these docs”). | Strong fit for **ambiguous** or multi-domain questions where retrieval depth should vary. |
| One retrieval pass loads chunks **up front** before generation; more context needs an explicit **app** re-query or another request. | Same turn can **invoke retrieval tools again** (new args, another server, paging) as the model reasons—**incremental** context; see [§3 — Context window management](#okp-structured-3-context-window). |
| Risk: retrieves when unnecessary → noise / cost. | Risk: model **skips** retrieval or **over-calls** → wrong grounding or latency. |

#### 2. Latency and token efficiency

| Aspect | Native RAG | RAG through MCP |
|--------|------------|-----------------|
| **Round trips** | Often **one logical** request path: retrieve then single completion (still: embed query + search + generate). | Typically **2+** LLM-facing steps when a tool round is needed (generate → tool → generate again). |
| **Context use** | May inject fixed top‑k even when irrelevant if policy is “always RAG.” | Retrieves only when the model invokes tools—but tool **arguments** and **results** still consume context. |
| **Overhead** | Embedding + vector search + prompt assembly. | Adds MCP **JSON-RPC**, auth, and tool plumbing on top of retrieval. |

**Rule of thumb:** Native tends to win on **tail latency** for “always ground” Q&A; MCP can win on **avoiding** unused chunks when the model correctly **omits** retrieval—**not** automatically on total tokens (multi-step tool chatter can cost more).

<a id="okp-structured-3-context-window"></a>

#### 3. Context window management

| Native RAG | RAG through MCP |
|------------|-----------------|
| App owns chunking, ranking, truncation into the model budget. | Server may cap or page results; **model** still receives bounded text and must fit the same context window. |
| Static “inject N chunks” pattern is common. | Enables **iterative** “search → read → refine query” flows when the agent loop is built that way. |

#### 4. Multi-source complexity

| Native RAG | RAG through MCP |
|------------|-----------------|
| New corpus often means **code / config / pipeline** changes and unified scoring across indexes. | New corpus often means **attach another MCP server**; integration surface is standardized. |
| One embedding model (or a controlled small set) is typical for merged indexes. | Each server may use **its own** embeddings and schema; the LLM stitches answers across tool results. |

#### 5. Observability and debugging

| Native RAG | RAG through MCP |
|------------|-----------------|
| Retrieval can be deep in app code; needs explicit spans (query, index id, chunk ids). | Tool calls show up in **traces** as named operations: easier to answer “what did the model ask for?” |
| “Why this chunk?” requires custom instrumentation (scores, rank). | Same ranking explainability **inside** the server unless it returns scores; boundary is clearer at the **tool** API. |

#### When to use which (design guide)

| Prefer native RAG when… | Prefer RAG through MCP when… |
|-------------------------|------------------------------|
| Single-domain Q&A with a **stable** corpus (e.g. shipped product docs). | Many **heterogeneous** sources (docs, tickets, APIs) owned by different teams. |
| **Latency** SLOs are tight and retrieval is on every query path. | **Flexibility** of retrieval depth per query matters more than best-case latency. |
| Retrieval policy must be **deterministic** / auditable as app logic. | You mainly **integrate** third-party knowledge products that ship MCP. |
| High volume, cost-sensitive **baseline** grounding. | Rapid iteration on **changing** sources without redeploying the core app. |

---

## The two corpora (problem shorthand)

In practice, operators and customers think in terms of **two** RAG-related surfaces. Today both are usually extra entries in `indexes[]`; under **Option B** (MCP-only for docs), that layout may no longer apply to those corpora:

1. **OpenShift product documentation** — curated, version-aligned OpenShift / OCP content that ships with or alongside the product (today often pulled via the RAG content image / `vector_db` layout described in project docs).
2. **BYOK (bring your own knowledge)** — customer-built indexes (separate images, paths, and lifecycle), used for org-specific runbooks, internal docs, or other knowledge not in the product bundle.

Those two differ in **ownership, build pipeline, embedding contract, release cadence, and support boundary**.

**Why this moment matters for BYOK:** Product docs and BYOK have **both** historically been modeled as **native FAISS** under `reference_content`—BYOK is often its own **image / path / sidecar** story, but the **same** in-process pattern and the **same** “one vector bundle shape” mental model. **OKP-backed product docs sit on Solr**; how that Solr is **fed into OLS** (static export, in-process provider, MCP, …) is what options and spikes decide. The shift is that **FAISS is weaker as a forced unifier** across the two corpora: the **new default to plan for** is **two independent storage and delivery stories**—product docs on whatever OKP and the program agree, BYOK on whatever fits the customer—then **explicit** wiring in OLS (native `indexes[]`, MCP tools, or hybrid) instead of assuming both sides must match one FAISS contract forever.

Any change to where “OpenShift docs” come from must preserve clear behavior for **BYOK** (unchanged expectations unless we explicitly scope a breaking change).

---

## What is changing (problem statement)

**OpenShift documentation** that today feeds the product-docs RAG path is expected to **come from OKP** instead of (or in addition to) the current packaging and delivery mechanism your team is migrating away from.

Beyond swapping a path, the program needs clear answers for:

- **Versioning** — Which OKP content matches this cluster’s OpenShift version (and add-ons)?
- **Delivery** — For OKP-backed product docs, whether OLS consumes them **natively** (`reference_content`) or through **MCP** (Options **A–C**); cadence and packaging follow that decision.
- **Embeddings** — Build-time and query-time models stay aligned (`embeddings_model_path` and existing RAG validation).
- **BYOK together** — OKP product docs plus BYOK `indexes[]` stay predictable (order, primary index, dedup per `.ai/spec/what/rag.md`).

**Unifying store vs two backends (simple view):**

1. **Today** — **FAISS** (same `reference_content` idea for both corpora) is the practical **unifier**: one loading model in OLS; product docs and BYOK both ship as on-disk indexes the service ingests at startup.
2. **Product documentation is in Solr** — OKP-backed OpenShift docs **live in Solr**; OLS still chooses **native vs MCP** to ground on them (Options **A–C**).
3. **BYOK is no longer locked to FAISS** — Product docs are no longer the same static-FAISS story as BYOK, so BYOK can use **other** storage (still FAISS, another vector DB, MCP in front of a service, …) without forcing one shared bundle shape for both corpora.
4. **Why FAISS is a weak default for churny BYOK** — BYOK changes often; each change usually means **rebuilding and shipping** a whole on-disk FAISS artifact into the cluster—**slow and heavy** for operators.
5. **Why look at a standalone DB (or service) for BYOK** — Customer knowledge can follow **incremental update** patterns in a database or hosted search layer instead of **regenerating the entire vector snapshot** on every BYOK change.

---

<a id="recommendation-context-dependent"></a>

## Recommendation

### Primary recommendation: **decide the approach**, then implement

Do **not** treat “ship FAISS like today” as the automatic outcome. **Recommend** that product and engineering **choose Option A, B, or C** below and capture it in an **ADR** (or equivalent) **before** large refactors, including explicit **non-goals**. Base the choice on evidence, for example:

- **SLOs** — tail latency, availability, cold start, cost per query.
- **Grounding policy** — must OpenShift/OKP docs be retrieved **deterministically** for regulated or support workflows?
- **OKP delivery** — Solr is the **source of truth** for OKP-backed docs; which **consumption surface** OLS uses (native export, stack provider, MCP, …) is the architectural choice.
- **BYOK** — embedding and lifecycle mismatch vs product docs; for **A** and the **native** leg of **C**, corpus precedence follows **`indexes[]`** / primary-index rules. For **B** and the **MCP** leg of **C**, precedence is **tool contracts, prompts, and tests**, not list order.

Spikes in [Implementation options](#implementation-options-concrete-artefacts) inform the decision; they do not replace it.

### Option A — Native for **both** OKP and BYOK

**What:** **Both** corpora—OKP-backed OpenShift product docs **and** BYOK—are served as **native** `ols_config.reference_content.indexes[]` entries (FAISS or agreed on-disk shape, `metadata.json` / index id, `embeddings_model_path` per index). Same readiness and partial-load rules as `.ai/spec/what/rag.md` for configured indexes.

**Advantages:**

1. **Preserves the current OLS implementation (mostly)** — `reference_content`, index load, readiness, and multi-index semantics stay on the **same** code path; change is mainly **what** OKP and BYOK **publish** (artifacts and config), not a new doc-retrieval protocol in-process.
2. **Tighter control over RAG population** — **When**, **how much**, and **from which index** context is retrieved is **application policy** (order, top‑k, thresholds, truncation) before text reaches the model—not LLM-initiated MCP tool rounds for those corpora.

**Tradeoffs:** Native multi-index **ordering**, **embedding alignment** across two builders, and **unified ranking** are entirely in your design space.

### Option B — MCP for **both** OKP and BYOK

**What:** **Both** corpora are retrieved **only** through MCP tools—for example [okp-mcp](https://github.com/rhel-lightspeed/okp-mcp) for the OKP knowledge surface and a **custom MCP server** for BYOK. `reference_content` for those slices is absent or unused, depending on product (BYOK may no longer ship as static FAISS in-cluster).

**Advantages:**

1. **MCP-first for all doc grounding** — One **integration pattern** (tools + JSON-RPC) for every knowledge source; easier for agent-style clients and partner surfaces that already speak MCP.
2. **Strong separation per corpus** — OKP and BYOK can use **different** embedding models, chunking, and **release cadences** without forcing a single in-process ranker or shared FAISS contract inside OLS.
3. **Federation instead of a unified native retriever** — Each corpus stays **behind its own server**; OLS does not have to merge heterogeneous indexes in one vector pipeline.
4. **RAG content delivery when required** — The model can **pull additional** grounding in later tool rounds (narrower query, paging, second corpus) instead of committing the full chunk bundle **up front** for every request (see [structured comparison §1](#okp-structured-1-retrieval-control)).

**Tradeoffs:**

- **Tool-mediated retrieval** — The LLM decides when to call each server; you inherit **skips**, **wrong tool choice**, and **extra** round trips as operational risks.
- **Two MCP fleets (minimum)** — OKP and BYOK each need **auth**, **networking**, **SLOs**, and **on-call** story; cost and latency add up versus one native load path.
- **Grounding is policy, not config alone** — “Always use product docs” (and BYOK priority) must be enforced with **prompts**, **forced tools**, or a **router**, then tested like any other requirement.

### Option C — Hybrid (**one** native, **one** MCP)

**What:** **Split transport by corpus**—each knowledge base uses **either** native `reference_content` **or** MCP, not both for the same corpus. Example: **native** OKP/OpenShift product docs and **MCP** for BYOK (or the **reverse** only if product mandates).

**Advantages:**

1. **Determinism and tail latency on one path** — The corpus that must stay **in-process**, **predictable**, and **application-controlled** for grounding can use native `reference_content` without subjecting it to MCP tool choice and round trips.
2. **MCP flexibility on the other** — The second corpus can use MCP when **customer content churns**, **a different team owns delivery**, or **embedding / stack choices** should not be folded into the OLS image and vector pipeline.
3. **Independent RAG engineering on the MCP corpus** — Like Option B’s per-server split, that corpus can use its **own** embedding model, chunking policy, and **release cadence** without forcing both knowledge bases into one shared native multi-index embedding and ranker contract inside OLS.

**Tradeoffs:** Operators and support must run **two** retrieval modes; two failure domains and config surfaces.

---

## Implementation options (concrete artefacts)

The links below are **spike ideas**, not delivery commitments. Work can touch one GitHub repo or several, depending on where retrieval runs:

- **`lightspeed-service`** (this repo) — OLS app: config, `reference_content` index load, query path, MCP client wiring.
- **`lightspeed-providers`** — Llama Stack–style **providers** (e.g. remote Solr vector I/O) used when retrieval is modeled as a stack provider.
- **`lightspeed-stack`** — Broader stack / pipeline work (e.g. cross-encoder rerank PRs) that may sit outside the OLS service process.

Spikes for a given OKP + OLS design might change only **`lightspeed-service`**, or also **`lightspeed-providers`** / **`lightspeed-stack`** if hot retrieval or reranking moves there.

For native RAG code paths already in this repo, see the [Code map](#code-map-lightspeed-service) under References.

### Direct / native-style retrieval (Solr + optional rerank)

| Component | Role | Link |
|-----------|------|------|
| **Solr vector I/O (Llama Stack provider)** | Llama Stack provider for Solr-backed vector retrieval; extend or adapt where the stack needs different auth, schemas, or lifecycle vs OKP Solr. | [lightspeed-providers …/remote/solr_vector_io/solr_vector_io](https://github.com/lightspeed-core/lightspeed-providers/tree/main/lightspeed_stack_providers/providers/remote/solr_vector_io/solr_vector_io) |
| **Cross-encoder reranking** | PR **LCORE-1723** — cross-encoder reranking for enhanced RAG in **lightspeed-stack**; use when a **second-stage ranker** on top of vector hits is desired (scope per merged PR; validate against your retrieval pipeline). | [lightspeed-core/lightspeed-stack#1566](https://github.com/lightspeed-core/lightspeed-stack/pull/1566) |

**Relation to OLS today:** `lightspeed-service` currently loads **pre-built FAISS** indexes under `reference_content` (see `.ai/spec/what/rag.md`). A **direct/Solr** path implies either feeding that pipeline from Solr-backed exports, or moving **hot retrieval** into the stack provider path above—**gap analysis** belongs in the spike work that informs implementation.

### MCP-based retrieval

| Component | Role | Link |
|-----------|------|------|
| **okp-mcp** | MCP server for the **Red Hat Offline Knowledge Portal (OKP)**; bridges tool calls to **OKP Solr** (upstream README: RHEL docs, CVEs, errata, solutions, articles). Transports and `MCP_SOLR_URL` are configurable. **Scope:** confirm with PM how **OpenShift product docs** align with this **RHEL-centric** portal corpus (subset, overlap, or separate OKP surface). | [rhel-lightspeed/okp-mcp](https://github.com/rhel-lightspeed/okp-mcp) |
| **BYOK MCP (custom)** | No first-party BYOK MCP is assumed here; implement a dedicated MCP server (or servers) that expose search/read tools over **customer** knowledge with the same operational concerns as any other MCP (auth, rate limits, observability). OLS still needs an agreed **tool contract** (names, args, chunk shape) for whatever server you ship. | *To be created / specified* |

**Illustrative ecosystem patterns (not exhaustive, not product endorsements):** Public and sample RAG MCP servers use different tool surfaces; spikes and BYOK MCP design should expect variation, not a single canonical tool list.

| Style / source | Example tools | Notes |
|----------------|---------------|-------|
| Mindset AI | `rag_search` (mandatory), `verify_document_access` | Enterprise-oriented; auth headers |
| Knowledge Base MCP | `list_knowledge_bases`, `retrieve_knowledge` | FAISS-oriented; auto-indexing flows |
| Tutorial-style servers | `rag_retrieve`, `rag_docs`, `search` | Minimal single-purpose tools |
| Semantic Search MCP | `search`, `set_index_path`, `build_index` | Code-search–oriented workflows |
| Hugging Face Hub MCP | `search_datasets`, `search_models`, `find_similar_*` | Hub discovery and similarity helpers |

**Note:** Solr-backed **direct** path and **okp-mcp** both assume Solr-shaped OKP indexes; pick transport and auth per spike. **Option A** uses native indexes only; **B/C** add MCP. Second-stage **rerank** (e.g. cross-encoder in lightspeed-stack) is a **retrieval pipeline** choice, independent of **MCP vs native `reference_content`** transport for a corpus.

---

## Phasing

1. **Decide the approach** — Pick **Option A, B, or C** (see [Recommendation](#recommendation-context-dependent)), record it (e.g. ADR), and align OKP delivery assumptions with that choice.
2. **Implement** — Execute against the **contracts and artefacts** OKP and the program actually provide (indexes, MCP servers, operator wiring, embeddings); this proposal’s implementation links are inputs to that work, not a substitute for those resources.

---

## References

### Specification

`.ai/spec/what/rag.md` — multi-index rules, primary vs secondary ordering, partial load, readiness, dedup by URL. **Hybrid RAG** there means **tool / skill selection**, not `reference_content` document retrieval.

<a id="code-map-lightspeed-service"></a>

### Code map (`lightspeed-service`)

Native `reference_content` vs MCP **tools** (separate config):

| Area | Path | Role |
|------|------|------|
| **Config models** | `ols/app/models/config.py` | `ReferenceContentIndex`, `ReferenceContent` (`indexes[]`, `embeddings_model_path`); **`MCPServerConfig`** / `OLSConfig.mcp_servers` — MCP **tools**, not the FAISS index pipeline. |
| **Index load** | `ols/src/rag_index/index_loader.py` | `IndexLoader`: loads each `product_docs_index_path`, builds retriever; `get_retriever()` used by the query path. |
| **App wiring** | `ols/utils/config.py` | Lazy `rag_index_loader` built from `ols_config.reference_content`; exposes `rag_index` / embedding model. |
| **Query path** | `ols/app/endpoints/ols.py` | `generate_response` passes `config.rag_index_loader.get_retriever()` into `DocsSummarizer` (streaming and non-streaming). |
| **Retrieve + truncate** | `ols/src/query_helpers/docs_summarizer.py` | `rag_retriever.retrieve(query)` and RAG context assembly / token charging for injected chunks. |
| **Readiness** | `ols/app/endpoints/health.py` | RAG readiness when `reference_content` is set but the index is not ready (per spec). |
| **Diagnostics** | `ols/src/config_status/config_status.py` | Surfaces loaded RAG index ids for status / debugging. |
| **Hybrid (not doc RAG)** | `ols/src/rag/hybrid_rag.py` | Dense + sparse retrieval for **tool filtering / skills**; do not conflate with OKP doc ingestion unless design explicitly merges concerns. |

MCP runtime: `ols/src/mcp/`; server validation: `ols/utils/checks.py` (e.g. `validate_mcp_servers`). OKP-on-MCP spikes extend that graph, not `IndexLoader` alone.

### Docs and external links

- **`README.md`** — RAG content image, `vector_db` / `embeddings_model`, BYOK flow, example `ols_config.reference_content.indexes` YAML.
- [lightspeed-providers — `solr_vector_io`](https://github.com/lightspeed-core/lightspeed-providers/tree/main/lightspeed_stack_providers/providers/remote/solr_vector_io/solr_vector_io)
- [lightspeed-stack PR 1566 — cross-encoder reranking (LCORE-1723)](https://github.com/lightspeed-core/lightspeed-stack/pull/1566)
- [rhel-lightspeed / okp-mcp](https://github.com/rhel-lightspeed/okp-mcp)
