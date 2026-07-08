# Configuration Model Guide

Read this when adding new configuration fields, new config classes, or modifying `ols/app/models/config.py`.

## Key Facts

- All config classes use Pydantic `BaseModel`
- Config is loaded from YAML via `ProviderConfig(data_dict)` and similar constructors — not from keyword args directly
- Validation errors must raise `checks.InvalidConfigurationError`, not Pydantic's built-in errors
- `extra="forbid"` is used on provider-specific configs — unknown fields cause a hard error
- Credentials are never stored in the YAML directly; they are read from files via `checks.read_secret()`

## Config Class Patterns

### Standard field with validation

```python
class MyConfig(BaseModel):
    name: str
    optional_field: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value:
            raise checks.InvalidConfigurationError("name must not be empty")
        return value
```

### Cross-field validation

```python
@model_validator(mode="after")
def validate_consistency(self) -> Self:
    if self.field_a and not self.field_b:
        raise checks.InvalidConfigurationError("field_b is required when field_a is set")
    return self
```

### Before-mode validator (for input transformation)

```python
@model_validator(mode="before")
@classmethod
def transform_inputs(cls, data: Any) -> Any:
    data["credentials"] = checks.read_secret(
        data, constants.CREDENTIALS_PATH_SELECTOR, constants.API_TOKEN_FILENAME
    )
    return data
```

## Reading Secrets

Credentials come from files, never from the YAML values directly:

```python
# Read API token from a file path specified in config
credentials = checks.read_secret(
    data,                               # the raw config dict
    constants.CREDENTIALS_PATH_SELECTOR,  # key = "credentials_path"
    constants.API_TOKEN_FILENAME,         # filename = "apitoken"
)
```

If `credentials_path` points to a directory, it reads `<directory>/apitoken`. If it points directly to a file, it reads that file. See `ols/utils/checks.py` for the full implementation.

## Adding a New Provider-Specific Config

1. Subclass `ProviderSpecificConfig` (which provides `url`, `token`, `api_key`):

```python
class MyProviderConfig(ProviderSpecificConfig, extra="forbid"):
    credentials_path: str
    my_specific_field: Optional[str] = None
```

2. Add an optional field to `ProviderConfig`:

```python
class ProviderConfig(BaseModel):
    ...
    my_provider_config: Optional[MyProviderConfig] = None
```

3. Add a `case` branch in `ProviderConfig.set_provider_specific_configuration()`:

```python
case constants.PROVIDER_MYPROVIDER:
    my_config = data.get("my_provider_config")
    self.check_provider_config(my_config)
    self.read_api_key(my_config)
    self.my_provider_config = MyProviderConfig(**my_config)
```

4. The config key in YAML follows the pattern `<provider_type>_config` (e.g. `openai_config`, `watsonx_config`).

## Config Validation Flow

Config classes have two validation phases:

1. **`__init__`** — Pydantic field validation + any custom `__init__` logic (reading secrets, setting derived fields)
2. **`validate_yaml()`** — Called explicitly after full config load; used for cross-object validation (e.g. TLS cert file existence checks)

If you add a new config class that needs file existence checks or cross-section validation, add a `validate_yaml()` method and call it from the appropriate parent config's `validate_yaml()`.

### Optional Solr hybrid RAG (`ols_config.solr_hybrid`)

Omit the key to leave Solr hybrid RAG off (no client, no tool). When present, values map to `SolrHybridSettings` and the feature is active. `solr_http_base` must be a valid `http` or `https` URL with a host (checked in `validate_yaml()`).

When ``solr_hybrid`` is defined, every entry under ``reference_content.indexes``
must include ``byok_index: true`` if any local indexes are configured. That rejects
Solr together with an unmarked local vector index (duplicate product RAG); omit
``indexes`` for Solr-only, or mark BYOK indexes explicitly.

```yaml
ols_config:
  solr_hybrid:
    solr_http_base: "https://solr.example.com:8983"
    # max_results, hybrid_*, max_expansion_neighbors: optional overrides
```

Do not duplicate the same product documentation in Solr and a local product index.
You may use ``solr_hybrid`` for product docs and ``reference_content`` for separate
BYOK indexes; each BYOK row must set ``byok_index: true`` when ``solr_hybrid`` is present.

### Hybrid search and chunk expansion

OLS sends a hybrid query to OKP Solr's ``/hybrid-search`` endpoint. The query
performs keyword retrieval first, then reranks candidates by KNN vector
similarity. See the
[OKP RAG Chunk Retrieval Strategy](https://docs.google.com/document/d/1W7G3Tbz5peMAh8cnGcpKvayQAciAyDXzXY_5KXdkY_0/edit?tab=t.0)
for the full specification.

Solr returns raw matched chunks. OLS implements chunk expansion client-side
(all logic in ``SolrHybridSearch`` in ``ols/src/rag_index/solr_support.py``):

1. **Deduplicate by parent** — keep only the first (highest-scored) chunk per
   ``parent_id``. Chunks without a ``parent_id`` are kept as-is.
2. **Cap results** — slice the deduped list to ``max_results``.
3. **Compute per-chunk budget** — divide the tool's ``tools_token_budget``
   (set by the execution framework via ``tool.metadata``) equally across the
   deduped chunks. When the budget is 0, skip expansion entirely and return
   the matched chunk as-is.
4. **Fetch family** — for each matched chunk, query Solr ``/select`` for all
   chunks sharing the same ``parent_id`` AND ``heading_id`` (ordered by
   ``chunk_index``). Orphan chunks (missing ``heading_id``) skip this step.
5. **Expand around match** — starting from the matched chunk, alternate
   between previous and next siblings until one of these limits is hit:
   - per-chunk token budget exhausted (tracked via ``num_tokens``)
   - ``max_expansion_neighbors`` reached on each side (default 2, configurable
     0–10 in ``SolrHybridSettings``; 0 disables expansion)
6. **Concatenate** — assemble the expanded chunks in ``chunk_index`` order,
   strip HTML, and return as the tool result (one content block per deduped
   match).

Relevant Solr fields for expansion:

| Field | Purpose |
|---|---|
| ``parent_id`` | Groups chunks by source document |
| ``chunk_index`` | Sequential ordering for neighbor expansion |
| ``heading_id`` | Family grouping (chunks under the same heading) |
| ``num_tokens`` | Token count per chunk for budget tracking |

Relevant config fields on ``SolrHybridSettings``:

| Field | Default | Purpose |
|---|---|---|
| ``max_results`` | 5 | Max deduped chunks returned |
| ``max_expansion_neighbors`` | 2 | Max siblings per side during expansion (0 disables) |

### OCP version resolution at startup

When ``solr_hybrid`` is configured, ``SolrHybridSearch.__init__`` resolves the
OCP product version for ``chunk_filter_query``:

1. Read the ``OCP_CLUSTER_VERSION`` environment variable (set by the operator).
   If not set, raise ``InvalidConfigurationError`` and stop.
2. Query Solr for available versions of ``openshift_container_platform``
   (facet on ``product_version`` with ``fq=product:openshift_container_platform``).
3. Clamp the requested version to the nearest available:
   - env version < lowest available → use lowest available
   - env version > highest available → use highest available
   - otherwise → use the closest available version ≤ requested
4. Build ``chunk_filter_query``:
   ``is_chunk:true AND product:openshift_container_platform AND product_version:<resolved>``


## Important Constants

Config-related constants live in `ols/constants.py`:

| Constant | Value | Used for |
|---|---|---|
| `CREDENTIALS_PATH_SELECTOR` | `"credentials_path"` | Key for secret file path in config dicts |
| `API_TOKEN_FILENAME` | `"apitoken"` | Default filename inside credentials directory |
| `DEFAULT_CONTEXT_WINDOW_SIZE` | (int) | Default if not set in model config |
| `DEFAULT_MAX_TOKENS_FOR_RESPONSE` | (int) | Default token limit |
| `SUPPORTED_PROVIDER_TYPES` | frozenset | Validated against in `ProviderConfig.set_provider_type()` |
| `SUPPORTED_AUTHENTICATION_MODULES` | frozenset | Validated in `AuthenticationConfig` |
| `tool_round_cap_fraction` | (YAML: `ols_config.tool_round_cap_fraction`) | Global per-round MCP tool budget cap; bounds `TOOL_ROUND_CAP_FRACTION_MIN` / `MAX` in `ols/constants.py` |
