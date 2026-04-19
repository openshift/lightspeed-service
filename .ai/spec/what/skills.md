# Skills

Skills are guided troubleshooting procedures that get injected into the system prompt when they match a user's query, enabling the LLM to follow domain-specific expert workflows instead of relying solely on its training data.

## Behavioral Rules

### Skill Definition

1. A skill is a directory containing a `skill.md` file with YAML frontmatter. The frontmatter must include a `name` field (string). The `description` field is optional but strongly recommended -- it is the primary text used for matching against user queries. The body of `skill.md` and any additional text files in the directory tree form the skill content.

2. The `skill.md` filename match is case-insensitive (`skill.md`, `SKILL.md`, `Skill.md` are all valid). Only one skill definition file per directory is recognized.

3. Each skill directory may contain any number of supporting files alongside `skill.md`. These files are included in the skill content when the skill is selected.

### Discovery

4. At startup, the system must scan the configured `skills_dir` directory. Each immediate subdirectory is treated as a potential skill. Directories are scanned in sorted order.

5. A subdirectory is registered as a skill only if it contains a valid `skill.md` file with parseable YAML frontmatter that includes a `name` field. Subdirectories without a valid `skill.md` must be silently skipped (logged at debug level).

6. If the configured `skills_dir` does not exist, the system must log a warning and proceed with no skills. This is not a fatal error.

7. If parsing the YAML frontmatter of a `skill.md` file fails, or if the frontmatter lacks a `name` field, the system must log a warning and skip that subdirectory.

8. Discovery indexes each skill's `name` and `description` into the hybrid RAG retrieval system for later matching. The index is populated eagerly at startup, but skill file content is loaded on demand only when selected.

### Selection

9. When a query arrives (see `what/query-processing.md`, Stage 5), the system must match the query against indexed skills using hybrid RAG retrieval (dense embedding similarity + sparse BM25 text matching). Dense and sparse scores are fused using a weighted linear combination controlled by the `skills.alpha` parameter.

10. Only the single best-matching skill is selected per query. There is no multi-skill composition.

11. The match must meet a minimum relevance threshold configured via `skills.threshold` (default 0.35). If no skill exceeds the threshold, no skill is applied and the query proceeds without skill content.

12. The skills threshold (default 0.35) is intentionally higher than the tool filtering threshold (default 0.01), because selecting the wrong skill degrades answer quality more than including an irrelevant tool.

### Token Budget Check

13. After selection, the system must load the skill's content and verify that the total content fits within the available token budget. Skill content is loaded on demand -- not at startup -- so large skill directories do not consume memory until needed.

14. If the skill's token cost exceeds 80% of the remaining token budget (after prompt and RAG), the system must skip the skill, emit a `skill_selected` streaming event with `skipped: true` and the reason, and proceed without skill content.

15. If the skill uses more than 50% but fits within 80% of the remaining budget, the system must log a warning but still use the skill.

16. If loading the skill content fails (filesystem error), the system must fall back to no skill without failing the request.

### File Concatenation Order

17. Skill content is assembled by recursively reading all files in the skill directory tree (`rglob`), sorted alphabetically. The concatenation order is:
    - The body of `skill.md` (everything after YAML frontmatter) is always placed first.
    - All other readable files follow, each prefixed with a Markdown header showing its relative path (e.g., `## subdir/checklist.md`).

18. Files that cannot be read as UTF-8 (binary files, encoding errors) must be silently skipped.

19. The parts are joined with double newlines (`\n\n`) between each section.

### Prompt Injection

20. When a skill is selected and passes the token budget check, the concatenated skill content is injected into the system prompt with the `USE_SKILL_INSTRUCTION` contextual instruction ("Follow the procedure below to address the user's request:"). The skill content is placed immediately after this instruction. See `what/query-processing.md`, Stage 6, for the full prompt assembly order.

21. When a skill is successfully selected, the system must emit a `skill_selected` streaming event with the skill name and confidence score.

### No Skill Selected

22. If no skill meets the threshold, or if skills are not configured (`ols_config.skills` is absent), or if the skills directory is empty or missing, the query proceeds without skill content. This is normal operation, not an error.

## Configuration Surface

- `ols_config.skills` -- Presence of this section enables skill selection. If absent, skills are entirely disabled.
  - `skills.skills_dir` -- Path to the directory containing skill subdirectories. Default: `"skills"`.
  - `skills.embed_model_path` -- Optional path to a sentence transformer model for skill matching embeddings. Falls back to the global RAG embedding model when available, or the default `sentence-transformers/all-mpnet-base-v2` model.
  - `skills.alpha` -- Weight for dense vs. sparse retrieval blending (0.0--1.0, default 0.8). A value of 1.0 means pure dense (semantic) retrieval; 0.0 means pure sparse (keyword BM25) retrieval.
  - `skills.threshold` -- Minimum relevance score to accept a skill match (0.0--1.0, default 0.35).

## Constraints

1. Only one skill may be selected per query. There is no mechanism for combining multiple skills in a single response.

2. Skill directories are read-only at runtime. The system never creates, modifies, or deletes skill files.

3. The embedding model used for skill retrieval at query time must be the same model used to populate the index at startup. There is no mechanism to change the model without restarting the service.

4. Skill content must fit within the available token budget. There is no chunking or summarization of skill content -- it is used in full or not at all.

5. The hybrid retrieval uses an in-memory Qdrant vector store. All indexed skills must fit in memory. The maximum top-k for retrieval is capped at 20, but effectively capped at the number of loaded skills when fewer than 20 exist.

6. Skill selection runs synchronously on the query path. Slow embedding models will add latency to every query when skills are configured.

## Planned Changes

- [PLANNED: OLS-2873] Create end-to-end tests to verify skills in OLS, covering skill discovery, selection, prompt injection, and token budget enforcement.
- [PLANNED: OLS-2874] Enable skills configuration in the OLSConfig CRD, so that skills can be configured via the OpenShift operator rather than only through the config file.
- [PLANNED: OLS-2842] Evaluate skills/tools usage in OLS to measure selection accuracy, false positive rates, and impact on response quality.
