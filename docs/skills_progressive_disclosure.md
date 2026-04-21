# Skills Progressive Disclosure in OLS

How OpenShift LightSpeed implements skills with progressive disclosure: metadata
retrieval, prompt injection from `skill.md`, optional on-demand support files via a
bound tool. This document describes **current behavior** in the codebase, not a
future design.

## Three levels of disclosure

### Level 1 — Metadata (always loaded)

Skill frontmatter (`name`, `description`) is indexed by SkillsRAG at startup.
Used for hybrid retrieval (dense + sparse) to match queries to skills.
No content is loaded into the LLM prompt at this stage.

### Level 2 — Skill summary (loaded when skill is selected)

When SkillsRAG selects a skill for a query, the `skill.md` body (everything after
frontmatter) is injected into the LLM prompt via `GeneratePrompt` / `docs_summarizer`.
This contains the overview, triage steps, and quality standards — enough for the
LLM to follow the runbook.

If the skill has support files (auto-discovered when skills are loaded from disk),
OLS appends an auto-generated manifest after the body:

```
---
Available support files (use load_skill_support_files to retrieve):
- basic.json (examples/basic.json)
- escalation.md (templates/escalation.md)
```

Each manifest line lists **display name** and **relative path** (see
`Skill.support_files`).

The manifest is only appended when support files exist. Single-file skills get no
manifest and no support-file tool — minimal overhead.

### Level 3 — Support files (loaded on LLM demand via tool call)

A LangChain `StructuredTool` named `load_skill_support_files` is registered for the
request when the selected skill has support files. The LLM reads the manifest in the
prompt, chooses relative paths, and calls the tool with `files: list[str]`.

The tool is built by `create_skill_support_tool(skill)` in `skills_rag.py`: an
async closure calls `skill.load_support_files(files)` for that `Skill` instance. No
`skill_name` argument is required — the tool is bound to the active skill for the
request.

`docs_summarizer` appends this tool to the same list as MCP tools (`all_mcp_tools`)
after `get_mcp_tools`, so it participates in normal tool execution (e.g.
`LLMExecutionAgent`, approvals, token budgets for tool definitions).

Key properties:

- **Selective loading**: the model requests only paths it needs for the query.
- **Path-based**: arguments are relative paths as shown in the manifest.
- **Resilience**: each requested path is opened under the skill directory; missing
  files or read errors produce an inline `ERROR:` section for that path in the tool
  output (no crash). There is **no** separate allowlist check against
  `support_files`; authors should treat the skill directory as the trust boundary.

**Trust assumption:** Skill trees are internal to OLS (shipped with the product or
otherwise operator-controlled). There is no support today for loading skills from
untrusted external sources; if that changes, path handling in
`load_support_files` should be hardened (normalize paths and enforce containment under
the skill directory).

## Implementation (current code)

### `Skill` and loading (`ols/src/skills/skills_rag.py`)

- **`support_files`**: `list[tuple[str, str]]` — `(filename, relative_path)` for each
  UTF-8 text file under the skill directory (excluding `skill.md`), filled at parse
  time in `_parse_skill_directory` / `_discover_support_files`.

- **`SkillLoadResult`**: `content: str | None`, `has_support_files: bool`, `ok: bool`.
  Returned by **`load_skill()`**. On read/parse failure, logs and returns
  `ok=False`, `content=None` (no exception to the caller).

- **`load_skill()`**: primary path for the prompt — `skill.md` body plus optional
  manifest. Replaces the old “always dump whole tree into the prompt” behavior for
  `docs_summarizer`.

- **`load_support_files(files)`**: reads only the requested relative paths, one
  `## path` header per file, concatenated with blank lines between entries.

- **`load_content()`**: still present — reads the entire skill directory tree into
  one string (useful for tests or tooling that wants everything at once). The
  summarizer’s request path uses **`load_skill()`**, not `load_content()`.

- **`create_skill_support_tool(skill)`**: returns `StructuredTool` via
  `StructuredTool.from_function(coroutine=..., name="load_skill_support_files", ...)`
  so async execution matches `execute_tool_call` / streaming tool execution.

### Integration with `docs_summarizer` (`ols/src/query_helpers/docs_summarizer.py`)

1. **Bindings**: `skill`, `skill_content`, and `has_support_files` are initialized
   before the skills block so later code always has defined locals (including when
   `config.skills_rag` is `None`).

2. **Selection and prompt text**: if `skills_rag` is configured, `retrieve_skill`
   then `load_skill()`; on success, `skill_content` and `has_support_files` are set;
   token budget may clear `skill_content` and emit `SKILL_SELECTED` with
   `skipped: True` if the body is too large.

3. **History and prompt**: conversation history is prepared; `_build_final_prompt`
   receives `skill_content` (may be `None`).

4. **MCP tools**: `get_mcp_tools` runs with a query string that includes
   `skill_content` when present (so discovery can use skill context).

5. **Support tool**: if `skill` is still active, `skill_content` is set, and
   `has_support_files` is true, **`create_skill_support_tool(skill)`** is appended to
   `all_mcp_tools`.

6. **Execution**: `LLMExecutionAgent.execute` runs with the combined tool list; no
   separate code path for “skill tools” beyond registration and naming.

### Skill authoring

Skill authors write `skill.md` (with YAML frontmatter including `name` and
`description`) and add optional files in the same directory (or subdirectories).
OLS discovers UTF-8 text files, builds `support_files`, and when needed the manifest
and tool.

Example structure:

```
skills/
  pod-failure-diagnosis/
    skill.md              ← level 2: summary + triage steps (+ manifest if extras)
    commands.md           ← level 3: detailed oc/kubectl commands
    escalation-matrix.md  ← level 3: escalation procedures
```

Name files descriptively so the manifest helps the model choose which paths to load
via `load_skill_support_files`.
