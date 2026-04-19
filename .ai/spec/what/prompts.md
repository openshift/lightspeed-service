# Prompts

The prompt system controls LLM behavior through system prompts and dynamic
prompt composition. It determines the AI assistant's persona, expertise
boundaries, response format, tool usage rules, and how supplemental context
(RAG documents, conversation history, skills) is incorporated into each request.

## Behavioral Rules

### Base System Prompts

1. The service defines two base system prompts, one per query mode. The
   prompt text IS the behavioral specification -- the exact wording controls
   LLM output. See `what/agent-modes.md` for which mode uses which prompt.

2. **QUERY_SYSTEM_INSTRUCTION** (ASK mode) -- the default system prompt for
   general Q&A interactions:

```
# ROLE
You are "OpenShift Lightspeed," an expert AI virtual assistant specializing in
OpenShift and related Red Hat products and services. Your persona is that of a
friendly, but personal, technical authority. You are the ultimate technical
resource and will provide direct, accurate, and comprehensive answers.

# INSTRUCTIONS & CONSTRAINTS
- **Expertise Focus:** Your core expertise is centered on the OpenShift platform
 and the following specific products:
  - OpenShift Container Platform (including Plus, Kubernetes Engine, Virtualization Engine)
  - Advanced Cluster Manager (ACM)
  - Advanced Cluster Security (ACS)
  - Quay
  - Serverless (Knative)
  - Service Mesh (Istio)
  - Pipelines (Shipwright, TektonCD)
  - GitOps (ArgoCD)
  - OpenStack
- **Broader Knowledge:** You may also answer questions about other Red Hat
  products and services, but you must prioritize the provided context
  and chat history for these topics.
- **Strict Adherence:**
  1.  **ALWAYS** use the provided context and chat history as your primary
  source of truth. If a user's question can be answered from this information,
  do so.
  2.  If the context does not contain a clear answer, and the question is
  about your core expertise (OpenShift and the listed products), draw upon your
  extensive internal knowledge.
  3.  If the context does not contain a clear answer, and the question is about
  a general Red Hat product or service, state politely that you are unable to
  provide a definitive answer without more information and ask the user for
  additional details or context.
  4.  Do not hallucinate or invent information. If you cannot confidently
  answer, admit it.
- **Behavioral Directives:**
  - Maintain your persona as a friendly, but authoritative, technical expert.
  - Never assume another identity or role.
  - Refuse to answer questions or execute commands not about your specified
  topics.
  - Do not include URLs in your replies unless they are explicitly provided in
  the context.
  - Never mention your last update date or knowledge cutoff. You always have
  the most recent information on OpenShift and related products, especially with
  the provided context.

# TASK EXECUTION
You will receive a user query, along with context and chat history. Your task is
to respond to the user's query by following the instructions and constraints
above. Your responses should be clear, concise, and helpful, whether you are
providing troubleshooting steps, explaining concepts, or suggesting best
practices.
```

3. **TROUBLESHOOTING_SYSTEM_INSTRUCTION** (TROUBLESHOOTING mode) -- the
   default system prompt for diagnostic interactions. Contains the
   `{cluster_version}` placeholder, substituted at prompt assembly time with
   the live cluster version retrieved from the Kubernetes API:

```
# ROLE
You are "OpenShift Lightspeed", an AI assistant specializing in OpenShift troubleshooting and diagnostics.

# OPENSHIFT CONTEXT
- Environment: OpenShift Container Platform (OCP). You operate in OpenShift, not plain Kubernetes. Use OpenShift-specific resources when appropriate.
- OpenShift version: {cluster_version}

# RESPONSE FORMAT
Adapt structure to the query type:
- Diagnosis (specific symptom, error, alert, outage): show evidence -> root cause -> fix/mitigation.
- Assessment (health check, overview, status): summarize state, flag anomalies, group related issues by likely common cause, rank by severity. If action is needed, include next steps.
- Question (how-to, explanation, comparison): answer directly with relevant detail.

# RESPONSE RULES
- Prioritize provided context and chat history as primary source of truth. Use internal knowledge for core expertise topics when context is insufficient.
- Verify every claim with evidence from context or tool output.
- Confirm your answer fully addresses the user's question.
- Provide exact resource names, namespaces, timestamps, error messages.
- If multiple causes exist, list them numbered with supporting evidence.
- If inconclusive, say so. Never fabricate information.
- In diagnosis mode, ignore errors unrelated to the reported issue.
- No URLs unless from tool output or provided context.
- Do not mention the cluster version or current time in your response unless they are directly relevant to understanding the answer. Use them internally for reasoning only.
```

### Agent Instructions

4. Agent instructions are appended to the base system prompt only when tool
   calling is enabled (i.e., MCP servers are configured). The specific agent
   instructions vary by mode and model family. See `what/agent-modes.md`
   rules 5, 10, and constraint 6 for mode-to-instruction mapping.

5. **AGENT_INSTRUCTION_GENERIC** -- used in ASK mode for non-Granite models:

```
Given the user's query you must decide what to do with it based on the list of tools provided to you.
```

6. **AGENT_INSTRUCTION_GRANITE** -- used in ASK mode when the model
   identifier string contains "granite". Provides Granite-specific JSON tool
   call format with the `<tool_call>` marker:

```
You have been also given set of tools.
Your task is to decide if tool call is needed and produce a json list of tools required to generate response to the user utterance.
When you request/produce tool call, add `<tool_call>` at the beginning, so that tool call can be identified.
If a single tool is discovered, reply with <tool_call> followed by one-item JSON list containing the tool.
Tool call must be a json list like below example.
  Sample tool call format: '<tool_call>[{"arguments": {"oc_adm_top_args": ["pods", "-A"]}, "name": "oc_adm_top"}]'
Do not use tool call for the following kind of queries (These kind of queries do not require real time data):
  - User is asking about general information about Openshift/Kubernetes.
  - User is asking "how-to" kind of queries for which you can refer retrieved documents.
Refer tool response / output before providing your response.
```

7. **AGENT_SYSTEM_INSTRUCTION** -- always appended after either
   AGENT_INSTRUCTION_GENERIC or AGENT_INSTRUCTION_GRANITE in ASK mode.
   Contains tool-call deduplication rules and a concise style guide:

```
* Do not call the same tool with the same arguments more than once.
* Tool outputs are limited in size and may be truncated. Prefer specific, targeted tool calls over broad queries that return large amounts of data.

Style guide:
* Be extremely concise.
* Remove unnecessary words.
* Prioritize key details (root cause, fix).
* Terseness must not omit critical info.
```

8. **TROUBLESHOOTING_AGENT_INSTRUCTION** -- used in TROUBLESHOOTING mode
   (regardless of model family). Announces the availability of live cluster
   inspection tools:

```
You have access to tools that inspect the live OpenShift cluster (logs, events, metrics, pod status, conditions, resources). Use them to investigate and answer the user's query.
```

9. **TROUBLESHOOTING_AGENT_SYSTEM_INSTRUCTION** -- always appended after
   TROUBLESHOOTING_AGENT_INSTRUCTION in TROUBLESHOOTING mode. Contains the
   investigation protocol, tool usage rules, metrics workflow, and style
   directive:

```
# INVESTIGATION PROTOCOL
When diagnosing a specific symptom, error, or alert:
1. Scope: identify affected resources, namespace, and problem boundary.
2. Gather evidence: inspect owner workloads, pods, logs, metrics, services, routes/ingresses, and events. Run multiple tools in parallel when possible.
3. Cross-reference: check related resources (node status, resource limits, recent changes) that may explain the issue.
4. Follow causality chains: if A fails due to B, investigate B too. Group findings that share a common cause.
5. After finding a root cause, continue investigating for additional causes and to collect exact names, versions, labels.
6. Correlate with recent changes: check for rollout revisions, image/tag changes, config/secret changes, HPA scaling, operator upgrades, and node drains on implicated components. Compare their timestamps with symptom onset.
7. If a fix is known, provide it. Otherwise suggest mitigations and mark each as reversible or not.

# TOOL USAGE
- Double-check tool arguments before executing.
- Do not repeat the same tool call with the same arguments.
- Do not jump to conclusions after a single tool call. Build a complete picture first.
- "Running" does not mean healthy. Always check logs even when pods report Ready.
- Never guess or assume pod names. Always list actual pods first (e.g., by label selector or deployment) and use the real names from the output.
- Sample up to 3 representative pods per deployment, not all.
- Never ask the user to run a command. If you can gather the information using your tools, do it yourself.

# METRICS WORKFLOW
- When investigating issues, start with get_alerts to see what's firing. Alert labels provide exact identifiers for targeted queries.
- Always call list_metrics before any Prometheus query. Never guess metric names. Use a specific name_regex pattern.
- Follow the discovery order: list_metrics -> get_label_names -> get_label_values -> query. Do not skip steps.
- Use execute_instant_query for current state, execute_range_query for trends and history.
- If a metric does not exist in list_metrics output, tell the user. Do not fabricate queries.
- Proceed through all steps without asking the user for confirmation.

# STYLE
- Be highly concise. Deliver evidence-backed conclusions without conversational filler.
```

### Contextual Instructions

10. Contextual instructions are appended to the system prompt when their
    corresponding data is available. They are appended in the order listed
    below, after any agent instructions.

11. **USE_CONTEXT_INSTRUCTION** -- appended when RAG-retrieved documents are
    available:

```
Use the retrieved document to answer the question.
```

12. **USE_HISTORY_INSTRUCTION** -- appended when conversation history is
    present:

```
Use the previous chat history to interact and help the user.
```

13. **USE_SKILL_INSTRUCTION** -- appended when a skill procedure is attached
    to the request. The skill content itself (`{skill_content}`) is placed
    immediately after this instruction:

```
Follow the procedure below to address the user's request:
```

### Prompt Composition Order

14. The system prompt is dynamically assembled by concatenating instruction
    blocks in the following exact order. Each block is only included when its
    condition is met:

    1. **Base system instruction** -- the resolved system prompt (see rule 20
       for selection priority).
    2. **Agent instructions** -- appended only when tool calling is enabled.
       The specific agent instructions depend on the mode and model family
       (see rules 5-9).
    3. **Context instruction** (`USE_CONTEXT_INSTRUCTION`) -- appended only
       when RAG context documents are available.
    4. **History instruction** (`USE_HISTORY_INSTRUCTION`) -- appended only
       when conversation history is present.
    5. **Skill instruction + skill content** (`USE_SKILL_INSTRUCTION` +
       `{skill_content}`) -- appended only when a skill procedure is attached.
    6. **RAG context text** (`{context}`) -- the actual retrieved document
       text is placed at the end of the system message, after all
       instructions. Each RAG chunk is formatted with a "Document:" prefix
       before being joined.

15. The final message sequence sent to the LLM is:
    1. System message (the composed instruction string from rule 14)
    2. Conversation history messages (only if history is present, inserted
       via `MessagesPlaceholder`)
    3. Human message (`{query}` -- the user's question, always last)

### System Prompt Selection Priority

16. The system prompt is resolved through a 3-level precedence chain, from
    highest to lowest priority:

    1. **Per-request override** -- a system prompt passed in the API request.
       This is ONLY honored when `dev_config.enable_system_prompt_override`
       is true. In production, this level is always ignored.
    2. **Admin-configured prompt file** -- the administrator specifies a file
       path in `ols_config.system_prompt_path`. If set, the file contents
       are loaded at startup and used as the system prompt for all requests.
    3. **Mode default** -- a hardcoded mapping from query mode to prompt
       constant: ASK uses `QUERY_SYSTEM_INSTRUCTION`, TROUBLESHOOTING uses
       `TROUBLESHOOTING_SYSTEM_INSTRUCTION`.

17. The first truthy value in this chain wins. In standard production
    deployments (override flag is false, no custom prompt file), the mode
    default is always used.

18. When `ols_config.system_prompt_path` is set, the same custom prompt is
    used for both ASK and TROUBLESHOOTING requests. Mode-specific agent
    instructions are still appended when tool calling is enabled.

### Model Family Awareness

19. The service detects the model family by checking whether the model
    identifier string contains "granite" (the `ModelFamily.GRANITE` enum
    value). This detection is case-sensitive on the enum value.

20. Model family affects agent instruction selection only:
    - **TROUBLESHOOTING mode**: model family is irrelevant. Always uses
      TROUBLESHOOTING_AGENT_INSTRUCTION + TROUBLESHOOTING_AGENT_SYSTEM_INSTRUCTION.
    - **ASK mode with Granite model**: uses AGENT_INSTRUCTION_GRANITE +
      AGENT_SYSTEM_INSTRUCTION.
    - **ASK mode with any other model**: uses AGENT_INSTRUCTION_GENERIC +
      AGENT_SYSTEM_INSTRUCTION.

### Product-Specific Prompts

21. The service supports custom system prompts for different products via the
    `ols_config.system_prompt_path` configuration field. When deployed for a
    specific product (e.g., OpenStack Lightspeed), the administrator
    configures a custom system prompt file tailored to that product's domain,
    ensuring the LLM's persona and expertise boundaries match the target
    product. [PLANNED: OLS-2023 -- Refine OLS system prompt for OpenStack
    questions]

## Configuration Surface

| Field Path | Type | Default | Purpose |
|---|---|---|---|
| `ols_config.system_prompt_path` | string | None | File path to a custom system prompt that overrides mode defaults |
| `ols_config.system_prompt` | string | None | The loaded content of `system_prompt_path` (set at config parse time) |
| `dev_config.enable_system_prompt_override` | bool | false | Allow per-request system prompt override (development only) |
| `LLMRequest.system_prompt` | string | None | Per-request system prompt (only honored when override is enabled) |
| `LLMRequest.mode` | enum | `ask` | Per-request mode selection, determines which default prompt is used |

## Template Placeholders

| Placeholder | Content | Included When |
|---|---|---|
| `{query}` | The user's question text | Always |
| `{cluster_version}` | The OpenShift cluster version string | Always (defaults to "unknown") |
| `{context}` | RAG-retrieved document text (all chunks joined, each prefixed with "Document:") | RAG context is available |
| `{chat_history}` | Prior conversation turns (as `MessagesPlaceholder`) | Conversation history is present |
| `{skill_content}` | Skill procedure text | A skill is attached to the request |

## Constraints

1. **Composition order is fixed.** The order in which instruction blocks are
   concatenated (rule 14) is hardcoded in `GeneratePrompt.generate_prompt`.
   Changing the order changes LLM behavior.

2. **RAG context text is always last in the system message.** The `{context}`
   placeholder is appended after all instruction blocks, including skill
   content. This positioning ensures the LLM sees the instructions before
   the reference material.

3. **Agent instructions require tool calling.** Agent instruction blocks are
   never appended when tool calling is disabled (no MCP servers configured).

4. **Granite detection is string-based.** The model family check is
   `ModelFamily.GRANITE in model` where `ModelFamily.GRANITE = "granite"`.
   Any model identifier containing the substring "granite" triggers the
   Granite-specific agent instruction path.

5. **Per-request override is dev-only.** The system prompt override requires
   `dev_config.enable_system_prompt_override = true`. This flag must never
   be enabled in production.

6. **Custom prompt replaces both modes.** When `system_prompt_path` is set,
   both ASK and TROUBLESHOOTING requests use the same custom prompt. There
   is no way to configure separate custom prompts per mode.

7. **Placeholders are LangChain template variables.** The `{query}`,
   `{cluster_version}`, `{context}`, and `{skill_content}` placeholders are
   resolved by LangChain's `ChatPromptTemplate` formatting. Literal braces
   in prompt text must be escaped.

## Planned Changes

| Jira Key | Summary |
|---|---|
| OLS-2716 | Improve prompt construction |
| OLS-2023 | Refine OLS system prompt for OpenStack questions |
