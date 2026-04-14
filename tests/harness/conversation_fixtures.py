"""Canned conversation histories for agent loop testing.

Each fixture returns a ``list[dict]`` of plain message dicts — no LangChain
types — so the harness stays framework-agnostic.  Convert to/from
``BaseMessage`` at the boundary if needed.

Fixtures are sized to exercise specific compaction and context-window
scenarios:

=================  ======  ==========================================
Fixture            Tokens  Purpose
=================  ======  ==========================================
short              ~2 k    Baseline — well under any context window
medium             ~8 k    Moderate history, no compaction expected
bloated_tool       ~50 k   Single oversized tool result triggers
                           Stage 1 truncation
sacred_first_msg   ~12 k   20 turns; first message carries critical
                           system context that must survive compaction
recent_tool        ~6 k    12 turns with tool call/result near the
                           end — split-point must not break the pair
=================  ======  ==========================================
"""

from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_FILLER = (
    "OpenShift Container Platform provides enterprise Kubernetes with "
    "automated operations, consistent security, and developer productivity. "
)


def _pad(text: str, target_words: int) -> str:
    """Repeat filler text until we hit roughly ``target_words`` words."""
    words = text.split()
    while len(words) < target_words:
        words.extend(_FILLER.split())
    return " ".join(words[:target_words])


def _msg(role: str, content: str, **extra: Any) -> dict[str, Any]:
    """Build a plain message dict."""
    m: dict[str, Any] = {"role": role, "content": content}
    m.update(extra)
    return m


# ---------------------------------------------------------------------------
# 1. Short conversation (~2k tokens / ~1.5k words)
# ---------------------------------------------------------------------------
def short_conversation() -> list[dict[str, Any]]:
    """Return a 4-message conversation well under any context window."""
    return [
        _msg("user", "What version of OpenShift am I running?"),
        _msg(
            "assistant",
            "Let me check your cluster version.",
            tool_calls=[
                {"name": "get_cluster_info", "args": {}, "id": "tc-short-1"}
            ],
        ),
        _msg(
            "tool",
            '{"version": "4.15.2", "nodes": 6, "status": "healthy"}',
            tool_call_id="tc-short-1",
        ),
        _msg(
            "assistant",
            "You are running OpenShift Container Platform version 4.15.2. "
            "Your cluster has 6 nodes and is reporting a healthy status.",
        ),
    ]


# ---------------------------------------------------------------------------
# 2. Medium conversation (~8k tokens / ~6k words)
# ---------------------------------------------------------------------------
def medium_conversation() -> list[dict[str, Any]]:
    """Return a 10-message conversation with moderate token count."""
    msgs: list[dict[str, Any]] = []
    topics = [
        ("How do I list pods in a namespace?", "You can use `oc get pods -n <namespace>`."),
        ("What about deployments?", "Use `oc get deployments -n <namespace>`."),
        ("How do I check node status?", "Run `oc get nodes` to see node conditions."),
        ("Can I drain a node safely?", "Yes, use `oc adm drain <node> --ignore-daemonsets`."),
        ("What is a PodDisruptionBudget?", _pad("A PDB limits voluntary disruptions.", 400)),
    ]
    for question, answer in topics:
        msgs.append(_msg("user", question))
        msgs.append(_msg("assistant", answer))
    return msgs


# ---------------------------------------------------------------------------
# 3. Bloated tool result (~50k tokens)
# ---------------------------------------------------------------------------
_HUGE_YAML_LINE = "  - name: nginx-{i}\n    image: nginx:1.25\n    status: Running\n"


def bloated_tool_result() -> list[dict[str, Any]]:
    """Return a conversation containing a single massive tool output.

    The tool result is ~50 000 tokens of YAML pod descriptions, designed to
    trigger Stage 1 truncation in the CompactionService.
    """
    huge_output = "".join(_HUGE_YAML_LINE.format(i=i) for i in range(6000))
    return [
        _msg("user", "Show me all pods across every namespace."),
        _msg(
            "assistant",
            "I'll list all pods cluster-wide.",
            tool_calls=[
                {"name": "list_pods", "args": {"namespace": "all"}, "id": "tc-bloat-1"}
            ],
        ),
        _msg("tool", huge_output, tool_call_id="tc-bloat-1"),
        _msg(
            "assistant",
            "Here is the full pod listing. There are approximately 6000 pods "
            "across all namespaces. Several are in CrashLoopBackOff state.",
        ),
    ]


# ---------------------------------------------------------------------------
# 4. Sacred first message (20 turns, ~12k tokens)
# ---------------------------------------------------------------------------
def sacred_first_message() -> list[dict[str, Any]]:
    """Return a 20-turn conversation where the first message is critical.

    The opening user message contains deployment constraints that the LLM must
    retain even after compaction.  Tests that the compaction split-point
    preserves the first message in the 'keep zone'.
    """
    sacred = (
        "IMPORTANT CONTEXT: I am operating a production OCP 4.15 cluster on "
        "AWS with FIPS mode enabled. All changes must be non-disruptive. "
        "The cluster runs PCI-DSS workloads — no public endpoints, no "
        "privileged containers, and all images must come from the internal "
        "registry at registry.internal.example.com. Compliance scans run "
        "every 4 hours. Do not suggest anything that violates these constraints."
    )
    msgs: list[dict[str, Any]] = [_msg("user", sacred)]

    filler_exchanges = [
        ("How do I check FIPS status?", "Run `oc get cm -n openshift-config`."),
        ("List my machinesets.", "Use `oc get machinesets -n openshift-machine-api`."),
        ("How do I add a worker node?", "Scale the machineset replicas."),
        ("What StorageClasses are available?", "Run `oc get sc`."),
        ("How do I configure an ImageContentSourcePolicy?", "Apply an ICSP manifest."),
        ("Show me the cluster operators.", "Run `oc get co`."),
        ("Any degraded operators?", "Check the DEGRADED column in `oc get co`."),
        ("How do I update the cluster?", "Use `oc adm upgrade --to=<version>`."),
        ("Can I pause machine config updates?", "Yes, pause the MCP."),
    ]
    for question, answer in filler_exchanges:
        msgs.append(_msg("user", question))
        msgs.append(
            _msg("assistant", _pad(answer, 80))
        )

    msgs.append(_msg("user", "Now drain node worker-3 for maintenance."))
    return msgs


# ---------------------------------------------------------------------------
# 5. Recent tool call (12 turns, ~6k tokens)
# ---------------------------------------------------------------------------
def recent_tool_conversation() -> list[dict[str, Any]]:
    """Return a 12-turn conversation with a tool call near the end.

    The tool_calls/tool message pair sits at positions -4/-3, so the
    compaction split-point logic must not break the pair even when the
    messages before it are in the compress zone.
    """
    msgs: list[dict[str, Any]] = []

    early_exchanges = [
        ("What is a DaemonSet?", "A DaemonSet ensures a pod runs on every node."),
        ("How about StatefulSets?", "StatefulSets manage stateful applications."),
        ("Explain PersistentVolumeClaims.", _pad("A PVC requests storage.", 120)),
        ("What networking plugin does OCP use?", "OCP uses OVN-Kubernetes by default."),
    ]
    for question, answer in early_exchanges:
        msgs.append(_msg("user", question))
        msgs.append(_msg("assistant", answer))

    msgs.append(_msg("user", "Scale my nginx deployment to 5 replicas."))
    msgs.append(
        _msg(
            "assistant",
            "I'll scale the nginx deployment for you.",
            tool_calls=[
                {
                    "name": "scale_deployment",
                    "args": {"name": "nginx", "replicas": 5},
                    "id": "tc-recent-1",
                }
            ],
        ),
    )
    msgs.append(
        _msg(
            "tool",
            '{"scaled": "nginx", "replicas": 5, "previous_replicas": 1}',
            tool_call_id="tc-recent-1",
        ),
    )
    msgs.append(
        _msg(
            "assistant",
            "Done — the nginx deployment has been scaled from 1 to 5 replicas.",
        ),
    )

    return msgs


# ---------------------------------------------------------------------------
# Registry — all fixtures keyed by name for parametrized tests
# ---------------------------------------------------------------------------
CONVERSATION_FIXTURES: dict[str, Any] = {
    "short": short_conversation,
    "medium": medium_conversation,
    "bloated_tool": bloated_tool_result,
    "sacred_first_msg": sacred_first_message,
    "recent_tool": recent_tool_conversation,
}
