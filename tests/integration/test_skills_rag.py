"""Integration tests for SkillsRAG with a real embedding model.

These tests validate that the hybrid RAG system selects the correct skill
for semantically meaningful queries using a real sentence-transformer model.
"""

import pytest
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ols.src.skills.skills_rag import Skill, SkillsRAG

_MODEL = "sentence-transformers/all-mpnet-base-v2"


@pytest.fixture(scope="module")
def embed_model() -> HuggingFaceEmbedding:
    """Load the embedding model once for all tests in this module."""
    return HuggingFaceEmbedding(model_name=_MODEL)


def _sample_skills() -> list[Skill]:
    """Return the production skill set for testing."""
    return [
        Skill(
            name="pod-failure-diagnosis",
            description=(
                "Troubleshoot CrashLoopBackOff, ImagePullBackOff, Pending, Error, "
                "or OOMKilled status. Use when a workload keeps restarting, fails "
                "to start, or is crash-looping."
            ),
            source_path="skills/pod-failure-diagnosis",
        ),
        Skill(
            name="degraded-operator-recovery",
            description=(
                "Troubleshoot ClusterOperator in Degraded, Unavailable, or not "
                "Progressing state. Use when operator status shows error conditions, "
                "reconciliation failures, or degraded health checks."
            ),
            source_path="skills/degraded-operator-recovery",
        ),
        Skill(
            name="node-not-ready",
            description=(
                "Troubleshoot NotReady or SchedulingDisabled node status. "
                "Use when a node is down, unschedulable, or needs to be "
                "drained and restored."
            ),
            source_path="skills/node-not-ready",
        ),
        Skill(
            name="route-ingress-troubleshooting",
            description=(
                "Troubleshoot Route or Ingress connectivity failures. Use when "
                "traffic returns 502, 503, connection refused, or the endpoint "
                "is not reachable externally."
            ),
            source_path="skills/route-ingress-troubleshooting",
        ),
        Skill(
            name="namespace-troubleshooting",
            description=(
                "Troubleshoot namespace stuck in Terminating state, ResourceQuota "
                "exhaustion, or RBAC permission denied errors. Use when resources "
                "cannot be created or forbidden errors occur."
            ),
            source_path="skills/namespace-troubleshooting",
        ),
    ]


@pytest.fixture(scope="module")
def populated_rag(embed_model: HuggingFaceEmbedding) -> SkillsRAG:
    """Create and populate a SkillsRAG with the real embedding model."""
    rag = SkillsRAG(
        encode_fn=embed_model.get_text_embedding,
        alpha=0.8,
        threshold=0.35,
    )
    rag.populate_skills(_sample_skills())
    return rag


_MIN_CONFIDENCE = 0.35


class TestSkillSelection:
    """Validate that the correct skill is selected for domain-specific queries."""

    @pytest.mark.parametrize(
        ("query", "expected_skill"),
        [
            ("my pod is stuck in CrashLoopBackOff", "pod-failure-diagnosis"),
            ("pod keeps restarting and crashing", "pod-failure-diagnosis"),
            ("container is in ImagePullBackOff state", "pod-failure-diagnosis"),
            (
                "the cluster operator is degraded and unavailable",
                "degraded-operator-recovery",
            ),
            (
                "ClusterOperator authentication is not progressing",
                "degraded-operator-recovery",
            ),
            ("node is in NotReady state", "node-not-ready"),
            ("worker node is unschedulable", "node-not-ready"),
            (
                "application is not reachable through route",
                "route-ingress-troubleshooting",
            ),
            (
                "getting 503 error on my OpenShift route",
                "route-ingress-troubleshooting",
            ),
            ("namespace is stuck in Terminating state", "namespace-troubleshooting"),
            ("resource quota exhausted in my namespace", "namespace-troubleshooting"),
            (
                "permission denied forbidden error in namespace",
                "namespace-troubleshooting",
            ),
        ],
    )
    def test_selects_correct_skill(
        self,
        populated_rag: SkillsRAG,
        query: str,
        expected_skill: str,
    ) -> None:
        """Verify the correct skill is selected with confidence above threshold."""
        skill, score = populated_rag.retrieve_skill(query)
        assert (
            skill is not None
        ), f"Expected '{expected_skill}' but got None for: {query}"
        assert skill.name == expected_skill, (
            f"Expected '{expected_skill}' but got '{skill.name}' "
            f"(score={score:.3f}) for: {query}"
        )
        assert (
            score >= _MIN_CONFIDENCE
        ), f"Score {score:.3f} is below threshold {_MIN_CONFIDENCE} for: {query}"


class TestNoSkillSelected:
    """Validate that no skill is selected for unrelated queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "what is the weather today",
            "how to cook pasta",
            "explain quantum computing",
            "how to create a deployment in OpenShift",
            "configure persistent volume for my application",
            "how to upgrade OpenShift cluster to a new version",
            "set up horizontal pod autoscaler",
            "what are the resource limits for a container",
        ],
    )
    def test_returns_none_for_unrelated_query(
        self,
        populated_rag: SkillsRAG,
        query: str,
    ) -> None:
        """Verify no skill is selected for queries outside the skill domain."""
        skill, score = populated_rag.retrieve_skill(query)
        assert skill is None, (
            f"Expected None but got '{skill.name}' "
            f"(score={score:.3f}) for unrelated query: {query}"
        )
