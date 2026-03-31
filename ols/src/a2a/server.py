"""A2A server components: Agent Card definition and OLS executor."""

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    Part,
    TextPart,
)

from ols import config, version
from ols.app.models.models import StreamChunkType
from ols.constants import SERVICE_NAME, QueryMode
from ols.src.query_helpers.docs_summarizer import DocsSummarizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

SKILL_ID_ASK = "openshift-ask"
SKILL_ID_TROUBLESHOOTING = "openshift-troubleshooting"

SKILL_ASK = AgentSkill(
    id=SKILL_ID_ASK,
    name="OpenShift and Kubernetes Q&A",
    description=(
        "Answers general questions about OpenShift and Kubernetes, "
        "generates YAML manifests, and explains concepts."
    ),
    tags=["openshift", "kubernetes", "containers", "yaml"],
    examples=[
        "How do I create a deployment in OpenShift?",
        "Generate a NetworkPolicy that allows traffic only from namespace 'frontend'.",
        "What is the difference between a StatefulSet and a Deployment?",
    ],
    input_modes=["text/plain"],
    output_modes=["text/plain"],
)

SKILL_TROUBLESHOOTING = AgentSkill(
    id=SKILL_ID_TROUBLESHOOTING,
    name="OpenShift Cluster Troubleshooting",
    description=(
        "Diagnoses and troubleshoots OpenShift and Kubernetes cluster issues. "
        "Uses cluster version context to provide version-specific guidance."
    ),
    tags=["openshift", "kubernetes", "troubleshooting", "diagnostics"],
    examples=[
        "Why is my pod in CrashLoopBackOff?",
        "My cluster upgrade to 4.15 is stuck, what should I check?",
        "How do I debug ImagePullBackOff errors?",
    ],
    input_modes=["text/plain"],
    output_modes=["text/plain"],
)

SKILL_ID_TO_MODE: dict[str, QueryMode] = {
    SKILL_ID_ASK: QueryMode.ASK,
    SKILL_ID_TROUBLESHOOTING: QueryMode.TROUBLESHOOTING,
}

DEFAULT_MODE = QueryMode.ASK

# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------


def build_agent_card(server_url: str) -> AgentCard:
    """Build an A2A AgentCard describing this OLS instance.

    Args:
        server_url: Base URL where the OLS server is reachable
            (e.g. "https://ols.example.com").

    Returns:
        A fully populated AgentCard.
    """
    return AgentCard(
        name=SERVICE_NAME,
        description=(
            "AI-powered assistant for OpenShift and Kubernetes. "
            "Answers questions, generates YAML, troubleshoots clusters, "
            "and provides guidance based on product documentation."
        ),
        url=server_url,
        version=version.__version__,
        provider=AgentProvider(
            organization="Red Hat",
            url="https://www.redhat.com",
        ),
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            state_transition_history=False,
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[SKILL_ASK, SKILL_TROUBLESHOOTING],
    )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class OLSAgentExecutor(AgentExecutor):
    """Bridge between the A2A protocol and the OLS query pipeline.

    Receives A2A messages, extracts text, runs DocsSummarizer, and
    publishes the response back through the A2A event queue.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the OLS query pipeline for an A2A request."""
        task_id = context.task_id
        context_id = context.context_id
        updater = TaskUpdater(event_queue, task_id, context_id)

        query = context.get_user_input()
        if not query.strip():
            error_msg = updater.new_agent_message(
                [Part(root=TextPart(text="Empty query received."))]
            )
            await updater.failed(message=error_msg)
            return

        skill_id = context.metadata.get("skill_id", "")
        mode = SKILL_ID_TO_MODE.get(skill_id, DEFAULT_MODE)

        logger.info("A2A task %s: query=%r, mode=%s", task_id, query[:200], mode.value)
        await updater.start_work()

        try:
            summarizer = DocsSummarizer(mode=mode, streaming=True)
            response_text = await self._run_summarizer(summarizer, query, context_id)
        except Exception:
            logger.exception("A2A task %s failed", task_id)
            error_msg = updater.new_agent_message(
                [
                    Part(
                        root=TextPart(
                            text="An error occurred while processing your request."
                        )
                    )
                ]
            )
            await updater.failed(message=error_msg)
            return

        await updater.add_artifact(
            [Part(root=TextPart(text=response_text))],
            name="response",
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel an in-progress task."""
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()

    @staticmethod
    async def _run_summarizer(
        summarizer: DocsSummarizer,
        query: str,
        conversation_id: str | None,
    ) -> str:
        """Run the DocsSummarizer and collect the full text response.

        Args:
            summarizer: Configured DocsSummarizer instance.
            query: User query text.
            conversation_id: A2A context_id used as OLS conversation_id.

        Returns:
            The complete response text.
        """
        chunks = [
            chunk.text
            async for chunk in summarizer.generate_response(
                query,
                config.rag_index_loader.get_retriever(),
                conversation_id=conversation_id,
            )
            if chunk.type == StreamChunkType.TEXT
        ]
        return "".join(chunks)
