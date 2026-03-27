"""Prompt generator based on model / context."""

from typing import Optional

from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from ols.constants import ModelFamily, QueryMode
from ols.src.prompts import prompts


def format_retrieved_chunk(rag_content: str) -> str:
    """Format RAG Docs."""
    return f"Document:\n{rag_content}"


class GeneratePrompt:
    """Generate prompt dynamically."""

    def __init__(
        self,
        query: str,
        rag_context: list[str] = [],
        history: list[BaseMessage] = [],
        system_instruction: str = prompts.QUERY_SYSTEM_INSTRUCTION,
        tool_call: bool = False,
        mode: QueryMode = QueryMode.ASK,
        cluster_version: str = "unknown",
        skill_content: Optional[str] = None,
    ) -> None:
        """Initialize prompt generator."""
        self._query = query
        self._rag_context = rag_context
        self._history = history
        self._sys_instruction = system_instruction
        self._tool_call = tool_call
        self._mode = mode
        self._cluster_version = cluster_version
        self._skill_content = skill_content

    def _get_agent_instructions(self, model: str) -> str:
        """Return agent instructions based on mode and model family."""
        if self._mode == QueryMode.TROUBLESHOOTING:
            return (
                prompts.TROUBLESHOOTING_AGENT_INSTRUCTION.strip()
                + "\n"
                + prompts.TROUBLESHOOTING_AGENT_SYSTEM_INSTRUCTION.strip()
            )

        agent_instructions = prompts.AGENT_INSTRUCTION_GENERIC.strip()
        if ModelFamily.GRANITE in model:
            agent_instructions = prompts.AGENT_INSTRUCTION_GRANITE.strip()
        return agent_instructions + "\n" + prompts.AGENT_SYSTEM_INSTRUCTION.strip()

    def generate_prompt(self, model: str) -> tuple[ChatPromptTemplate, dict]:
        """Generate prompt."""
        prompt_message = []
        sys_intruction = self._sys_instruction.strip()
        llm_input_values: dict = {
            "query": self._query,
            "cluster_version": self._cluster_version,
        }

        if self._tool_call:
            agent_instructions = self._get_agent_instructions(model)
            sys_intruction = sys_intruction + "\n" + agent_instructions

        if len(self._rag_context) > 0:
            llm_input_values["context"] = "\n".join(self._rag_context)
            sys_intruction = (
                sys_intruction + "\n" + prompts.USE_CONTEXT_INSTRUCTION.strip()
            )

        if len(self._history) > 0:
            llm_input_values["chat_history"] = self._history

            sys_intruction = (
                sys_intruction + "\n" + prompts.USE_HISTORY_INSTRUCTION.strip()
            )

        if self._skill_content is not None:
            llm_input_values["skill_content"] = self._skill_content
            sys_intruction = (
                sys_intruction
                + "\n"
                + prompts.USE_SKILL_INSTRUCTION.strip()
                + "\n{skill_content}"
            )

        if "context" in llm_input_values:
            sys_intruction = sys_intruction + "\n{context}"

        prompt_message.append(SystemMessagePromptTemplate.from_template(sys_intruction))

        if "chat_history" in llm_input_values:
            prompt_message.append(MessagesPlaceholder("chat_history"))

        prompt_message.append(HumanMessagePromptTemplate.from_template("{query}"))
        return ChatPromptTemplate.from_messages(prompt_message), llm_input_values
