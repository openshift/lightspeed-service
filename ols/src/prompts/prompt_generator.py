"""Prompt generator based on model / context."""

from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from ols.constants import ModelFamily

from .prompts import (
    AGENT_INSTRUCTION_GENERIC,
    AGENT_INSTRUCTION_GRANITE,
    AGENT_SYSTEM_INSTRUCTION,
    QUERY_SYSTEM_INSTRUCTION,
    USE_CONTEXT_INSTRUCTION,
    USE_HISTORY_INSTRUCTION,
)


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
        system_instruction: str = QUERY_SYSTEM_INSTRUCTION,
        tool_call: bool = False,
    ):
        """Initialize prompt generator."""
        self._query = query
        self._rag_context = rag_context
        self._history = history
        self._sys_instruction = system_instruction
        self._tool_call = tool_call

    def generate_prompt(self, model: str) -> tuple[ChatPromptTemplate, dict]:
        """Generate prompt."""
        prompt_message = []
        sys_intruction = self._sys_instruction.strip()
        llm_input_values: dict = {"query": self._query}

        if self._tool_call:
            agent_instructions = AGENT_INSTRUCTION_GENERIC.strip()
            if ModelFamily.GRANITE in model:
                agent_instructions = AGENT_INSTRUCTION_GRANITE.strip()
            agent_instructions = (
                agent_instructions + "\n" + AGENT_SYSTEM_INSTRUCTION.strip()
            )
            sys_intruction = sys_intruction + "\n" + agent_instructions

        if len(self._rag_context) > 0:
            llm_input_values["context"] = "\n".join(self._rag_context)
            sys_intruction = sys_intruction + "\n" + USE_CONTEXT_INSTRUCTION.strip()

        if len(self._history) > 0:
            llm_input_values["chat_history"] = self._history

            sys_intruction = sys_intruction + "\n" + USE_HISTORY_INSTRUCTION.strip()

        if "context" in llm_input_values:
            sys_intruction = sys_intruction + "\n{context}"

        prompt_message.append(SystemMessagePromptTemplate.from_template(sys_intruction))

        if "chat_history" in llm_input_values:
            prompt_message.append(MessagesPlaceholder("chat_history"))

        prompt_message.append(HumanMessagePromptTemplate.from_template("{query}"))
        return ChatPromptTemplate.from_messages(prompt_message), llm_input_values
