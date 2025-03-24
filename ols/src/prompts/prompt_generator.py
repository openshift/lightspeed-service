"""Prompt generator based on model / context."""

from ast import literal_eval
from json import dumps

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ols.constants import PROVIDER_WATSONX, ModelFamily

from .prompts import (
    AGENT_INSTRUCTION_GRANITE,
    AGENT_SYSTEM_INSTRUCTION,
    QUERY_SYSTEM_INSTRUCTION,
    USE_CONTEXT_INSTRUCTION,
    USE_HISTORY_INSTRUCTION,
)
SYSTEM_PROMPT_S = """
You are a helpful assistant with access to the following function calls. 
Your task is to produce a list of function calls necessary to generate response to the user utterance.
Use tools only if it is required. 
Execute as many tools as required to find out correct answer.
Use the following function calls as required.
"""

PROMPT_TAGS = {
    ModelFamily.GRANITE: {
        "role_prefix": "<|start_of_role|>",
        "role_suffix": "<|end_of_role|>",
        "msg_suffix": "<|end_of_text|>",
    }
}


def format_retrieved_chunk(rag_content: str, n_order=int) -> str:
    """Format RAG Docs."""
    # return f"Document {n_order + 1}:\n{rag_content}"
    return f"[Document]\n{rag_content}[END]"


def format_prompt_message_granite(message):
    """Format prompt message for granite."""
    prompt_tags = PROMPT_TAGS[ModelFamily.GRANITE]
    role_prefix = prompt_tags["role_prefix"]
    role_suffix = prompt_tags["role_suffix"]
    msg_suffix = prompt_tags["msg_suffix"]

    role = message.type
    content = message.content

    msg = role_prefix
    if role == "system":
        msg += "system" + role_suffix + content
    elif role == "human":
        msg += "user" + role_suffix + content
    elif role == "ai":
        msg += "assistant" + role_suffix
        if content.startswith("<tool_call>"):
            content = content.lstrip("<tool_call>")
            msg += "<tool_call>" + dumps(literal_eval(content), indent=4)
        msg += content
    elif role == "tools":
        # Tools input definition
        msg += "tools" + role_suffix + content
    elif role == "tool":
        # Tools output
        msg += "tool_response" + role_suffix + content
    else:
        raise Exception("Invalid message role")

    return msg + msg_suffix


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

    def _generate_prompt(self, provider, model) -> list:
        """Generate prompt."""
        prompt_messages = []
        sys_intruction = self._sys_instruction.strip()

        if self._rag_context:
            sys_intruction = sys_intruction + "\n" + USE_CONTEXT_INSTRUCTION.strip()

        if self._history:
            sys_intruction = sys_intruction + "\n" + USE_HISTORY_INSTRUCTION.strip()
        if self._tool_call:
            agent_instructions = AGENT_SYSTEM_INSTRUCTION.strip()
            if provider == PROVIDER_WATSONX and ModelFamily.GRANITE in model:
                agent_instructions = (
                    AGENT_INSTRUCTION_GRANITE.strip() + "\n" + agent_instructions
                    # AGENT_INSTRUCTION_GRANITE.strip()
                )
            sys_intruction = sys_intruction + "\n" + agent_instructions
        # prompt_messages.append(SystemMessage(sys_intruction))

        if self._rag_context:
            # sys_intruction = "Retrieved Context:"
            sys_intruction = sys_intruction + "\n" + "\n".join(self._rag_context)
            # prompt_messages.append(HumanMessage(sys_intruction))
            # print("\n".join(
            #     [
            #         format_retrieved_chunk(chunk, idx)
            #         for idx, chunk in enumerate(self._rag_context)
            #     ]
            # ))
        # if self._tool_call:
        #     agent_instructions = AGENT_SYSTEM_INSTRUCTION.strip()
        #     if provider == PROVIDER_WATSONX and ModelFamily.GRANITE in model:
        #         agent_instructions = (
        #             # AGENT_INSTRUCTION_GRANITE.strip() + "\n" + agent_instructions
        #             agent_instructions + "\n" + AGENT_INSTRUCTION_GRANITE.strip()
        #         )
        #     sys_intruction = sys_intruction + "\n" + agent_instructions
        # TODO: Use generic template
        # prompt_messages.append(SystemMessage(SYSTEM_PROMPT_S))
        prompt_messages.append(SystemMessage(sys_intruction))

        prompt_messages.extend(self._history)

        prompt_messages.append(HumanMessage(self._query))
        return prompt_messages

    def generate_prompt(self, provider: str, model: str) -> list | str:
        """Generate prompt."""
        prompt_message = self._generate_prompt(provider, model)
        # if provider == PROVIDER_WATSONX and ModelFamily.GRANITE in model:
            # from langchain_core.messages import convert_to_openai_messages
            # prompt_message = convert_to_openai_messages(prompt_message)
            # new_msg = []
            # for i in prompt_message:
            #     if i.type == "human":
            #         i.type = "user"
            #     if i.type == "ai":
            #         i.type = "assistant"

            # prompt_message = "\n".join(
            #     format_prompt_message_granite(message) for message in prompt_message
            # )
            # prompt_tags = PROMPT_TAGS[ModelFamily.GRANITE]
            # role_prefix = prompt_tags["role_prefix"]
            # role_suffix = prompt_tags["role_suffix"]
            # prompt_message += "\n" + role_prefix + "assistant" + role_suffix

        return prompt_message
