"""Unit tests for PromptGenerator."""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from ols.constants import ModelFamily
from ols.src.prompts.prompt_generator import GeneratePrompt, format_retrieved_chunk

model = ["some-granite-model", "some-gpt-model"]

system_instruction = """
Answer user queries in the context of openshift.
"""
agent_instruction_granite = """
Agent Instruction Granite.
"""
agent_instruction_generic = """
Agent Instruction generic.
"""
agent_system_instruction = """
Agent Instruction default.
"""
query = "What is Kubernetes?"
rag_context = ["context 1", "context 2"]
conversation_history = [
    HumanMessage("First human message"),
    AIMessage("First AI message"),
]


@pytest.mark.parametrize("model", model)
def test_generate_prompt_default_prompt(model):
    """Test if prompt generator returns default prompt for given input."""
    rag_formatted = [format_retrieved_chunk(text) for text in rag_context]

    prompt, llm_input_values = GeneratePrompt(
        query,
        rag_formatted,
        conversation_history,
        system_instruction,
    ).generate_prompt(model)

    assert set(prompt.input_variables) == {"chat_history", "context", "query"}
    assert set(llm_input_values.keys()) == set(prompt.input_variables)

    assert type(prompt) is ChatPromptTemplate
    assert len(prompt.messages) == 3
    assert type(prompt.messages[0]) is SystemMessagePromptTemplate
    assert type(prompt.messages[1]) is MessagesPlaceholder
    assert type(prompt.messages[2]) is HumanMessagePromptTemplate

    assert prompt.messages[0].prompt.template == (
        "Answer user queries in the context of openshift.\n"
        "Use the retrieved document to answer the question.\n"
        "Use the previous chat history to interact and help the user.\n"
        "{context}"
    )
    assert prompt.format(**llm_input_values) == (
        "System: Answer user queries in the context of openshift.\n"
        "Use the retrieved document to answer the question.\n"
        "Use the previous chat history to interact and help the user.\n"
        "Document:\n"
        "context 1\n"
        "Document:\n"
        "context 2\n"
        "Human: First human message\n"
        "AI: First AI message\n"
        f"Human: {query}"
    )


@pytest.mark.parametrize("model", model)
def test_generate_prompt_without_rag_context(model):
    """Test what prompt will be returned for non-existent RAG context."""
    prompt, llm_input_values = GeneratePrompt(
        query,
        [],
        conversation_history,
        system_instruction,
    ).generate_prompt(model)

    assert set(prompt.input_variables) == {"chat_history", "query"}
    assert set(llm_input_values.keys()) == set(prompt.input_variables)

    assert type(prompt) is ChatPromptTemplate
    assert len(prompt.messages) == 3
    assert type(prompt.messages[0]) is SystemMessagePromptTemplate
    assert type(prompt.messages[1]) is MessagesPlaceholder
    assert type(prompt.messages[2]) is HumanMessagePromptTemplate

    assert prompt.messages[0].prompt.template == (
        "Answer user queries in the context of openshift.\n"
        "Use the previous chat history to interact and help the user."
    )
    assert prompt.format(**llm_input_values) == (
        "System: Answer user queries in the context of openshift.\n"
        "Use the previous chat history to interact and help the user.\n"
        "Human: First human message\n"
        "AI: First AI message\n"
        f"Human: {query}"
    )


@pytest.mark.parametrize("model", model)
def test_generate_prompt_without_history(model):
    """Test if prompt generator returns prompt without history."""
    rag_context = ["context 1"]
    rag_formatted = [format_retrieved_chunk(text) for text in rag_context]

    prompt, llm_input_values = GeneratePrompt(
        query,
        rag_formatted,
        [],
        system_instruction,
    ).generate_prompt(model)

    assert set(prompt.input_variables) == {"context", "query"}
    assert set(llm_input_values.keys()) == set(prompt.input_variables)

    assert type(prompt) is ChatPromptTemplate
    assert len(prompt.messages) == 2
    assert type(prompt.messages[0]) is SystemMessagePromptTemplate
    assert type(prompt.messages[1]) is HumanMessagePromptTemplate

    assert prompt.messages[0].prompt.template == (
        "Answer user queries in the context of openshift.\n"
        "Use the retrieved document to answer the question.\n"
        "{context}"
    )
    assert prompt.format(**llm_input_values) == (
        "System: Answer user queries in the context of openshift.\n"
        "Use the retrieved document to answer the question.\n"
        "Document:\n"
        "context 1\n"
        f"Human: {query}"
    )


@pytest.mark.parametrize("model", model)
def test_generate_prompt_without_rag_without_history(model):
    """Test if prompt generator returns prompt without RAG and without history."""
    prompt, llm_input_values = GeneratePrompt(
        query,
        [],
        [],
        system_instruction,
    ).generate_prompt(model)

    assert prompt.input_variables == ["query"]
    assert set(llm_input_values.keys()) == set(prompt.input_variables)

    assert type(prompt) is ChatPromptTemplate
    assert len(prompt.messages) == 2
    assert type(prompt.messages[0]) is SystemMessagePromptTemplate
    assert type(prompt.messages[1]) is HumanMessagePromptTemplate
    assert prompt.messages[0].prompt.template == (
        "Answer user queries in the context of openshift."
    )
    assert prompt.format(**llm_input_values) == (
        "System: Answer user queries in the context of openshift.\n" f"Human: {query}"
    )


@pytest.mark.parametrize("model", model)
def test_generate_prompt_with_tool_call(model):
    """Test prompt when tool call is enabled."""
    rag_formatted = [format_retrieved_chunk(text) for text in rag_context]

    with (
        patch(
            "ols.customize.prompts.AGENT_INSTRUCTION_GRANITE",
            agent_instruction_granite,
        ),
        patch(
            "ols.customize.prompts.AGENT_INSTRUCTION_GENERIC",
            agent_instruction_generic,
        ),
        patch(
            "ols.customize.prompts.AGENT_SYSTEM_INSTRUCTION",
            agent_system_instruction,
        ),
    ):
        prompt, llm_input_values = GeneratePrompt(
            query, rag_formatted, conversation_history, system_instruction, True
        ).generate_prompt(model)

    assert set(prompt.input_variables) == {"chat_history", "context", "query"}
    assert set(llm_input_values.keys()) == set(prompt.input_variables)

    assert type(prompt) is ChatPromptTemplate
    assert len(prompt.messages) == 3
    assert type(prompt.messages[0]) is SystemMessagePromptTemplate
    assert type(prompt.messages[1]) is MessagesPlaceholder
    assert type(prompt.messages[2]) is HumanMessagePromptTemplate

    agent_instruction = agent_instruction_generic.strip()
    if ModelFamily.GRANITE in model:
        agent_instruction = agent_instruction_granite.strip()
    agent_instruction = agent_instruction + "\n" + agent_system_instruction.strip()

    assert prompt.messages[0].prompt.template == (
        "Answer user queries in the context of openshift.\n"
        f"{agent_instruction}\n"
        "Use the retrieved document to answer the question.\n"
        "Use the previous chat history to interact and help the user.\n"
        "{context}"
    )
    assert prompt.format(**llm_input_values) == (
        "System: Answer user queries in the context of openshift.\n"
        f"{agent_instruction}\n"
        "Use the retrieved document to answer the question.\n"
        "Use the previous chat history to interact and help the user.\n"
        "Document:\n"
        "context 1\n"
        "Document:\n"
        "context 2\n"
        "Human: First human message\n"
        "AI: First AI message\n"
        f"Human: {query}"
    )
