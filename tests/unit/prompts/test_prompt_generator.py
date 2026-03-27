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

from ols.constants import ModelFamily, QueryMode
from ols.src.prompts import prompts
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
    assert set(prompt.input_variables) <= set(llm_input_values.keys())

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
    assert set(prompt.input_variables) <= set(llm_input_values.keys())

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
    assert set(prompt.input_variables) <= set(llm_input_values.keys())

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
    assert set(prompt.input_variables) <= set(llm_input_values.keys())

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
            "ols.src.prompts.prompts.AGENT_INSTRUCTION_GRANITE",
            agent_instruction_granite,
        ),
        patch(
            "ols.src.prompts.prompts.AGENT_INSTRUCTION_GENERIC",
            agent_instruction_generic,
        ),
        patch(
            "ols.src.prompts.prompts.AGENT_SYSTEM_INSTRUCTION",
            agent_system_instruction,
        ),
    ):
        prompt, llm_input_values = GeneratePrompt(
            query, rag_formatted, conversation_history, system_instruction, True
        ).generate_prompt(model)

    assert set(prompt.input_variables) == {"chat_history", "context", "query"}
    assert set(prompt.input_variables) <= set(llm_input_values.keys())

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


troubleshooting_system_instruction = """
Troubleshooting system instruction.
"""
troubleshooting_agent_instruction = """
Troubleshooting agent instruction.
"""
troubleshooting_agent_system_instruction = """
Troubleshooting agent system instruction.
"""


@pytest.mark.parametrize("model", model)
def test_generate_prompt_troubleshooting_mode_with_tool_call(model):
    """Test that troubleshooting mode selects troubleshooting agent instructions."""
    rag_formatted = [format_retrieved_chunk(text) for text in rag_context]

    with (
        patch(
            "ols.src.prompts.prompts.AGENT_INSTRUCTION_GRANITE",
            agent_instruction_granite,
        ),
        patch(
            "ols.src.prompts.prompts.AGENT_INSTRUCTION_GENERIC",
            agent_instruction_generic,
        ),
        patch(
            "ols.src.prompts.prompts.AGENT_SYSTEM_INSTRUCTION",
            agent_system_instruction,
        ),
        patch(
            "ols.src.prompts.prompts.TROUBLESHOOTING_AGENT_INSTRUCTION",
            troubleshooting_agent_instruction,
        ),
        patch(
            "ols.src.prompts.prompts.TROUBLESHOOTING_AGENT_SYSTEM_INSTRUCTION",
            troubleshooting_agent_system_instruction,
        ),
    ):
        prompt, _llm_input_values = GeneratePrompt(
            query,
            rag_formatted,
            conversation_history,
            troubleshooting_system_instruction,
            True,
            QueryMode.TROUBLESHOOTING,
        ).generate_prompt(model)

    assert set(prompt.input_variables) == {"chat_history", "context", "query"}

    # In troubleshooting mode, agent instructions should come from
    # troubleshooting constants, not the generic/granite ones.
    expected_agent = (
        troubleshooting_agent_instruction.strip()
        + "\n"
        + troubleshooting_agent_system_instruction.strip()
    )

    assert prompt.messages[0].prompt.template == (
        "Troubleshooting system instruction.\n"
        f"{expected_agent}\n"
        "Use the retrieved document to answer the question.\n"
        "Use the previous chat history to interact and help the user.\n"
        "{context}"
    )


@pytest.mark.parametrize("model", model)
def test_generate_prompt_ask_mode_ignores_troubleshooting_agent_instructions(model):
    """Test that ask mode uses generic/granite agent instructions, not troubleshooting."""
    with (
        patch(
            "ols.src.prompts.prompts.AGENT_INSTRUCTION_GENERIC",
            agent_instruction_generic,
        ),
        patch(
            "ols.src.prompts.prompts.AGENT_SYSTEM_INSTRUCTION",
            agent_system_instruction,
        ),
        patch(
            "ols.src.prompts.prompts.TROUBLESHOOTING_AGENT_INSTRUCTION",
            troubleshooting_agent_instruction,
        ),
        patch(
            "ols.src.prompts.prompts.TROUBLESHOOTING_AGENT_SYSTEM_INSTRUCTION",
            troubleshooting_agent_system_instruction,
        ),
    ):
        prompt, _ = GeneratePrompt(
            query, [], [], system_instruction, True, QueryMode.ASK
        ).generate_prompt(model)

    template = prompt.messages[0].prompt.template
    assert "Troubleshooting agent instruction" not in template
    assert "Agent Instruction" in template


@pytest.mark.parametrize("model", model)
def test_generate_prompt_resolves_version_variable(model):
    """Test that {cluster_version} in system instruction is resolved."""
    instruction_with_version = "OpenShift version: {cluster_version}."

    prompt, llm_input_values = GeneratePrompt(
        query,
        [],
        [],
        instruction_with_version,
        cluster_version="4.17.3",
    ).generate_prompt(model)

    assert llm_input_values["cluster_version"] == "4.17.3"

    formatted = prompt.format(**llm_input_values)
    assert "{cluster_version}" not in formatted
    assert "4.17.3" in formatted


@pytest.mark.parametrize("model", model)
def test_generate_prompt_with_skill_content(model):
    """Test prompt includes skill instruction and content when skill_content is provided."""
    skill_body = "Step 1: Check pod status\nStep 2: Review logs"

    prompt, llm_input_values = GeneratePrompt(
        query,
        [],
        [],
        system_instruction,
        skill_content=skill_body,
    ).generate_prompt(model)

    assert set(prompt.input_variables) == {"query", "skill_content"}
    assert llm_input_values["skill_content"] == skill_body

    expected_template = (
        "Answer user queries in the context of openshift.\n"
        + prompts.USE_SKILL_INSTRUCTION.strip()
        + "\n{skill_content}"
    )
    assert prompt.messages[0].prompt.template == expected_template

    formatted = prompt.format(**llm_input_values)
    assert skill_body in formatted
    assert prompts.USE_SKILL_INSTRUCTION.strip() in formatted


@pytest.mark.parametrize("model", model)
def test_generate_prompt_skill_content_none_has_no_effect(model):
    """Test that skill_content=None does not alter the prompt."""
    prompt_with_none, vals_with_none = GeneratePrompt(
        query, [], [], system_instruction, skill_content=None
    ).generate_prompt(model)
    prompt_without, _vals_without = GeneratePrompt(
        query, [], [], system_instruction
    ).generate_prompt(model)

    assert prompt_with_none.input_variables == prompt_without.input_variables
    assert "skill_content" not in vals_with_none
    assert (
        prompt_with_none.messages[0].prompt.template
        == prompt_without.messages[0].prompt.template
    )
