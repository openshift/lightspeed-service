"""Unit tests for PromptGenerator."""

import pytest
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage

from ols.constants import ModelFamily
from ols.src.prompts.prompt_generator import (
    GeneratePrompt,
    restructure_history,
    restructure_rag_context_post,
    restructure_rag_context_pre,
)

model = ["some-granite-model", "some-gpt-model"]

system_instruction = """
Answer user queries in the context of openshift.
"""
query = "What is Kubernetes?"
rag_context = ["context 1", "context 2"]
conversation_history = [
    HumanMessage("First human message"),
    AIMessage("First AI message"),
]


def _restructure_prompt_input(rag_context, conversation_history, model):
    """Restructure prompt input."""
    rag_formatted = [
        restructure_rag_context_post(restructure_rag_context_pre(text, model), model)
        for text in rag_context
    ]
    history_formatted = [
        restructure_history(history, model) for history in conversation_history
    ]
    return rag_formatted, history_formatted


@pytest.mark.parametrize("model", model)
def test_generate_prompt_default_prompt(model):
    """Test if prompt generator returns default prompt for given input."""
    rag_formatted, history_formatted = _restructure_prompt_input(
        rag_context, conversation_history, model
    )

    prompt, llm_input_values = GeneratePrompt(
        query,
        rag_formatted,
        history_formatted,
        system_instruction,
    ).generate_prompt(model)

    assert set(prompt.input_variables) == {"chat_history", "context", "query"}
    assert set(llm_input_values.keys()) == set(prompt.input_variables)

    if ModelFamily.GRANITE in model:
        assert type(prompt) is PromptTemplate
        assert prompt.template == (
            "<|system|>\n"
            "Answer user queries in the context of openshift.\n"
            "\n"
            "Use the retrieved document to answer the question.\n"
            "Use the previous chat history to interact and help the user.\n"
            "{context}\n"
            "{chat_history}\n"
            "<|user|>\n"
            "{query}\n"
            "<|assistant|>"
            "\n"
        )
        assert prompt.format(**llm_input_values) == (
            "<|system|>\n"
            "Answer user queries in the context of openshift.\n"
            "\n"
            "Use the retrieved document to answer the question.\n"
            "Use the previous chat history to interact and help the user.\n"
            "\n"
            "[Document]\n"
            "context 1\n"
            "[End]\n"
            "[Document]\n"
            "context 2\n"
            "[End]\n"
            "\n"
            "<|user|>\n"
            "First human message\n"
            "<|assistant|>\n"
            "First AI message\n"
            "<|user|>\n"
            "What is Kubernetes?\n"
            "<|assistant|>"
            "\n"
        )
    else:
        assert type(prompt) is ChatPromptTemplate
        assert len(prompt.messages) == 3
        assert type(prompt.messages[0]) is SystemMessagePromptTemplate
        assert type(prompt.messages[1]) is MessagesPlaceholder
        assert type(prompt.messages[2]) is HumanMessagePromptTemplate
        assert prompt.messages[0].prompt.template == (
            "Answer user queries in the context of openshift.\n"
            "\n"
            "Use the retrieved document to answer the question.\n"
            "Use the previous chat history to interact and help the user.\n"
            "{context}"
        )
        assert prompt.format(**llm_input_values) == (
            "System: Answer user queries in the context of openshift.\n"
            "\n"
            "Use the retrieved document to answer the question.\n"
            "Use the previous chat history to interact and help the user.\n"
            "\n"
            "Document:\n"
            "context 1\n"
            "\n"
            "Document:\n"
            "context 2\n"
            "\n"
            "Human: First human message\n"
            "AI: First AI message\n"
            f"Human: {query}"
        )


@pytest.mark.parametrize("model", model)
def test_generate_prompt_without_rag_context(model):
    """Test what prompt will be returned for non-existent RAG context."""
    rag_context = []
    rag_formatted, history_formatted = _restructure_prompt_input(
        rag_context, conversation_history, model
    )

    prompt, llm_input_values = GeneratePrompt(
        query,
        rag_formatted,
        history_formatted,
        system_instruction,
    ).generate_prompt(model)

    assert set(prompt.input_variables) == {"chat_history", "query"}
    assert set(llm_input_values.keys()) == set(prompt.input_variables)

    if ModelFamily.GRANITE in model:
        assert type(prompt) is PromptTemplate
        assert prompt.template == (
            "<|system|>\n"
            "Answer user queries in the context of openshift.\n"
            "\n"
            "Use the previous chat history to interact and help the user.\n"
            "{chat_history}\n"
            "<|user|>\n"
            "{query}\n"
            "<|assistant|>"
            "\n"
        )
        assert prompt.format(**llm_input_values) == (
            "<|system|>\n"
            "Answer user queries in the context of openshift.\n"
            "\n"
            "Use the previous chat history to interact and help the user.\n"
            "\n"
            "<|user|>\n"
            "First human message\n"
            "<|assistant|>\n"
            "First AI message\n"
            "<|user|>\n"
            "What is Kubernetes?\n"
            "<|assistant|>"
            "\n"
        )
    else:
        assert type(prompt) is ChatPromptTemplate
        assert len(prompt.messages) == 3
        assert type(prompt.messages[0]) is SystemMessagePromptTemplate
        assert type(prompt.messages[1]) is MessagesPlaceholder
        assert type(prompt.messages[2]) is HumanMessagePromptTemplate
        assert prompt.messages[0].prompt.template == (
            "Answer user queries in the context of openshift.\n"
            "\n"
            "Use the previous chat history to interact and help the user."
        )
        assert prompt.format(**llm_input_values) == (
            "System: Answer user queries in the context of openshift.\n"
            "\n"
            "Use the previous chat history to interact and help the user.\n"
            "Human: First human message\n"
            "AI: First AI message\n"
            f"Human: {query}"
        )


@pytest.mark.parametrize("model", model)
def test_generate_prompt_without_history(model):
    """Test if prompt generator returns prompt without history."""
    conversation_history = []
    rag_context = ["context 1"]

    rag_formatted, history_formatted = _restructure_prompt_input(
        rag_context, conversation_history, model
    )

    prompt, llm_input_values = GeneratePrompt(
        query,
        rag_formatted,
        history_formatted,
        system_instruction,
    ).generate_prompt(model)

    assert set(prompt.input_variables) == {"context", "query"}
    assert set(llm_input_values.keys()) == set(prompt.input_variables)

    if ModelFamily.GRANITE in model:
        assert type(prompt) is PromptTemplate
        assert prompt.template == (
            "<|system|>\n"
            "Answer user queries in the context of openshift.\n"
            "\n"
            "Use the retrieved document to answer the question.\n"
            "{context}\n"
            "<|user|>\n"
            "{query}\n"
            "<|assistant|>"
            "\n"
        )
        assert prompt.format(**llm_input_values) == (
            "<|system|>\n"
            "Answer user queries in the context of openshift.\n"
            "\n"
            "Use the retrieved document to answer the question.\n"
            "\n"
            "[Document]\n"
            "context 1\n"
            "[End]\n"
            "<|user|>\n"
            "What is Kubernetes?\n"
            "<|assistant|>"
            "\n"
        )
    else:
        assert type(prompt) is ChatPromptTemplate
        assert len(prompt.messages) == 2
        assert type(prompt.messages[0]) is SystemMessagePromptTemplate
        assert type(prompt.messages[1]) is HumanMessagePromptTemplate
        assert prompt.messages[0].prompt.template == (
            "Answer user queries in the context of openshift.\n"
            "\n"
            "Use the retrieved document to answer the question.\n"
            "{context}"
        )
        assert prompt.format(**llm_input_values) == (
            "System: Answer user queries in the context of openshift.\n"
            "\n"
            "Use the retrieved document to answer the question.\n"
            "\n"
            "Document:\n"
            "context 1\n"
            "\n"
            f"Human: {query}"
        )


@pytest.mark.parametrize("model", model)
def test_generate_prompt_without_rag_without_history(model):
    """Test if prompt generator returns prompt without RAG and without history."""
    conversation_history = []
    rag_context = []

    rag_formatted, history_formatted = _restructure_prompt_input(
        rag_context, conversation_history, model
    )

    prompt, llm_input_values = GeneratePrompt(
        query,
        rag_formatted,
        history_formatted,
        system_instruction,
    ).generate_prompt(model)

    assert prompt.input_variables == ["query"]
    assert set(llm_input_values.keys()) == set(prompt.input_variables)

    if ModelFamily.GRANITE in model:
        assert type(prompt) is PromptTemplate
        assert prompt.template == (
            "<|system|>\n"
            "Answer user queries in the context of openshift.\n"
            "\n"
            "<|user|>\n"
            "{query}\n"
            "<|assistant|>"
            "\n"
        )
        assert prompt.format(**llm_input_values) == (
            "<|system|>\n"
            "Answer user queries in the context of openshift.\n"
            "\n"
            "<|user|>\n"
            "What is Kubernetes?\n"
            "<|assistant|>"
            "\n"
        )
    else:
        assert type(prompt) is ChatPromptTemplate
        assert len(prompt.messages) == 2
        assert type(prompt.messages[0]) is SystemMessagePromptTemplate
        assert type(prompt.messages[1]) is HumanMessagePromptTemplate
        assert prompt.messages[0].prompt.template == (
            "Answer user queries in the context of openshift.\n"
        )
        assert prompt.format(**llm_input_values) == (
            "System: Answer user queries in the context of openshift.\n"
            "\n"
            f"Human: {query}"
        )
