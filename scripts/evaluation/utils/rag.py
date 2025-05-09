"""Utility for evaluation."""

from langchain_core.messages import AIMessage
from ols import config
from ols.constants import RAG_CONTENT_LIMIT
from ols.src.prompts.prompt_generator import GeneratePrompt
from ols.utils.token_handler import TokenHandler


def retrieve_rag_chunks(query, model, model_config):
    """Retrieve rag chunks."""
    token_handler = TokenHandler()
    temp_prompt, temp_prompt_input = GeneratePrompt(
        query, ["sample"], [AIMessage(content="sample")]
    ).generate_prompt(model)
    available_tokens = token_handler.calculate_and_check_available_tokens(
        temp_prompt.format(**temp_prompt_input),
        model_config.context_window_size,
        model_config.parameters.max_tokens_for_response,
    )

    retriever = config.rag_index.as_retriever(similarity_top_k=RAG_CONTENT_LIMIT)
    rag_chunks, _ = token_handler.truncate_rag_context(
        retriever.retrieve(query), available_tokens
    )
    return [rag_chunk.text for rag_chunk in rag_chunks]
