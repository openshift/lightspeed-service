"""Utility for evaluation."""

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from ols import config
from ols.constants import RAG_CONTENT_LIMIT, GenericLLMParameters
from ols.src.llms.providers.azure_openai import AzureOpenAI
from ols.src.llms.providers.openai import OpenAI
from ols.src.llms.providers.watsonx import Watsonx
from ols.src.prompts.prompt_generator import GeneratePrompt
from ols.utils.token_handler import TokenHandler


class OpenAIVanilla(OpenAI):
    """OpenAI provider."""

    @property
    def default_params(self):
        """Default LLM params."""
        self.url = str(self.provider_config.url)
        self.credentials = self.provider_config.credentials
        # provider-specific configuration has precendence over regular configuration
        if self.provider_config.openai_config is not None:
            openai_config = self.provider_config.openai_config
            self.url = str(openai_config.url)
            if openai_config.api_key is not None:
                self.credentials = openai_config.api_key

        return {
            "base_url": self.url,
            "openai_api_key": self.credentials,
            "model": self.model,
        }


class AzureOpenAIVanilla(AzureOpenAI):
    """Azure OpenAI provider."""

    @property
    def default_params(self):
        """Default LLM params."""
        self.url = str(self.provider_config.url or self.url)
        self.credentials = self.provider_config.credentials
        deployment_name = self.provider_config.deployment_name
        azure_config = self.provider_config.azure_config

        # provider-specific configuration has precendence over regular configuration
        if azure_config is not None:
            self.url = str(azure_config.url)
            deployment_name = azure_config.deployment_name
            if azure_config.api_key is not None:
                self.credentials = azure_config.api_key

        default_parameters = {
            "azure_endpoint": self.url,
            "api_version": "2024-02-15-preview",
            "deployment_name": deployment_name,
            "model": self.model,
        }

        if self.credentials is not None:
            # if credentials with API key is set, use it to call Azure OpenAI endpoints
            default_parameters["api_key"] = self.credentials
        else:
            # credentials for API key is not set -> azure AD token is
            # obtained through azure config parameters (tenant_id,
            # client_id and client_secret)
            access_token = self.resolve_access_token(azure_config)
            default_parameters["azure_ad_token"] = access_token
        return default_parameters


class WatsonxVanilla(Watsonx):
    """Watsonx provider."""

    @property
    def default_params(self):
        """Default LLM params."""
        return {
            GenTextParamsMetaNames.MAX_NEW_TOKENS: 1024,
        }


VANILLA_MODEL = {
    "watsonx": WatsonxVanilla,
    "openai": OpenAIVanilla,
    "azure_openai": AzureOpenAIVanilla,
}

MODEL_OLS_PARAM = {"watsonx": Watsonx, "openai": OpenAI, "azure_openai": AzureOpenAI}

BASIC_PROMPT = """
You are a helpful assistant.
"""


def _retrieve_rag_chunks(query, model, model_config):
    """Retrieve rag chunks."""
    token_handler = TokenHandler()
    temp_prompt, temp_prompt_input = GeneratePrompt(
        query, ["sample"], ["ai: sample"]
    ).generate_prompt(model)
    available_tokens = token_handler.calculate_and_check_available_tokens(
        temp_prompt.format(**temp_prompt_input),
        model_config.context_window_size,
        model_config.parameters.max_tokens_for_response,
    )

    retriever = config.rag_index.as_retriever(similarity_top_k=RAG_CONTENT_LIMIT)
    rag_chunks, _ = token_handler.truncate_rag_context(
        retriever.retrieve(query), model, available_tokens
    )
    return [rag_chunk.text for rag_chunk in rag_chunks]


def get_model_response(query, provider, model, mode, timeout, api_client=None):
    """Get response depending upon the mode."""
    if mode == "ols":
        response = api_client.post(
            "/v1/query",
            json={
                "query": query,
                "provider": provider,
                "model": model,
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            raise Exception(response)
        return response.json()["response"].strip()

    prompt = PromptTemplate.from_template("{query}")
    prompt_input = {"query": query}
    provider_config = config.config.llm_providers.providers[provider]
    model_config = provider_config.models[model]
    llm = VANILLA_MODEL[provider](model, provider_config).load()

    if mode == "ols_param":
        max_resp_tokens = model_config.parameters.max_tokens_for_response
        override_params = {
            GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: max_resp_tokens
        }
        llm = MODEL_OLS_PARAM[provider](model, provider_config, override_params).load()
    if mode == "ols_prompt":
        prompt, prompt_input = GeneratePrompt(query, [], []).generate_prompt(model)
    if mode == "ols_rag":
        rag_chunks = _retrieve_rag_chunks(query, model, model_config)
        prompt, prompt_input = GeneratePrompt(
            query, rag_chunks, [], BASIC_PROMPT
        ).generate_prompt(model)

    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return llm_chain(inputs=prompt_input)["text"].strip()
