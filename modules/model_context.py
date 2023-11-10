# base python things
import os

# external deps
from llama_index.embeddings import TextEmbeddingsInference
from llama_index import ServiceContext
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials


def get_watsonx_predictor(model, min_new_tokens=1, max_new_tokens=256, **kwargs):
    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    else:
        verbose = False

    api_key = os.getenv("BAM_API_KEY", None)
    api_url = os.getenv("BAM_URL", None)
    creds = Credentials(api_key, api_endpoint=api_url)

    params = GenerateParams(decoding_method="greedy", min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens)

    predictor = LangChainInterface(
        model=model, params=params, credentials=creds, verbose=verbose
    )
    return predictor


def get_watsonx_context(model, url='local', tei_embedding_model = None, **kwargs):

    if url != 'local':
        # MUST set tei_embedding_model to do this
        # TODO: make this appropriately blow up
        embed_model = TextEmbeddingsInference(
            model_name=tei_embedding_model,
            base_url=url,
        )
    else:
        embed_model = "local:BAAI/bge-base-en"

    predictor = get_watsonx_predictor(model)

    service_context = ServiceContext.from_defaults(
        chunk_size=1024, llm=predictor, embed_model=embed_model, **kwargs
    )

    return service_context
