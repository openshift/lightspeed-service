import os
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from llama_index.llms import LangChainLLM
from llama_index import ServiceContext

from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials


def get_watsonx_predictor(model, min_new_tokens=1):
    api_key = os.getenv("BAM_API_KEY", None)
    api_url = os.getenv("BAM_URL", None)
    creds = Credentials(api_key, api_endpoint=api_url)

    params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: 256,
    }

    predictor = LangChainInterface(
        model=model, params=params, credentials=creds, verbose=True
    )
    return predictor


def get_watsonx_context(model, **kwargs):
    embed_model = "local:BAAI/bge-base-en"
    predictor = get_watsonx_predictor(model)

    service_context = ServiceContext.from_defaults(
        chunk_size=1024, llm=predictor, embed_model=embed_model, **kwargs
    )

    return service_context
