"""Constants used in business logic."""

import os
from enum import StrEnum


# Query validation methods
class QueryValidationMethod(StrEnum):
    """Possible options for query validation method."""

    KEYWORD = "keyword"
    LLM = "llm"
    DISABLED = "disabled"


# Query validation responses
SUBJECT_REJECTED = "REJECTED"
SUBJECT_ALLOWED = "ALLOWED"


# Default responses
INVALID_QUERY_RESP = (
    "Hi, I'm the OpenShift Lightspeed assistant, I can help you with questions about OpenShift, "
    "please ask me a question related to OpenShift."
)


# providers
PROVIDER_BAM = "bam"
PROVIDER_OPENAI = "openai"
PROVIDER_AZURE_OPENAI = "azure_openai"
PROVIDER_WATSONX = "watsonx"
SUPPORTED_PROVIDER_TYPES = frozenset(
    {PROVIDER_BAM, PROVIDER_OPENAI, PROVIDER_AZURE_OPENAI, PROVIDER_WATSONX}
)

# models

# BAM
GRANITE_13B_CHAT_V1 = "ibm/granite-13b-chat-v1"
GRANITE_13B_CHAT_V2 = "ibm/granite-13b-chat-v2"
GRANITE_20B_CODE_INSTRUCT_V1 = "ibm/granite-20b-code-instruct-v1"

# OpenAI & Azure OpenAI
GPT35_TURBO_1106 = "gpt-3.5-turbo-1106"
GPT35_TURBO = "gpt-3.5-turbo"

GPT4_TURBO = "gpt-4-turbo"


class GenericLLMParameters:
    """Generic LLM parameters that can be mapped into LLM provider-specific parameters."""

    MIN_NEW_TOKENS = "min_new_tokens"
    MAX_NEW_TOKENS = "max_new_tokens"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TEMPERATURE = "temperature"


# Token related constants
DEFAULT_CONTEXT_WINDOW_SIZE = 8192
DEFAULT_RESPONSE_TOKEN_LIMIT = 512
MINIMUM_CONTEXT_TOKEN_LIMIT = 1

# Provider and Model-specific context window size
# see https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4
# and https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x?topic=models-supported-foundation
CONTEXT_WINDOW_SIZES = {
    PROVIDER_BAM: {
        GRANITE_13B_CHAT_V1: 8192,
        GRANITE_13B_CHAT_V2: 8192,
    },
    PROVIDER_WATSONX: {
        GRANITE_13B_CHAT_V1: 8192,
        GRANITE_13B_CHAT_V2: 8192,
    },
    PROVIDER_AZURE_OPENAI: {
        GPT4_TURBO: 128000,
        GPT35_TURBO: 16384,
    },
    PROVIDER_OPENAI: {
        GPT4_TURBO: 128000,
        GPT35_TURBO: 16384,
    },
}

DEFAULT_TOKENIZER_MODEL = "cl100k_base"

# Example: 1.05 means we increase by 5%.
TOKEN_BUFFER_WEIGHT = 1.1


# RAG related constants

# This is used to decide how many matching chunks we want to retrieve as context.
# (in descending order of similarity between query & chunk)
# Currently we want to fetch best matching chunk, hence the value is set to 1.
# If we want to fetch multiple chunks, then this value will increase accordingly.

# This also depends on chunk_size used during index creation,
# if chunk_size is small, we need to set a higher value, so that we will get
# more context. If chunk_size is more, then we need to set a low value as we may
# end up using too much context. Precise context will get us better response.
RAG_CONTENT_LIMIT = 1

# Once the chunk is retrived we need to check similarity score, so that we won't
# pick any random matching chunk.
# Currently we use Inner product based FAISS index. Higher score means query & chunk
# are more similar. Chunk node score should be greater than the cutoff value.
# If we set a very low cutoff, then we may end up picking irrelevant chunks.
# And if we set a very high value, then there is risk of discarding all the chunks,
# as there won't be perfect matching chunk.
# Range: 0 to 1
RAG_SIMILARITY_CUTOFF = 0.3


# cache constants
CACHE_TYPE_MEMORY = "memory"
IN_MEMORY_CACHE_MAX_ENTRIES = 1000
CACHE_TYPE_REDIS = "redis"
REDIS_CACHE_HOST = "lightspeed-redis-server.openshift-lightspeed.svc"
REDIS_CACHE_PORT = 6379
REDIS_CACHE_MAX_MEMORY = "1024mb"
REDIS_CACHE_MAX_MEMORY_POLICY = "allkeys-lru"
REDIS_CACHE_MAX_MEMORY_POLICIES = frozenset({"allkeys-lru", "volatile-lru"})
REDIS_RETRY_ON_ERROR = True
REDIS_RETRY_ON_TIMEOUT = True
REDIS_NUMBER_OF_RETRIES = 3
CACHE_TYPE_POSTGRES = "postgres"
POSTGRES_CACHE_HOST = "localhost"
POSTGRES_CACHE_PORT = 5432
POSTGRES_CACHE_DBNAME = "cache"
POSTGRES_CACHE_USER = "postgres"
POSTGRES_CACHE_MAX_ENTRIES = 1000

# look at https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNECT-SSLMODE
# for all possible options
POSTGRES_CACHE_SSL_MODE = "prefer"


# default indentity for local testing and deployment
# "nil" UUID is used on purpose, because it will be easier to
# filter these values in CSV export with user feedbacks etc.
DEFAULT_USER_UID = "00000000-0000-0000-0000-000000000000"
DEFAULT_USER_NAME = "OLS"


# HTTP headers to redact from FastAPI HTTP logs
HTTP_REQUEST_HEADERS_TO_REDACT = frozenset(
    {"authorization", "proxy-authorization", "cookie"}
)
HTTP_RESPONSE_HEADERS_TO_REDACT = frozenset(
    {"www-authenticate", "proxy-authenticate", "set-cookie"}
)


# Tells if the code is running in a cluster or not. It depends on
# specific envs that k8s/ocp sets to pod.
RUNNING_IN_CLUSTER = (
    "KUBERNETES_SERVICE_HOST" in os.environ and "KUBERNETES_SERVICE_PORT" in os.environ
)
