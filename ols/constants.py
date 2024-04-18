"""Constants used in business logic."""

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
    "I'm sorry, this question does not appear to be about OpenShift or Kubernetes.  "
    "I can only answer questions related to those topics, please rephrase or ask another question."
)
NO_RAG_CONTENT_RESP = (
    "The following response was generated without access to reference content:\n\n"
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


# Token related constants
DEFAULT_CONTEXT_WINDOW_SIZE = 8000
DEFAULT_RESPONSE_TOKEN_LIMIT = 500
MINIMUM_CONTEXT_TOKEN_LIMIT = 1

# Model-specific context window size
# see https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4
# and https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x?topic=models-supported-foundation
CONTEXT_WINDOW_SIZES = {
    GRANITE_13B_CHAT_V1: 8192,
    GRANITE_13B_CHAT_V2: 8192,
    GPT4_TURBO: 128000,
    GPT35_TURBO: 16385,
}

DEFAULT_TOKENIZER_MODEL = "cl100k_base"


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
# Currently we use L2/KNN based FAISS index. And this cutoff signifies distance
# between chunk and query in vector space. Lower distance means query & chunk are
# more similar. So lower cutoff value is better.

# If we set a very high cutoff, then we may end up picking irrelevant chunks.
# And if we set a very low value, then there is risk of discarding all the chunks,
# as there won't be perfect matching chunk. This also depends on embedding model
# used during index creation/retrieval.
# Range: positive float value (can be > 1)
RAG_SIMILARITY_CUTOFF_L2 = 0.75


# cache constants
IN_MEMORY_CACHE = "memory"
IN_MEMORY_CACHE_MAX_ENTRIES = 1000
REDIS_CACHE = "redis"
REDIS_CACHE_HOST = "lightspeed-redis-server.openshift-lightspeed.svc"
REDIS_CACHE_PORT = 6379
REDIS_CACHE_MAX_MEMORY = "1024mb"
REDIS_CACHE_MAX_MEMORY_POLICY = "allkeys-lru"
REDIS_CACHE_MAX_MEMORY_POLICIES = frozenset({"allkeys-lru", "volatile-lru"})
REDIS_RETRY_ON_ERROR = True
REDIS_RETRY_ON_TIMEOUT = True
REDIS_NUMBER_OF_RETRIES = 3
POSTGRES_CACHE = "postgres"
POSTGRES_CACHE_HOST = "localhost"
POSTGRES_CACHE_PORT = 5432
POSTGRES_CACHE_DBNAME = "cache"
POSTGRES_CACHE_USER = "postgres"
POSTGRES_CACHE_MAX_ENTRIES = 1000

# look at https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNECT-SSLMODE
# for all possible options
POSTGRES_CACHE_SSL_MODE = "prefer"


# default indentity for local testing and deployment
DEFAULT_USER_UID = "c1143120-551e-4a47-ad47-2748d6f3c81c"
DEFAULT_USER_NAME = "OLS"
# TO-DO: make this UUID dynamic per cluster (dynamic uuid for each cluster)
DEFAULT_KUBEADMIN_UID = "b6553200-0f7b-4c82-b1c5-9303ff18e5f0"
