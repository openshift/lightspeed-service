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
POSSIBLE_QUESTION_VALIDATOR_RESPONSES = (
    SUBJECT_REJECTED,
    SUBJECT_ALLOWED,
)


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

# Model configs
DEFAULT_CONTEXT_WINDOW_SIZE = 8000
DEFAULT_RESPONSE_TOKEN_LIMIT = 500


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


# default indentity for local testing and deployment
DEFAULT_USER_UID = "c1143120-551e-4a47-ad47-2748d6f3c81c"
DEFAULT_USER_NAME = "OLS"
# TO-DO: make this UUID dynamic per cluster (dynamic uuid for each cluster)
DEFAULT_KUBEADMIN_UID = "b6553200-0f7b-4c82-b1c5-9303ff18e5f0"
