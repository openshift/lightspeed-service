"""Constants used in business logic."""

import os
import ssl
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


# providers
PROVIDER_BAM = "bam"
PROVIDER_OPENAI = "openai"
PROVIDER_AZURE_OPENAI = "azure_openai"
PROVIDER_WATSONX = "watsonx"
PROVIDER_RHOAI_VLLM = "rhoai_vllm"
PROVIDER_RHELAI_VLLM = "rhelai_vllm"
PROVIDER_FAKE = "fake_provider"
SUPPORTED_PROVIDER_TYPES = frozenset(
    {
        PROVIDER_BAM,
        PROVIDER_OPENAI,
        PROVIDER_AZURE_OPENAI,
        PROVIDER_WATSONX,
        PROVIDER_RHOAI_VLLM,
        PROVIDER_RHELAI_VLLM,
        PROVIDER_FAKE,
    }
)
DEFAULT_AZURE_API_VERSION = "2024-02-15-preview"


# models
class ModelFamily(StrEnum):
    """Different LLM models family/group."""

    GPT = "gpt"
    GRANITE = "granite"


class GenericLLMParameters:
    """Generic LLM parameters that can be mapped into LLM provider-specific parameters."""

    MIN_TOKENS_FOR_RESPONSE = "min_tokens_for_response"
    MAX_TOKENS_FOR_RESPONSE = "max_tokens_for_response"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TEMPERATURE = "temperature"


# Max Iteration for tool calling
MAX_ITERATIONS = 5


# Token related constants

# It is important to set correct context window, otherwise there will be potential
# error due to context window limit or unnecessary truncation may happen.
# For Provider and Model-specific context window size, Please refer
# their official documentations. If not set, default value will be used.
DEFAULT_CONTEXT_WINDOW_SIZE = 128000

# Max tokens reserved for response may also vary depending upon provider/model & query.
# Ex: Larger models tends give more descriptive response,
# Also response with YAML generally uses more tokens (when large model is used)
# It should be in reasonable proportion to context window limit; otherwise unnecessary
# truncation will happen. If not set, default value will be used.
DEFAULT_MAX_TOKENS_FOR_RESPONSE = 4096


# Tokenizer model to generate tokens (for an approximated token calculation)
DEFAULT_TOKENIZER_MODEL = "cl100k_base"

# Example: 1.05 means we increase by 5%.
TOKEN_BUFFER_WEIGHT = 1.1

# Tool output token limits
# Maximum tokens for a single tool output before truncation
DEFAULT_MAX_TOKENS_PER_TOOL_OUTPUT = 8000
# Total tokens reserved for all tool outputs (only used when MCP servers configured)
DEFAULT_MAX_TOKENS_FOR_TOOLS = 32000


# RAG related constants

# Minimum tokens to have some meaningful RAG context including special tags.
MINIMUM_CONTEXT_TOKEN_LIMIT = 10

# This is used to decide how many matching chunks we want to retrieve as context.
# (in descending order of similarity between query & chunk)

# This also depends on chunk_size used during index creation,
# if chunk_size is small, we need to set a higher value, so that we will get
# more context. If chunk_size is more, then we need to set a low value as we may
# end up using too much context. Precise context will get us better response.
RAG_CONTENT_LIMIT = 5

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
CACHE_TYPE_POSTGRES = "postgres"
POSTGRES_CACHE_HOST = "localhost"
POSTGRES_CACHE_PORT = 5432
POSTGRES_CACHE_DBNAME = "cache"
POSTGRES_CACHE_USER = "postgres"
POSTGRES_CACHE_MAX_ENTRIES = 1000

# look at https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNECT-SSLMODE
# for all possible options
POSTGRES_CACHE_SSL_MODE = "prefer"

# look at https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNECT-GSSENCMODE
# for all possible options
POSTGRES_CACHE_GSSENCMODE = "prefer"


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

# Supported attachment types
ATTACHMENT_TYPES = frozenset(
    {
        "alert",
        "api object",
        "configuration",
        "error message",
        "event",
        "log",
        "stack trace",
    }
)

# Supported attachment content types
ATTACHMENT_CONTENT_TYPES = frozenset(
    {"text/plain", "application/json", "application/yaml", "application/xml"}
)

# Default name of file containing API token
API_TOKEN_FILENAME = "apitoken"  # noqa: S105  # nosec: B105

# Default name of file containing client ID to Azure OpenAI
AZURE_CLIENT_ID_FILENAME = "client_id"

# Default name of file containing tenant ID to Azure OpenAI
AZURE_TENANT_ID_FILENAME = "tenant_id"

# Default name of file containing client secret to Azure OpenAI
AZURE_CLIENT_SECRET_FILENAME = "client_secret"  # noqa: S105  # nosec: B105

# Selectors for fields from configuration structure
CREDENTIALS_PATH_SELECTOR = "credentials_path"

# Default directory where standard and extra certificates will be stored
DEFAULT_CERTIFICATE_DIRECTORY = "/tmp"  # noqa: S108

# Certificate storage filename
CERTIFICATE_STORAGE_FILENAME = "ols.pem"

# Default SSL version used by FastAPI REST API
DEFAULT_SSL_VERSION = ssl.PROTOCOL_TLS_SERVER

# Default SSL ciphers used by FastAPI REST API
DEFAULT_SSL_CIPHERS = "TLSv1"

# Default authentication module
DEFAULT_AUTHENTICATION_MODULE = "k8s"

# All supported authentication modules
SUPPORTED_AUTHENTICATION_MODULES = {"k8s", "noop", "noop-with-token"}

# Default configuration file name
DEFAULT_CONFIGURATION_FILE = "olsconfig.yaml"

# Configuration can be dumped into this file
CONFIGURATION_DUMP_FILE_NAME = "configuration.json"

# Environment variable containing configuration file name to override default
# configuration file
CONFIGURATION_FILE_NAME_ENV_VARIABLE = "OLS_CONFIG_FILE"

# Response streaming media types
MEDIA_TYPE_TEXT = "text/plain"
MEDIA_TYPE_JSON = "application/json"

# default value for token when no token is provided
NO_USER_TOKEN = ""

# quota limiters constants
USER_QUOTA_LIMITER = "user_limiter"
CLUSTER_QUOTA_LIMITER = "cluster_limiter"

# MCP transport types
MCP_TRANSPORT_STDIO = "stdio"
MCP_TRANSPORT_SSE = "sse"
SSE_TRANSPORT_DEFAULT_TIMEOUT = 5  # in seconds
SSE_TRANSPORT_DEFAULT_READ_TIMEOUT = 10  # in seconds
STDIO_TRANSPORT_DEFAULT_ENCODING = "utf-8"
STDIO_TRANSPORT_DEFAULT_ENV: dict[str, str | int] = {}
STDIO_TRANSPORT_DEFAULT_CWD = "."
STREAMABLE_HTTP_TRANSPORT_DEFAULT_TIMEOUT = 5  # in seconds
STREAMABLE_HTTP_TRANSPORT_DEFAULT_READ_TIMEOUT = 10  # in seconds

# timeout value for a single llm with tools round
# Keeping it really high at this moment (until this is configurable)
TOOL_CALL_ROUND_TIMEOUT = 300
