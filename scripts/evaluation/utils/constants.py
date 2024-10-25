"""Constants."""

# Same Provider/Model combination must be used while launching OLS.
INSCOPE_MODELS = {
    "bam+ibm/granite-13b-chat-v2": ("bam", "ibm/granite-13b-chat-v2"),
    "watsonx+ibm/granite-13b-chat-v2": ("watsonx", "ibm/granite-13b-chat-v2"),
    "openai+gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "azure_openai+gpt-4o-mini": ("azure_openai", "gpt-4o-mini"),
    "azure_openai+gpt-4o": ("azure_openai", "gpt-4o"),
}

SCORE_DESCRIPTION = {
    "cos_score": "Cosine Similarity Score (mpnet)",  # with mpnet sentence embedding model
    "euc_score": "Euclidean Distance Score (mpnet)",  # with mpnet sentence embedding model
    "len_score": "Character length delta Score",
    "rougeL_precision": "RougeL Precision Score",
    "rougeL_recall": "RougeL Recall Score",
    "rougeL_f1": "RougeL F1 Score",
    "answer_relevancy": "Answer relevancy score against query",
}

EVAL_MODES = {
    "vanilla",  # Vanilla model (with default parameters, no prompt/rag)
    "ols_param",  # Model with OLS parameters (no prompt/rag)
    "ols_prompt",  # Model with OLS prompt (default parameters, no rag)
    "ols_rag",  # Model with RAG (default parameters & simple prompt/format)
    "ols",  # With OLS API
}

# Use below as system instruction to override OLS instruction.
# Instruction for RAG/Context is still used with proper document format.
BASIC_PROMPT = """
You are a helpful assistant.
"""

DEFAULT_QNA_FILE = "question_answer_pair.json"
DEFAULT_CONFIG_FILE = "olsconfig.yaml"

DEFAULT_INPUT_DIR = "eval_data"
DEFAULT_RESULT_DIR = "eval_result"

# Retry settings for LLM calls used when model does not respond reliably in 100% cases
MAX_RETRY_ATTEMPTS = 10
REST_API_TIMEOUT = 120
TIME_TO_BREATH = 10

# Cut-off similarity score used for response evaluation.
EVAL_THRESHOLD = 0.3  # low score is better
