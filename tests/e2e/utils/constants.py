"""Constants for end-to-end tests."""

# timeout settings
BASIC_ENDPOINTS_TIMEOUT = 5
NON_LLM_REST_API_TIMEOUT = 20
LLM_REST_API_TIMEOUT = 90
CONVERSATION_ID = "12345678-abcd-0000-0123-456789abcdef"

# Cut-off similarity score used for response evaluation.
EVAL_THRESHOLD = 0.2  # low score is better
