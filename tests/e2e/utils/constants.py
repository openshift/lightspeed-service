"""Constants for end-to-end tests."""

# timeout settings
BASIC_ENDPOINTS_TIMEOUT = 5
NON_LLM_REST_API_TIMEOUT = 20
LLM_REST_API_TIMEOUT = 90
CONVERSATION_ID = "12345678-abcd-0000-0123-456789abcdef"

# Cut-off similarity score used for response evaluation.

# TODO: It was observed that granite responses are very consistent for all providers.
# However, this is not same for gpt. Especially for generated YAMLs.
# We are going to add query/provider/model specific cut-offs, this will enable us
# to use separate cut-off values.
EVAL_THRESHOLD = 0.35  # low score is better
