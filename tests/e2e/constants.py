"""Constants for end-to-end tests."""

# timeout settings
BASIC_ENDPOINTS_TIMEOUT = 5
NON_LLM_REST_API_TIMEOUT = 20
LLM_REST_API_TIMEOUT = 90
CONVERSATION_ID = "12345678-abcd-0000-0123-456789abcdef"

# TODO: Currently threshold score is very high. Practically not helpful.
# But as we are changing embedding model, We will have to revisit
# Predefined QnA, Score calculation based on new embedding.
# Vector representation varies with different model. And this will have
# impact on distance metrics.
EVAL_THRESHOLD = 0.7  # low score is better
