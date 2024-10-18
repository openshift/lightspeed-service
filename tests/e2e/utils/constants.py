"""Constants for end-to-end tests."""

# timeout settings
BASIC_ENDPOINTS_TIMEOUT = 5
NON_LLM_REST_API_TIMEOUT = 20
LLM_REST_API_TIMEOUT = 90
CONVERSATION_ID = "12345678-abcd-0000-0123-456789abcdef"

# constant from tests/config/cluster_install/ols_manifests.yaml
OLS_USER_DATA_PATH = "/app-root/ols-user-data"
OLS_USER_DATA_COLLECTION_INTERVAL = 10
OLS_COLLECTOR_DISABLING_FILE = OLS_USER_DATA_PATH + "/disable_collector"
