"""Constants for end-to-end tests."""
import os

# timeout settings
BASIC_ENDPOINTS_TIMEOUT = 5
NON_LLM_REST_API_TIMEOUT = 20
LLM_REST_API_TIMEOUT = 90
CONVERSATION_ID = "12345678-abcd-0000-0123-456789abcdef"

# constant from tests/config/cluster_install/ols_manifests.yaml
OLS_USER_DATA_PATH = "/app-root/ols-user-data"

# Collection intervals for data exporter
# Set once at test suite startup to 3600s (1 hour) so no data is sent during tests
# For the data collection test specifically, we prune the dir, reset to 5s, and test
OLS_USER_DATA_COLLECTION_INTERVAL_LONG = (
    3600  # 1 hour - set at startup, prevents data collection
)
OLS_USER_DATA_COLLECTION_INTERVAL_SHORT = (
    5  # 5 seconds - used only in data collection test
)

OLS_SERVICE_DEPLOYMENT = "lightspeed-stack-deployment" if os.getenv("LCORE", 'False').lower() in ('true', '1', 't') else "lightspeed-app-server"
