# OpenShift Lightspeed - Evaluation

Evaluate OLS response quality using the [lightspeed-evaluation tool](https://github.com/lightspeed-core/lightspeed-evaluation).

## Prerequisites

- **Python 3.11+** (required by evaluation framework)
- **OLS service** running (default at `http://localhost:8080`)

```bash
# Check Python version
python3 --version  # Should be 3.11.0 or higher
```

## Setup & Usage

### Start OLS Service

```bash
# In lightspeed-service root (separate terminal)
cd ..
uv sync
make run
```

### Install Evaluation Framework

```bash
# In eval/ directory
python3 -m venv venv
source venv/bin/activate
pip install git+https://github.com/lightspeed-core/lightspeed-evaluation.git
```

### RBAC Setup for Cluster-Updates Evaluation

**IMPORTANT:** Cluster-updates evaluation requires proper RBAC permissions for OLS to access cluster resources via MCP servers.

```bash
# Apply the RBAC manifest (creates ServiceAccount with cluster-reader + monitoring-edit permissions)
oc apply -f eval/rbac-cluster-updates.yaml

# Generate an API token for the service account (valid for 24 hours)
export API_KEY=$(oc create token cluster-update-user -n openshift-lightspeed --duration=24h)
```

**Required Permissions:**

- **cluster-reader**: Access to ClusterVersion, ClusterOperator, Node resources
- **monitoring-edit**: Access to Prometheus metrics and Alertmanager alerts

**Why These Permissions Matter:**

Without proper RBAC, OLS will respond with "unable to retrieve... due to access restrictions" instead of providing accurate cluster status analysis. The evaluation will still run but will fail quality checks with ~79% pass rate instead of the expected 85%+.

### Run Evaluation

```bash
# In eval/ directory

# Activate environment
source venv/bin/activate

# Setup Judge-LLM & OLS API env variables
export API_KEY="your-api-endpoint-key"  # for OLS
# for judgeLLM, based on the provider used in system.yaml. refer lightspeed-evaluation for other providers.
export OPENAI_API_KEY="your-evaluation-judge-llm-key"  # example for OpenAI

# Small dataset (10 questions)
lightspeed-eval --system-config system.yaml --eval-data eval_data_short.yaml --output-dir ./results

# Full evaluation (797 questions)
lightspeed-eval --system-config system.yaml --eval-data eval_data.yaml --output-dir ./results

# Cluster-updates evaluation (19 conversations, 20 test turns) - uses optimized config
lightspeed-eval --system-config system_cluster_updates.yaml \
                --eval-data eval_data_cluster_updates.yaml \
                --output-dir ./results

# Run specific cluster-updates test category (e.g., critical tests)
lightspeed-eval --system-config system_cluster_updates.yaml \
                --eval-data eval_data_cluster_updates.yaml \
                --tags cluster-updates-critical \
                --output-dir ./results
```

## What's Included

### Datasets
- **`eval_data_short.yaml`**: 10 conversations (quick smoke test)
- **`eval_data.yaml`**: 797 general OpenShift knowledge questions (conv_001-797)
- **`eval_data_cluster_updates.yaml`**: 19 cluster-updates test conversations (conv_001-019, 20 test turns)

### Test Categories (by tag)
- **cluster-updates-scenarios**: Comprehensive health assessment with extensive constraints (conv_001-005, 5 conversations)
- **cluster-updates-critical**: Condition status interpretation - MUST pass 100% (conv_006)
- **cluster-updates-format**: Output format compliance (Summary + TL;DR) (conv_007)
- **cluster-updates-blockers**: Admin-ack gates and upgrade blockers (conv_008)
- **cluster-updates-risks**: Conditional update risk analysis (conv_009)
- **cluster-updates-path**: Upgrade path validation (conv_010)
- **cluster-updates-troubleshoot**: Upgrade failure diagnosis and remediation (conv_011)
- **cluster-updates-conversation**: Multi-turn conversation handling (conv_012, 2 turns)
- **cluster-updates-no-updates**: Cluster at latest version scenarios (conv_013)
- **cluster-updates-channels**: Update channel understanding (conv_014)
- **cluster-updates-mcp**: MachineConfigPool upgrade behavior (conv_015)
- **cluster-updates-pdb**: PodDisruptionBudget impact on upgrades (conv_016)
- **cluster-updates-eus**: Extended Update Support (EUS) upgrade paths (conv_017)
- **cluster-updates-conditions**: Condition status interpretation (conv_018, conv_019)

### Configuration Files

Two configuration files are available depending on your use case:

#### `system.yaml` - Default Configuration
- **Use for:** General OpenShift knowledge evaluation (conv_001-797)
- **API Base:** `http://localhost:8080` (local development)
- **Max Tokens:** 512 (standard responses)
- **API Provider:** `openai`
- **Metrics:** All standard metrics available (Ragas, DeepEval, custom)

#### `system_cluster_updates.yaml` - Cluster-Updates Optimized
- **Use for:** Cluster-updates evaluation (conv_001-019)
- **API Base:** `http://localhost:8080` (same as default)
- **Max Tokens:** 2048 (detailed cluster analysis - 4x larger for complex responses)
- **API Provider:** `openai` (cluster-specific configuration)
- **Output Directory:** `./results` (organized test output)
- **Available Metrics:**
  - `custom:answer_correctness` - Basic correctness evaluation
  - `geval:condition_status_accuracy` - Kubernetes condition interpretation (threshold: 0.99 - CRITICAL!)
  - `geval:output_format_compliance` - Response format validation (threshold: 0.80)
  - `geval:technical_accuracy` - OpenShift/Kubernetes domain knowledge (threshold: 0.80)
  - `geval:actionable_guidance` - Specific remediation steps (threshold: 0.7)


## Results

Results are saved in output directories:
- `evaluation_*_detailed.csv`
- `evaluation_*_summary.json`
- `graphs/` - Visualization charts


## Data & Eval system setup
Refer [Lightspeed Evaluation tool](https://github.com/lightspeed-core/lightspeed-evaluation#readme)
