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
pdm install
`pdm venv activate`
make run
```

### Install Evaluation Framework

```bash
# In eval/ directory
python3 -m venv venv
source venv/bin/activate
pip install git+https://github.com/lightspeed-core/lightspeed-evaluation.git
```

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
```

## What's Included

### Datasets
- **`eval_data_short.yaml`**: 10 conversations
- **`eval_data.yaml`**: 797 conversations

### Configuration
- **`system.yaml`**: Pre-configured for OLS at `localhost:8080`
- **Default metrics**: answer correctness


## Results

Results are saved in output directories:
- `evaluation_*_detailed.csv`
- `evaluation_*_summary.json`
- `graphs/` - Visualization charts


## Data & Eval system setup
Refer [Lightspeed Evaluation tool](https://github.com/lightspeed-core/lightspeed-evaluation#readme)