# Usage:
#   make all                        # Run all NetObserv evals
#   make dns_nxdomain               # Run a single eval by tag
#
# Prerequisites:
#   make setup-lseval               # once — isolated venv (see below)
#   oc login ...                    # OpenShift cluster with NetObserv FlowCollector Ready
#   OPENAI_API_KEY or openai_api_key.txt at repo root — judge LLM
#   OLS running with obs-mcp (default api_base http://localhost:8080)
#
# setup-lseval uses a dedicated .lseval-venv: the repo's lseval extra (v0.4.0) does not
# support api.extra_request_params (mode: troubleshooting), and newer lseval pins conflict
# with OLS langchain deps — so eval CLI is installed separately.

EVAL_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
REPO_ROOT := $(abspath $(EVAL_DIR)../..)
LSEVAL_VENV := $(EVAL_DIR).lseval-venv
LSEVAL := $(LSEVAL_VENV)/bin/lightspeed-eval
LSEVAL_GIT := git+https://github.com/lightspeed-core/lightspeed-evaluation.git@32c80e1c9555ccc3f3573bc073cd5a4c6595c414
OPENAI_KEY_FILE ?= $(REPO_ROOT)/openai_api_key.txt

.PHONY: all setup-lseval check-lseval \
        dns_latency dns_nxdomain packet_drops_kernel packet_drops_policy \
        tls_issues tcp_rtt

check-lseval:
	@test -x "$(LSEVAL)" || (echo "Run: make setup-lseval" >&2; exit 1)

ifneq ($(MAKECMDGOALS),)
$(filter-out setup-lseval check-lseval,$(MAKECMDGOALS)): check-lseval
endif

setup-lseval:
	@command -v uv >/dev/null || (echo "Install uv: https://docs.astral.sh/uv/" >&2; exit 1)
	uv venv --python 3.12 "$(LSEVAL_VENV)"
	uv pip install --python "$(LSEVAL_VENV)/bin/python" -U pip "$(LSEVAL_GIT)" 'langchain-community<0.4'

EVAL_CMD = @set -e; \
	if [ -z "$$OPENAI_API_KEY" ] && [ -f "$(OPENAI_KEY_FILE)" ]; then \
		export OPENAI_API_KEY=$$(tr -d '\n' < "$(OPENAI_KEY_FILE)"); \
	fi; \
	if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "Set OPENAI_API_KEY or create $(OPENAI_KEY_FILE)" >&2; exit 1; \
	fi; \
	API_KEY=$$(oc whoami -t) "$(LSEVAL)" \
		--system-config system.yaml \
		--eval-data scenario_evals.yaml \
		--output-dir results

all: dns_latency dns_nxdomain packet_drops_kernel packet_drops_policy tls_issues tcp_rtt

dns_latency:
	$(EVAL_CMD) --tag dns_latency

dns_nxdomain:
	$(EVAL_CMD) --tag dns_nxdomain

packet_drops_kernel:
	$(EVAL_CMD) --tag packet_drops_kernel

packet_drops_policy:
	$(EVAL_CMD) --tag packet_drops_policy

tls_issues:
	$(EVAL_CMD) --tag tls_issues

tcp_rtt:
	$(EVAL_CMD) --tag tcp_rtt
