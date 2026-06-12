# Usage:
#   make setup-env                    # Install Istio + Kiali + Bookinfo (once per cluster)
#   make all                          # Run all Kiali/OSSM evals
#   make check_mesh_status            # Run a single eval by tag
#
# Prerequisites:
#   make setup-lseval                 # once — isolated venv (see below)
#   oc login / kubectl ctx            # cluster with admin access
#   OPENAI_API_KEY or openai_api_key.txt at repo root — judge LLM
#   OLS running with kubernetes-mcp-server (default api_base http://localhost:8080)
#
# setup-lseval uses a dedicated .lseval-venv: the repo's lseval extra (v0.4.0) does not
# support api.extra_request_params (mode: troubleshooting), and newer lseval pins conflict
# with OLS langchain deps — so eval CLI is installed separately.

EVAL_DIR    := $(dir $(lastword $(MAKEFILE_LIST)))
REPO_ROOT   := $(abspath $(EVAL_DIR)../..)
LSEVAL_VENV := $(EVAL_DIR).lseval-venv
LSEVAL      := $(LSEVAL_VENV)/bin/lightspeed-eval
LSEVAL_GIT  := git+https://github.com/lightspeed-core/lightspeed-evaluation.git@32c80e1c9555ccc3f3573bc073cd5a4c6595c414
OPENAI_KEY_FILE ?= $(REPO_ROOT)/openai_api_key.txt

ISTIOCTL      = $(EVAL_DIR)_output/bin/istioctl
ISTIO_ADDONS  = $(EVAL_DIR)_output/istio-addons
ISTIO_SAMPLES = $(EVAL_DIR)_output/istio-samples
ISTIO_VERSION ?= 1.28.0
KIALI_VERSION ?= v2.27.0
# Pinned commit for the sail-operator ingress-gateway sample manifest.
# Update by running: git ls-remote https://github.com/istio-ecosystem/sail-operator HEAD
SAIL_OPERATOR_COMMIT ?= eb874d1605c0401887c30d96312c6db1006d89dd

.PHONY: all setup-lseval check-lseval \
        setup-env istioctl install-istio install-bookinfo update-kiali \
        check_mesh_status check_mesh_status_no_kiali \
        check_bookinfo_services check_latency_bookinfo_issue \
        fix_bookinfo_routing fix_bookinfo_fault_injection \
        troubleshoot_latency_trace

check-lseval:
	@test -x "$(LSEVAL)" || (echo "Run: make setup-lseval" >&2; exit 1)

ENV_TARGETS := setup-env istioctl install-istio install-bookinfo update-kiali
ifneq ($(MAKECMDGOALS),)
$(filter-out setup-lseval check-lseval $(ENV_TARGETS),$(MAKECMDGOALS)): check-lseval
endif

setup-lseval:
	@command -v uv >/dev/null || (echo "Install uv: https://docs.astral.sh/uv/" >&2; exit 1)
	uv venv --python 3.12 "$(LSEVAL_VENV)"
	uv pip install --python "$(LSEVAL_VENV)/bin/python" -U pip "$(LSEVAL_GIT)" 'langchain-community<0.4'

##@ Environment setup (Istio + Kiali + Bookinfo)

# Download the pinned Istio release tarball, verify its SHA256, then extract
# istioctl and the addon/bookinfo samples into _output/.
# Uses the GitHub release tarball directly — no curl-pipe-to-shell.
istioctl:
	@{ \
		set -e; \
		ARCH=$$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/'); \
		OS=$$(uname -s | tr '[:upper:]' '[:lower:]'); \
		TARBALL=istio-$(ISTIO_VERSION)-$$OS-$$ARCH.tar.gz; \
		BASE_URL=https://github.com/istio/istio/releases/download/$(ISTIO_VERSION); \
		echo "Downloading $$TARBALL from $$BASE_URL ..."; \
		mkdir -p $(shell dirname $(ISTIOCTL)); \
		TMPDIR=$$(mktemp -d); \
		curl -fsSL -o "$$TMPDIR/$$TARBALL"        "$$BASE_URL/$$TARBALL"; \
		curl -fsSL -o "$$TMPDIR/$$TARBALL.sha256" "$$BASE_URL/$$TARBALL.sha256"; \
		cd "$$TMPDIR"; \
		SHASUM=$$(command -v sha256sum 2>/dev/null || echo "shasum -a 256"); \
		$$SHASUM -c "$$TARBALL.sha256"; \
		tar xzf "$$TARBALL"; \
		ISTIODIR=$$(ls -d istio-* | head -n1); \
		cp "$$ISTIODIR/bin/istioctl" $(ISTIOCTL); \
		mkdir -p $(ISTIO_ADDONS) $(ISTIO_SAMPLES); \
		cp "$$ISTIODIR/samples/addons/jaeger.yaml" \
		   "$$ISTIODIR/samples/addons/prometheus.yaml" \
		   "$$ISTIODIR/samples/addons/kiali.yaml" \
		   $(ISTIO_ADDONS)/; \
		cp -r "$$ISTIODIR/samples/bookinfo" $(ISTIO_SAMPLES)/; \
		sed -i '/ tracing:/,/ identity:/ { s/ enabled: false/ enabled: true\n        in_cluster_url: "http:\/\/tracing.istio-system:16685\/jaeger"\n        use_grpc: true/ }' \
		   $(ISTIO_ADDONS)/kiali.yaml; \
		cd - >/dev/null; \
		rm -rf "$$TMPDIR"; \
	}

# Install Istio (demo profile) + Prometheus + Kiali + Jaeger, enable sidecar injection
install-istio: istioctl
	$(ISTIOCTL) install --set profile=demo \
		--set meshConfig.defaultConfig.tracing.zipkin.address=zipkin.istio-system:9411 \
		-y
	kubectl apply -f $(ISTIO_ADDONS)/prometheus.yaml -n istio-system
	kubectl apply -f $(ISTIO_ADDONS)/kiali.yaml -n istio-system
	kubectl apply -f $(ISTIO_ADDONS)/jaeger.yaml -n istio-system
	kubectl wait --namespace istio-system --for=condition=available deployment/kiali --timeout=300s
	kubectl wait --namespace istio-system --for=condition=available deployment/prometheus --timeout=300s
	kubectl wait --for=condition=Ready pod --all -n istio-system --timeout=300s
	kubectl label namespace default istio-injection=enabled --overwrite

# Patch Kiali image to the pinned version
update-kiali:
	kubectl patch deployment kiali -n istio-system \
		-p '{"spec":{"template":{"spec":{"containers":[{"name":"kiali","image":"quay.io/kiali/kiali:$(KIALI_VERSION)"}]}}}}'
	kubectl delete pod -l app=kiali -n istio-system
	kubectl wait --for=condition=available deployment/kiali -n istio-system --timeout=300s

# Deploy Bookinfo using manifests from the already-downloaded, checksum-verified
# Istio release tarball. The sail-operator ingress-gateway is fetched at the
# pinned commit SHA defined by SAIL_OPERATOR_COMMIT.
install-bookinfo:
	kubectl create ns bookinfo --dry-run=client -o yaml | kubectl apply -f -
	kubectl label namespace bookinfo istio-injection=enabled --overwrite
	kubectl apply -f $(ISTIO_SAMPLES)/bookinfo/platform/kube/bookinfo.yaml -n bookinfo
	kubectl apply -f $(ISTIO_SAMPLES)/bookinfo/networking/bookinfo-gateway.yaml -n bookinfo
	kubectl apply -n bookinfo -f https://raw.githubusercontent.com/istio-ecosystem/sail-operator/$(SAIL_OPERATOR_COMMIT)/chart/samples/ingress-gateway.yaml
	kubectl wait --for=condition=Ready pod --all -n bookinfo --timeout=300s

# Full environment setup: Istio + pinned Kiali + Bookinfo
setup-env: install-istio update-kiali install-bookinfo

##@ Evaluations

EVAL_CMD = @set -e; \
	if [ -z "$$OPENAI_API_KEY" ] && [ -f "$(OPENAI_KEY_FILE)" ]; then \
		export OPENAI_API_KEY=$$(tr -d '\n' < "$(OPENAI_KEY_FILE)"); \
	fi; \
	if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "Set OPENAI_API_KEY or create $(OPENAI_KEY_FILE)" >&2; exit 1; \
	fi; \
	TOKEN=$$(oc whoami -t 2>/dev/null || true); \
	if [ -z "$$TOKEN" ]; then \
		echo "Not logged in to OpenShift; run 'oc login'" >&2; exit 1; \
	fi; \
	API_KEY="$$TOKEN" "$(LSEVAL)" \
		--system-config system.yaml \
		--eval-data scenario_evals.yaml \
		--output-dir results

all:
	$(EVAL_CMD)

check_mesh_status:
	$(EVAL_CMD) --tags check_mesh_status

check_mesh_status_no_kiali:
	$(EVAL_CMD) --tags no_kiali

check_bookinfo_services:
	$(EVAL_CMD) --tags check_bookinfo_services

check_latency_bookinfo_issue:
	$(EVAL_CMD) --tags latency_bookinfo_issue

fix_bookinfo_routing:
	$(EVAL_CMD) --tags fix_bookinfo_routing

fix_bookinfo_fault_injection:
	$(EVAL_CMD) --tags fault_injection_bookinfo

troubleshoot_latency_trace:
	$(EVAL_CMD) --tags troubleshoot_latency_trace
