# Put targets here if there is a risk that a target name might conflict with a filename.
# this list is probably overkill right now.
# See: https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
.PHONY: test test-unit test-e2e images run format verify

ARTIFACT_DIR := $(if $(ARTIFACT_DIR),$(ARTIFACT_DIR),tests/test_results)
TEST_TAGS := $(if $(TEST_TAGS),$(TEST_TAGS),"")
SUITE_ID := $(if $(SUITE_ID),$(SUITE_ID),"nosuite")
PROVIDER := $(if $(PROVIDER),$(PROVIDER),"openai")
MODEL := $(if $(MODEL),$(MODEL),"gpt-4o-mini")
INTROSPECTION_ENABLED := $(if $(INTROSPECTION_ENABLED),$(INTROSPECTION_ENABLED),"n")
PATH_TO_PLANTUML := ~/bin

# Python registry to where the package should be uploaded
PYTHON_REGISTRY = testpypi

default: help

images: ## Build container images
	scripts/build-container.sh

install-tools:	install-woke ## Install required utilities/tools
	# OLS 1085: Service build failure issue caused by newest PDM version
	# (right now we need to stick to PDM specified in pyproject.toml file)
	@command -v pdm > /dev/null || { echo >&2 "pdm is not installed. Installing..."; pip install pdm; }
	pdm --version
	# this is quick fix for OLS-758: "Verify" CI job is broken after new Mypy 1.10.1 was released 2 days ago
	# CI job configuration would need to be updated in follow-up task
	# pip uninstall -v -y mypy 2> /dev/null || true
	# display setuptools version
	pip show setuptools
	export PIP_DEFAULT_TIMEOUT=100
	# install all dependencies, including devel ones
	@for a in 1 2 3 4 5; do pdm install --group default,dev,evaluation --fail-fast -v && break || sleep 15; done
	# check that correct mypy version is installed
	# mypy --version
	pdm run mypy --version
	# check that correct Black version is installed
	pdm run black --version
	# check that correct Ruff version is installed
	pdm run ruff --version
	# check that correct Pydocstyle version is installed
	pdm run pydocstyle --version


install-woke: ## Install woke, required for Inclusive Naming scan
	@command -v ./woke > /dev/null || { echo >&2 "woke is not installed. Installing..."; curl -sSfL https://git.io/getwoke | bash -s -- -b ./; }

pdm-lock-check: ## Check that the pdm.lock file is in a good shape
	pdm lock --check

install-deps: install-tools pdm-lock-check ## Install all required dependencies needed to run the service, according to pdm.lock
	@for a in 1 2 3 4 5; do pdm sync && break || sleep 15; done

install-deps-test: install-tools pdm-lock-check ## Install all required dev dependencies needed to test the service, according to pdm.lock
	@for a in 1 2 3 4 5; do pdm sync --dev && break || sleep 15; done

update-deps: ## Check pyproject.toml for changes, update the lock file if needed, then sync.
	pdm update

run: ## Run the service locally
	python runner.py

print-version: ## Print the service version
	python runner.py --version

test: test-e2e ## Run all tests

test-e2e: ## Run e2e tests - requires running OLS server
	@echo "Running e2e tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	pdm run pytest tests/e2e --ignore=tests/e2e/evaluation -s --durations=0 -o junit_suite_name="${SUITE_ID}" -m "${TEST_TAGS}" --junit-prefix="${SUITE_ID}" --junit-xml="${ARTIFACT_DIR}/junit_e2e_${SUITE_ID}.xml" \
	--eval_provider ${PROVIDER} --eval_model ${MODEL} --eval_out_dir ${ARTIFACT_DIR} --rp_name=ols-e2e-tests

test-eval: ## Run evaluation tests - requires running OLS server
	@echo "Running evaluation tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	pdm run pytest tests/e2e/evaluation -vv -s --durations=0 -o junit_suite_name="${SUITE_ID}" --junit-prefix="${SUITE_ID}" --junit-xml="${ARTIFACT_DIR}/junit_e2e_${SUITE_ID}.xml" \
	--eval_out_dir ${ARTIFACT_DIR}


format: ## Format the code into unified format
	pdm run black .
	pdm run ruff check . --fix

verify:	install-woke install-deps-test ## Verify the code using various linters
	pdm run black . --check
	pdm run ruff check .
	./woke . --exit-1-on-failure
	pylint our_ols scripts tests runner.py
	pdm run mypy --explicit-package-bases --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs our_ols/

requirements.txt:	pyproject.toml pdm.lock ## Generate requirements.txt file containing hashes for all non-devel packages
	pdm export --prod --format requirements --output requirements.txt --no-extras --without evaluation

get-rag: ## Download a copy of the RAG embedding model and vector database
	podman create --replace --name tmp-rag-container $$(grep 'ARG LIGHTSPEED_RAG_CONTENT_IMAGE' Containerfile | awk 'BEGIN{FS="="}{print $$2}') true
	rm -rf vector_db embeddings_model
	podman cp tmp-rag-container:/rag/vector_db vector_db
	podman cp tmp-rag-container:/rag/embeddings_model embeddings_model
	podman rm tmp-rag-container

help: ## Show this help screen
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-33s\033[0m %s\n", $$1, $$2}'
	@echo ''
