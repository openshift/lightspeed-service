# Put targets here if there is a risk that a target name might conflict with a filename.
# this list is probably overkill right now.
# See: https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
.PHONY: test test-unit test-e2e images run format verify

ARTIFACT_DIR := $(if $(ARTIFACT_DIR),$(ARTIFACT_DIR),tests/test_results)
TEST_TAGS := $(if $(TEST_TAGS),$(TEST_TAGS),"")
SUITE_ID := $(if $(SUITE_ID),$(SUITE_ID),"nosuite")
PROVIDER := $(if $(PROVIDER),$(PROVIDER),"openai")
MODEL := $(if $(MODEL),$(MODEL),"gpt-4o-mini")
OLS_CONFIG_SUFFIX := $(if $(OLS_CONFIG_SUFFIX),$(OLS_CONFIG_SUFFIX),"default")
PATH_TO_PLANTUML := ~/bin

# Python registry to where the package should be uploaded
PYTHON_REGISTRY = testpypi

default: help

images: ## Build container images
	scripts/build-container.sh

install-tools:	install-woke ## Install required utilities/tools
	@command -v uv > /dev/null || { echo >&2 "uv is not installed. Installing..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv --version
	# install all dependencies, including devel ones
	uv sync --group dev --extra evaluation
	# check that correct mypy version is installed
	uv run mypy --version
	# check that correct Black version is installed
	uv run black --version
	# check that correct Ruff version is installed
	uv run ruff --version
	# check that correct Pydocstyle version is installed
	uv run pydocstyle --version


install-woke: ## Install woke, required for Inclusive Naming scan
	@command -v ./woke > /dev/null || { echo >&2 "woke is not installed. Installing..."; curl -sSfL https://git.io/getwoke | bash -s -- -b ./; }

uv-lock-check: ## Check that the uv.lock file is in a good shape
	uv lock --check

install-deps: install-tools uv-lock-check ## Install all required dependencies needed to run the service, according to uv.lock
	@for a in 1 2 3 4 5; do uv sync && break || sleep 15; done

install-deps-test: install-tools uv-lock-check ## Install all required dev dependencies needed to test the service, according to uv.lock
	@for a in 1 2 3 4 5; do uv sync --group dev && break || sleep 15; done

update-deps: ## Check pyproject.toml for changes, update the lock file if needed, then sync.
	uv lock --upgrade && uv sync

run: ## Run the service locally
	uv run python runner.py

print-version: ## Print the service version
	uv run python runner.py --version

test: test-unit test-integration test-e2e ## Run all tests

benchmarks: ## Run benchmarks
	@echo "Running benchmarks..."
	uv run pytest tests/benchmarks --benchmark-histogram

test-unit: ## Run the unit tests
	@echo "Running unit tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	COVERAGE_FILE="${ARTIFACT_DIR}/.coverage.unit" uv run pytest tests/unit tests/mcp_local --cov=ols --cov=mcp_local --cov=runner --cov-report term-missing --cov-report "json:${ARTIFACT_DIR}/coverage_unit.json" --junit-xml="${ARTIFACT_DIR}/junit_unit.xml"
	uv run scripts/transform_coverage_report.py "${ARTIFACT_DIR}/coverage_unit.json" "${ARTIFACT_DIR}/coverage_unit.out"
	scripts/codecov.sh "${ARTIFACT_DIR}/coverage_unit.out"

test-integration: ## Run integration tests tests
	@echo "Running integration tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	COVERAGE_FILE="${ARTIFACT_DIR}/.coverage.integration" uv run pytest tests/integration --cov=ols --cov=runner --cov-report term-missing --cov-report "json:${ARTIFACT_DIR}/coverage_integration.json" --junit-xml="${ARTIFACT_DIR}/junit_integration.xml" --cov-fail-under=60
	uv run scripts/transform_coverage_report.py "${ARTIFACT_DIR}/coverage_integration.json" "${ARTIFACT_DIR}/coverage_integration.out"
	scripts/codecov.sh "${ARTIFACT_DIR}/coverage_integration.out"

check-coverage: test-unit test-integration  ## Unit tests and integration tests overall code coverage check
	uv run coverage combine --keep "${ARTIFACT_DIR}/.coverage.unit" "${ARTIFACT_DIR}/.coverage.integration"
	# the threshold should be very high there, in theory it should reach 100%
	uv run coverage report -m --fail-under=94

test-e2e: ## Run e2e tests - requires running OLS server
	@echo "Running e2e tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	uv run pytest tests/e2e --ignore=tests/e2e/evaluation -s --durations=0 -o junit_suite_name="${SUITE_ID}" -m "${TEST_TAGS}" --junit-prefix="${SUITE_ID}" --junit-xml="${ARTIFACT_DIR}/junit_e2e_${SUITE_ID}.xml" \
	--eval_provider ${PROVIDER} --eval_model ${MODEL} --eval_out_dir ${ARTIFACT_DIR} --rp_name=ols-e2e-tests

test-eval: ## Run evaluation tests - requires running OLS server
	@echo "Running evaluation tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	uv run pytest tests/e2e/evaluation -vv -s --durations=0 -o junit_suite_name="${SUITE_ID}" --junit-prefix="${SUITE_ID}" --junit-xml="${ARTIFACT_DIR}/junit_e2e_${SUITE_ID}.xml" \
	--eval_out_dir ${ARTIFACT_DIR}

coverage-report:	unit-tests-coverage-report integration-tests-coverage-report ## Export coverage reports into interactive HTML

unit-tests-coverage-report:	test-unit ## Export unit test coverage report into interactive HTML
	uv run coverage html --data-file="${ARTIFACT_DIR}/.coverage.unit" -d htmlcov-unit

integration-tests-coverage-report:	test-integration ## Export integration test coverage report into interactive HTML
	uv run coverage html --data-file="${ARTIFACT_DIR}/.coverage.integration" -d htmlcov-integration

check-types: ## Checks type hints in sources
	uv run mypy --explicit-package-bases --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs ols/

security-check: ## Check the project for security issues
	uv run bandit -c pyproject.toml -r .

format: ## Format the code into unified format
	uv run black .
	uv run ruff check . --fix

verify:	install-woke install-deps-test ## Verify the code using various linters
	uv run black . --check
	uv run ruff check .
	./woke . --exit-1-on-failure
	uv run --extra evaluation pylint ols scripts tests runner.py
	uv run mypy --explicit-package-bases --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs ols/

schema:	## Generate OpenAPI schema file
	uv run python scripts/generate_openapi_schema.py docs/openapi.json

requirements.txt:	pyproject.toml uv.lock ## Generate requirements.txt file containing hashes for all non-devel packages
	uv export --format requirements.txt --no-dev --no-extra evaluation --output-file requirements.txt

verify-packages-completeness:	requirements.txt ## Verify that requirements.txt file contains complete list of packages
	pip download -d /tmp/ --use-pep517 --verbose -r requirements.txt

get-rag: ## Download a copy of the RAG embedding model and vector database
	podman create --replace --name tmp-rag-container $$(grep 'ARG LIGHTSPEED_RAG_CONTENT_IMAGE' Containerfile | awk 'BEGIN{FS="="}{print $$2}') true
	rm -rf vector_db embeddings_model
	podman cp tmp-rag-container:/rag/vector_db vector_db
	podman cp tmp-rag-container:/rag/embeddings_model embeddings_model
	podman rm tmp-rag-container

config.puml: ## Generate PlantUML class diagram for configuration
	pyreverse ols/app/models/config.py --output puml --output-directory=docs/
	mv docs/classes.puml docs/config.puml

docs/config.png:	docs/config.puml ## Generate an image with configuration graph
	pushd docs && \
	java -jar ${PATH_TO_PLANTUML}/plantuml.jar --theme rose config.puml && \
	mv classes.png config.png && \
	popd

llms.puml: ## Generate PlantUML class diagram for LLM plugin system
	pyreverse ols/src/llms/ --output puml --output-directory=docs/
	mv docs/classes.puml docs/llms_classes.uml
	mv docs/packages.puml docs/llms_packages.uml

distribution-archives: ## Generate distribution archives to be uploaded into Python registry
	uv run python -m build

upload-distribution-archives: ## Upload distribution archives into Python registry
	uv run python -m twine upload --repository ${PYTHON_REGISTRY} dist/*

shellcheck: ## Run shellcheck
	wget -qO- "https://github.com/koalaman/shellcheck/releases/download/stable/shellcheck-stable.linux.x86_64.tar.xz" | tar -xJv \
	shellcheck --version
	shellcheck -- */*.sh

help: ## Show this help screen
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-33s\033[0m %s\n", $$1, $$2}'
	@echo ''
