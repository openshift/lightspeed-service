# Put targets here if there is a risk that a target name might conflict with a filename.
# this list is probably overkill right now.
# See: https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
.PHONY: test test-unit test-e2e images run format verify

ARTIFACT_DIR := $(if $(ARTIFACT_DIR),$(ARTIFACT_DIR),tests/test_results)
TEST_TAGS := $(if $(TEST_TAGS),$(TEST_TAGS),"")
SUITE_ID := $(if $(SUITE_ID),$(SUITE_ID),"nosuite")
PROVIDER := $(if $(PROVIDER),$(PROVIDER),"openai")
MODEL := $(if $(MODEL),$(MODEL),"gpt-4o-mini")

# Python registry to where the package should be uploaded
PYTHON_REGISTRY = testpypi


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
	pdm install --dev --fail-fast -v
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
	pdm sync

install-deps-test: install-tools pdm-lock-check ## Install all required dev dependencies needed to test the service, according to pdm.lock
	pdm sync --dev

update-deps: ## Check pyproject.toml for changes, update the lock file if needed, then sync.
	pdm install
	pdm install --dev

run: ## Run the service locally
	python runner.py

print-version: ## Print the service version
	python runner.py --version

test: test-unit test-integration test-e2e ## Run all tests

benchmarks: ## Run benchmarks
	@echo "Running benchmarks..."
	python -m pytest tests/benchmarks --benchmark-histogram

test-unit: ## Run the unit tests
	@echo "Running unit tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	export NO_CUDA=1
	export CUDA_VISIBLE_DEVICES=""
	NO_CUDE=1 CUDA_VISIBLE_DEVICES="" COVERAGE_FILE="${ARTIFACT_DIR}/.coverage.unit" python -m pytest tests/unit --cov=ols --cov=runner --cov-report term-missing --cov-report "json:${ARTIFACT_DIR}/coverage_unit.json" --junit-xml="${ARTIFACT_DIR}/junit_unit.xml"
	python scripts/transform_coverage_report.py "${ARTIFACT_DIR}/coverage_unit.json" "${ARTIFACT_DIR}/coverage_unit.out"
	scripts/codecov.sh "${ARTIFACT_DIR}/coverage_unit.out"

test-integration: ## Run integration tests tests
	@echo "Running integration tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	COVERAGE_FILE="${ARTIFACT_DIR}/.coverage.integration" python -m pytest -m 'not redis' tests/integration --cov=ols --cov=runner --cov-report term-missing --cov-report "json:${ARTIFACT_DIR}/coverage_integration.json" --junit-xml="${ARTIFACT_DIR}/junit_integration.xml" --cov-fail-under=60
	python scripts/transform_coverage_report.py "${ARTIFACT_DIR}/coverage_integration.json" "${ARTIFACT_DIR}/coverage_integration.out"
	scripts/codecov.sh "${ARTIFACT_DIR}/coverage_integration.out"

check-coverage: test-unit test-integration  ## Unit tests and integration tests overall code coverage check
	coverage combine --keep "${ARTIFACT_DIR}/.coverage.unit" "${ARTIFACT_DIR}/.coverage.integration"
	# the threshold should be very high there, in theory it should reach 100%
	coverage report -m --fail-under=94

test-e2e: ## Run e2e tests - requires running OLS server
	@echo "Running e2e tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	python -m pytest tests/e2e -s --durations=0 -o junit_suite_name="${SUITE_ID}" -m "${TEST_TAGS}" --junit-prefix="${SUITE_ID}" --junit-xml="${ARTIFACT_DIR}/junit_e2e_${SUITE_ID}.xml" \
	--eval_provider ${PROVIDER} --eval_model ${MODEL} --eval_out_dir ${ARTIFACT_DIR} --rp_name=ols-e2e-tests

coverage-report:	unit-tests-coverage-report integration-tests-coverage-report ## Export coverage reports into interactive HTML

unit-tests-coverage-report:	test-unit ## Export unit test coverage report into interactive HTML
	coverage html --data-file="${ARTIFACT_DIR}/.coverage.unit" -d htmlcov-unit

integration-tests-coverage-report:	test-integration ## Export integration test coverage report into interactive HTML
	coverage html --data-file="${ARTIFACT_DIR}/.coverage.integration" -d htmlcov-integration

check-types: ## Checks type hints in sources
	pdm run mypy --explicit-package-bases --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs ols/

security-check: ## Check the project for security issues
	bandit -c pyproject.toml -r .

format: ## Format the code into unified format
	pdm run black .
	pdm run ruff check . --fix --per-file-ignores=tests/*:S101 --per-file-ignores=scripts/*:S101

verify:	install-woke install-deps-test ## Verify the code using various linters
	pdm run black . --check
	pdm run ruff check . --per-file-ignores=tests/*:S101 --per-file-ignores=scripts/*:S101
	./woke . --exit-1-on-failure

schema:	## Generate OpenAPI schema file
	python scripts/generate_openapi_schema.py docs/openapi.json

requirements.txt:	pyproject.toml pdm.lock ## Generate requirements.txt file containing hashes for all non-devel packages
	pdm export --prod --format requirements --no-extras --output requirements.txt

verify-packages-completeness:	requirements.txt ## Verify that requirements.txt file contains complete list of packages
	pip download -d /tmp/ --use-pep517 --verbose -r requirements.txt

get-rag: ## Download a copy of the RAG embedding model and vector database
	podman create --replace --name tmp-rag-container $$(cat build.args | awk 'BEGIN{FS="="}{print $$2}') true
	rm -rf vector_db embeddings_model
	podman cp tmp-rag-container:/rag/vector_db vector_db
	podman cp tmp-rag-container:/rag/embeddings_model embeddings_model
	podman rm tmp-rag-container

config.puml: ## Generate PlantUML class diagram for configuration
	pyreverse ols/app/models/config.py --output puml --output-directory=docs/
	mv docs/classes.puml docs/config.puml

llms.puml: ## Generate PlantUML class diagram for LLM plugin system
	pyreverse ols/src/llms/ --output puml --output-directory=docs
	mv docs/classes.puml docs/llms_classes.uml

distribution-archives: ## Generate distribution archives to be uploaded into Python registry
	pdm run python -m build

upload-distribution-archives: ## Upload distribution archives into Python registry
	pdm run python -m twine upload --repository ${PYTHON_REGISTRY} dist/*

help: ## Show this help screen
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-33s\033[0m %s\n", $$1, $$2}'
	@echo ''
